"""
UR10e no MuJoCo: marcação de evidências forenses.

O robô (com uma caneta no flange) executa três fases:
  1) move cada junta individualmente, mostrando sua faixa de movimento;
  2) faz um movimento coordenado (todas as juntas juntas) do repouso até a parede;
  3) traça círculos de raio R ao redor de marcas de projétil (parede vertical)
     e de fissuras de faca (placa horizontal sobre a mesa).

Uso:
  python main.py                      # abre o visualizador interativo
  python main.py --raio 0.05          # muda o raio dos círculos (metros)
  python main.py --headless --video simulacao.mp4   # grava vídeo sem janela

Referências: Craig, "Robótica" (3a ed.), caps. 2, 3, 5, 7, 9 e 10;
Corke, "Robotics, Vision and Control" (2011), seções 7.4, 7.6, 8.2 e 8.4.
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

AQUI = Path(__file__).resolve().parent
CENA = AQUI / "cena_forense.xml"

NOMES_JUNTAS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Faixas usadas na demonstração de cada junta (rad). None = usa o limite do
# próprio modelo (+-360 graus; cotovelo +-180). Ombro e cotovelo usam uma
# subfaixa porque o braço bateria na parede ou no chão (verificado com
# detecção de colisão do MuJoCo). Altere e confira no visualizador.
FAIXA_DEMO = {
    "shoulder_pan_joint":  None,
    "shoulder_lift_joint": (-3.1, -1.4),
    "elbow_joint":         (-2.7, 2.7),
    "wrist_1_joint":       None,
    "wrist_2_joint":       None,
    "wrist_3_joint":       None,
}

# Velocidades máximas das juntas do UR10e (datasheet): 120 graus/s na base e no
# ombro, 180 graus/s nas demais. Usadas para escolher a duração dos movimentos.
VEL_MAX = np.radians([120, 120, 180, 180, 180, 180])


# ----------------------------------------------------------------------------
# 1. Trajetórias (Craig cap. 7.3 / Corke 3.1.1 e 7.4.1)
# ----------------------------------------------------------------------------
def quintica(t, T):
    """Perfil s(t) de 5a ordem: s(0)=0, s(T)=1, velocidade e aceleração nulas
    nas extremidades. Equivale ao tpoly do Corke."""
    tau = np.clip(t / T, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


# ----------------------------------------------------------------------------
# 2. Cinemática (Craig caps. 2, 3, 5 / Corke 7.2, 8.1, 8.4)
# ----------------------------------------------------------------------------
class Cinematica:
    """Usa uma cópia de MjData só para contas de cinemática, sem física."""

    def __init__(self, m, nome_site="ponta"):
        self.m = m
        self.d = mujoco.MjData(m)
        self.site = m.site(nome_site).id
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.lo = m.jnt_range[:, 0].copy()
        self.hi = m.jnt_range[:, 1].copy()

    def direta(self, q):
        """Cinemática direta: q -> (posição, matriz de rotação) da ponta da caneta."""
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # necessário antes de mj_jacSite
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    @staticmethod
    def erro_pose(p_des, R_des, p, R):
        """Vetor 6x1 [erro de posição; erro de orientação] no referencial do mundo
        (equivale ao tr2delta do Corke, seção 8.2)."""
        q_des = np.zeros(4); q_atual = np.zeros(4); q_conj = np.zeros(4); q_err = np.zeros(4)
        mujoco.mju_mat2Quat(q_des, R_des.flatten())
        mujoco.mju_mat2Quat(q_atual, R.flatten())
        mujoco.mju_negQuat(q_conj, q_atual)
        mujoco.mju_mulQuat(q_err, q_des, q_conj)
        w = np.zeros(3)
        mujoco.mju_quat2Vel(w, q_err, 1.0)
        return np.concatenate([p_des - p, w])

    def inversa(self, p_des, R_des, q0, iters=200, tol=1e-5, amort=1e-2, passo_max=0.2):
        """Cinemática inversa numérica por mínimos quadrados amortecidos
        (Corke 8.2.1 e 8.4; Craig 5.7 e 5.8):
            dq = J^T (J J^T + lambda^2 I)^-1 e
        Parte de q0 (a solução anterior), o que mantém a mesma configuração
        (ombro/cotovelo/punho) ao longo do círculo."""
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            p, R = self.direta(q)
            e = self.erro_pose(p_des, R_des, p, R)
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.vstack([self.jacp, self.jacr])
            dq = J.T @ np.linalg.solve(J @ J.T + amort**2 * np.eye(6), e)
            n = np.linalg.norm(dq)
            if n > passo_max:
                dq *= passo_max / n
            q = np.clip(q + dq, self.lo, self.hi)
        return q, np.linalg.norm(e[:3])

    def manipulabilidade(self, q):
        """sqrt(det(J J^T)) — Craig 8.4 / Corke 8.1.4. Perto de zero = singularidade."""
        self.direta(q)
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
        J = np.vstack([self.jacp, self.jacr])
        return np.sqrt(max(np.linalg.det(J @ J.T), 0.0))


# ----------------------------------------------------------------------------
# 3. Superfícies e marcas (Craig 2.2-2.3: sistemas de referência)
# ----------------------------------------------------------------------------
def referencial_superficie(normal, u):
    """Monta {S}: z = normal que sai da superfície (aponta para o robô),
    x = u (direção no plano). Retorna a matriz de rotação 3x3 [x y z]."""
    z = np.asarray(normal, float); z /= np.linalg.norm(z)
    x = np.asarray(u, float); x -= z * (x @ z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def orientacao_caneta(R_sup):
    """A caneta aponta CONTRA a superfície: z_caneta = -z_sup.
    Gira 180 graus em torno de x_sup para manter uma base dextrogira."""
    return R_sup @ np.diag([1.0, -1.0, -1.0])


def ponto_circulo(centro, R_sup, raio, fi, altura=0.0):
    """Ponto do círculo no referencial do mundo:
    p = c + raio*(cos(fi) x_S + sin(fi) y_S) + altura * z_S."""
    x, y, z = R_sup[:, 0], R_sup[:, 1], R_sup[:, 2]
    return centro + raio * (np.cos(fi) * x + np.sin(fi) * y) + altura * z


# ----------------------------------------------------------------------------
# 4. Simulação
# ----------------------------------------------------------------------------
class Simulacao:
    def __init__(self, headless=False, video=None, fps=30, largura=1280, altura=720):
        self.m = mujoco.MjModel.from_xml_path(str(CENA))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        mujoco.mj_forward(self.m, self.d)
        self.cin = Cinematica(self.m)
        self.q_cmd = self.d.qpos.copy()  # último comando enviado aos atuadores
        # ganhos do servo de posição definidos no MJCF: biasprm = [0, -kp, -kv]
        self.kp = self.m.actuator_gainprm[:, 0].copy()
        self.kv_total = -self.m.actuator_biasprm[:, 2] + self.m.dof_damping  # + atrito viscoso
        self.rastro = []                 # pontos "pintados" pela caneta
        self.log = []                    # (t, erro de rastreamento) para análise
        self.headless = headless
        self.viewer = None
        self.renderer = None
        self.gravador = None
        self.video = video
        self.fps = fps
        self._prox_frame = 0.0
        if video:
            self.renderer = mujoco.Renderer(self.m, altura, largura, max_geom=5000)
            self.cam = self.m.camera("vista").id
            import imageio.v2 as imageio  # grava quadro a quadro (não guarda tudo na RAM)
            self.gravador = imageio.get_writer(video, fps=fps, macro_block_size=8)
        if not headless:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m, self.d)

    # ---- laço de controle: 1 passo de física -------------------------------
    def passo(self, q_ref, desenhando=False):
        """Um passo de física com controle de posição por junta.

        O atuador do Menagerie é um servo PD (Craig 9.4-9.5):
            tau = kp*(ctrl - q) - kv*qd          (saturado em forcerange)
        Se mandarmos ctrl = q_ref, a junta fica atrasada (kv/kp = 0,1 s) e a
        gravidade deixa erro de regime. Por isso somamos ao comando um
        feedforward de velocidade e de gravidade/Coriolis (Craig 9.6 e 10.4,
        Corke 9.4.3), escrito como deslocamento da referência:
            ctrl = q_ref + (kv*qd_ref + bias(q, qd)) / kp
        o que dá tau = kp*(q_ref - q) + kv*(qd_ref - qd) + bias.
        """
        q_ref = np.asarray(q_ref, float)
        qd_ref = (q_ref - self.q_cmd) / self.m.opt.timestep
        self.d.ctrl[:] = q_ref + (self.kv_total * qd_ref + self.d.qfrc_bias) / self.kp
        mujoco.mj_step(self.m, self.d)
        self.q_cmd = q_ref.copy()

        if desenhando:
            p = self.d.site_xpos[self.cin.site].copy()
            if not self.rastro or np.linalg.norm(p - self.rastro[-1]) > 0.003:
                self.rastro.append(p)

        self._exibir()

    def _desenhar_rastro(self, scn):
        """Adiciona esferas vermelhas na cena (só visual) onde a caneta passou."""
        for p in self.rastro:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.0035, 0, 0]), p, np.eye(3).flatten(),
                                np.array([0.85, 0.05, 0.05, 1.0], dtype=np.float32))
            scn.ngeom += 1

    def _exibir(self):
        t = self.d.time
        if self.renderer is not None and t >= self._prox_frame:
            self.renderer.update_scene(self.d, camera=self.cam)
            self._desenhar_rastro(self.renderer.scene)
            self.gravador.append_data(self.renderer.render())
            self._prox_frame += 1.0 / self.fps
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise SystemExit
            if int(t / self.m.opt.timestep) % 8 == 0:  # ~60 Hz de tela
                with self.viewer.lock():
                    self.viewer.user_scn.ngeom = 0
                    self._desenhar_rastro(self.viewer.user_scn)
                self.viewer.sync()
                # mantém tempo real
                atraso = self.d.time - (time.perf_counter() - self._t0_real)
                if atraso > 0:
                    time.sleep(atraso)

    def esperar(self, segundos):
        for _ in range(int(segundos / self.m.opt.timestep)):
            self.passo(self.q_cmd)

    # ---- movimentos ---------------------------------------------------------
    def mover_juntas(self, q_alvo, duracao=None):
        """Movimento coordenado no espaço de juntas: todas as juntas saem e
        chegam juntas, com perfil quíntico (Craig 7.3 / Corke jtraj).
        Sem duração, escolhe a menor que respeita VEL_MAX: o pico de velocidade
        da quíntica é 1,875*|dq|/T (Craig, exercício 7.16)."""
        q0 = self.q_cmd.copy()
        q_alvo = np.asarray(q_alvo, float)
        if duracao is None:
            duracao = max(1.0, np.max(1.875 * np.abs(q_alvo - q0) / VEL_MAX))
        n = int(duracao / self.m.opt.timestep)
        for k in range(1, n + 1):
            s = quintica(k * self.m.opt.timestep, duracao)
            self.passo(q0 + s * (q_alvo - q0))

    def mover_cartesiano(self, p_alvo, R_alvo, duracao, desenhando=False):
        """Linha reta da ponta da caneta até p_alvo (Craig 7.4 / Corke ctraj),
        resolvendo a cinemática inversa a cada passo."""
        p0, _ = self.cin.direta(self.q_cmd)
        n = int(duracao / self.m.opt.timestep)
        q = self.q_cmd.copy()
        for k in range(1, n + 1):
            s = quintica(k * self.m.opt.timestep, duracao)
            q, _ = self.cin.inversa(p0 + s * (p_alvo - p0), R_alvo, q, iters=20)
            self.passo(q, desenhando)

    def tracar_circulo(self, centro, R_sup, raio, duracao):
        """Fi vai de 0 a 2*pi com perfil quíntico; a IK é resolvida a cada passo
        partindo da solução anterior (controle por taxa resolvida em malha
        fechada, como o sl_rrmc2 do Corke, seção 8.2)."""
        R_can = orientacao_caneta(R_sup)
        n = int(duracao / self.m.opt.timestep)
        q = self.q_cmd.copy()
        for k in range(1, n + 1):
            t = k * self.m.opt.timestep
            fi = 2 * np.pi * quintica(t, duracao)
            p_des = ponto_circulo(centro, R_sup, raio, fi)
            q, _ = self.cin.inversa(p_des, R_can, q, iters=20)
            self.passo(q, desenhando=True)
            # erro real (física) entre a ponta e o círculo desejado
            p_real = self.d.site_xpos[self.cin.site]
            self.log.append((self.d.time, np.linalg.norm(p_real - p_des)))

    def marcar(self, centro, R_sup, raio, aproximacao=0.06):
        """Sequência completa para uma marca: aproximar, descer, círculo, subir."""
        R_can = orientacao_caneta(R_sup)
        inicio = ponto_circulo(centro, R_sup, raio, 0.0)
        acima = ponto_circulo(centro, R_sup, raio, 0.0, altura=aproximacao)
        # 1) IK do ponto de aproximação e movimento no espaço de juntas até ele
        q_acima, err = self.cin.inversa(acima, R_can, self.q_cmd)
        if err > 1e-3:
            print(f"  aviso: ponto fora do alcance (erro {err*1000:.1f} mm)")
        self.mover_juntas(q_acima, 2.5)
        # 2) desce em linha reta até tocar a superfície
        self.mover_cartesiano(inicio, R_can, 1.0)
        # 3) traça o círculo
        self.tracar_circulo(centro, R_sup, raio, duracao=4.0)
        # 4) sobe em linha reta
        self.mover_cartesiano(acima, R_can, 0.8)

    # ---- fases do roteiro ---------------------------------------------------
    def fase1_juntas_individuais(self):
        print("\nFASE 1 — cada junta individualmente")
        q_home = self.m.key("home").qpos.copy()
        for i, nome in enumerate(NOMES_JUNTAS):
            lo_mod, hi_mod = self.m.jnt_range[self.m.joint(nome).id]
            lo, hi = FAIXA_DEMO[nome] if FAIXA_DEMO[nome] else (lo_mod, hi_mod)
            print(f"  {nome:20s} limite do modelo [{np.degrees(lo_mod):7.1f}, {np.degrees(hi_mod):6.1f}] graus"
                  f" | demo [{np.degrees(lo):7.1f}, {np.degrees(hi):6.1f}]")
            for alvo in (lo, hi):
                q = q_home.copy(); q[i] = alvo
                self.mover_juntas(q)
                self.esperar(0.3)
            self.mover_juntas(q_home)

    def fase2_movimento_coordenado(self):
        print("\nFASE 2 — movimento coordenado: repouso -> posição de inspeção da parede")
        p_parede = np.array([0.0, 0.70, 0.95])  # 20 cm da parede, na frente das marcas
        R_sup = referencial_superficie(normal=[0, -1, 0], u=[1, 0, 0])
        q_insp, err = self.cin.inversa(p_parede, orientacao_caneta(R_sup), self.q_cmd)
        print("  q de inspeção (graus):", np.degrees(q_insp).round(1), f"| erro IK {err*1000:.2f} mm")
        self.mover_juntas(q_insp, 3.0)
        self.esperar(0.5)

    def fase3_circulos(self, raio):
        print(f"\nFASE 3 — círculos de raio {raio*100:.1f} cm ao redor das marcas")
        alvos = [  # (nome do site, normal da superfície, direção x no plano)
            ("tiro_1", [0, -1, 0], [1, 0, 0]),
            ("tiro_2", [0, -1, 0], [1, 0, 0]),
            ("tiro_3", [0, -1, 0], [1, 0, 0]),
            ("faca_1", [0, 0, 1], [1, 0, 0]),
            ("faca_2", [0, 0, 1], [1, 0, 0]),
        ]
        na_mesa = False
        for nome, normal, u in alvos:
            R_sup = referencial_superficie(normal, u)
            centro = self.d.site_xpos[self.m.site(nome).id].copy()  # posição da marca no mundo
            print(f"  {nome}: centro {centro.round(3)}")
            if nome.startswith("faca") and not na_mesa:
                # troca de superfície: recua 25 cm da parede antes de girar para a mesa
                p, R = self.cin.direta(self.q_cmd)
                self.mover_cartesiano(p + np.array([0.0, -0.25, 0.0]), R, 1.0)
                na_mesa = True
            self.marcar(centro, R_sup, raio)
        self.mover_juntas(self.m.key("home").qpos, 3.0)

    def rodar(self, raio):
        self._t0_real = time.perf_counter()
        self.esperar(0.5)
        self.fase1_juntas_individuais()
        self.fase2_movimento_coordenado()
        self.fase3_circulos(raio)
        self.esperar(1.0)
        if self.log:
            e = np.array(self.log)[:, 1] * 1000
            print(f"\nErro de rastreamento nos círculos: médio {e.mean():.2f} mm, máximo {e.max():.2f} mm")
        if self.gravador is not None:
            self.gravador.close()
            print(f"Vídeo salvo em {self.video} ({self.d.time:.0f} s de simulação)")
        if self.viewer is not None:
            print("Fim. Feche a janela para sair.")
            while self.viewer.is_running():
                self.viewer.sync(); time.sleep(0.05)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raio", type=float, default=0.04, help="raio dos círculos em metros")
    ap.add_argument("--headless", action="store_true", help="sem janela (útil para gravar vídeo)")
    ap.add_argument("--video", type=str, default=None, help="arquivo .mp4 de saída")
    args = ap.parse_args()
    Simulacao(headless=args.headless, video=args.video).rodar(args.raio)
