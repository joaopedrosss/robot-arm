# UR10e no MuJoCo: marcação de evidências forenses

Simulação de um **Universal Robots UR10e** com uma caneta marcadora que traça
círculos de raio *R* ao redor de marcas numa superfície: furos de projétil numa
parede e fissuras de faca numa placa sobre a mesa.

- **Simulador:** MuJoCo 3.16.0 (pacote `mujoco` do pip; funciona em 3.2 ou mais nova)
- **Robô:** UR10e do [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur10e) (licença BSD-3, ver `LICENSE_menagerie_ur10e`)
- **Linguagem:** Python 3.10+

## Como instalar e rodar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # abre o visualizador e roda as 3 fases
python main.py --raio 0.06         # círculos de 6 cm
python main.py --headless --video simulacao.mp4   # grava vídeo sem janela
```

No macOS o visualizador passivo precisa ser aberto com `mjpython main.py`
(o `mjpython` vem junto com o pacote `mujoco`).

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `cena_forense.xml` | Cena (MJCF): chão, parede com marcas de tiro, mesa com placa e fissuras, caixas, faixa de isolamento, câmera |
| `ur10e_caneta.xml` | Modelo do UR10e do Menagerie com a caneta (corpo `caneta`, site `ponta` = TCP) adicionada ao flange |
| `assets/` | Malhas (.obj) do robô |
| `main.py` | Trajetórias, cinemática direta/inversa, controle e as 3 fases |

## O que a simulação faz

1. **Cada junta individualmente:** varre do limite inferior ao superior e volta,
   respeitando as velocidades máximas do datasheet (120 °/s na base e no ombro, 180 °/s nas demais).
   Ombro e cotovelo usam uma subfaixa porque o braço colidiria com a parede ou com o chão.
2. **Movimento coordenado:** do repouso (`home`) até a postura de inspeção da parede,
   todas as juntas ao mesmo tempo com perfil quíntico.
3. **Círculos:** para cada marca, aproxima no espaço de juntas, desce em linha reta,
   traça o círculo (cinemática inversa numérica a cada passo) e sobe.
   O rastro vermelho é desenhado na cena; ao final o terminal mostra o erro de rastreamento.

## Configuração do UR10e

- 6 graus de liberdade, todas as juntas rotativas (6R).
- Limites (datasheet): ±360° em todas as juntas. No modelo do Menagerie o cotovelo é limitado a ±180°.
- Alcance de 1300 mm, carga de 12,5 kg, repetibilidade de ±0,05 mm.
- Espaço de trabalho aproximadamente uma esfera de raio 1,3 m centrada no ombro, menos um
  cilindro na vertical da base, onde o robô não consegue posicionar a ferramenta.

## Uso de IA

Basicamente a única ferramenta que de IA generativa que precisei foi o Claude e Claude Code, nos quais alternei entre os modelos Opus para as tarefas de simulação e o Sonnet para pesquisas rápidas e tarefas mais simples.

A consulta a IA foi basante útil afim corrigir erros de instalação. Por exemplo, não conseguia de jeito nenhum abrir a tela de simulação da cena com o robô. Após quebrar a cabeça e consultar a IA, percebi que a causa era por que eu estava tentando trabalhar com o projeto no codespaces: o MuJoCo estava tentando localizar um monitor, mas não o achando dado que codespaces é uma IDE na nuvem. Então transferir o projeto para meu PC local e consegui trabalhar com o projeto.

A IA me foi muito útil para o meu aprendizado de forma direta: pedi para ela montar um plano para eu fazer o projeto e enteder o que estava fazendo no processo, com referencias à capítulos de livros e exercícios em cada passo.

No entanto a IA não me foi tão útil para a cinemática inversa, quando foi para o robô tracar o circulos na cena do crime. Talvez tenha sido em parte culpa minha por não ter calibriado o prompt de forma correta, mas ela não conseguiu fazer o robô traçar a trajetória desejada. Então tive que recorrer a recursos externos na internet e usei a IA para integrar o código que achei no meu projeto.