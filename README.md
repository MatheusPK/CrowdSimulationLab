# CrowdSimulationLab

Simulador 2D de evacuação de emergência com Deep Reinforcement Learning e modelo emocional FSM.
Desenvolvido como pesquisa de mestrado sobre como emoções afetam o comportamento coletivo em situações de emergência.

Compara quatro políticas de navegação — **Random, A\*, A\*+FSM e DQN+FSM** — em cinco cenários com hazards, gargalos e dilemas de rota.

---

## Índice

1. [Instalação](#instalação)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Configuração](#configuração)
4. [Como executar](#como-executar)
   - [Visualização interativa](#visualização-interativa)
   - [Rodar A\* ou Random](#rodar-a-ou-random)
   - [Rodar o DQN treinado](#rodar-o-dqn-treinado)
   - [Treino completo com currículo](#treino-completo-com-currículo)
   - [Avaliação e comparação de políticas](#avaliação-e-comparação-de-políticas)
5. [As quatro políticas](#as-quatro-políticas)
6. [Modelo emocional FSM](#modelo-emocional-fsm)
7. [Observação do DQN](#observação-do-dqn-17-features)
8. [Reward](#reward)
9. [Currículo de treino](#currículo-de-treino)
10. [Mapas](#mapas)
11. [Métricas](#métricas)
12. [Parâmetros de simulação](#parâmetros-de-simulação)
13. [Decisões de implementação](#decisões-de-implementação)
14. [Estimativa de tempo de treino](#estimativa-de-tempo-de-treino)

---

## Instalação

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 2. Instalar dependências
pip install -r requirements.txt
```

**Dependências:** `pygame >= 2.0`, `torch >= 2.0`

CUDA é detectado automaticamente. Em Mac com Apple Silicon, o PyTorch usa MPS. Sem GPU, roda em CPU sem nenhuma alteração de código.

---

## Estrutura do projeto

```
CrowdSimulationLab/
│
├── main.py                    # entrypoint: rodar/visualizar qualquer política
├── config.py                  # configuração central do experimento
├── train_curriculum.py        # treino DQN com currículo progressivo
├── simulation_params.py       # TODOS os hiperparâmetros — edite apenas aqui
├── requirements.txt
│
├── core/
│   ├── app_scenario.py        # enum: RANDOM | ASTAR | ASTAR_FSM | DQN_FSM
│   ├── direction.py           # enum: N NE E SE S SW W NW + vetores
│   ├── dqn_mode.py            # enum: TRAIN | EVAL
│   ├── fsm_state.py           # enum: CALM | EVACUATE | PANIC
│   └── map_data.py            # dataclass MapData (grid, exits, spawns, …)
│
├── entities/
│   ├── agent.py               # Agent: posição, emoção, FSMState, métricas
│   ├── exit.py
│   ├── hazard.py
│   └── obstacle.py
│
├── environment/
│   ├── environment.py         # step · reward · FSM · BFS dist_map · métricas
│   ├── astar.py               # AStarPlanner com clearance e approach cell
│   └── map_loader.py          # parser .txt → MapData
│
├── policies/
│   ├── astar_policy.py        # A* com cache de path e detecção de stuck
│   ├── dqn_policy.py          # DQN · ReplayBuffer · target network
│   └── random_policy.py
│
├── factories/
│   └── policy_factory.py      # instancia a política correta pelo config
│
├── rendering/
│   └── renderer.py            # pygame · cor do agente por emotion_level
│
├── models/                    # criado automaticamente pelo treino
│   ├── dqn_fsm.pth            # modelo principal
│   ├── ckpt_sN_epM.pth        # checkpoints por stage
│   └── curriculum_state.json  # estado do currículo para retomada
│
├── logs/                      # criado com --log ou pelo currículo
│   ├── results.csv            # resultados de main.py --log
│   └── training_log.csv       # log episódio-a-episódio do treino
│
└── maps/
    ├── train/                 # 12 mapas do currículo
    ├── eval/                  # 5 mapas de avaliação
    └── di_style.txt           # mapa de fine-tuning (estilo DI)
```

---

## Configuração

**Todo o experimento é controlado por `config.py`.** As três linhas mais importantes:

```python
SCENARIO = AppScenario.ASTAR_FSM   # política a rodar
MAP      = ALL_MAPS["mall_panic"]  # mapa
AGENTS   = 8                       # número de agentes
```

### Cenários disponíveis

```python
from core.app_scenario import AppScenario

AppScenario.RANDOM     # baseline aleatório, sem FSM
AppScenario.ASTAR      # A* puro, sem FSM, sem custo de hazard
AppScenario.ASTAR_FSM  # A* com FSM emocional, penaliza tiles H no path
AppScenario.DQN_FSM    # DQN com FSM emocional (requer modelo treinado)
```

### Mapas disponíveis

```python
# Treino (currículo)
ALL_MAPS["mall_small"]             # stage 1
ALL_MAPS["school_small"]           # stage 2
ALL_MAPS["office_wing_small"]      # stage 3
ALL_MAPS["library_small"]          # stage 4
ALL_MAPS["library_medium"]         # stage 5
ALL_MAPS["hazard_corridor_small"]  # stage 6 ★ dilema rota
ALL_MAPS["school_floor"]           # stage 7
ALL_MAPS["office_wing_medium"]     # stage 8
ALL_MAPS["hazard_bypass_medium"]   # stage 9 ★ bypass
ALL_MAPS["mall_medium"]            # stage 10
ALL_MAPS["hazard_dense_office"]    # stage 11 ★ alta densidade H
ALL_MAPS["library_hard"]           # stage 12

# Avaliação (experimentos do mestrado)
ALL_MAPS["library_bottleneck"]     # gargalo + hazard lateral
ALL_MAPS["office_single_exit"]     # exit único, alta densidade
ALL_MAPS["mall_panic"]             # dilema rota curta/perigosa vs longa/segura
ALL_MAPS["school_evacuation"]      # exits assimétricos, hazard central
ALL_MAPS["di_emergency"]           # estilo DI, hazard perto do exit superior
```

### Parâmetros gerais de `config.py`

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `RENDER` | `True` | `False` remove a janela pygame — treino ~2–3× mais rápido |
| `FPS` | `30` | Frames por segundo da visualização |
| `DT` | `0.1` | Timestep de simulação (segundos por step) |
| `MAX_STEPS` | `400` | Limite de steps por episódio |
| `EVAL_EPISODES` | `10` | Episódios ao rodar `main.py` sem flag `--episodes` |

### Bloco DQN em `config.py`

Controla o DQN ao rodar via `main.py`:

```python
DQN = {
    "mode":       DQNMode.EVAL,          # EVAL para avaliar, TRAIN para treino rápido
    "model_path": "models/dqn_fsm.pth",  # caminho do modelo a carregar/salvar
    "episodes":   500,                   # episódios (só se mode=TRAIN)
    # demais hiperparâmetros são lidos de simulation_params.py
}
```

> Para treino sério, use `train_curriculum.py`. O `mode=TRAIN` no `main.py` serve apenas para experimentos rápidos sem currículo.

---

## Como executar

### Visualização interativa

Edite `SCENARIO` e `MAP` em `config.py`, depois:

```bash
python main.py
```

Controles durante a janela:
- **Espaço** → pausa / retoma a renderização
- **Fechar janela** → encerra o episódio atual

### Rodar A\* ou Random

```bash
# 1. Em config.py, ajuste SCENARIO e MAP, por exemplo:
#    SCENARIO = AppScenario.ASTAR_FSM
#    MAP      = ALL_MAPS["mall_panic"]
#    AGENTS   = 12

# 2. Execute:
python main.py --no-render --episodes 30 --log
```

Flags disponíveis:

| Flag | Efeito |
|---|---|
| `--no-render` | Desativa pygame (mais rápido, útil para batch) |
| `--episodes N` | Número de episódios a rodar |
| `--log` | Salva resultados em `logs/results.csv` |

### Rodar o DQN treinado

```bash
# Em config.py:
#   SCENARIO = AppScenario.DQN_FSM
#   DQN["mode"] = DQNMode.EVAL
#   DQN["model_path"] = "models/dqn_fsm.pth"

python main.py --no-render --episodes 30 --log
```

Para avaliar um checkpoint específico do currículo:

```python
# Em config.py:
DQN["model_path"] = "models/ckpt_s9_ep1200.pth"
```

### Treino completo com currículo

```bash
# Treino completo (12 stages), retoma automaticamente se interrompido
python train_curriculum.py

# Silencioso — recomendado para treinos longos
python train_curriculum.py --quiet

# Com visualização
python train_curriculum.py --render

# Começar a partir de um stage específico
python train_curriculum.py --stage 6

# Usar um checkpoint como ponto de partida
python train_curriculum.py --model models/ckpt_s5_ep600.pth

# Fine-tuning em di_style após o currículo principal
python train_curriculum.py --fine-tune

# Todas as flags combinam livremente
python train_curriculum.py --stage 9 --quiet --model models/dqn_fsm.pth
```

O treino salva automaticamente:
- `models/dqn_fsm.pth` — modelo principal (atualizado a cada stage)
- `models/ckpt_sN_epM.pth` — checkpoint a cada 50 episódios
- `models/curriculum_state.json` — estado para retomada automática
- `logs/training_log.csv` — log completo episódio a episódio

### Avaliação e comparação de políticas

#### Avaliar o DQN nos 5 mapas de avaliação

```bash
python train_curriculum.py --eval --episodes 30
# Com modelo específico:
python train_curriculum.py --eval --episodes 30 --model models/dqn_fsm.pth
```

Output esperado:

```
Mapa                      evac  panic  peak_em  haz_ct  r_util    var   time
library_bottleneck        0.88   0.25    0.312    0.12   1.000  0.0231   187
office_single_exit        0.94   0.38    0.441    0.25   1.000  0.0312   203
mall_panic                0.81   0.62    0.588    0.44   0.750  0.0891   312
school_evacuation         0.86   0.31    0.389    0.18   0.833  0.0412   228
di_emergency              0.79   0.55    0.521    0.31   0.667  0.0671   298
```

#### Comparar todas as 4 políticas num mesmo mapa

Rode sequencialmente trocando `SCENARIO` em `config.py`:

```bash
# 1. SCENARIO = AppScenario.RANDOM
python main.py --no-render --episodes 30 --log

# 2. SCENARIO = AppScenario.ASTAR
python main.py --no-render --episodes 30 --log

# 3. SCENARIO = AppScenario.ASTAR_FSM
python main.py --no-render --episodes 30 --log

# 4. SCENARIO = AppScenario.DQN_FSM  (DQN["mode"] = DQNMode.EVAL)
python main.py --no-render --episodes 30 --log
```

Todos os resultados vão para `logs/results.csv`, que acumula sem sobrescrever. A coluna `scenario` identifica cada política.

**Protocolo estatístico recomendado:** 30 episódios por política por mapa, Mann-Whitney U test (não-paramétrico), p < 0.05. Total: 4 políticas × 5 mapas × 30 episódios = 600 runs (~15–25 min sem render).

---

## As quatro políticas

| Cenário | Política | FSM emocional | Custo de hazard no path |
|---|---|---|---|
| `RANDOM` | Ação aleatória entre 8 direções | não | — |
| `ASTAR` | A\* com clearance de agente | não | não (ignora hazard) |
| `ASTAR_FSM` | A\* com clearance de agente | **sim** | **sim** (`ASTAR_HAZARD_COST = 25`) |
| `DQN_FSM` | Deep Q-Network (17 features, 2×128) | **sim** | aprende por experiência |

### Como o A\* funciona internamente

O `AStarPolicy` planeja caminhos com 8-conectividade (incluindo diagonais) e clearance baseado no raio físico do agente (`AGENT_RADIUS + PLANNER_RADIUS_MARGIN = 11px`). Isso impede que o caminho passe por espaços estreitos demais para o agente físico atravessar.

O goal do A\* **não é o centro do tile `E`**, mas a célula livre mais próxima cujo centro consegue tocar o exit com o raio físico do agente (6px). Isso resolve o caso de exits embutidos em paredes onde o centro do tile está inacessível.

Quando `ASTAR_FSM` está ativo e o mapa tem mais de um exit, o A\* usa o `_dist_map_safe` (Dijkstra com penalidade de hazard) para escolher qual exit perseguir — garantindo que em mapas como `mall_panic` e `di_emergency` o agente vá para o exit seguro, mesmo que o exit perigoso seja mais próximo em linha reta.

O anti-stuck replaneja o caminho se o agente não mudar de célula por `ASTAR_STUCK_THRESHOLD = 5` steps consecutivos.

### Como o DQN funciona internamente

O DQN usa uma rede 17 → 128 → 128 → 8, onde as 8 saídas são as 8 direções de movimento. A ação é escolhida por epsilon-greedy durante o treino e greedy durante a avaliação.

**Multi-agent parameter sharing:** todos os agentes do episódio compartilham a mesma rede e contribuem transições independentes ao replay buffer. Com 12 agentes × 450 steps, cada episódio gera ~5400 transições, capturando dinâmicas coletivas (contágio emocional, congestionamento, dilemas de rota) que treino single-agent não reproduz.

---

## Modelo emocional FSM

A FSM tem três estados com **histerese** — os limiares de subida e descida são diferentes para evitar oscilação:

```
CALM ──(emotion ≥ 0.35)──► EVACUATE ──(emotion ≥ 0.72)──► PANIC
     ◄──(emotion < 0.25)──           ◄──(emotion < 0.62)──
```

Cada estado altera a velocidade e os parâmetros de avoidance:

| Estado | Velocidade | Avoidance de obstáculos | Avoidance de agentes | Smoothing |
|---|---|---|---|---|
| CALM | 1.0× (50 px/s) | forte (dist=16, str=80) | forte (dist=14, str=95) | 0.20 |
| EVACUATE | 1.15× (57.5 px/s) | médio (dist=12, str=70) | médio (dist=10, str=85) | 0.16 |
| PANIC | 1.30× (65 px/s) | fraco (dist=8, str=55) | fraco (dist=7, str=70) | 0.10 |

No pânico, o avoidance fraco causa aglomeração realista — agentes se empurram em vez de se desviar, simulando pisoteamento.

### Evolução do emotion_level por step

O `emotion_level` varia em [0, 1] e é atualizado a cada step:

| Evento | Delta |
|---|---|
| Decaimento passivo (sempre) | −0.02 |
| Contato direto com hazard | +0.15 |
| Hazard visível (raio 40px, sem contato) | +0.08 |
| Contágio emocional | +0.04 × média de emoção dos vizinhos no raio de 35px |

O contágio é proporcional à emoção média dos vizinhos — um grupo em pânico eleva os agentes próximos mais do que um grupo calmo. O contágio só ocorre com agentes no raio de 35px (~4 tiles).

---

## Observação do DQN (17 features)

```
[0]   pos_x / mapa_width                         posição normalizada
[1]   pos_y / mapa_height
[2]   (exit_cx - agent_x) / mapa_width           vetor para o exit mais próximo por BFS
[3]   (exit_cy - agent_y) / mapa_height
[4]   dist_BFS_pixels / diagonal_mapa            distância BFS normalizada
[5]   atan2(dy, dx) / π                          ângulo para o exit em [-1, 1]
[6]   1 se hazard visível (raio 40px), 0 c.c.
[7]   dist_hazard / (2 × raio_visão)             distância ao hazard normalizada
[8]   1 se em contato com hazard, 0 c.c.
[9]   emotion_level                              [0, 1]
[10]  fsm_state / 2                              0=CALM, 0.5=EVACUATE, 1.0=PANIC
[11]  vx / current_speed                         velocidade normalizada
[12]  vy / current_speed
[13]  min(densidade_local / 8, 1.0)              agentes no raio de 35px, normalizado
[14]  dist_obstáculo / (2 × obs_avoidance_dist)  distância à parede mais próxima
[15]  current_speed / base_speed                 efeito da FSM na velocidade
[16]  1 se evacuado, 0 c.c.
```

**Por que BFS em vez de euclidiana nas features [2–4]:** em mapas com corredores, a euclidiana subestima a distância real em até 6×, fazendo o vetor de direção apontar para fora das paredes. O BFS reflete a geometria real do mapa.

**Por que o DQN não recebe `_dist_map_safe`:** se a feature [4] já penalizasse hazard, o DQN receberia a resposta pré-calculada sobre qual rota é mais segura. A separação força o DQN a aprender esse tradeoff por experiência, que é o comportamento central a investigar.

---

## Reward

O reward é calculado em três camadas por step, por agente:

### Camada 1 — Navegação (sinal principal)

| Componente | Fórmula | Valor padrão |
|---|---|---|
| Progresso BFS | `+15 × (prev_dist − new_dist) / diagonal` | escala com o progresso |
| Evacuação | `+80` (terminal, uma vez) | `REWARD_EVACUATED = 80` |
| Penalidade de tempo | `−0.05` por step | `REWARD_TIME_PENALTY = -0.05` |
| Retrocesso real | `−1.0` se `progress < -1e-4` | `REWARD_NO_PROGRESS = -1.0` |

A penalidade de retrocesso (`REWARD_NO_PROGRESS`) só é ativada quando o agente efetivamente recua — `progress < -1e-4`. Estagnação (`progress == 0`) **não é penalizada**: com `VELOCITY_SMOOTHING = 0.18`, o agente leva ~4 steps para mudar de célula BFS enquanto acelera. Penalizar estagnação mascarava o sinal de progresso em ~94% dos steps.

### Camada 2 — Hazard e emoção

| Componente | Fórmula | Valor padrão |
|---|---|---|
| Contato com hazard | `−3.0` por step | `REWARD_HAZARD_CONTACT = -3.0` |
| Calma perto do hazard | `+0.4 × (1 − emotion)` (só se visível e sem contato) | `REWARD_HAZARD_VISIBLE_CALM = 0.4` |
| Pânico perto do hazard | `−0.5` se `emotion > 0.5` e hazard visível | `REWARD_HAZARD_PANIC = -0.5` |

> **Atenção:** `REWARD_HAZARD_VISIBLE_CALM` incentiva calma perto do hazard, não aproximação. Se `hazard_contact_rate` subir nos stages 4–5, reduzir para 0.2 ou zerar.

### Camada 3 — Interação social

| Componente | Fórmula | Valor padrão |
|---|---|---|
| Colisão com parede ou agente | `−0.3` | `REWARD_COLLISION = -0.3` |
| Densidade local | `−0.1 × min(densidade / 8, 1.0)` | `REWARD_DENSITY_SCALE = -0.1` |

---

## Currículo de treino

O treino usa 12 stages progressivos. Promoção ocorre quando `evacuation_rate ≥ 0.80` nos últimos 30 episódios. Patience: 300 episódios por stage.

| Stage | Mapa | Agentes | Max steps | O que ensina |
|---|---|---|---|---|
| 1 | mall_small | 4 | 300 | Navegação pura, sem hazard |
| 2 | school_small | 4 | 300 | Single-exit, sem hazard |
| 3 | office_wing_small | 4 | 300 | Corredor, sem hazard |
| 4 | library_small | 4 | 300 | Primeiro hazard + FSM |
| 5 | library_medium | 10 | 400 | Hazard médio, multi-exit |
| 6 | hazard_corridor_small ★ | 4 | 350 | Dilema: rota perigosa vs segura |
| 7 | school_floor | 10 | 400 | Hazard em corredor |
| 8 | office_wing_medium | 10 | 400 | Hazard em salas densas |
| 9 | hazard_bypass_medium ★ | 12 | 450 | H flanqueia exit principal, bypass |
| 10 | mall_medium | 12 | 450 | Ambiente aberto, hazard distribuído |
| 11 | hazard_dense_office ★ | 12 | 450 | Hazard em 8% da área livre |
| 12 | library_hard | 12 | 450 | Gargalo + hazard |

★ Mapas criados especificamente para cobrir lacunas do currículo: o currículo original não tinha nenhum mapa com dilema de rota (padrão central dos mapas de eval) e o hazard máximo era 1.6% vs 12.2% do `mall_panic`.

### Progressão de agentes

A oscilação N=4 → N=10 → N=4 (stage 6) → N=10 → N=12 é intencional:
- Stages 1–4 (N=4): aprender navegação sem overhead de dinâmica coletiva
- Stages 5, 7–8 (N=10): introduzir dinâmica coletiva (Xu et al. 2021: mínimo 10)
- Stage 6 (N=4): mapa small, mantém 4 para caber nos spawns disponíveis
- Stages 9–12 (N=12): contágio emocional consistente (Lv et al. 2022: mínimo 12)

### Fine-tuning opcional

```bash
python train_curriculum.py --fine-tune
```

Treina em `maps/di_style.txt` após o currículo principal. Prepara para `di_emergency` (hazard perto do exit superior). Roda por até 300 episódios com critério de promoção em 80%.

---

## Mapas

### Formato de mapa (.txt)

Cada caractere representa um tile de 8×8 px:

| Char | Tile |
|---|---|
| `O` | Parede / obstáculo |
| `.` | Espaço livre |
| `E` | Exit (saída de evacuação) |
| `H` | Hazard (incêndio, fumaça) |
| `S` | Spawn (posição inicial de agente) |

**Regras:**
- Borda de `O` em todo o perímetro
- Número de `S` deve ser ≥ número de agentes configurados
- Tiles `E` devem ser alcançáveis por BFS a partir dos tiles `S`
- Tiles `H` adjacentes contam como uma região de hazard única (Union-Find)

**Exemplo mínimo (11×6):**

```
OOOOOOOOOOO
OEEE.......O
O....S.....O
O....S.....O
O..........O
OOOOOOOOOOO
```

**Adicionando ao projeto:**
```python
# Em config.py:
ALL_MAPS["meu_mapa"] = "maps/train/meu_mapa.txt"

# Para incluir no currículo, adicione em CURRICULUM dentro de train_curriculum.py:
("meu_mapa", "maps/train/meu_mapa.txt", 8, 400),
```

### Mapas de avaliação

| Mapa | H% área livre | Exits | Fenômeno testado |
|---|---|---|---|
| `library_bottleneck` | 2.1% | 1 | Gargalo com hazard lateral |
| `office_single_exit` | 1.3% | 1 | Exit único, alta densidade |
| `mall_panic` | 12.2% | 3 | Dilema: exit perigoso vs exits seguros |
| `school_evacuation` | 2.8% | 2 | Exits assimétricos, hazard central |
| `di_emergency` | 2.6% | 1 | Hazard perto do exit superior |

Todos com N=12 agentes. Spawns verificados: mínimo 15, máximo 29 por mapa.

---

## Métricas

Todas retornadas por `env.get_episode_metrics()` e registradas em CSV.

### Grupo 1 — Eficiência de evacuação

| Métrica | Tipo | Descrição |
|---|---|---|
| `evacuation_rate` | [0, 1] | Fração de agentes evacuados no episódio |
| `all_evacuated` | bool | `True` se 100% evacuaram |
| `mean_evacuation_time` | steps | Média de tempo de evacuação (só dos que evacuaram) |
| `steps` | int | Steps totais do episódio |

### Grupo 2 — Comportamento emocional (contribuição central)

| Métrica | Tipo | Descrição |
|---|---|---|
| `mean_emotion_final` | [0, 1] | Média de `emotion_level` no último step |
| `mean_peak_emotion` | [0, 1] | Média do pico emocional por agente ao longo do episódio |
| `emotion_variance` | float | Variância de `emotion_level` entre agentes |
| `panic_rate` | [0, 1] | Fração de agentes que atingiram `FSMState.PANIC` |

`emotion_variance` é a **métrica central do paper**: mede divergência emocional entre agentes. Variância alta durante o episódio indica dilema de rota ativo; convergência para baixo indica que os agentes encontraram uma rota segura consistente.

### Grupo 3 — Qualidade de rota e interação

| Métrica | Tipo | Descrição |
|---|---|---|
| `mean_speed_ratio` | float | `current_speed / base_speed` — efeito da FSM na velocidade |
| `hazard_contact_rate` | [0, 1] | Fração de agentes que tocaram o hazard em algum momento |
| `exit_utilization` | [0, 1] | `min(uso_exit) / max(uso_exit)` entre grupos de exit. 1.0 = uso perfeitamente igual; 0.0 = todos num exit só |

`exit_utilization` usa Union-Find para agrupar tiles `E` adjacentes em uma mesma saída antes de contar — sem isso, um exit de 22 tiles reportaria 22 saídas distintas.

---

## Parâmetros de simulação

**Edite apenas `simulation_params.py`.** Todos os outros arquivos importam daqui. Nunca altere valores diretamente em `environment.py`, `agent.py` ou `dqn_policy.py`.

### Agente — física

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `AGENT_RADIUS` | 6.0 px | Raio físico para colisão e detecção de exit |
| `PLANNER_RADIUS_MARGIN` | 5.0 px | Margem extra do A* para evitar paredes. Planner radius = 11px. Reduzir permite passar por corredores mais estreitos, mas aumenta risco de colisão. Mínimo recomendado: 2.0 |
| `AGENT_BASE_SPEED` | 50.0 px/s | Velocidade no estado CALM. Com `dt=0.1`, equivale a 5px/step. Aumentar → evacuação mais rápida; diminuir → mais tempo para desenvolver emoção |
| `VELOCITY_SMOOTHING` | 0.18 | Filtro exponencial de velocidade [0–1]. 0 = inerte, 1 = instantâneo. Valor baixo → movimento fluido mas ~4 steps para mudar de célula BFS |

### Agente — avoidance base (sobrescrito pela FSM quando `use_fsm=True`)

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `OBSTACLE_AVOIDANCE_DISTANCE` | 14.0 px | Distância em que a repulsão de paredes começa |
| `OBSTACLE_AVOIDANCE_STRENGTH` | 70.0 | Magnitude da força repulsiva |
| `AGENT_AVOIDANCE_DISTANCE` | 12.0 px | Distância de início da repulsão entre agentes |
| `AGENT_AVOIDANCE_STRENGTH` | 90.0 | Magnitude. Alto → agentes muito dispersos; baixo → aglomeração e mais contágio |

### FSM — transições

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `FSM_CALM_TO_EVACUATE` | 0.35 | Limiar de subida CALM → EVACUATE |
| `FSM_EVACUATE_TO_CALM` | 0.25 | Limiar de descida EVACUATE → CALM (histerese) |
| `FSM_EVACUATE_TO_PANIC` | 0.72 | Limiar de subida EVACUATE → PANIC |
| `FSM_PANIC_TO_EVACUATE` | 0.62 | Limiar de descida PANIC → EVACUATE (histerese) |

### FSM — velocidades

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `FSM_SPEED_CALM` | 1.00 | Multiplicador de `AGENT_BASE_SPEED` no estado CALM |
| `FSM_SPEED_EVACUATE` | 1.15 | +15% de velocidade |
| `FSM_SPEED_PANIC` | 1.30 | +30% de velocidade |
| `FSM_SPEED_SMOOTHING` | 0.20 | Suavização da transição de velocidade entre estados |

### Emoção — deltas

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `EMOTION_DECAY` | −0.02 | Decaimento passivo por step. Mais negativo → calma mais rápida |
| `EMOTION_DELTA_HAZARD_CONTACT` | +0.15 | Delta ao tocar o hazard |
| `EMOTION_DELTA_HAZARD_VISIBLE` | +0.08 | Delta ao ver o hazard sem contato |
| `EMOTION_DELTA_CONTAGION` | +0.04 | Coeficiente de contágio: `delta += 0.04 × avg_emoção_vizinhos` |
| `CONTAGION_RADIUS` | 35.0 px | Raio de contágio (~4 tiles). Aumentar → pânico se propaga a distância maior |
| `HAZARD_VISION_RADIUS` | 40.0 px | Raio de visão do hazard (5 tiles). Não igualar ao planner radius |

### Percepção

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `LOCAL_DENSITY_RADIUS` | 35.0 px | Raio para contar agentes vizinhos (densidade e contágio) |

### A* / Planner

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `ASTAR_HAZARD_COST` | 25.0 | Custo extra por tile H no pathfinding. 0 → ignora hazard completamente; 25 → evita rotas com H se a alternativa for até 25 tiles mais longa; 40+ → nunca atravessa H se houver alternativa |
| `ASTAR_STUCK_THRESHOLD` | 5 | Steps sem mudar de célula antes de replanejar |

### DQN — rede e treino

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `DQN_HIDDEN_DIM` | 128 | Neurônios por camada oculta (2 camadas). Aumentar para 256 em mapas mais complexos |
| `DQN_BUFFER_CAPACITY` | 100 000 | Transições no replay buffer (~28 episódios com N=12) |
| `DQN_TRAIN_START_SIZE` | 2 000 | Buffer mínimo antes de começar updates. Com N=12 agentes, atingido em <1 episódio |
| `DQN_BATCH_SIZE` | 64 | Amostras por update de gradiente |
| `DQN_GAMMA` | 0.99 | Desconto temporal. 0.99 → horizonte ~100 steps; 0.95 → ~20 steps |
| `DQN_LR` | 1e-3 | Learning rate do Adam. Reduzir para 1e-4 se treino instável |
| `DQN_TARGET_UPDATE_FREQ` | 300 | Updates de gradiente entre cada sync da target network |
| `DQN_GRAD_CLIP_NORM` | 10.0 | Gradient clipping (estabiliza treino com reward shaping) |

### DQN — epsilon-greedy

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `DQN_EPSILON_START` | 1.0 | Exploração inicial (100% aleatório) |
| `DQN_EPSILON_END` | 0.05 | Exploração mínima (5% aleatório) |
| `DQN_EPSILON_DECAY` | 1 600 000 | Transições para decair de `START` até `END`. Calibrado para ε≈0.15 no stage 9 (dilema de rota). Com 200k, epsilon chegava a 0.05 já no stage 2 |

### Reward

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `REWARD_PROGRESS_SCALE` | 15.0 | Peso do progresso BFS. Aumentar se o agente não converge para o exit |
| `REWARD_EVACUATED` | 80.0 | Recompensa terminal por evacuação |
| `REWARD_TIME_PENALTY` | −0.05 | Urgência por step. Mais negativo → agente mais apressado |
| `REWARD_NO_PROGRESS` | −1.0 | Penalidade por retrocesso real (`progress < -1e-4`) |
| `REWARD_HAZARD_CONTACT` | −3.0 | Por step dentro do hazard |
| `REWARD_HAZARD_VISIBLE_CALM` | +0.4 | Bônus × (1 − emotion) ao ver hazard sem entrar. Monitorar `hazard_contact_rate` nos stages 4–5 |
| `REWARD_HAZARD_PANIC` | −0.5 | Por step em pânico perto do hazard |
| `REWARD_COLLISION` | −0.3 | Por colisão com parede ou agente |
| `REWARD_DENSITY_SCALE` | −0.1 | × densidade normalizada. Penaliza aglomeração extrema |

### Currículo

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `CURRICULUM_PROMOTION_THRESHOLD` | 0.80 | `evacuation_rate` mínima nos últimos N episódios para avançar de stage |
| `CURRICULUM_EVAL_WINDOW` | 30 | Janela de episódios para calcular a média de promoção |
| `CURRICULUM_PATIENCE` | 300 | Máximo de episódios por stage antes de avançar mesmo sem promoção |
| `CURRICULUM_SAVE_EVERY` | 50 | Frequência de checkpoint (episódios) |

---

## Decisões de implementação

**Distância BFS em vez de euclidiana** — o ambiente pré-computa um `_dist_map` por BFS multi-source a partir de todos os exits. Em mapas com corredores, a euclidiana subestima a distância real em até 6×, gerando reward negativo para ações corretas (ex: subir um corredor para encontrar a passagem). O BFS garante que o reward de progresso é sempre positivo quando o agente está no caminho ótimo.

**Dois dist_maps separados** — `_dist_map` (BFS puro, sem custo de hazard) é usado pelo DQN para que aprenda o tradeoff rota-perigosa/rota-segura por experiência. `_dist_map_safe` (Dijkstra com `ASTAR_HAZARD_COST = 25` por tile H) é usado exclusivamente pelo A\*+FSM para escolher qual exit perseguir — garantindo que o A\* roteia pelo exit acessível mesmo quando o mais próximo está bloqueado por hazard.

**FSM com histerese** — limiares assimétricos (subida em 0.35, descida em 0.25) evitam oscilação rápida entre estados quando `emotion_level` está próximo do limiar.

**Contágio emocional ponderado** — `delta += 0.04 × avg_emoção_vizinhos` em vez de um valor fixo por densidade. Agentes rodeados de outros em pânico sobem emoção mais rapidamente do que agentes rodeados de agentes calmos.

**A\* com approach cell** — o goal do A\* não é o centro do tile E, mas a célula livre mais próxima cujo centro consegue tocar o exit com o raio físico do agente (6px). Resolve exits embutidos em paredes onde o tile E é inacessível diretamente.

**Clamping da força de obstacle avoidance** — mapas com tiles O individuais acumulam forças de dezenas de obstáculos adjacentes. Sem clamping, 56 tiles de parede na row 0 geravam ~1568 px/s de força (vs velocidade de 50 px/s). A força total é limitada a `agent.current_speed`.

**Cache de densidade por step** — `local_density()` era chamada N vezes dentro de um loop de N agentes → O(N³) por step. `_compute_density_cache()` roda uma vez em O(N²) por step e todos os agentes consultam em O(1).

**`REWARD_NO_PROGRESS` só em retrocesso real** — com `VELOCITY_SMOOTHING = 0.18`, o agente leva ~4 steps para mudar de célula BFS enquanto acelera. Penalizar estagnação (`progress == 0`) junto com retrocesso mascarava o sinal de progresso em ~94% dos steps. Agora a penalidade só é aplicada quando `progress < -1e-4`.

**exit_utilization com Union-Find** — `map_loader` cria 1 objeto `Exit` por tile E. Para calcular distribuição de uso entre saídas, tiles E adjacentes são agrupados por Union-Find antes de contar. Sem isso, um exit de 22 tiles (como `mall_panic`) reportaria 22 saídas distintas, com cada agente usando uma — resultado sempre 1.0.

---

## Estimativa de tempo de treino

Medições para **MacBook M3 Max** (bottleneck = loops Python da física, não PyTorch):

| Cenário | Sem render | Com render |
|---|---|---|
| Otimista (~35% da patience por stage) | ~2h | ~4h |
| Central (estimativa realista) | ~3h | ~6h |
| Pessimista (stages difíceis esgotam patience) | ~5h | ~10h |

Os stages mais pesados são 9 (`hazard_bypass_medium`), 11 (`hazard_dense_office`) e 12 (`library_hard`) — N=12 agentes com física O(N²) e mapas onde o DQN pode precisar de muitos episódios para aprender o bypass.

**Recomendações:**
- Use `--quiet --no-render` para treinos completos. Economiza ~50% do tempo.
- Monitore `logs/training_log.csv` em tempo real com `tail -f logs/training_log.csv`.
- Use `--stage N` para retomar a partir de um stage específico se o treino for interrompido.
- O estado é salvo automaticamente em `models/curriculum_state.json` a cada 50 episódios.