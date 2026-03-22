# CrowdSimulationLab

Simulador 2D de evacuação de emergência com Deep Reinforcement Learning e modelo emocional FSM.
Pesquisa de mestrado sobre como emoções coletivas afetam o comportamento de evacuação.

Compara quatro políticas — **Random, A\*, A\*+FSM e DQN+FSM** — em cinco cenários com hazards, gargalos e dilemas de rota.

---

## Índice

1. [Instalação](#instalação)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Configuração](#configuração)
4. [Como executar](#como-executar)
5. [As quatro políticas](#as-quatro-políticas)
6. [Modelo emocional FSM](#modelo-emocional-fsm)
7. [Observação do DQN — 17 features](#observação-do-dqn--17-features)
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
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Dependências:** `pygame >= 2.0`, `torch >= 2.0`

CUDA e Apple MPS são detectados automaticamente. Sem GPU, roda em CPU sem alteração de código.

---

## Estrutura do projeto

```
CrowdSimulationLab/
│
├── main.py                    # entrypoint: rodar/visualizar qualquer política
├── config.py                  # configuração central do experimento
├── train_curriculum.py        # treino DQN com currículo progressivo de 18 stages
├── simulation_params.py       # TODOS os hiperparâmetros — edite apenas aqui
├── requirements.txt
│
├── core/
│   ├── app_scenario.py        # enum: RANDOM | ASTAR | ASTAR_FSM | DQN_FSM
│   ├── direction.py           # enum: N NE E SE S SW W NW + vetores
│   ├── dqn_mode.py            # enum: TRAIN | EVAL
│   ├── fsm_state.py           # enum: CALM | EVACUATE | PANIC
│   └── map_data.py            # dataclass MapData
│
├── entities/
│   ├── agent.py
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
│   ├── dqn_policy.py          # DQN · PER · target network · UPDATE_EVERY
│   └── random_policy.py
│
├── factories/
│   └── policy_factory.py
│
├── rendering/
│   └── renderer.py            # pygame com suporte a sprites e escala visual
│
├── models/
│   ├── dqn_fsm.pth            # modelo principal (atualizado a cada stage)
│   ├── ckpt_sN_epM.pth        # checkpoint a cada SAVE_EVERY episódios
│   ├── ckpt_sN_final.pth      # checkpoint ao final de cada stage
│   └── curriculum_state.json  # estado para retomada automática
│
├── logs/
│   ├── results.csv            # resultados de main.py --log
│   └── training_log.csv       # log episódio-a-episódio do treino
│
└── maps/
    ├── train/                 # 18 mapas do currículo (+ di_style para fine-tuning)
    └── eval/                  # 5 mapas de avaliação
```

---

## Configuração

**`config.py`** controla o experimento via `main.py`. As três linhas principais:

```python
SCENARIO = AppScenario.ASTAR_FSM   # política a rodar
MAP      = EVAL_MAPS["mall_panic"] # mapa
AGENTS   = 12                      # número de agentes
```

### Cenários

| Cenário | Política | FSM | Custo de hazard no path |
|---|---|---|---|
| `RANDOM` | Ação aleatória | não | — |
| `ASTAR` | A\* com clearance | não | não |
| `ASTAR_FSM` | A\* com clearance | **sim** | **sim** (25 por tile H) |
| `DQN_FSM` | Deep Q-Network | **sim** | aprende por experiência |

### Parâmetros gerais

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `RENDER` | `True` | `False` remove pygame (~2–3× mais rápido) |
| `FPS` | 30 | Frames por segundo |
| `DT` | 0.1 | Timestep (s/step) |
| `MAX_STEPS` | 450 | Limite de steps por episódio |
| `EVAL_EPISODES` | 10 | Episódios padrão em `main.py` |

---

## Como executar

### Visualização interativa

```bash
# Edite SCENARIO, MAP e AGENTS em config.py, depois:
python main.py

# Janela 2× maior (sem impacto na simulação):
python main.py --scale 2
```

Controles: **Espaço** pausa/retoma · fechar janela encerra o episódio.

### Rodar qualquer política

```bash
python main.py --no-render --episodes 30 --log
```

| Flag | Efeito |
|---|---|
| `--no-render` | Sem pygame |
| `--episodes N` | Número de episódios |
| `--scale N` | Escala visual da janela (ex: `--scale 2`) |
| `--log` | Salva em `logs/results.csv` |

### Treino completo com currículo

```bash
python train_curriculum.py                           # treino completo (18 stages)
python train_curriculum.py --quiet                   # sem prints detalhados
python train_curriculum.py --render --scale 2        # com visualização 2×
python train_curriculum.py --stage 9                 # começa no stage 9
python train_curriculum.py --model models/ckpt.pth   # usa checkpoint específico
python train_curriculum.py --fine-tune               # fine-tuning em di_style
```

O treino salva automaticamente:
- `models/dqn_fsm.pth` — modelo principal (atualizado ao fim de cada stage)
- `models/ckpt_sN_epM.pth` — checkpoint a cada 50 episódios
- `models/ckpt_sN_final.pth` — checkpoint ao final de cada stage (promovido ou não)
- `models/curriculum_state.json` — estado para retomada automática
- `logs/training_log.csv` — log completo episódio a episódio

### Avaliar o DQN nos 5 mapas de eval

```bash
python train_curriculum.py --eval --episodes 30
python train_curriculum.py --eval --episodes 30 --model models/ckpt_s15_final.pth
```

Output:

```
======================================================
AVALIAÇÃO — models/dqn_fsm.pth
======================================================
Mapa                        evac  panic  peak_em  haz_ct  r_util     var   time
---------------------------------------------------------------------------
  library_bottleneck        0.88   0.14    0.136    0.00   0.950  0.0744    61
  office_single_exit        0.62   0.16    0.162    0.08   0.800  0.0412   203
  mall_panic                0.74   0.17    0.198    0.05   0.667  0.1338    74
  school_evacuation         0.58   0.09    0.111    0.00   0.500  0.0457   136
  di_emergency              0.36   0.15    0.145    0.00   1.000  0.0041    51
```

### Comparar as 4 políticas

```bash
# Trocar SCENARIO em config.py e rodar cada um:
python main.py --no-render --episodes 30 --log  # repita para RANDOM, ASTAR, ASTAR_FSM, DQN_FSM
```

Todos os resultados acumulam em `logs/results.csv`. A coluna `scenario` identifica cada política.

**Protocolo recomendado:** 30 episódios × 4 políticas × 5 mapas = 600 runs. Mann-Whitney U test (não-paramétrico), p < 0.05.

---

## As quatro políticas

### Random
Ação aleatória entre 8 direções. Baseline inferior — nenhuma estrutura de navegação.

### A\* e A\*+FSM
Pathfinding com 8-conectividade e clearance de 11px (raio físico + margem do planner). O goal não é o centro do tile `E`, mas a célula livre mais próxima que consegue tocar o exit com o raio físico do agente (6px) — resolve exits embutidos em paredes.

`ASTAR_FSM` usa `_dist_map_safe` (Dijkstra com penalidade de 25 por tile H) para escolher qual exit perseguir em mapas com múltiplos exits. Em `mall_panic`, por exemplo, isso faz o agente ignorar os exits bloqueados por hazard e ir pelo exit seguro à direita.

Anti-stuck: replaneja se o agente não muda de célula por 5 steps consecutivos.

### DQN+FSM
Rede 17 → 128 → 128 → 8. Multi-agent parameter sharing: todos os agentes compartilham a rede e contribuem transições independentes ao buffer. Com N=12 agentes × 450 steps, cada episódio gera ~5400 transições — captura contágio emocional, congestionamento e dilemas de rota que treino single-agent não reproduz.

Usa Prioritized Experience Replay (PER, Schaul et al. 2015): transições com TD-error alto (evacuações bem-sucedidas, primeiros contatos com hazard) são amostradas com mais frequência, resolvendo o sinal esparso de mapas com hazard.

---

## Modelo emocional FSM

```
CALM ──(≥ 0.35)──► EVACUATE ──(≥ 0.72)──► PANIC
     ◄──(< 0.25)──           ◄──(< 0.62)──
```

A histerese (limiares de subida > descida) evita oscilação quando `emotion_level` está próximo do limiar.

### Efeitos por estado

| Estado | Velocidade | Avoidance obstáculo | Avoidance agente |
|---|---|---|---|
| CALM | 1.0× (50 px/s) | dist=16, str=80 | dist=14, str=95 |
| EVACUATE | 1.15× | dist=12, str=70 | dist=10, str=85 |
| PANIC | 1.30× | dist=8, str=55 | dist=7, str=70 |

No pânico, avoidance fraco causa aglomeração realista — agentes se empurram, simulando pisoteamento.

### Evolução do emotion_level por step

| Evento | Delta |
|---|---|
| Decaimento passivo | −0.02 |
| Contato com hazard | +0.15 |
| Hazard visível (raio 40px, sem contato) | +0.08 |
| Contágio emocional | +0.04 × média de emoção dos vizinhos (raio 35px) |

Para stages com N≥12 e hazard >5%, o raio de contágio é reduzido automaticamente de 35px para 20px — limita cascata emocional a vizinhos imediatos.

---

## Observação do DQN — 17 features

| Idx | Feature | Descrição |
|---|---|---|
| 0–1 | `pos_x / width`, `pos_y / height` | Posição normalizada |
| 2–3 | `(exit_cx − x) / width`, `(exit_cy − y) / height` | Vetor para o exit mais próximo por BFS |
| 4 | `dist_BFS / diagonal` | Distância BFS normalizada |
| 5 | `atan2(dy, dx) / π` | Ângulo para o exit em [−1, 1] |
| 6 | `1 se hazard visível` | Booleano |
| 7 | `dist_hazard / (2 × raio_visão)` | Distância ao hazard mais próximo |
| 8 | `1 se em contato com hazard` | Booleano |
| 9 | `emotion_level` | [0, 1] |
| 10 | `fsm_state / 2` | 0=CALM, 0.5=EVACUATE, 1=PANIC |
| 11–12 | `vx / speed`, `vy / speed` | Velocidade normalizada |
| 13 | `min(densidade / 8, 1)` | Agentes no raio de 35px |
| 14 | `dist_obstáculo / (2 × avoidance_dist)` | Distância à parede mais próxima |
| 15 | `current_speed / base_speed` | Efeito da FSM na velocidade |
| 16 | `1 se evacuado` | Booleano |

O DQN recebe `_dist_map` (BFS puro, sem custo de hazard) nas features 2–4, não o `_dist_map_safe`. Isso força o DQN a aprender o tradeoff rota-perigosa/segura por experiência — que é o comportamento central investigado.

---

## Reward

```
R = PROGRESS_SCALE × (prev_dist − new_dist) / diagonal   ← sinal principal
  + EVACUATED                                              ← terminal (+80)
  + TIME_PENALTY                                           ← por step (−0.05)
  + NO_PROGRESS         se progress < −1e-4               ← retrocesso real (−0.3)
  + HAZARD_CONTACT      por step dentro do hazard          ← (−3.0)
  + HAZARD_VISIBLE_CALM × (1 − emotion)                   ← (+0.10)
  + HAZARD_PANIC        se emotion > 0.5 e hazard visível  ← (−0.5)
  + COLLISION                                              ← (−0.3)
  + DENSITY_SCALE × densidade_norm                         ← (−0.1)
```

`REWARD_NO_PROGRESS` só é ativado em retrocesso real (`progress < −1e-4`), não em estagnação — com `VELOCITY_SMOOTHING = 0.18`, o agente leva ~4 steps para mudar de célula BFS ao acelerar.

---

## Currículo de treino

18 stages progressivos. Promoção: `evacuation_rate ≥ 0.80` nos últimos 30 episódios. Patience: 500 episódios por stage. Early patience: se `avg30 < 0.15` após 200 episódios, avança e reseta o buffer.

| Stage | Mapa | N | O que ensina |
|---|---|---|---|
| 1 | mall_small | 4 | Navegação pura |
| 2 | school_small | 4 | Single-exit sem hazard |
| 3 | office_wing_small | 4 | Corredor sem hazard |
| 4 | library_small | 4 | Primeiro hazard + FSM |
| 5 | library_medium | 6 | Hazard médio, multi-exit |
| 6 | hazard_corridor_small ★ | 4 | Dilema: rota perigosa vs segura |
| 7 | hazard_near_exit_small ★★ | 6 | Hazard frente ao exit — passar perto do hazard |
| 8 | school_floor | 8 | Escala coletiva gradual |
| 9 | office_wing_medium | 10 | Base antes de N=12 |
| 10 | hazard_near_exit_medium ★★ | 10 | Réplica do padrão office_single_exit |
| 11–14 | bridge_* (4 mapas) | 11→12 | Ponte N=11→12 sem hazard → hazard suave |
| 15 | hazard_bypass_medium ★ | 12 | H flanqueia exit principal, bypass |
| 16 | mall_medium | 12 | Ambiente aberto, hazard distribuído |
| 17 | hazard_dense_office ★ | 12 | Hazard em 8% da área |
| 18 | library_hard | 12 | Gargalo + hazard |

★ Dilema de rota · ★★ Hazard frente ao exit (padrão dos mapas de eval)

A progressão de N (4→6→8→10→11→12) introduz dinâmica coletiva gradualmente. O salto direto para N=12 com hazard alto causava cascata emocional irrecuperável nos ciclos anteriores.

### Buffer reset entre stages

Se um stage esgota a patience sem promover, o replay buffer é descartado antes do próximo stage. Isso elimina a contaminação por transições ruins que causava *catastrophic forgetting* nos stages seguintes.

### Fine-tuning opcional

```bash
python train_curriculum.py --fine-tune
```

Treina em `di_style.txt` após o currículo principal. Prepara especificamente para `di_emergency` (hazard perto do exit superior).

---

## Mapas

### Formato (.txt)

Cada caractere = tile de 8×8 px:

| Char | Tile |
|---|---|
| `O` | Parede / obstáculo |
| `.` | Espaço livre |
| `E` | Exit |
| `H` | Hazard |
| `S` | Spawn |

**Regras:** borda de `O` no perímetro · spawns ≥ agentes configurados · exits alcançáveis por BFS.

### Adicionar um mapa ao projeto

```python
# config.py — para visualizar com main.py:
ALL_MAPS["meu_mapa"] = "maps/train/meu_mapa.txt"

# train_curriculum.py — para incluir no currículo:
# Adicione uma linha em CURRICULUM:
("meu_mapa", "maps/train/meu_mapa.txt", 8, 400, 190_000),
# (nome, caminho, n_agents, max_steps, epsilon_decay)
# epsilon_decay = ep_alvo × (N × max_steps / 4)
```

### Mapas de avaliação

| Mapa | H% | Exits | Fenômeno |
|---|---|---|---|
| `library_bottleneck` | 2.1% | 1 | Gargalo com hazard lateral |
| `office_single_exit` | 1.3% | 1 | Exit único com hazard no corredor |
| `mall_panic` | 12.2% | 3 | Dilema: exit perigoso vs exits seguros |
| `school_evacuation` | 2.8% | 2 | Hazard frente ao exit principal |
| `di_emergency` | 2.6% | 1 | Hazard perto do exit superior |

Todos avaliados com N=12 agentes.

---

## Métricas

### Eficiência de evacuação

| Métrica | Tipo | Descrição |
|---|---|---|
| `evacuation_rate` | [0, 1] | Fração de agentes evacuados |
| `all_evacuated` | bool | 100% evacuaram |
| `mean_evacuation_time` | steps | Média de tempo dos que evacuaram |
| `steps` | int | Steps totais do episódio |

### Comportamento emocional

| Métrica | Tipo | Descrição |
|---|---|---|
| `mean_emotion_final` | [0, 1] | Média de `emotion_level` no último step |
| `mean_peak_emotion` | [0, 1] | Média do pico emocional por agente |
| `emotion_variance` | float | Variância entre agentes — métrica central do paper |
| `panic_rate` | [0, 1] | Fração que atingiu PANIC |

`emotion_variance` alta durante o episódio indica dilema de rota ativo; convergência para baixo indica rota segura encontrada.

### Qualidade de rota

| Métrica | Tipo | Descrição |
|---|---|---|
| `mean_speed_ratio` | float | `current_speed / base_speed` |
| `hazard_contact_rate` | [0, 1] | Fração que tocou o hazard |
| `exit_utilization` | [0, 1] | Equilíbrio de uso entre exits. 1.0 = perfeito; 0.0 = todos num exit só |

`exit_utilization` usa Union-Find para agrupar tiles `E` adjacentes antes de contar — sem isso, um exit de 22 tiles reportaria 22 saídas distintas.

---

## Parâmetros de simulação

**Edite apenas `simulation_params.py`.** Todos os outros arquivos importam daqui.

### Agente

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `AGENT_RADIUS` | 6.0 px | Raio físico |
| `PLANNER_RADIUS_MARGIN` | 5.0 px | Margem do A* (planner radius = 11px total) |
| `AGENT_BASE_SPEED` | 50.0 px/s | Velocidade no estado CALM (= 5px/step com dt=0.1) |
| `VELOCITY_SMOOTHING` | 0.18 | Filtro exponencial [0–1] |
| `OBSTACLE_AVOIDANCE_DISTANCE` | 14.0 px | Início da repulsão de paredes |
| `OBSTACLE_AVOIDANCE_STRENGTH` | 70.0 | Magnitude |
| `AGENT_AVOIDANCE_DISTANCE` | 12.0 px | Início da repulsão entre agentes |
| `AGENT_AVOIDANCE_STRENGTH` | 90.0 | Magnitude |

### FSM

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `FSM_CALM_TO_EVACUATE` | 0.35 | Limiar CALM → EVACUATE |
| `FSM_EVACUATE_TO_CALM` | 0.25 | Limiar EVACUATE → CALM (histerese) |
| `FSM_EVACUATE_TO_PANIC` | 0.72 | Limiar EVACUATE → PANIC |
| `FSM_PANIC_TO_EVACUATE` | 0.62 | Limiar PANIC → EVACUATE (histerese) |
| `FSM_SPEED_PANIC` | 1.30 | Velocidade em pânico (×base) |

### Emoção e contágio

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `EMOTION_DECAY` | −0.02 | Decaimento passivo por step |
| `EMOTION_DELTA_HAZARD_CONTACT` | +0.15 | Delta por contato com hazard |
| `EMOTION_DELTA_HAZARD_VISIBLE` | +0.08 | Delta por hazard visível |
| `EMOTION_DELTA_CONTAGION` | +0.04 | Coeficiente de contágio |
| `CONTAGION_RADIUS` | 35.0 px | Raio padrão de contágio |
| `CONTAGION_RADIUS_HIGH_N` | 20.0 px | Raio reduzido para N≥12 e hazard >5% |
| `HAZARD_VISION_RADIUS` | 40.0 px | Raio de visão do hazard |

### A\*

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `ASTAR_HAZARD_COST` | 25.0 | Custo extra por tile H. 0 = ignora; 25 = evita se alternativa ≤25 tiles mais longa |
| `ASTAR_STUCK_THRESHOLD` | 5 | Steps sem mudar de célula antes de replanejar |

### DQN

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `DQN_HIDDEN_DIM` | 128 | Neurônios por camada oculta |
| `DQN_BUFFER_CAPACITY` | 100 000 | Capacidade do replay buffer |
| `DQN_BATCH_SIZE` | 64 | Amostras por gradient step |
| `DQN_GAMMA` | 0.99 | Fator de desconto |
| `DQN_LR` | 1e-3 | Learning rate Adam |
| `DQN_TARGET_UPDATE_FREQ` | 300 | Gradient steps entre sincronizações da target net |
| `DQN_UPDATE_EVERY` | 4 | 1 gradient step a cada N transições (ratio 4:1 — padrão DQN original) |
| `PER_ALPHA` | 0.6 | Grau de priorização do PER |
| `PER_BETA_START` | 0.4 | IS weights iniciais |

### Reward

| Parâmetro | Padrão |
|---|---|
| `REWARD_PROGRESS_SCALE` | 15.0 |
| `REWARD_EVACUATED` | 80.0 |
| `REWARD_TIME_PENALTY` | −0.05 |
| `REWARD_NO_PROGRESS` | −0.3 |
| `REWARD_HAZARD_CONTACT` | −3.0 |
| `REWARD_HAZARD_VISIBLE_CALM` | 0.10 |
| `REWARD_HAZARD_PANIC` | −0.5 |
| `REWARD_COLLISION` | −0.3 |
| `REWARD_DENSITY_SCALE` | −0.1 |

### Currículo

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `CURRICULUM_PROMOTION_THRESHOLD` | 0.80 | evacuation_rate para promover |
| `CURRICULUM_EVAL_WINDOW` | 30 | Janela de episódios |
| `CURRICULUM_PATIENCE` | 500 | Máx episódios por stage |
| `CURRICULUM_SAVE_EVERY` | 50 | Frequência de checkpoint |
| `CURRICULUM_EARLY_PATIENCE_AFTER` | 200 | Episódios antes de verificar early stop |
| `CURRICULUM_EARLY_PATIENCE_THRESHOLD` | 0.15 | avg30 mínimo para continuar |

---

## Decisões de implementação

**BFS em vez de distância euclidiana** — em mapas com corredores, a euclidiana subestima a distância real em até 6×, gerando reward negativo para ações corretas. O BFS reflete a geometria real do mapa.

**Dois dist_maps separados** — `_dist_map` (BFS puro) para as features do DQN; `_dist_map_safe` (Dijkstra com custo de hazard) exclusivamente para o A\*+FSM. O DQN aprende o tradeoff rota-perigosa/segura por experiência.

**Histerese na FSM** — limiares assimétricos evitam oscilação quando `emotion_level` está próximo do limiar.

**Contágio emocional ponderado** — `delta += 0.04 × avg_emoção_vizinhos` em vez de valor fixo. Grupos em pânico elevam mais a emoção dos vizinhos.

**A\* com approach cell** — o goal é a célula livre mais próxima que toca o exit com raio físico 6px, não o centro do tile E. Resolve exits embutidos em paredes.

**Clamping da força de avoidance** — sem clamping, 56 tiles de parede em sequência acumulam ~1568 px/s de força repulsiva (vs 50 px/s de velocidade). A força total é limitada a `agent.current_speed`.

**Cache de densidade por step** — `local_density()` era O(N²) chamada N vezes → O(N³)/step. `_compute_density_cache()` roda uma vez em O(N²)/step.

**`REWARD_NO_PROGRESS` só em retrocesso real** — estagnação não é penalizada porque com `VELOCITY_SMOOTHING=0.18` o agente leva ~4 steps para mudar de célula BFS ao acelerar.

**exit_utilization com Union-Find** — agrupa tiles E adjacentes em saídas antes de contar distribuição de uso. Sem isso, um exit de 22 tiles reportaria 22 saídas distintas.

**Buffer reset entre stages** — se um stage esgota a patience, o buffer é descartado antes do próximo. Elimina *catastrophic forgetting* causado por transições ruins de stages anteriores.

**UPDATE_EVERY=4** — 1 gradient step a cada 4 transições (ratio 4:1, padrão DQN original). Com N=12 agentes, o código anterior fazia 12 gradient steps por env step, dominando o tempo de treino mesmo sem render.

**Escala visual no renderer** — `--scale N` amplia a janela N× sem afetar coordenadas físicas. Útil para visualização em monitores de alta resolução ou mapas pequenos.

---

## Estimativa de tempo de treino

Medido em MacBook M3 Max (bottleneck = loops Python da física, não PyTorch):

| Cenário | Sem render | Com render |
|---|---|---|
| Otimista (~35% da patience por stage) | ~3h | ~5h |
| Central | ~5h | ~8h |
| Pessimista (stages difíceis esgotam patience) | ~8h | ~14h |

Os stages mais pesados são 15 (`hazard_bypass_medium`), 17 (`hazard_dense_office`) e 18 (`library_hard`) — N=12 com física O(N²) e dilemas de rota que exigem exploração longa.

**Recomendações:**
```bash
# Treino silencioso sem render — mais rápido
python train_curriculum.py --quiet

# Monitorar log em tempo real
tail -f logs/training_log.csv

# Retomar após interrupção (automático via curriculum_state.json)
python train_curriculum.py

# Retomar a partir de stage específico
python train_curriculum.py --stage 15 --model models/ckpt_s14_final.pth
```