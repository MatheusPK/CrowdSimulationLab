# CrowdSimulationLab

Simulador de evacuação de multidões com Deep Reinforcement Learning e FSM emocional.  
Desenvolvido como parte de pesquisa de mestrado sobre como emoções afetam o comportamento individual em situações de emergência.

---

## Visão geral

O sistema compara quatro políticas de navegação em cenários de evacuação:

| Cenário | Política | FSM emocional |
|---|---|---|
| `RANDOM` | Movimento aleatório | não |
| `ASTAR` | A\* com clearance de agente | não |
| `ASTAR_FSM` | A\* com clearance de agente | **sim** |
| `DQN_FSM` | Deep Q-Network | **sim** |

A FSM modela três estados emocionais — **CALM → EVACUATE → PANIC** — com transições baseadas em proximidade de hazards, contágio emocional por densidade e decaimento passivo. Cada estado altera velocidade, força de desvio e suavização de movimento do agente.

---

## Instalação

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 2. Instalar dependências
pip install -r requirements.txt
```

**Dependências:** `pygame >= 2.0`, `torch >= 2.0`  
PyTorch com suporte a CUDA é detectado automaticamente; caso não haja GPU, roda em CPU.

---

## Configuração

Edite **apenas** `config.py` para mudar o que vai rodar. As três linhas relevantes:

```python
SCENARIO = AppScenario.ASTAR_FSM   # <-- cenário
MAP      = MAPS["office_complex"]  # <-- mapa
AGENTS   = 10                      # <-- número de agentes
```

### Cenários disponíveis

```python
AppScenario.RANDOM      # baseline aleatório
AppScenario.ASTAR       # A* puro, sem emoção
AppScenario.ASTAR_FSM   # A* com FSM emocional
AppScenario.DQN_FSM     # DQN com FSM emocional
```

### Mapas disponíveis

```python
# Pequenos — desenvolvimento e testes rápidos
"small_easy" | "small_medium" | "small_hard"

# Médios — experimentos principais
"medium_easy" | "medium_medium" | "medium_hard"

# Office — corredores e layouts de escritório
"office_simple" | "office_openplan" | "office_hazard"
"office_junction" | "office_complex"

# Mall — espaços comerciais
"mall_corridor" | "mall_atrium" | "mall_emergency"

# Experimentais — cenários específicos do mestrado
"bottleneck_hazard"     # duas salas + corredor estreito com hazard
"single_exit_room"      # sala grande com saída única (gargalo máximo)
"dual_exit_asymmetric"  # hazard bloqueia caminho curto; dilema de rota
"panic_corridor"        # 3 barreiras progressivamente mais estreitas
"open_hazard_field"     # campo aberto, 2 focos de hazard, contágio puro

# Planta real
"DI_primeiro_andar"
```

### Outros parâmetros em `config.py`

```python
RENDER        = True    # False para rodar sem janela (mais rápido)
FPS           = 30
MAX_STEPS     = 300     # limite de steps por episódio
EVAL_EPISODES = 10      # episódios de avaliação

ASTAR_HAZARD_COST = 8.0  # penalidade por tile de hazard no A*
```

Para DQN, o bloco `DQN = { ... }` dentro de `config.py` controla `mode` (TRAIN ou EVAL), `model_path`, e todos os hiperparâmetros.

---

## Como rodar

### Avaliação / visualização de uma política

```bash
python main.py
```

Flags opcionais:
```bash
python main.py --no-render        # sem janela pygame (mais rápido)
python main.py --episodes 20      # sobrescreve eval_episodes do config
python main.py --log              # salva métricas em logs/results.csv
```

Durante a janela pygame, pressione **Espaço** para pausar/retomar a renderização.

### Treino DQN com currículo progressivo

```bash
python train_curriculum.py
```

O currículo tem 14 stages (9 gerais + 5 experimentais), ordenados por complexidade crescente. O agente avança de stage automaticamente quando atinge 80% de evacuation rate nos últimos 30 episódios.

Flags úteis:
```bash
python train_curriculum.py --stage 3          # começa no stage 3
python train_curriculum.py --render           # renderiza durante o treino
python train_curriculum.py --eval             # só avalia o modelo salvo
python train_curriculum.py --model meu.pth   # usa modelo específico
python train_curriculum.py --quiet            # suprime prints detalhados
```

O treino salva checkpoints em `models/` a cada 50 episódios e retoma de onde parou se interrompido.

---

## Formato dos mapas

Mapas são arquivos `.txt` onde cada caractere representa um tile de 8×8 px:

| Char | Significado |
|---|---|
| `O` | parede / obstáculo |
| `.` | espaço livre |
| `E` | exit (saída de evacuação) |
| `H` | hazard (perigo — incêndio, fumaça) |
| `S` | spawn (posição inicial de agente) |

Para criar um novo mapa, crie um `.txt` seguindo a legenda, adicione a entrada no dicionário `MAPS` em `config.py` e coloque o arquivo em `maps/experimental/`.

---

## Métricas registradas

Cada episódio registra as métricas abaixo, disponíveis em `logs/results.csv` com `--log`:

| Métrica | Descrição |
|---|---|
| `evacuation_rate` | fração de agentes que evacuaram |
| `all_evacuated` | 1 se todos evacuaram, 0 caso contrário |
| `mean_evac_time` | tempo médio de evacuação (steps) |
| `mean_emotion_final` | média de `emotion_level` no último step |
| `emotion_variance` | variância emocional entre agentes |
| `mean_speed_ratio` | velocidade média / base\_speed (efeito da FSM) |

---

## Estrutura do projeto

```
CrowdSimulationLab/
├── main.py                    # entrypoint — rodar um cenário
├── config.py                  # configuração central (edite aqui)
├── train_curriculum.py        # treino DQN com currículo progressivo
├── requirements.txt
│
├── core/                      # tipos e enums compartilhados
│   ├── app_scenario.py        # RANDOM | ASTAR | ASTAR_FSM | DQN_FSM
│   ├── direction.py           # N NE E SE S SW W NW + vetores
│   ├── dqn_mode.py            # TRAIN | EVAL
│   ├── fsm_state.py           # CALM | EVACUATE | PANIC
│   └── map_data.py            # dataclass MapData
│
├── entities/                  # objetos do mundo
│   ├── agent.py               # Agent com emotion_level, FSMState, peak_emotion
│   ├── exit.py
│   ├── hazard.py
│   └── obstacle.py
│
├── environment/               # lógica central da simulação
│   ├── environment.py         # step · reward · FSM · BFS dist_map
│   ├── astar.py               # AStarPlanner com clearance e approach cell
│   └── map_loader.py          # parser de .txt → MapData
│
├── policies/
│   ├── astar_policy.py        # A* com cache de path e detecção de stuck
│   ├── dqn_policy.py          # DQN · ReplayBuffer · target network
│   └── random_policy.py
│
├── factories/
│   └── policy_factory.py      # constrói a política a partir do config
│
├── rendering/
│   └── renderer.py            # pygame · cor por emotion_level
│
├── models/                    # pesos salvos (gerado automaticamente)
│   └── dqn_fsm.pth
│
├── logs/                      # CSVs de experimento (gerado com --log)
│   ├── results.csv
│   └── training_log.csv
│
└── maps/
    ├── small/                 # 30×50 tiles
    ├── medium/                # 60×100 tiles
    ├── test_maps/             # office_* e mall_*
    ├── experimental/          # cenários do mestrado
    └── DI_primeiro_andar.txt  # planta real
```

---

## Decisões de implementação relevantes para o mestrado

**Distância BFS em vez de euclidiana** — o ambiente pré-computa um `dist_map` por BFS multi-source a partir de todos os exits. Essa distância é usada tanto na feature de observação do DQN quanto no reward de progresso. Em mapas com corredores fechados a euclidiana subestimava a distância real em até 6×, gerando reward negativo para ações corretas.

**FSM com histerese** — as transições de estado usam limiares assimétricos (ex: CALM→EVACUATE em `emotion ≥ 0.35`, EVACUATE→CALM em `emotion < 0.25`) para evitar oscilação nos limiares.

**Contágio emocional ponderado** — o delta emocional inclui `0.04 × avg_emotion_vizinhos` em vez de um valor fixo por densidade. Agentes perto de outros em pânico sobem de emoção mais rápido do que agentes perto de agentes calmos.

**Reward em 3 camadas** — navegação (progresso BFS normalizado pela diagonal), hazard (bônus por calma, penalidade por pânico perto do perigo) e interação social (colisão, densidade). A normalização pela diagonal torna o sinal comparável entre mapas de tamanhos diferentes.

**A\* com approach cell** — o goal do A* não é o centro do tile `E`, mas a célula livre mais próxima cujo centro toca o exit com o raio físico do agente (6px). Isso resolve o bug onde exits embutidos em paredes tornavam a evacuação impossível.