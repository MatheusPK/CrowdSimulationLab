
# ══════════════════════════════════════════════════════════════════════
# 1. MUNDO
# ══════════════════════════════════════════════════════════════════════

TILE_SIZE = 8  # px por tile

# ══════════════════════════════════════════════════════════════════════
# 2. AGENTE
# ══════════════════════════════════════════════════════════════════════

AGENT_RADIUS          = 6.0   # px — raio físico para colisão e detecção de exit
PLANNER_RADIUS_MARGIN = 5.0   # px — margem extra do A* (planner_radius = 11px total)

AGENT_BASE_SPEED      = 50.0  # px/s — velocidade no estado CALM

OBSTACLE_AVOIDANCE_DISTANCE = 12.0  # px — início da repulsão de obstáculos
OBSTACLE_AVOIDANCE_STRENGTH = 70.0  # magnitude da força repulsiva

AGENT_AVOIDANCE_DISTANCE    = 12.0  # px — início da repulsão entre agentes
AGENT_AVOIDANCE_STRENGTH    = 75.0  # magnitude

VELOCITY_SMOOTHING    = 0.25  # [0..1] — fração da velocidade target aplicada por step

# ══════════════════════════════════════════════════════════════════════
# 3. FSM / EMOÇÃO
# ══════════════════════════════════════════════════════════════════════

EMOTION_DECAY                = -0.02  # decaimento passivo por step
EMOTION_DELTA_HAZARD_CONTACT = +0.15  # incremento por contato com hazard
EMOTION_DELTA_HAZARD_VISIBLE = +0.08  # incremento por hazard visível sem contato
EMOTION_DELTA_CONTAGION      = +0.04  # coeficiente de contágio entre vizinhos

CONTAGION_RADIUS        = 35.0  # px — raio de influência emocional entre agentes

# Raio alternativo ativado por _stage_contagion_radius() para N≥12 e hazard >5%
CONTAGION_RADIUS_HIGH_N = 20.0
N_CONTAGION_THRESHOLD   = 12
HAZARD_CONTAGION_PCTG   = 0.05  # fração H/total de tiles do mapa

# Thresholds FSM com histerese
FSM_CALM_TO_EVACUATE  = 0.35
FSM_EVACUATE_TO_CALM  = 0.25
FSM_EVACUATE_TO_PANIC = 0.72
FSM_PANIC_TO_EVACUATE = 0.62

# Velocidade por estado (múltiplo de AGENT_BASE_SPEED)
FSM_SPEED_CALM      = 1.00
FSM_SPEED_EVACUATE  = 1.15
FSM_SPEED_PANIC     = 1.30
FSM_SPEED_SMOOTHING = 0.20

# Avoidance — CALM
FSM_CALM_OBS_DIST   = 16.0
FSM_CALM_OBS_STR    = 80.0
FSM_CALM_AGENT_DIST = 14.0
FSM_CALM_AGENT_STR  = 95.0
FSM_CALM_VEL_SMOOTH = 0.25

# Avoidance — EVACUATE
FSM_EVAC_OBS_DIST   = 12.0
FSM_EVAC_OBS_STR    = 70.0
FSM_EVAC_AGENT_DIST = 10.0
FSM_EVAC_AGENT_STR  = 85.0
FSM_EVAC_VEL_SMOOTH = 0.16

# Avoidance — PANIC
FSM_PANIC_OBS_DIST   = 8.0
FSM_PANIC_OBS_STR    = 55.0
FSM_PANIC_AGENT_DIST = 7.0
FSM_PANIC_AGENT_STR  = 70.0
FSM_PANIC_VEL_SMOOTH = 0.10

# ══════════════════════════════════════════════════════════════════════
# 4. PERCEPÇÃO
# ══════════════════════════════════════════════════════════════════════

HAZARD_VISION_RADIUS = 5 * TILE_SIZE  # 40px — raio de visão do hazard
LOCAL_DENSITY_RADIUS = 35.0           # px — raio para cálculo de densidade local

# ══════════════════════════════════════════════════════════════════════
# 5. A* / PLANNER
# ══════════════════════════════════════════════════════════════════════

ASTAR_HAZARD_COST     = 25.0  # custo extra por tile H no pathfinding
ASTAR_STUCK_THRESHOLD = 5     # steps sem mudar de célula antes de replanejar

# ══════════════════════════════════════════════════════════════════════
# 6. DQN
# ══════════════════════════════════════════════════════════════════════

DQN_HIDDEN_DIM         = 128        # neurônios por camada oculta (rede 17→128→128→8)
DQN_BUFFER_CAPACITY    = 100_000    # capacidade do replay buffer
DQN_TRAIN_START_SIZE   = 2_000      # mínimo de transições antes de iniciar updates
DQN_BATCH_SIZE         = 64         # amostras por gradient step
DQN_GAMMA              = 0.99       # fator de desconto
DQN_LR                 = 1e-3       # learning rate Adam
DQN_TARGET_UPDATE_FREQ = 300        # gradient steps entre sincronizações da target net
DQN_EPSILON_START      = 1.0        # epsilon inicial por stage
DQN_EPSILON_END        = 0.05       # epsilon mínimo
DQN_EPSILON_DECAY      = 1_600_000  # fallback global (stages usam o decay do CURRICULUM)
DQN_GRAD_CLIP_NORM     = 10.0       # max_norm para clip_grad_norm_

# PER — Prioritized Experience Replay (Schaul et al. 2015)
PER_ALPHA       = 0.6        # grau de priorização [0=uniforme, 1=total]
PER_BETA_START  = 0.4        # IS weights iniciais (cresce linearmente até 1.0)
PER_BETA_FRAMES = 3_000_000  # transições para beta chegar a 1.0

# ══════════════════════════════════════════════════════════════════════
# 7. RECOMPENSA
# ══════════════════════════════════════════════════════════════════════
#
# R = PROGRESS_SCALE × (prev_dist - new_dist) / diagonal
#   + EVACUATED                        (terminal)
#   + TIME_PENALTY                     (por step)
#   + NO_PROGRESS                      (se progress <= 0)
#   + HAZARD_CONTACT                   (por step dentro do hazard)
#   + HAZARD_VISIBLE_CALM × (1-emotion)(ao ver hazard sem contato)
#   + HAZARD_PANIC                     (se emotion > 0.5 perto do hazard)
#   + COLLISION                        (por colisão)
#   + DENSITY_SCALE × densidade_norm

REWARD_PROGRESS_SCALE     =  25.0   # peso do progresso BFS em direção à saída
REWARD_EVACUATED          =  100.0   # recompensa terminal por evacuação
REWARD_TIME_PENALTY       =  -0.05  # por step
REWARD_NO_PROGRESS        =  -5.0   # se progress <= 0 (inclui zero — urgência para mover)
REWARD_STAGNATION         =  -3.0   # se estagnado >= STAGNATION_THRESHOLD steps consecutivos
STAGNATION_THRESHOLD      =  8      # steps sem progresso antes de penalizar (2× o tempo legítimo ~4 steps)
REWARD_HAZARD_CONTACT     =  -3.0   # por step dentro do hazard
REWARD_HAZARD_VISIBLE_CALM =  0.10  # bônus × (1 - emotion) ao ver hazard sem contato
REWARD_HAZARD_PANIC       =  -0.5   # por step em pânico perto do hazard
REWARD_COLLISION          =  -0.3   # por colisão
REWARD_DENSITY_SCALE      =  -0.1   # × densidade normalizada

# ══════════════════════════════════════════════════════════════════════
# 8. CURRÍCULO
# ══════════════════════════════════════════════════════════════════════

CURRICULUM_PROMOTION_THRESHOLD = 0.9  # evacuation_rate média para promover
CURRICULUM_EVAL_WINDOW         = 30    # janela de episódios para a média
CURRICULUM_PATIENCE            = 500   # máx de episódios por stage
CURRICULUM_SAVE_EVERY          = 50    # salva checkpoint a cada N episódios

# Early patience: avança e reseta buffer se avg30 < THRESHOLD após AFTER episódios
CURRICULUM_EARLY_PATIENCE_AFTER     = 250
CURRICULUM_EARLY_PATIENCE_THRESHOLD = 0.15

# ══════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE
# ══════════════════════════════════════════════════════════════════════

# 1 gradient step a cada UPDATE_EVERY transições
DQN_UPDATE_EVERY = 4