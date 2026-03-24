"""
simulation_params.py — parâmetros de simulação para DQN+FSM+ORCA.

A física de avoidance agente-agente e agente-parede é gerenciada pelo ORCA
(environment_orca.py). Os parâmetros de força do modelo antigo foram removidos.

Grupos:
  1. MUNDO
  2. AGENTE      — geometria e velocidade base
  3. FSM/EMOÇÃO  — thresholds, velocidade e smoothing por estado
  4. PERCEPÇÃO   — raios de visão e densidade
  5. A*          — custo de hazard no planner (A* e A*+FSM policies)
  6. DQN         — arquitetura e hiperparâmetros
  7. RECOMPENSA
  8. CURRÍCULO
  9. PERFORMANCE
"""

# ══════════════════════════════════════════════════════════════════════
# 1. MUNDO
# ══════════════════════════════════════════════════════════════════════

TILE_SIZE = 8  # px por tile

# ══════════════════════════════════════════════════════════════════════
# 2. AGENTE
# ══════════════════════════════════════════════════════════════════════

AGENT_RADIUS          = 6.0    # px — raio físico para colisão e detecção de exit
PLANNER_RADIUS_MARGIN = 5.0    # px — margem extra do A* (planner_radius = 11px total)
AGENT_BASE_SPEED      = 50.0   # px/s — velocidade no estado CALM

# Velocity smoothing base — sobrescrito pelo update_fsm por estado FSM
VELOCITY_SMOOTHING    = 0.25

# ══════════════════════════════════════════════════════════════════════
# 3. FSM / EMOÇÃO
# ══════════════════════════════════════════════════════════════════════

EMOTION_DECAY                = -0.02   # decaimento passivo por step
EMOTION_DELTA_HAZARD_CONTACT = +0.15   # incremento por contato com hazard
EMOTION_DELTA_HAZARD_VISIBLE = +0.08   # incremento por hazard visível sem contato
EMOTION_DELTA_CONTAGION      = +0.04   # coeficiente de contágio entre vizinhos

CONTAGION_RADIUS        = 35.0   # px — raio de influência emocional
CONTAGION_RADIUS_HIGH_N = 20.0   # raio reduzido para N≥12 com hazard >5%
N_CONTAGION_THRESHOLD   = 12
HAZARD_CONTAGION_PCTG   = 0.05

# Thresholds FSM com histerese
FSM_CALM_TO_EVACUATE  = 0.35
FSM_EVACUATE_TO_CALM  = 0.25
FSM_EVACUATE_TO_PANIC = 0.72
FSM_PANIC_TO_EVACUATE = 0.62

# Velocidade por estado (múltiplo de AGENT_BASE_SPEED)
# Também modula max_speed no ORCA — agentes em pânico se movem mais rápido
FSM_SPEED_CALM      = 1.00
FSM_SPEED_EVACUATE  = 1.15
FSM_SPEED_PANIC     = 1.30
FSM_SPEED_SMOOTHING = 0.20

# Velocity smoothing por estado — modula responsividade a mudanças de direção
# PANIC=0.10: reage rápido (empurra-empurra); CALM=0.25: movimentos suaves
FSM_CALM_VEL_SMOOTH  = 0.25
FSM_EVAC_VEL_SMOOTH  = 0.16
FSM_PANIC_VEL_SMOOTH = 0.10

# ══════════════════════════════════════════════════════════════════════
# 4. PERCEPÇÃO
# ══════════════════════════════════════════════════════════════════════

HAZARD_VISION_RADIUS = 5 * TILE_SIZE   # 40px — raio de visão do hazard
LOCAL_DENSITY_RADIUS = 35.0            # px — raio para cálculo de densidade local

# ══════════════════════════════════════════════════════════════════════
# 5. A* / PLANNER
# ══════════════════════════════════════════════════════════════════════

ASTAR_HAZARD_COST     = 25.0   # custo extra por tile H no pathfinding (A* e A*+FSM)
ASTAR_STUCK_THRESHOLD = 5      # steps sem mudar de célula antes de replanejar

# ══════════════════════════════════════════════════════════════════════
# 6. DQN
# ══════════════════════════════════════════════════════════════════════

DQN_HIDDEN_DIM         = 128
DQN_BUFFER_CAPACITY    = 100_000
DQN_TRAIN_START_SIZE   = 2_000
DQN_BATCH_SIZE         = 64
DQN_GAMMA              = 0.99
DQN_LR                 = 1e-3
DQN_TARGET_UPDATE_FREQ = 300
DQN_EPSILON_START      = 1.0
DQN_EPSILON_END        = 0.05
DQN_EPSILON_DECAY      = 1_600_000
DQN_GRAD_CLIP_NORM     = 10.0

# PER — Prioritized Experience Replay (Schaul et al. 2015)
PER_ALPHA       = 0.6
PER_BETA_START  = 0.4
PER_BETA_FRAMES = 3_000_000

# ══════════════════════════════════════════════════════════════════════
# 7. RECOMPENSA
# ══════════════════════════════════════════════════════════════════════
#
# R = PROGRESS_SCALE × (prev_dist - new_dist)   ← sinal denso principal
#   + EVACUATED                                  (terminal)
#   + TIME_PENALTY                               (por step)
#   + NO_PROGRESS                                (se progress ≤ 0)
#   + HAZARD_CONTACT                             (por step dentro do hazard)
#   + HAZARD_VISIBLE_CALM × (1 - emotion)        (ao ver hazard sem contato)
#   + HAZARD_PANIC                               (se emotion > 0.5 perto do hazard)
#   + DENSITY_SCALE × densidade_norm

REWARD_PROGRESS_SCALE      =  60.0   # era 40 — sinal denso mais dominante
REWARD_EVACUATED           =  80.0
REWARD_TIME_PENALTY        =  -0.05
REWARD_NO_PROGRESS         =  -1.5   # suave — ORCA pode parar brevemente para desviar
REWARD_HAZARD_CONTACT      =  -3.0
REWARD_HAZARD_VISIBLE_CALM =   0.10
REWARD_HAZARD_PANIC        =  -0.1   # era -0.5 — reduzido para não sobrescrever BFS em pânico
REWARD_COLLISION           =   0.0   # não usado — ORCA garante separação física
REWARD_DENSITY_SCALE       =  -0.1

# Stagnation inteligente — baseado em histórico de posições (12 steps / 1.2s).
# Distingue parado de oscilando. Mais suave que NO_PROGRESS porque pode ser
# física legítima do ORCA desviando de outro agente.
REWARD_STAGNATION_STUCK      = -2.0   # agente quase imóvel (var < 4px²)
REWARD_STAGNATION_OSCILLATE  = -3.0   # agente oscilando sem progredir (mais grave)

# ══════════════════════════════════════════════════════════════════════
# 8. CURRÍCULO
# ══════════════════════════════════════════════════════════════════════

CURRICULUM_PROMOTION_THRESHOLD = 0.80
CURRICULUM_EVAL_WINDOW         = 30
CURRICULUM_PATIENCE            = 500
CURRICULUM_SAVE_EVERY          = 50

CURRICULUM_EARLY_PATIENCE_AFTER     = 250
CURRICULUM_EARLY_PATIENCE_THRESHOLD = 0.15

# ══════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE
# ══════════════════════════════════════════════════════════════════════

DQN_UPDATE_EVERY = 4