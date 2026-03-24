"""
simulation_params.py — fonte única de todos os parâmetros de simulação.

Changelog:
  - Avoidance strengths reduzidos ~35% em todos os estados FSM.
    Motivo: avoidance forte sobrescrevia a direção escolhida pelo DQN.
  - Rewards e curriculum alinhados com melhor resultado de eval.
"""

# ══════════════════════════════════════════════════════════════════════
# 1. MUNDO
# ══════════════════════════════════════════════════════════════════════

TILE_SIZE = 8

# ══════════════════════════════════════════════════════════════════════
# 2. AGENTE
# ══════════════════════════════════════════════════════════════════════

AGENT_RADIUS          = 6.0
PLANNER_RADIUS_MARGIN = 5.0
AGENT_BASE_SPEED      = 50.0

# Strengths reduzidos de 70/90 para 45/50 — não sobrescreve a política DQN
OBSTACLE_AVOIDANCE_DISTANCE = 12.0
OBSTACLE_AVOIDANCE_STRENGTH = 45.0   # era 70.0

AGENT_AVOIDANCE_DISTANCE    = 12.0
AGENT_AVOIDANCE_STRENGTH    = 50.0   # era 90.0

VELOCITY_SMOOTHING    = 0.25

# ══════════════════════════════════════════════════════════════════════
# 3. FSM / EMOÇÃO
# ══════════════════════════════════════════════════════════════════════

EMOTION_DECAY                = -0.02
EMOTION_DELTA_HAZARD_CONTACT = +0.15
EMOTION_DELTA_HAZARD_VISIBLE = +0.08
EMOTION_DELTA_CONTAGION      = +0.04

CONTAGION_RADIUS        = 35.0
CONTAGION_RADIUS_HIGH_N = 20.0
N_CONTAGION_THRESHOLD   = 12
HAZARD_CONTAGION_PCTG   = 0.05

FSM_CALM_TO_EVACUATE  = 0.35
FSM_EVACUATE_TO_CALM  = 0.25
FSM_EVACUATE_TO_PANIC = 0.72
FSM_PANIC_TO_EVACUATE = 0.62

FSM_SPEED_CALM      = 1.00
FSM_SPEED_EVACUATE  = 1.15
FSM_SPEED_PANIC     = 1.30
FSM_SPEED_SMOOTHING = 0.20

# CALM — strengths reduzidos de 80/95 para 50/60
FSM_CALM_OBS_DIST   = 16.0
FSM_CALM_OBS_STR    = 50.0   # era 80.0
FSM_CALM_AGENT_DIST = 14.0
FSM_CALM_AGENT_STR  = 60.0   # era 95.0
FSM_CALM_VEL_SMOOTH = 0.25

# EVACUATE — strengths reduzidos de 70/85 para 42/52
FSM_EVAC_OBS_DIST   = 12.0
FSM_EVAC_OBS_STR    = 42.0   # era 70.0
FSM_EVAC_AGENT_DIST = 10.0
FSM_EVAC_AGENT_STR  = 52.0   # era 85.0
FSM_EVAC_VEL_SMOOTH = 0.16

# PANIC — strengths reduzidos de 55/70 para 35/42
FSM_PANIC_OBS_DIST   = 8.0
FSM_PANIC_OBS_STR    = 35.0   # era 55.0
FSM_PANIC_AGENT_DIST = 7.0
FSM_PANIC_AGENT_STR  = 42.0   # era 70.0
FSM_PANIC_VEL_SMOOTH = 0.10

# ══════════════════════════════════════════════════════════════════════
# 4. PERCEPÇÃO
# ══════════════════════════════════════════════════════════════════════

HAZARD_VISION_RADIUS = 5 * TILE_SIZE  # 40px
LOCAL_DENSITY_RADIUS = 35.0

# ══════════════════════════════════════════════════════════════════════
# 5. A* / PLANNER
# ══════════════════════════════════════════════════════════════════════

ASTAR_HAZARD_COST     = 25.0
ASTAR_STUCK_THRESHOLD = 5

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

PER_ALPHA       = 0.6
PER_BETA_START  = 0.4
PER_BETA_FRAMES = 3_000_000

# ══════════════════════════════════════════════════════════════════════
# 7. RECOMPENSA
# ══════════════════════════════════════════════════════════════════════

REWARD_PROGRESS_SCALE      =  40.0   # era 25 — sinal denso mais forte
REWARD_EVACUATED           =  80.0   # era 100 — reduzido para equilibrar com progress
REWARD_TIME_PENALTY        =  -0.05
REWARD_NO_PROGRESS         =  -1.5   # era -5 — ORCA às vezes para brevemente para desviar
REWARD_HAZARD_CONTACT      =  -3.0
REWARD_HAZARD_VISIBLE_CALM =   0.10
REWARD_HAZARD_PANIC        =  -0.5
REWARD_COLLISION           =   0.0   # removido — ORCA garante física
REWARD_DENSITY_SCALE       =  -0.1

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