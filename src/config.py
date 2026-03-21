"""
config.py — ponto único de configuração da simulação.

Para rodar um experimento, edite as três linhas marcadas com  <--
e execute:  python main.py

Cenários disponíveis:
    AppScenario.RANDOM      baseline aleatório, sem FSM
    AppScenario.ASTAR       A* puro,            sem FSM
    AppScenario.ASTAR_FSM   A* com emoção,      com FSM
    AppScenario.DQN_FSM     DQN com emoção,     com FSM

Para treino DQN com currículo progressivo:
    python train_curriculum.py
"""

from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from simulation_params import (
    ASTAR_HAZARD_COST,
    DQN_HIDDEN_DIM, DQN_BUFFER_CAPACITY, DQN_TRAIN_START_SIZE,
    DQN_BATCH_SIZE, DQN_GAMMA, DQN_LR, DQN_TARGET_UPDATE_FREQ,
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY,
)

# ── Mapas de treino (usados pelo currículo em train_curriculum.py) ────────────

TRAIN_MAPS = {
    # Pequenos — 26×56, sem hazard (navegação pura)
    "library_small":      "maps/train/library_small.txt",
    "office_wing_small":  "maps/train/office_wing_small.txt",
    "mall_small":         "maps/train/mall_small.txt",
    "school_small":       "maps/train/school_small.txt",

    # Médios — 36×80, hazard leve (introdução da FSM)
    "library_medium":     "maps/train/library_medium.txt",
    "mall_food_court":    "maps/train/mall_food_court.txt",
    "school_floor":       "maps/train/school_floor.txt",
    "office_wing_medium": "maps/train/office_wing_medium.txt",

    # Médios — 36×80, hazard e layout mais complexo
    "office_complex_real":"maps/train/office_complex_real.txt",
    "mall_medium":        "maps/train/mall_medium.txt",
    "library_hard":       "maps/train/library_hard.txt",
    "di_style":           "maps/train/di_style.txt",
}

# ── Mapas de avaliação (cenários do mestrado) ─────────────────────────────────

EVAL_MAPS = {
    "library_bottleneck": "maps/eval/library_bottleneck.txt",
    "office_single_exit": "maps/eval/office_single_exit.txt",
    "mall_panic":         "maps/eval/mall_panic.txt",
    "school_evacuation":  "maps/eval/school_evacuation.txt",
    "di_emergency":       "maps/eval/di_emergency.txt",
}

# ── Todos os mapas (treino + eval + real) — para testes rápidos no main.py ────

ALL_MAPS = {
    **TRAIN_MAPS,
    **EVAL_MAPS,
    "di_primeiro_andar": "maps/DI_primeiro_andar.txt",
}

SCENARIO = AppScenario.ASTAR_FSM          # <-- cenário
MAP      = ALL_MAPS["di_emergency"]         # <-- mapa
AGENTS   = 10                             # <-- número de agentes

# ── Parâmetros gerais ─────────────────────────────────────────────────────────

RENDER        = True
FPS           = 30
DT            = 0.1
MAX_STEPS     = 400
EVAL_EPISODES = 10
ASTAR_HAZARD_COST = ASTAR_HAZARD_COST  # importado de simulation_params

# ── Parâmetros DQN ────────────────────────────────────────────────────────────

DQN = {
    "mode":               DQNMode.EVAL,
    "model_path":         "models/dqn_fsm.pth",
    "episodes":           500,
    "hidden_dim":         DQN_HIDDEN_DIM,
    "batch_size":         DQN_BATCH_SIZE,
    "gamma":              DQN_GAMMA,
    "lr":                 DQN_LR,
    "buffer_capacity":    DQN_BUFFER_CAPACITY,
    "target_update_freq": DQN_TARGET_UPDATE_FREQ,
    "train_start_size":   DQN_TRAIN_START_SIZE,
    "epsilon_start":      DQN_EPSILON_START,
    "epsilon_end":        DQN_EPSILON_END,
    "epsilon_decay":      DQN_EPSILON_DECAY,
}