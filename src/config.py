"""
config.py — configuração da simulação para main.py.

Edite as três linhas marcadas com <-- e execute: python main.py

Cenários disponíveis:
    AppScenario.RANDOM      baseline aleatório, sem FSM
    AppScenario.ASTAR       A* puro,            sem FSM
    AppScenario.ASTAR_FSM   A* com emoção,      com FSM
    AppScenario.DQN_FSM     DQN com emoção,     com FSM
"""

from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from simulation_params import (
    ASTAR_HAZARD_COST,
    DQN_HIDDEN_DIM, DQN_BUFFER_CAPACITY, DQN_TRAIN_START_SIZE,
    DQN_BATCH_SIZE, DQN_GAMMA, DQN_LR, DQN_TARGET_UPDATE_FREQ,
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY,
)

# ── Mapas de treino ───────────────────────────────────────────────────────────

TRAIN_MAPS = {
    "mall_small":              "maps/train/mall_small.txt",
    "school_small":            "maps/train/school_small.txt",
    "office_wing_small":       "maps/train/office_wing_small.txt",
    "library_small":           "maps/train/library_small.txt",
    "library_medium":          "maps/train/library_medium.txt",
    "hazard_corridor_small":   "maps/train/hazard_corridor_small.txt",
    "hazard_near_exit_small":  "maps/train/hazard_near_exit_small.txt",
    "school_floor":            "maps/train/school_floor.txt",
    "office_wing_medium":      "maps/train/office_wing_medium.txt",
    "hazard_near_exit_medium": "maps/train/hazard_near_exit_medium.txt",
    "bridge_open_medium":      "maps/train/bridge_open_medium.txt",
    "bridge_corridor_medium":  "maps/train/bridge_corridor_medium.txt",
    "bridge_hazard_intro":     "maps/train/bridge_hazard_intro.txt",
    "bridge_multi_exit":       "maps/train/bridge_multi_exit.txt",
    "hazard_bypass_medium":    "maps/train/hazard_bypass_medium.txt",
    "mall_medium":             "maps/train/mall_medium.txt",
    "hazard_dense_office":     "maps/train/hazard_dense_office.txt",
    "library_hard":            "maps/train/library_hard.txt",
    "di_style":                "maps/train/di_style.txt",
}

# ── Mapas de avaliação ────────────────────────────────────────────────────────

EVAL_MAPS = {
    "library_bottleneck": "maps/eval/library_bottleneck.txt",
    "office_single_exit": "maps/eval/office_single_exit.txt",
    "mall_panic":         "maps/eval/mall_panic.txt",
    "school_evacuation":  "maps/eval/school_evacuation.txt",
    "di_emergency":       "maps/eval/di_emergency.txt",
}

ALL_MAPS = {
    **TRAIN_MAPS,
    **EVAL_MAPS,
    "di_primeiro_andar": "maps/DI_primeiro_andar.txt",
}

# ── Configuração do experimento ───────────────────────────────────────────────

SCENARIO = AppScenario.ASTAR_FSM          # <-- cenário
MAP      = EVAL_MAPS["library_bottleneck"]         # <-- mapa
AGENTS   = 12                              # <-- número de agentes

RENDER        = True
FPS           = 30
DT            = 0.1
MAX_STEPS     = 450
EVAL_EPISODES = 10

# ── DQN ──────────────────────────────────────────────────────────────────────

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