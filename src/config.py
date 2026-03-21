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

ALL_MAPS = [
    "maps/train/library_small.txt",
    "maps/train/office_wing_small.txt",
    "maps/train/mall_small.txt",
    "maps/train/school_small.txt",
    "maps/train/library_medium.txt",
    "maps/train/mall_food_court.txt",
    "maps/train/school_floor.txt",
    "maps/train/office_wing_medium.txt",    
    "maps/train/office_complex_real.txt",
    "maps/train/mall_medium.txt",
    "maps/train/library_hard.txt",
    "maps/train/di_style.txt",
    "maps/eval/library_bottleneck.txt",
    "maps/eval/office_single_exit.txt",
    "maps/eval/mall_panic.txt",
    "maps/eval/school_evacuation.txt",
    "maps/eval/di_emergency.txt",
    "maps/DI_primeiro_andar.txt",
]

# ── Configuração ativa  <-- edite aqui ───────────────────────────────────────

SCENARIO = AppScenario.ASTAR_FSM          # <-- cenário
MAP      = ALL_MAPS[16] #TRAIN_MAPS["library_hard"]         # <-- mapa
AGENTS   = 15                             # <-- número de agentes

# ── Parâmetros gerais ─────────────────────────────────────────────────────────

RENDER        = True
FPS           = 30
DT            = 0.1
MAX_STEPS     = 400
EVAL_EPISODES = 10
ASTAR_HAZARD_COST = 8.0

# ── Parâmetros DQN ────────────────────────────────────────────────────────────

DQN = {
    "mode":               DQNMode.EVAL,
    "model_path":         "models/dqn_fsm.pth",
    "episodes":           500,
    "hidden_dim":         128,
    "batch_size":         64,
    "gamma":              0.99,
    "lr":                 1e-3,
    "buffer_capacity":    50_000,
    "target_update_freq": 300,
    "train_start_size":   1_000,
    "epsilon_start":      1.0,
    "epsilon_end":        0.05,
    "epsilon_decay":      30_000,
}