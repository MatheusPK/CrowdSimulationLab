from enum import Enum

class AppScenario(Enum):
    ASTAR = "astar"
    ASTAR_FSM = "astar_fsm"
    RANDOM = "random"
    DQN_FSM = "dqn_fsm"