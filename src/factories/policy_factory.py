import config as cfg
from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from environment.environment import Environment
from policies.astar_policy import AStarPolicy
from policies.random_policy import RandomPolicy
from policies.dqn_policy import DQNPolicy
from simulation_params import PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES


class PolicyFactory:
    @staticmethod
    def build(env: Environment) -> AStarPolicy | RandomPolicy | DQNPolicy:
        scenario = cfg.SCENARIO

        if scenario in (AppScenario.ASTAR, AppScenario.ASTAR_FSM):
            return AStarPolicy(hazard_cost=cfg.ASTAR_HAZARD_COST)

        if scenario == AppScenario.RANDOM:
            return RandomPolicy()

        if scenario == AppScenario.DQN_FSM:
            d = cfg.DQN
            return DQNPolicy(
                mode=d["mode"],
                model_path=d["model_path"],
                state_dim=env.OBS_DIM,
                action_dim=8,
                hidden_dim=d["hidden_dim"],
                batch_size=d["batch_size"],
                gamma=d["gamma"],
                lr=d["lr"],
                buffer_capacity=d["buffer_capacity"],
                target_update_freq=d["target_update_freq"],
                train_start_size=d["train_start_size"],
                epsilon_start=d["epsilon_start"],
                epsilon_end=d["epsilon_end"],
                epsilon_decay=d["epsilon_decay"],
                per_alpha=PER_ALPHA,
                per_beta_start=PER_BETA_START,
                per_beta_frames=PER_BETA_FRAMES,
            )

        raise ValueError(f"Cenário desconhecido: {scenario}")