from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from environment.environment import Environment
from policies.astar_policy import AStarPolicy
from policies.random_policy import RandomPolicy
from policies.dqn_policy import DQNPolicy


class PolicyFactory:
    @staticmethod
    def build(config, env: Environment):
        scenario = config["scenario"]

        if scenario in (AppScenario.ASTAR, AppScenario.ASTAR_FSM):
            hazard_cost = config.get("astar", {}).get("hazard_cost", 8.0)
            return AStarPolicy(hazard_cost=hazard_cost)

        if scenario == AppScenario.RANDOM:
            return RandomPolicy()

        if scenario == AppScenario.DQN_FSM:
            dqn_cfg = config.get("dqn", {})
            return DQNPolicy(
                mode=dqn_cfg.get("mode", DQNMode.TRAIN),
                model_path=dqn_cfg.get("model_path", "models/dqn_model.pth"),
                state_dim=env.OBS_DIM,               # vem direto do environment
                action_dim=8,
                hidden_dim=dqn_cfg.get("hidden_dim", 128),
                batch_size=dqn_cfg.get("batch_size", 64),
                gamma=dqn_cfg.get("gamma", 0.99),
                lr=dqn_cfg.get("lr", 1e-3),
                buffer_capacity=dqn_cfg.get("buffer_capacity", 20_000),
                target_update_freq=dqn_cfg.get("target_update_freq", 200),
                train_start_size=dqn_cfg.get("train_start_size", 1_000),
                epsilon_start=dqn_cfg.get("epsilon_start", 1.0),
                epsilon_end=dqn_cfg.get("epsilon_end", 0.05),
                epsilon_decay=dqn_cfg.get("epsilon_decay", 20_000),
            )

        raise ValueError(f"Unknown scenario: {scenario}")