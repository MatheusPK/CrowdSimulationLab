from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode


class ScenarioConfigFactory:
    @staticmethod
    def build(scenario, map_path, agents):
        base_config = {
            "map_path": map_path,
            "render": True,
            "max_steps": 300,
            "dt": 0.1,
            "fps": 30,
            "agents": agents,
            "astar": {
                "hazard_cost": 8.0,
            }
        }

        if scenario == AppScenario.ASTAR:
            return {
                **base_config,
                "scenario": AppScenario.ASTAR,
                "eval_episodes": 5,
                "use_fsm": False,
            }

        if scenario == AppScenario.ASTAR_FSM:
            return {
                **base_config,
                "scenario": AppScenario.ASTAR_FSM,
                "eval_episodes": 5,
                "use_fsm": True,
            }

        if scenario == AppScenario.RANDOM:
            return {
                **base_config,
                "scenario": AppScenario.RANDOM,
                "eval_episodes": 10,
                "use_fsm": False,
            }

        if scenario == AppScenario.DQN_FSM:
            return {
                **base_config,
                "scenario": AppScenario.DQN_FSM,
                "eval_episodes": 10,
                "use_fsm": True,
                "dqn": {
                    "mode": DQNMode.TRAIN,
                    "model_path": "models/dqn_model.pth",
                    "episodes": 500,
                    "batch_size": 64,
                    "gamma": 0.99,
                    "lr": 1e-3,
                    "buffer_capacity": 20000,
                    "target_update_freq": 200,
                    "train_start_size": 1000,
                    "epsilon_start": 1.0,
                    "epsilon_end": 0.05,
                    "epsilon_decay": 20000,
                    "hidden_dim": 128,
                },
            }

        raise ValueError(f"Cenário não suportado: {scenario}")