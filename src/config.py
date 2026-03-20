from core.app_scenario import AppScenario
from factories.scenario_config_factory import ScenarioConfigFactory

maps = [
    "maps/small/small_easy.txt",
    "maps/small/small_medium.txt",
    "maps/small/small_hard.txt",
    "maps/medium/medium_easy.txt",
    "maps/medium/medium_medium.txt",
    "maps/medium/medium_hard.txt",
    "maps/DI_primeiro_andar.txt"
]

CONFIG = ScenarioConfigFactory.build(
    scenario=AppScenario.DQN_FSM,
    map_path=maps[0],
    agents=10
)