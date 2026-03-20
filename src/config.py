from core.app_scenario import AppScenario
from factories.scenario_config_factory import ScenarioConfigFactory

maps = [
    "maps/small/small_easy.txt",
    "maps/small/small_medium.txt",
    "maps/small/small_hard.txt",
    "maps/medium/medium_easy.txt",
    "maps/medium/medium_medium.txt",
    "maps/medium/medium_hard.txt",
    "maps/DI_primeiro_andar.txt",
    "maps/test_maps/mall_atrium.txt",
    "maps/test_maps/mall_corridor.txt",
    "maps/test_maps/mall_emergency.txt",
    "maps/test_maps/office_complex.txt",
    "maps/test_maps/office_hazard.txt",
    "maps/test_maps/office_junction.txt",
    "maps/test_maps/office_openplan.txt",
    "maps/test_maps/office_simple.txt",
]

CONFIG = ScenarioConfigFactory.build(
    scenario=AppScenario.ASTAR_FSM,
    map_path=maps[10],
    agents=10
)