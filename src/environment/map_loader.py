from entities.obstacle import Obstacle
from entities.exit import Exit
from entities.hazard import Hazard
from core.map_data import MapData

TILE_SIZE = 8

def load_map(file_path):
    map_data = MapData()
    map_data.tile_size = TILE_SIZE

    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]

    map_data.rows = len(lines)
    map_data.cols = len(lines[0]) if lines else 0
    map_data.width = map_data.cols * TILE_SIZE
    map_data.height = map_data.rows * TILE_SIZE
    map_data.grid = [list(line) for line in lines]

    for row, line in enumerate(lines):
        for col, char in enumerate(line):
            x = col * TILE_SIZE
            y = row * TILE_SIZE

            if char == "O":
                map_data.obstacles.append(Obstacle(x, y, TILE_SIZE, TILE_SIZE))
            elif char == "E":
                map_data.exits.append(Exit(x, y, TILE_SIZE, TILE_SIZE))
            elif char == "H":
                map_data.hazards.append(Hazard(x, y, TILE_SIZE, TILE_SIZE, intensity=1))
            elif char == "S":
                map_data.spawns.append((row, col))

    return map_data