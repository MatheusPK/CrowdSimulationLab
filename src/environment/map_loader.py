"""
map_loader.py — lê um arquivo .txt e constrói o MapData.

Legenda de tiles:
    O  parede / obstáculo
    .  espaço livre
    E  exit (saída de evacuação)
    H  hazard (perigo — incêndio, fumaça, etc.)
    S  spawn (posição inicial de agente)
"""

from entities.obstacle import Obstacle
from entities.exit import Exit
from entities.hazard import Hazard
from core.map_data import MapData
from simulation_params import TILE_SIZE


def load_map(file_path: str) -> MapData:
    map_data = MapData()
    map_data.tile_size = TILE_SIZE

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    map_data.rows   = len(lines)
    map_data.cols   = max(len(line) for line in lines) if lines else 0
    map_data.width  = map_data.cols * TILE_SIZE
    map_data.height = map_data.rows * TILE_SIZE

    map_data.grid = [
        list(line.ljust(map_data.cols, "O"))
        for line in lines
    ]

    for row, line in enumerate(map_data.grid):
        for col, char in enumerate(line):
            x = col * TILE_SIZE
            y = row * TILE_SIZE

            if char == "O":
                map_data.obstacles.append(Obstacle(x, y, TILE_SIZE, TILE_SIZE))
            elif char == "E":
                map_data.exits.append(Exit(x, y, TILE_SIZE, TILE_SIZE))
            elif char == "H":
                map_data.hazards.append(Hazard(x, y, TILE_SIZE, TILE_SIZE))
            elif char == "S":
                map_data.spawns.append((row, col))

    return map_data