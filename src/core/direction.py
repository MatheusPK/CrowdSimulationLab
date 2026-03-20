from enum import Enum

class Direction(Enum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7

DIRECTION_VECTORS = {
    Direction.N:  (0, -1),
    Direction.NE: (1, -1),
    Direction.E:  (1, 0),
    Direction.SE: (1, 1),
    Direction.S:  (0, 1),
    Direction.SW: (-1, 1),
    Direction.W:  (-1, 0),
    Direction.NW: (-1, -1),
}