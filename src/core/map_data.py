class MapData:
    def __init__(self):
        self.obstacles = []
        self.exits = []
        self.hazards = []
        self.spawns = []

        self.width = 0
        self.height = 0
        self.tile_size = 0

        self.grid = []
        self.rows = 0
        self.cols = 0