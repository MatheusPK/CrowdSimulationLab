import constants as K

class Cell:
    def __init__(self, type, entity=None):
        self.type = type
        self.entity = entity
    
    def __repr__(self):
        return f"<Cell type={self.type} entity={self.entity}>"

class Map:
    def __init__(self, filename=None, agents=None):
        if filename:
            self.load_from_file(filename)
        else:
            self.width = 0
            self.height = 0
            self.map = []

        if agents:
            self.place_agents(agents)

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            self.map = []
            self.width, self.height = map(int, f.readline().strip().split())

            for i in range(self.height):
                line = f.readline().strip()
                row = [Cell(type=int(char), entity=None) for char in line]
                if len(row) != self.width: raise ValueError(f"Line {i+2} in map file does not match specified width.")
                self.map.append(row)

    def place_agents(self, agents):
        for agent in agents:
            if self.map[agent.y][agent.x].type == K.EMPTY:
                self.map[agent.y][agent.x] = Cell(type=K.AGENT, entity=agent)
            else:
                raise ValueError(f"Cannot place agent {agent.id} at occupied position ({agent.x}, {agent.y}).")
            
    def __repr__(self):
        repr_str = ""
        for row in self.map:
            repr_str += ''.join(str(cell.type) for cell in row) + "\n"
        return repr_str


