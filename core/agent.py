import uuid

class Agent:
    def __init__(self, agent_id=None, x=0, y=0, policy=None):
        self.id = agent_id if agent_id is not None else str(uuid.uuid4())
        self.x = x
        self.y = y
        self.policy = policy
        self.done = False
        # futuros atributos: self.rotation, self.panic etc.

    def __repr__(self):
        return f"<Agent id={self.id} pos=({self.x}, {self.y}) done={self.done}>"