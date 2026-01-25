import uuid

class AgentEntity:
    def __init__(self, agent_id=None, x=0, y=0, policy=None):
        self.id = agent_id if agent_id is not None else str(uuid.uuid4())
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.vx_half = 0.0
        self.vy_half = 0.0
        self._lf_initialized = False
        self.policy = policy
        self.done = False
        self.mass = 80.0
        self.v_desired = 2.0
        self.radius = 0.25

    def __repr__(self):
        return f"<Agent id={self.id} pos=({self.x:.2f}, {self.y:.2f}) v=({self.vx:.2f},{self.vy:.2f}) done={self.done}>"