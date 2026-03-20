from core.direction import Direction
from core.fsm_state import FSMState


class Agent:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

        # Geometria
        self.radius = 6.0
        self.planner_radius_margin = 5.0

        # Movimento
        self.base_speed = 50.0
        self.current_speed = self.base_speed
        self.vx = 0.0
        self.vy = 0.0
        self.last_action = 0

        # Direção / estado
        self.direction = Direction.N
        self.state = FSMState.CALM
        self.emotion_level = 0.0

        # Status
        self.evacuated = False
        self.alive = True

        # Avoidance de obstáculos
        self.obstacle_avoidance_distance = 14.0
        self.obstacle_avoidance_strength = 70.0

        # Avoidance de outros agentes
        self.agent_avoidance_distance = 12.0
        self.agent_avoidance_strength = 90.0

        # Suavização da velocidade
        self.velocity_smoothing = 0.18

    def apply_position(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0