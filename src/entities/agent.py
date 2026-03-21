from core.direction import Direction
from core.fsm_state import FSMState
from simulation_params import (
    AGENT_RADIUS,
    PLANNER_RADIUS_MARGIN,
    AGENT_BASE_SPEED,
    OBSTACLE_AVOIDANCE_DISTANCE,
    OBSTACLE_AVOIDANCE_STRENGTH,
    AGENT_AVOIDANCE_DISTANCE,
    AGENT_AVOIDANCE_STRENGTH,
    VELOCITY_SMOOTHING,
)


class Agent:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

        # Geometria
        self.radius               = AGENT_RADIUS
        self.planner_radius_margin = PLANNER_RADIUS_MARGIN

        # Movimento
        self.base_speed    = AGENT_BASE_SPEED
        self.current_speed = self.base_speed
        self.vx = 0.0
        self.vy = 0.0
        self.last_action = 0

        # Direção / estado
        self.direction    = Direction.N
        self.state        = FSMState.CALM
        self.emotion_level = 0.0
        self.peak_emotion  = 0.0

        # Status
        self.evacuated      = False
        self.alive          = True
        self.evacuation_time = None

        # Avoidance base (sobrescrito pelo update_fsm quando use_fsm=True)
        self.obstacle_avoidance_distance = OBSTACLE_AVOIDANCE_DISTANCE
        self.obstacle_avoidance_strength = OBSTACLE_AVOIDANCE_STRENGTH
        self.agent_avoidance_distance    = AGENT_AVOIDANCE_DISTANCE
        self.agent_avoidance_strength    = AGENT_AVOIDANCE_STRENGTH
        self.velocity_smoothing          = VELOCITY_SMOOTHING

    def apply_position(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0

    def update_peak_emotion(self):
        if self.emotion_level > self.peak_emotion:
            self.peak_emotion = self.emotion_level