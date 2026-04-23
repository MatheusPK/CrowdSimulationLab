from core.direction import Direction
from core.fsm_state import FSMState
from simulation_params import *

class Agent:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)
        self.radius               = AGENT_RADIUS
        self.planner_radius_margin = PLANNER_RADIUS_MARGIN
        self.base_speed    = AGENT_BASE_SPEED
        self.current_speed = self.base_speed
        self.vx = 0.0
        self.vy = 0.0
        self.last_action = 0
        self.direction     = Direction.N
        self.state         = FSMState.CALM
        self.emotion_level = 0.0
        self.peak_emotion  = 0.0
        self.velocity_smoothing = VELOCITY_SMOOTHING
        self.evacuated       = False
        self.alive           = True
        self.evacuation_time = None
        self.touched_hazard  = False
        self.exit_used       = None
        self._pos_history = []

    def apply_position(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def stop(self):
        self.vx = 0.0
        self.vy = 0.0

    def update_peak_emotion(self):
        if self.emotion_level > self.peak_emotion:
            self.peak_emotion = self.emotion_level