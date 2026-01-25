import random
import constants as K

class Policy:
    def __init__(self):
        self.action_space = [
            K.N,
            K.S,
            K.E,
            K.W,
            K.NE,
            K.NW,
            K.SE,
            K.SW
        ]

    def select_action(self, state):
        # Placeholder for actual policy logic
        return random.choice(self.action_space)
    
    def train(self, state, action, reward, next_state, done):
        # Placeholder for training logic
        pass