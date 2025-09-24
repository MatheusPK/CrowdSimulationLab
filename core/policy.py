import random
import constants as K

class Policy:
    def __init__(self):
        self.action_space = [
            K.NORTH,
            K.SOUTH,
            K.EAST,
            K.WEST,
            K.NORTHEAST,
            K.NORTHWEST,
            K.SOUTHEAST,
            K.SOUTHWEST
        ]

    def select_action(self, state):
        # Placeholder for actual policy logic
        return random.choice(self.action_space)