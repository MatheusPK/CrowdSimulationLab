import random

class RandomPolicy:
    def choose_action(self, env, agent, exit_obj=None):
        return random.randint(0, 7)