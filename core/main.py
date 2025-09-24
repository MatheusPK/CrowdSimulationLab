from environment import *
from policy import *
from agent import *
from map import *
from renderer import *
from trainer import *

if __name__ == "__main__":
    policy = Policy()

    agents = [
        Agent(agent_id="Naruto", x=0, y=0, policy=policy),
        Agent(agent_id="Sasuke", x=5, y=3, policy=policy),
        Agent(agent_id="Sakura", x=2, y=4, policy=policy),
        Agent(agent_id="Kakashi", x=10, y=4, policy=policy)
    ]

    map = Map(filename="../maps/basic.txt", agents=agents)
    env = Environment(agents=agents, map=map)
    renderer = Renderer(environment=env, cell_size=20, fps=120)
    trainer = Trainer(episodes=3, max_steps=500, environment=env, renderer=renderer)

    trainer.train()