import sys
from environment import *
from policy import *
from agent import *
from map import *
from renderer import *
from trainer import *

if __name__ == "__main__":
    # trainer params: episodes, max_steps, render, output_model_file
    # runner params: cell_size, fps, model_file
    # process command line args

    policy = Policy()

    agents = [
        Agent(agent_id="Naruto", x=0, y=0, policy=policy),
        Agent(agent_id="Sasuke", x=5, y=3, policy=policy),
        Agent(agent_id="Sakura", x=2, y=4, policy=policy),
        Agent(agent_id="Kakashi", x=10, y=4, policy=policy)
    ]

    map = Map(filename="../maps/normal.txt", agents=agents)
    env = Environment(agents=agents, map=map)
    renderer = Renderer(environment=env, cell_size=20, fps=24)
    trainer = Trainer(episodes=1000, max_steps=200, environment=env, renderer=renderer)

    trainer.train()