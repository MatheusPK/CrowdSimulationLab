import constants as K
from environment import *
from policy import *
from agent import * 
from map import *
from renderer import *

def main():
    policy = Policy()

    agents = [
        Agent(agent_id="Naruto", x=0, y=0, policy=policy),
        Agent(agent_id="Sasuke", x=5, y=3, policy=policy),
        Agent(agent_id="Sakura", x=2, y=4, policy=policy),
        Agent(agent_id="Kakashi", x=10, y=4, policy=policy)
    ]

    map = Map()
    map.init_from_file("../maps/basic.txt")
    map.place_agents(agents)
    print("Initial Map:")
    print(map)

    env = Environment(agents=agents, map=map)

    renderer = Renderer(environment=env, fps=60)
    renderer.initialize()

    while True:
        for agent in env.agents.values():
            if agent.done:
                continue
            action = agent.policy.select_action("state placeholder")
            obs, reward, done = env.act(agent, action)
            print(f"Agent {agent.id} took action {action}, reward: {reward}, done: {done}")
        
        renderer.render()


if __name__ == "__main__":
    main()