class Trainer:
    def __init__(self, episodes, max_steps, environment, renderer):
        self.episodes = episodes
        self.max_steps = max_steps
        self.environment = environment
        self.renderer = renderer
        self.render = renderer is not None
    
    def train(self):
        if self.render: 
            self.renderer.initialize()

        for episode in range(self.episodes):
            self.environment.reset()
            step = 0
            done_agents = set()

            print(f"Starting episode {episode+1}/{self.episodes}")

            for step in range(self.max_steps):
                if len(done_agents) == len(self.environment.agents):
                    break

                for agent in self.environment.agents:
                    if agent.done: 
                        continue

                    state = self.environment.get_microscopic_observation(agent)
                    action = agent.policy.select_action(state)
                    obs, reward, done = self.environment.act(agent, action)
                    agent.policy.train(state, action, reward, obs, done)

                    if done: 
                        done_agents.add(agent)

                    if self.render: 
                        self.renderer.render()

            print(f"Episode {episode+1} finished in {step+1} steps with {len(done_agents)}/{len(self.environment.agents)} done agents.")

