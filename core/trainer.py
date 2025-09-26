class Trainer:
    def __init__(self, episodes, max_steps, environment, renderer):
        self.episodes = episodes
        self.max_steps = max_steps
        self.environment = environment
        self.renderer = renderer
        self.render = renderer
    
    def train(self):
        self.renderer.initialize()

        for episode in range(self.episodes):
            self.environment.reset()
            steps_taken = 0
            done_agents = set()

            print(f"\nStarting episode {episode+1}/{self.episodes}")

            for steps_taken in range(self.max_steps):
                if len(done_agents) == len(self.environment.agents):
                    break

                for agent in self.environment.agents:
                    if agent.done: 
                        continue

                    state = self.environment.get_microscopic_observation(agent)
                    action = agent.policy.select_action(state)

                    next_state, reward, done = self.environment.act(agent, action)
                    agent.policy.train(state, action, reward, next_state, done)

                    if done: 
                        done_agents.add(agent)

                self.renderer.render()

            print(
                f"Episode {episode+1}/{self.episodes} | "
                f"Steps: {steps_taken+1} | "
                f"Done: {len(done_agents)}/{len(self.environment.agents)}",
                end="\n\n"
            )

