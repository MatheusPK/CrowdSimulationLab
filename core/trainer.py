class Trainer:
    def __init__(self, episodes, max_steps, environment, renderer):
        self.episodes = episodes
        self.max_steps = max_steps
        self.environment = environment
        self.renderer = renderer

    def train(self):
        self.renderer.initialize()

        for episode in range(self.episodes):
            self.environment.reset()
            done_ids = set()

            print(f"\nStarting episode {episode+1}/{self.episodes}")

            for step in range(self.max_steps):
                if len(done_ids) == len(self.environment.agents):
                    break

                # 1) act
                actions_by_id = {}
                prev_states_by_id = {}

                for agent in self.environment.agents:
                    if agent.id in done_ids:
                        continue

                    s = self.environment.get_microscopic_observation(agent)
                    a = agent.policy.select_action(s)
                    prev_states_by_id[agent.id] = s
                    actions_by_id[agent.id] = a

                # 2) step
                transitions = self.environment.step(actions_by_id)

                # 3) train
                for agent_id, (prev_state, reward, next_state, done) in transitions.items():
                    a = actions_by_id[agent_id]
                    agent = next(a for a in self.environment.agents if a.id == agent_id)  # ou mantenha dict id->agent

                    agent.policy.train(prev_state, a, reward, next_state, done)

                    if done:
                        done_ids.add(agent_id)

                self.renderer.render()

            print(
                f"Episode {episode+1}/{self.episodes} | "
                f"Steps: {step+1} | "
                f"Done: {len(done_ids)}/{len(self.environment.agents)}\n"
            )

