from config import CONFIG
from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from environment.environment import Environment
from environment.map_loader import load_map
from factories.policy_factory import PolicyFactory
from rendering.renderer import Renderer


# ---------------------------------------------------------------------------
# Loop de episódio genérico (A*, Random, DQN eval)
# ---------------------------------------------------------------------------

def run_policy_episode(env, policy, renderer=None, max_steps=300):
    observations = env.reset()
    print(
        f"agents_requested={env.num_agents_requested} | "
        f"agents_spawned={len(env.agents)}"
    )

    step_count = 0
    running = True
    done = False
    total_reward = 0.0
    total_agent_rewards = [0.0 for _ in env.agents]

    while running and not done and step_count < max_steps:
        if renderer is not None:
            running = renderer.poll_events()
            if not running:
                break

        actions = []
        for agent in env.agents:
            if agent.evacuated:
                actions.append(None)
                continue
            exit_obj = env.get_nearest_exit(agent)
            action = policy.choose_action(env, agent, exit_obj)
            actions.append(action)

        observations, reward_list, done, info = env.step(actions)

        step_reward = sum(reward_list)
        total_reward += step_reward
        step_count += 1

        for i, reward in enumerate(reward_list):
            total_agent_rewards[i] += reward

        active_count   = sum(1 for a in env.agents if not a.evacuated)
        evacuated_count = sum(1 for a in env.agents if a.evacuated)

        print(
            f"step={step_count} | active={active_count} | "
            f"evacuated={evacuated_count}/{len(env.agents)} | "
            f"step_reward={step_reward:.2f} | done={done}"
        )

        for i, agent in enumerate(env.agents):
            print(
                f"  agent={i} | pos=({agent.x:.1f},{agent.y:.1f}) | "
                f"speed={agent.current_speed:.1f} | action={actions[i]} | "
                f"reward={reward_list[i]:.2f} | emotion={agent.emotion_level:.2f} | "
                f"state={agent.state.name} | evacuated={agent.evacuated} | "
                f"collided={info['collided'][i]}"
            )

        if renderer is not None:
            renderer.render(env)

    evacuated_count  = sum(1 for a in env.agents if a.evacuated)
    evacuation_rate  = evacuated_count / max(1, len(env.agents))

    return {
        "success":         evacuated_count == len(env.agents),
        "evacuated_count": evacuated_count,
        "evacuation_rate": evacuation_rate,
        "steps":           step_count,
        "total_reward":    total_reward,
        "agent_rewards":   total_agent_rewards,
    }


# ---------------------------------------------------------------------------
# Loop de treino DQN (armazena transições e salva o modelo ao final)
# ---------------------------------------------------------------------------

def run_dqn_training(env, policy, renderer=None, config=None):
    dqn_cfg    = config.get("dqn", {})
    episodes   = dqn_cfg.get("episodes", 500)
    max_steps  = config.get("max_steps", 300)

    results = []

    for episode in range(episodes):
        obs_list = env.reset()
        prev_obs = {id(a): obs_list[i] for i, a in enumerate(env.agents)}

        step_count = 0
        running    = True
        done       = False
        ep_reward  = 0.0

        while running and not done and step_count < max_steps:
            if renderer is not None:
                running = renderer.poll_events()
                if not running:
                    break

            actions = []
            for agent in env.agents:
                if agent.evacuated:
                    actions.append(None)
                    continue
                exit_obj = env.get_nearest_exit(agent)
                action   = policy.choose_action(env, agent, exit_obj)
                actions.append(action)

            next_obs_list, reward_list, done, info = env.step(actions)

            # Armazena uma transição por agente ativo
            for i, agent in enumerate(env.agents):
                action = actions[i]
                if action is None:
                    continue
                policy.store_transition(
                    obs      = prev_obs[id(agent)],
                    action   = action,
                    reward   = reward_list[i],
                    next_obs = next_obs_list[i],
                    done     = done,
                )

            prev_obs = {id(a): next_obs_list[i] for i, a in enumerate(env.agents)}
            ep_reward += sum(reward_list)
            step_count += 1

            if renderer is not None:
                renderer.render(env)

        evacuated_count = sum(1 for a in env.agents if a.evacuated)
        evacuation_rate = evacuated_count / max(1, len(env.agents))

        result = {
            "success":         evacuated_count == len(env.agents),
            "evacuated_count": evacuated_count,
            "evacuation_rate": evacuation_rate,
            "steps":           step_count,
            "total_reward":    ep_reward,
            "epsilon":         policy.current_epsilon(),
        }
        results.append(result)

        print(
            f"[DQN] ep={episode + 1}/{episodes} | "
            f"steps={step_count} | "
            f"evacuated={evacuated_count}/{len(env.agents)} | "
            f"reward={ep_reward:.1f} | "
            f"eps={policy.current_epsilon():.3f}"
        )

    # Salva modelo ao final do treino
    policy.save()
    print(f"[DQN] Modelo salvo em {policy.model_path}")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_renderer(config, map_data):
    if not config["render"]:
        return None

    renderer = Renderer(
        width=map_data.width,
        height=map_data.height,
        title=f"Scenario: {config['scenario'].value}",
        fps=config["fps"],
        draw_grid=True,
        tile_size=map_data.tile_size,
    )
    renderer.initialize()
    return renderer


def print_summary(config, results):
    success_rate       = sum(1 for r in results if r["success"]) / max(1, len(results))
    avg_steps          = sum(r["steps"] for r in results) / max(1, len(results))
    avg_reward         = sum(r["total_reward"] for r in results) / max(1, len(results))
    avg_evacuation_rate = sum(r["evacuation_rate"] for r in results) / max(1, len(results))

    print("\n=== Summary ===")
    print(f"scenario={config['scenario'].value}")
    print(f"agents_requested={config['agents']}")
    print(f"success_rate={success_rate:.2f}")
    print(f"avg_evacuation_rate={avg_evacuation_rate:.2f}")
    print(f"avg_steps={avg_steps:.2f}")
    print(f"avg_reward={avg_reward:.2f}")


# ---------------------------------------------------------------------------
# Cenários
# ---------------------------------------------------------------------------

def run_scenario(config):
    map_data = load_map(config["map_path"])

    env = Environment(
        map_data=map_data,
        dt=config["dt"],
        max_steps=config["max_steps"],
        use_fsm=config["use_fsm"],
        num_agents=config["agents"],
    )

    renderer = build_renderer(config, map_data)
    policy   = PolicyFactory.build(config, env)

    dqn_mode = config.get("dqn", {}).get("mode")

    if dqn_mode == DQNMode.TRAIN:
        results = run_dqn_training(
            env=env,
            policy=policy,
            renderer=renderer,
            config=config,
        )
    else:
        episodes = config["eval_episodes"]
        results  = []
        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            result = run_policy_episode(
                env=env,
                policy=policy,
                renderer=renderer,
                max_steps=config["max_steps"],
            )
            results.append(result)
            print(result)

    if renderer is not None:
        renderer.close()

    print_summary(config, results)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    scenario = CONFIG["scenario"]

    if scenario in (
        AppScenario.ASTAR,
        AppScenario.ASTAR_FSM,
        AppScenario.RANDOM,
        AppScenario.DQN_FSM,
    ):
        run_scenario(CONFIG)
        return

    raise ValueError(f"Cenário desconhecido: {scenario}")


if __name__ == "__main__":
    main()