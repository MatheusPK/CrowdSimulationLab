"""
main.py — entrypoint para rodar e avaliar qualquer política.

Uso:
    python main.py                        # usa config.py como está
    python main.py --no-render            # sem janela (mais rápido)
    python main.py --episodes 30          # número de episódios
    python main.py --log                  # salva resultados em logs/results.csv
    python main.py --no-render --episodes 30 --log  # combinando flags

Para treino DQN com currículo progressivo:
    python train_curriculum.py            # treino completo
    python train_curriculum.py --stage 6  # começa no stage 6
    python train_curriculum.py --eval     # avalia nos 5 mapas eval

NOTA: main.py é exclusivamente para execução e avaliação de políticas
já treinadas. Nunca use main.py com DQN mode=TRAIN — use train_curriculum.py.
"""

import argparse
import csv
import os
from pathlib import Path

import config as cfg
from core.app_scenario import AppScenario
from core.dqn_mode import DQNMode
from environment.environment import Environment
from environment.map_loader import load_map
from factories.policy_factory import PolicyFactory
from rendering.renderer import Renderer


# ── Renderer ─────────────────────────────────────────────────────────────────

def build_renderer(map_data, title: str) -> Renderer | None:
    if not cfg.RENDER:
        return None
    r = Renderer(
        width=map_data.width,
        height=map_data.height,
        title=title,
        fps=cfg.FPS,
        draw_grid=True,
        tile_size=map_data.tile_size,
    )
    r.initialize()
    return r


# ── Loop de episódio ──────────────────────────────────────────────────────────

def run_episode(env: Environment, policy, renderer: Renderer | None) -> dict:
    """Roda um episódio completo e retorna as métricas."""
    obs_list = env.reset()

    done    = False
    running = True

    while running and not done:
        if renderer is not None:
            running = renderer.poll_events()
            if not running:
                break

        actions = [
            policy.choose_action(env, agent, env.get_nearest_exit(agent))
            if not agent.evacuated else None
            for agent in env.agents
        ]

        next_obs_list, reward_list, done, _ = env.step(actions)

        if renderer is not None:
            renderer.render(env)

    return env.get_episode_metrics()


# ── Logging de resultados ─────────────────────────────────────────────────────

LOG_PATH = Path("logs/results.csv")

def init_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "scenario", "map", "episode",
                "evacuation_rate", "all_evacuated",
                "mean_evac_time", "steps",
                "mean_emotion_final", "mean_peak_emotion", "emotion_variance",
                "panic_rate", "hazard_contact_rate", "exit_utilization",
                "mean_speed_ratio",
            ])

def log_result(episode: int, metrics: dict):
    fields = [
        "scenario", "map", "episode",
        "evacuation_rate", "all_evacuated",
        "mean_evac_time", "steps",
        "mean_emotion_final", "mean_peak_emotion", "emotion_variance",
        "panic_rate", "hazard_contact_rate", "exit_utilization",
        "mean_speed_ratio",
    ]
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerow({
            "scenario":            cfg.SCENARIO.value,
            "map":                 os.path.basename(cfg.MAP),
            "episode":             episode,
            "evacuation_rate":     round(metrics.get("evacuation_rate", 0), 4),
            "all_evacuated":       int(metrics.get("all_evacuated", False)),
            "mean_evac_time":      round(metrics.get("mean_evacuation_time", cfg.MAX_STEPS), 2),
            "steps":               metrics.get("steps", cfg.MAX_STEPS),
            "mean_emotion_final":  round(metrics.get("mean_emotion_final", 0), 4),
            "mean_peak_emotion":   round(metrics.get("mean_peak_emotion", 0), 4),
            "emotion_variance":    round(metrics.get("emotion_variance", 0), 4),
            "panic_rate":          round(metrics.get("panic_rate", 0), 4),
            "hazard_contact_rate": round(metrics.get("hazard_contact_rate", 0), 4),
            "exit_utilization":    round(metrics.get("exit_utilization", 0), 4),
            "mean_speed_ratio":    round(metrics.get("mean_speed_ratio", 1), 4),
        })


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict]):
    n = max(1, len(all_results))
    print(f"\n{'='*50}")
    print(f"SCENARIO : {cfg.SCENARIO.value}")
    print(f"MAP      : {cfg.MAP}")
    print(f"AGENTS   : {cfg.AGENTS}")
    print(f"EPISODES : {n}")
    print(f"{'─'*50}")
    print(f"evacuation_rate   : {sum(r['evacuation_rate'] for r in all_results)/n:.3f}")
    print(f"all_evacuated     : {sum(r.get('all_evacuated',False) for r in all_results)}/{n}")
    print(f"mean_evac_time    : {sum(r.get('mean_evacuation_time',cfg.MAX_STEPS) for r in all_results)/n:.1f} steps")
    print(f"mean_emotion_final: {sum(r.get('mean_emotion_final',0) for r in all_results)/n:.3f}")
    print(f"emotion_variance  : {sum(r.get('emotion_variance',0) for r in all_results)/n:.4f}")
    print(f"mean_speed_ratio  : {sum(r.get('mean_speed_ratio',1) for r in all_results)/n:.3f}")
    print(f"{'='*50}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-render",  action="store_true", help="Desativa renderização")
    p.add_argument("--episodes",   type=int, default=None)
    p.add_argument("--log",        action="store_true", help="Salva resultados em CSV")
    p.add_argument("--seed",       type=int, default=None,
                   help="Seed para reproducibilidade (spawns e aleatóriedade)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_render:
        cfg.RENDER = False

    # Guarda contra uso acidental de main.py para treino DQN
    if cfg.SCENARIO == AppScenario.DQN_FSM and cfg.DQN.get("mode") == DQNMode.TRAIN:
        print("ERRO: main.py não deve ser usado para treino DQN.")
        print("Use: python train_curriculum.py")
        return

    episodes = args.episodes or cfg.EVAL_EPISODES

    if args.log:
        init_log()

    map_data = load_map(cfg.MAP)
    env = Environment(
        map_data=map_data,
        dt=cfg.DT,
        max_steps=cfg.MAX_STEPS,
        use_fsm=(cfg.SCENARIO in (AppScenario.ASTAR_FSM, AppScenario.DQN_FSM)),
        num_agents=cfg.AGENTS,
    )

    policy = PolicyFactory.build(env)

    renderer = build_renderer(map_data, title=f"{cfg.SCENARIO.value} | {os.path.basename(cfg.MAP)}")

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        import numpy as _np
        _np.random.seed(args.seed)

    all_results = []
    for ep in range(1, episodes + 1):
        print(f"\n── Episode {ep}/{episodes} ──────────────────────────────")
        metrics = run_episode(env, policy, renderer)
        all_results.append(metrics)

        print(
            f"  evac={metrics['evacuation_rate']:.2f}  "
            f"steps={metrics['steps']}  "
            f"emotion={metrics.get('mean_emotion_final',0):.2f}  "
            f"speed={metrics.get('mean_speed_ratio',1):.2f}"
        )

        if args.log:
            log_result(ep, metrics)

    if renderer is not None:
        renderer.close()

    print_summary(all_results)


if __name__ == "__main__":
    main()