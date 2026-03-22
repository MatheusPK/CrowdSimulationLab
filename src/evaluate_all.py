"""
evaluate_all.py — compara todas as políticas nos mapas de eval.

Uso:
    python evaluate_all.py                        # 30 episódios, sem render
    python evaluate_all.py --episodes 50          # mais episódios
    python evaluate_all.py --seed 42              # reproducibilidade
    python evaluate_all.py --render               # com visualização
    python evaluate_all.py --policies dqn astar   # só algumas políticas
    python evaluate_all.py --csv results.csv      # salva CSV

Políticas disponíveis: random, astar, astar_fsm, dqn
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# ── Mapas de eval ─────────────────────────────────────────────────────────────

# EVAL_MAPS = [
#     ("library_bottleneck", "maps/eval/library_bottleneck.txt", 12, 450),
#     ("office_single_exit", "maps/eval/office_single_exit.txt", 12, 450),
#     ("mall_panic",         "maps/eval/mall_panic.txt",         12, 450),
#     ("school_evacuation",  "maps/eval/school_evacuation.txt",  12, 450),
#     ("di_emergency",       "maps/eval/di_emergency.txt",       12, 500),
# ]

EVAL_MAPS = [
    ("mall_small",              "maps/train/mall_small.txt",              4,  300),
    ("school_small",            "maps/train/school_small.txt",            4,  300),
    ("office_wing_small",       "maps/train/office_wing_small.txt",       4,  300),
    ("library_small",           "maps/train/library_small.txt",           4,  300),
    ("library_medium",          "maps/train/library_medium.txt",          6,  400),
    ("hazard_corridor_small",   "maps/train/hazard_corridor_small.txt",   4,  350),
    ("hazard_near_exit_small",  "maps/train/hazard_near_exit_small.txt",  6,  350),
    ("school_floor",            "maps/train/school_floor.txt",            8,  400),
    ("office_wing_medium",      "maps/train/office_wing_medium.txt",     10,  400),
    ("hazard_near_exit_medium", "maps/train/hazard_near_exit_medium.txt",10,  420),
    ("bridge_open_medium",      "maps/train/bridge_open_medium.txt",     11,  400),
    ("bridge_corridor_medium",  "maps/train/bridge_corridor_medium.txt", 11,  400),
    ("bridge_hazard_intro",     "maps/train/bridge_hazard_intro.txt",    12,  420),
    ("bridge_multi_exit",       "maps/train/bridge_multi_exit.txt",      12,  420),
    ("hazard_bypass_medium",    "maps/train/hazard_bypass_medium.txt",   12,  450),
    ("mall_medium",             "maps/train/mall_medium.txt",            12,  450),
    ("hazard_dense_office",     "maps/train/hazard_dense_office.txt",    12,  450),
    ("library_hard",            "maps/train/library_hard.txt",           12,  450),
]

DQN_MODEL = "models/dqn_fsm.pth"

# ── Imports do projeto ────────────────────────────────────────────────────────

from core.dqn_mode import DQNMode
from environment.environment import Environment
from environment.map_loader import load_map
from policies.astar_policy import AStarPolicy
from policies.random_policy import RandomPolicy
from policies.dqn_policy import DQNPolicy
from rendering.renderer import Renderer
from simulation_params import (
    ASTAR_HAZARD_COST,
    DQN_HIDDEN_DIM, DQN_BATCH_SIZE, DQN_GAMMA, DQN_LR,
    DQN_BUFFER_CAPACITY, DQN_TARGET_UPDATE_FREQ, DQN_TRAIN_START_SIZE,
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY, DQN_UPDATE_EVERY,
    PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES,
)

DQN_CFG = dict(
    state_dim=17, action_dim=8,
    hidden_dim=DQN_HIDDEN_DIM,
    batch_size=DQN_BATCH_SIZE,
    gamma=DQN_GAMMA,
    lr=DQN_LR,
    buffer_capacity=DQN_BUFFER_CAPACITY,
    target_update_freq=DQN_TARGET_UPDATE_FREQ,
    train_start_size=DQN_TRAIN_START_SIZE,
    epsilon_start=DQN_EPSILON_START,
    epsilon_end=DQN_EPSILON_END,
    epsilon_decay=DQN_EPSILON_DECAY,
    per_alpha=PER_ALPHA,
    per_beta_start=PER_BETA_START,
    per_beta_frames=PER_BETA_FRAMES,
)

# ── Builders ──────────────────────────────────────────────────────────────────

def build_env(map_path, n_agents, max_steps, use_fsm):
    return Environment(
        map_data=load_map(map_path), dt=0.1,
        max_steps=max_steps, use_fsm=use_fsm,
        num_agents=n_agents,
    )

def build_policy(policy_name, env):
    if policy_name == "astar":
        return AStarPolicy(hazard_cost=ASTAR_HAZARD_COST)
    if policy_name == "astar_fsm":
        return AStarPolicy(hazard_cost=ASTAR_HAZARD_COST)
    if policy_name == "dqn":
        return DQNPolicy(
            mode=DQNMode.EVAL,
            model_path=DQN_MODEL,
            **DQN_CFG,
        )
    raise ValueError(f"Política desconhecida: {policy_name}")

def policy_uses_fsm(policy_name):
    # astar_fsm e dqn usam FSM; astar puro e random não
    return policy_name in ("astar_fsm", "dqn")

def build_renderer(env, title, scale, render):
    if not render:
        return None
    r = Renderer(
        width=env.map_data.width, height=env.map_data.height,
        title=title, fps=30, draw_grid=True,
        tile_size=env.map_data.tile_size, scale=scale,
    )
    r.initialize()
    return r

# ── Episódio ──────────────────────────────────────────────────────────────────

def run_episode(env, policy, renderer=None):
    obs_list = env.reset()
    done = False
    running = True

    while running and not done:
        if renderer is not None:
            running = renderer.poll_events()
            if not running:
                break

        actions = [
            policy.choose_action(env, a, env.get_nearest_exit_bfs(a, use_safe_map=True))
            if not a.evacuated else None
            for a in env.agents
        ]

        _, _, done, _ = env.step(actions)

        if renderer is not None:
            renderer.render(env)

    return env.get_episode_metrics()

# ── Avaliação de uma política num mapa ───────────────────────────────────────

def evaluate_policy_on_map(policy_name, map_name, map_path, n_agents,
                            max_steps, n_episodes, render, scale):
    use_fsm = policy_uses_fsm(policy_name)
    env = build_env(map_path, n_agents, max_steps, use_fsm)
    policy = build_policy(policy_name, env)
    renderer = build_renderer(env, f"{policy_name} | {map_name}", scale, render)

    results = []
    for ep in range(n_episodes):
        metrics = run_episode(env, policy, renderer)
        results.append(metrics)

    if renderer:
        renderer.close()

    N = max(1, n_episodes)
    return {
        "policy":              policy_name,
        "map":                 map_name,
        "evac_rate":           sum(r.get("evacuation_rate", 0)        for r in results) / N,
        "evac_time":           sum(r.get("mean_evacuation_time", max_steps) for r in results) / N,
        "panic_rate":          sum(r.get("panic_rate", 0)             for r in results) / N,
        "peak_emotion":        sum(r.get("mean_peak_emotion", 0)      for r in results) / N,
        "emotion_var":         sum(r.get("emotion_variance", 0)       for r in results) / N,
        "hazard_contact_rate": sum(r.get("hazard_contact_rate", 0)    for r in results) / N,
        "exit_utilization":    sum(r.get("exit_utilization", 0)       for r in results) / N,
    }

# ── Impressão da tabela ───────────────────────────────────────────────────────

POLICY_LABELS = {
    "astar":    "A*",
    "astar_fsm": "A*+FSM",
    "dqn":      "DQN+FSM",
}

METRICS = [
    ("evac_rate",           "evac",    6,  ".2f"),
    ("evac_time",           "time",    6,  ".0f"),
    ("panic_rate",          "panic",   6,  ".2f"),
    ("peak_emotion",        "peak_em", 8,  ".3f"),
    ("emotion_var",         "var",     7,  ".4f"),
    ("hazard_contact_rate", "haz_ct",  7,  ".2f"),
    ("exit_utilization",    "r_util",  7,  ".3f"),
]

def print_results(all_results, policies):
    """Imprime tabela comparativa por mapa."""
    map_names = list(dict.fromkeys(r["map"] for r in all_results))

    for map_name in map_names:
        print(f"\n{'='*70}")
        print(f"  {map_name}")
        print(f"{'='*70}")

        # Cabeçalho
        header = f"  {'Política':<12}"
        for _, col_label, width, _ in METRICS:
            header += f"  {col_label:>{width}}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for pol in policies:
            row_data = next(
                (r for r in all_results if r["policy"] == pol and r["map"] == map_name),
                None
            )
            if row_data is None:
                continue
            label = POLICY_LABELS.get(pol, pol)
            row = f"  {label:<12}"
            for key, _, width, fmt in METRICS:
                val = row_data.get(key, 0)
                row += f"  {val:{width}{fmt}}"
            print(row)

def print_summary(all_results, policies):
    """Imprime médias por política (resumo geral)."""
    print(f"\n{'='*70}")
    print("  RESUMO — médias sobre todos os mapas")
    print(f"{'='*70}")

    header = f"  {'Política':<12}"
    for _, col_label, width, _ in METRICS:
        header += f"  {col_label:>{width}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for pol in policies:
        pol_rows = [r for r in all_results if r["policy"] == pol]
        if not pol_rows:
            continue
        label = POLICY_LABELS.get(pol, pol)
        row = f"  {label:<12}"
        for key, _, width, fmt in METRICS:
            avg = sum(r.get(key, 0) for r in pol_rows) / max(1, len(pol_rows))
            row += f"  {avg:{width}{fmt}}"
        print(row)

# ── Exportação CSV ────────────────────────────────────────────────────────────

def save_csv(all_results, path):
    keys = ["policy", "map"] + [k for k, *_ in METRICS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\nResultados salvos em: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global DQN_MODEL
    p = argparse.ArgumentParser(description="Avalia políticas nos mapas de eval")
    p.add_argument("--episodes",  type=int,   default=30,
                   help="Episódios por mapa por política (default: 30)")
    p.add_argument("--seed",      type=int,   default=None,
                   help="Seed para reproducibilidade")
    p.add_argument("--render",    action="store_true",
                   help="Renderiza os episódios")
    p.add_argument("--scale",     type=int,   default=2,
                   help="Escala do render (default: 2)")
    p.add_argument("--policies",  nargs="+",
                   default=["astar", "astar_fsm", "dqn"],
                   choices=["astar", "astar_fsm", "dqn"],
                   help="Políticas a avaliar")
    p.add_argument("--maps",      nargs="+",  default=None,
                   help="Subset de mapas (nomes, default: todos)")
    p.add_argument("--csv",       type=str,   default=None,
                   help="Caminho para salvar CSV com resultados")
    p.add_argument("--model",     type=str,   default=DQN_MODEL,
                   help=f"Caminho do modelo DQN (default: {DQN_MODEL})")
    args = p.parse_args()

    DQN_MODEL = args.model

    if args.seed is not None:
        import random as _r
        import numpy as _np
        _r.seed(args.seed)
        _np.seed = args.seed

    eval_maps = EVAL_MAPS
    if args.maps:
        eval_maps = [(n, p, a, s) for n, p, a, s in EVAL_MAPS if n in args.maps]
        if not eval_maps:
            print(f"Nenhum mapa encontrado: {args.maps}")
            sys.exit(1)

    total = len(args.policies) * len(eval_maps)
    done  = 0
    all_results = []

    print(f"\nAvaliando {len(args.policies)} política(s) × "
          f"{len(eval_maps)} mapa(s) × {args.episodes} episódios")
    print(f"Total: {total} configurações\n")

    for pol in args.policies:
        for map_name, map_path, n_agents, max_steps in eval_maps:
            done += 1
            print(f"  [{done}/{total}] {POLICY_LABELS.get(pol, pol):<10} | {map_name} ...",
                  end=" ", flush=True)
            result = evaluate_policy_on_map(
                pol, map_name, map_path, n_agents, max_steps,
                args.episodes, args.render, args.scale,
            )
            all_results.append(result)
            print(f"evac={result['evac_rate']:.2f}  "
                  f"panic={result['panic_rate']:.2f}  "
                  f"time={result['evac_time']:.0f}")

    print_results(all_results, args.policies)
    print_summary(all_results, args.policies)

    if args.csv:
        save_csv(all_results, args.csv)

if __name__ == "__main__":
    main()