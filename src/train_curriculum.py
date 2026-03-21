"""
train_curriculum.py — treino DQN com currículo progressivo de 12 stages.

Todos os mapas são plantas de edifícios reais (biblioteca, escritório, shopping, escola).

Uso:
    python train_curriculum.py                         # treino completo
    python train_curriculum.py --stage 5               # começa no stage 5
    python train_curriculum.py --eval                  # avalia em todos os mapas eval
    python train_curriculum.py --render                # com visualização
    python train_curriculum.py --quiet                 # sem prints detalhados

Currículo — 12 stages, complexidade crescente:
    Stage  1  library_small        pequeno, sem hazard, navegação pura
    Stage  2  office_wing_small    pequeno, sem hazard, fileira de salas
    Stage  3  mall_small           pequeno, sem hazard, shopping
    Stage  4  school_small         pequeno, sem hazard, escola
    Stage  5  library_medium       médio, hazard leve, 2 exits
    Stage  6  mall_food_court      médio, hazard leve, praça central
    Stage  7  school_floor         médio, hazard, corredores
    Stage  8  office_wing_medium   médio, hazard, escritório completo
    Stage  9  office_complex_real  médio, hazard perto do exit, L-shape
    Stage 10  mall_medium          médio, hazard, shopping anel
    Stage 11  library_hard         médio, gargalo interno + hazard
    Stage 12  di_style             médio, auditório + salas satélite

Promoção:
    evacuation_rate média ≥ 80% nos últimos 30 episódios → avança.
    Após PATIENCE episódios sem promoção → avança mesmo assim.
"""

import argparse
import csv
import json
import os
from collections import deque
from pathlib import Path

from core.dqn_mode import DQNMode
from environment.environment import Environment
from environment.map_loader import load_map
from policies.dqn_policy import DQNPolicy
from rendering.renderer import Renderer
from simulation_params import (
    DQN_HIDDEN_DIM, DQN_BATCH_SIZE, DQN_GAMMA, DQN_LR,
    DQN_BUFFER_CAPACITY, DQN_TARGET_UPDATE_FREQ, DQN_TRAIN_START_SIZE,
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY,
    CURRICULUM_PROMOTION_THRESHOLD, CURRICULUM_EVAL_WINDOW,
    CURRICULUM_PATIENCE, CURRICULUM_SAVE_EVERY,
)

# ── Currículo ─────────────────────────────────────────────────────────────────

CURRICULUM = [
    # (nome,               caminho,                                  n_agents, max_steps)
    ("library_small",      "maps/train/library_small.txt",           4,  300),
    ("office_wing_small",  "maps/train/office_wing_small.txt",       4,  300),
    ("mall_small",         "maps/train/mall_small.txt",              4,  300),
    ("school_small",       "maps/train/school_small.txt",            4,  300),
    ("library_medium",     "maps/train/library_medium.txt",          6,  400),
    ("mall_food_court",    "maps/train/mall_food_court.txt",         6,  400),
    ("school_floor",       "maps/train/school_floor.txt",            6,  400),
    ("office_wing_medium", "maps/train/office_wing_medium.txt",      6,  400),
    ("office_complex_real","maps/train/office_complex_real.txt",     8,  450),
    ("mall_medium",        "maps/train/mall_medium.txt",             8,  450),
    ("library_hard",       "maps/train/library_hard.txt",            8,  450),
    ("di_style",           "maps/train/di_style.txt",                8,  500),
]

EVAL_MAPS = [
    ("library_bottleneck", "maps/eval/library_bottleneck.txt",  8, 450),
    ("office_single_exit", "maps/eval/office_single_exit.txt",  8, 450),
    ("mall_panic",         "maps/eval/mall_panic.txt",          8, 450),
    ("school_evacuation",  "maps/eval/school_evacuation.txt",   8, 450),
    ("di_emergency",       "maps/eval/di_emergency.txt",        8, 500),
]

PROMOTION_THRESHOLD = CURRICULUM_PROMOTION_THRESHOLD
EVAL_WINDOW         = CURRICULUM_EVAL_WINDOW
PATIENCE            = CURRICULUM_PATIENCE
SAVE_EVERY          = CURRICULUM_SAVE_EVERY

MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs")
MODEL_PATH = MODEL_DIR / "dqn_fsm.pth"
LOG_PATH   = LOG_DIR / "training_log.csv"
STATE_PATH = MODEL_DIR / "curriculum_state.json"

DQN_CFG = dict(
    hidden_dim=DQN_HIDDEN_DIM, batch_size=DQN_BATCH_SIZE,
    gamma=DQN_GAMMA, lr=DQN_LR,
    buffer_capacity=DQN_BUFFER_CAPACITY, target_update_freq=DQN_TARGET_UPDATE_FREQ,
    train_start_size=DQN_TRAIN_START_SIZE, epsilon_start=DQN_EPSILON_START,
    epsilon_end=DQN_EPSILON_END, epsilon_decay=DQN_EPSILON_DECAY,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

def save_state(stage, ep):
    STATE_PATH.write_text(json.dumps({"stage": stage, "episode_global": ep}))

def load_state():
    if STATE_PATH.exists():
        d = json.loads(STATE_PATH.read_text())
        return d.get("stage", 0), d.get("episode_global", 0)
    return 0, 0

def init_log():
    LOG_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode_global","stage","map_name",
                "evacuation_rate","all_evacuated","mean_evac_time","steps",
                "mean_emotion_final","emotion_variance","mean_speed_ratio",
                "total_reward","epsilon",
            ])

def log_ep(row):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

def build_env(map_path, n_agents, max_steps):
    return Environment(
        map_data=load_map(map_path), dt=0.1,
        max_steps=max_steps, use_fsm=True, num_agents=n_agents,
    )

def build_policy(env, model_path, mode):
    return DQNPolicy(mode=mode, model_path=model_path,
                     state_dim=env.OBS_DIM, action_dim=8, **DQN_CFG)

def build_renderer(env, title, fps=30):
    r = Renderer(width=env.map_data.width, height=env.map_data.height,
                 title=title, fps=fps, draw_grid=True,
                 tile_size=env.map_data.tile_size)
    r.initialize()
    return r

# ── Loop de episódio ──────────────────────────────────────────────────────────

def run_episode(env, policy, renderer=None, train=True):
    obs_list = env.reset()
    prev_obs = {id(a): obs_list[i] for i, a in enumerate(env.agents)}
    ep_reward = 0.0
    done = False
    running = True

    while running and not done:
        if renderer is not None:
            running = renderer.poll_events()
            if not running:
                break

        actions = [
            policy.choose_action(env, a, env.get_nearest_exit(a))
            if not a.evacuated else None
            for a in env.agents
        ]
        next_obs, rewards, done, _ = env.step(actions)

        for a in env.agents:
            a.update_peak_emotion()

        if train:
            for i, a in enumerate(env.agents):
                if actions[i] is None:
                    continue
                policy.store_transition(
                    obs=prev_obs[id(a)], action=actions[i],
                    reward=rewards[i], next_obs=next_obs[i], done=done,
                )

        prev_obs = {id(a): next_obs[i] for i, a in enumerate(env.agents)}
        ep_reward += sum(rewards)

        if renderer is not None:
            renderer.render(env)

    metrics = env.get_episode_metrics()
    metrics["total_reward"] = ep_reward
    return metrics

# ── Treino de um stage ────────────────────────────────────────────────────────

def train_stage(stage_idx, policy, ep_global_start, render=False, verbose=True):
    name, map_path, n_agents, max_steps = CURRICULUM[stage_idx]
    env = build_env(map_path, n_agents, max_steps)
    renderer = build_renderer(env, f"Stage {stage_idx+1}: {name}") if render else None

    recent = deque(maxlen=EVAL_WINDOW)
    ep_global = ep_global_start
    stage_ep  = 0
    promoted  = False

    if verbose:
        print(f"\n{'='*55}")
        print(f"STAGE {stage_idx+1}/12: {name}")
        print(f"  {map_path}  agents={n_agents}  max_steps={max_steps}")
        print(f"{'='*55}")

    while stage_ep < PATIENCE:
        m = run_episode(env, policy, renderer, train=True)
        recent.append(m["evacuation_rate"])
        ep_global += 1; stage_ep += 1

        log_ep({
            "episode_global":    ep_global,
            "stage":             stage_idx + 1,
            "map_name":          name,
            "evacuation_rate":   round(m["evacuation_rate"], 4),
            "all_evacuated":     int(m.get("all_evacuated", False)),
            "mean_evac_time":    round(m.get("mean_evacuation_time", max_steps), 2),
            "steps":             m.get("steps", max_steps),
            "mean_emotion_final":round(m.get("mean_emotion_final", 0), 4),
            "emotion_variance":  round(m.get("emotion_variance", 0), 4),
            "mean_speed_ratio":  round(m.get("mean_speed_ratio", 1), 4),
            "total_reward":      round(m["total_reward"], 2),
            "epsilon":           round(policy.current_epsilon(), 4),
        })

        if verbose and stage_ep % 10 == 0:
            avg = sum(recent) / len(recent)
            print(f"  [s{stage_idx+1}|ep{stage_ep:4d}|g{ep_global:5d}] "
                  f"evac={m['evacuation_rate']:.2f}  avg{EVAL_WINDOW}={avg:.2f}  "
                  f"reward={m['total_reward']:6.1f}  eps={policy.current_epsilon():.3f}  "
                  f"emotion={m.get('mean_emotion_final',0):.2f}")

        if stage_ep % SAVE_EVERY == 0:
            policy.save(str(MODEL_DIR / f"ckpt_s{stage_idx+1}_ep{ep_global}.pth"))
            save_state(stage_idx, ep_global)

        if len(recent) >= EVAL_WINDOW and sum(recent)/len(recent) >= PROMOTION_THRESHOLD:
            promoted = True
            if verbose:
                print(f"\n  ✓ PROMOVIDO! avg={sum(recent)/len(recent):.2f}")
            break

    if renderer:
        renderer.close()

    policy.save(str(MODEL_PATH))
    save_state(stage_idx + (1 if promoted else 0), ep_global)

    if verbose and not promoted:
        print(f"\n  → Patience esgotada. Avançando.")

    return ep_global, promoted

# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate(model_path, n_episodes=20, render=False):
    print(f"\n{'='*55}\nAVALIAÇÃO — {model_path}\n{'='*55}")
    for name, map_path, n_agents, max_steps in EVAL_MAPS:
        env = build_env(map_path, n_agents, max_steps)
        policy = build_policy(env, model_path, DQNMode.EVAL)
        renderer = build_renderer(env, name) if render else None
        results = [run_episode(env, policy, renderer, train=False)
                   for _ in range(n_episodes)]
        if renderer:
            renderer.close()
        avg_evac    = sum(r["evacuation_rate"] for r in results) / n_episodes
        avg_emotion = sum(r.get("mean_emotion_final", 0) for r in results) / n_episodes
        avg_time    = sum(r.get("mean_evacuation_time", max_steps) for r in results) / n_episodes
        print(f"  {name:<25}  evac={avg_evac:.2f}  emotion={avg_emotion:.2f}  time={avg_time:.0f}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",    type=int, default=None)
    p.add_argument("--eval",     action="store_true")
    p.add_argument("--render",   action="store_true")
    p.add_argument("--model",    type=str, default=str(MODEL_PATH))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--quiet",    action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    init_log()

    if args.eval:
        evaluate(args.model, args.episodes, args.render)
        return

    saved_stage, ep_global = load_state()
    start = (args.stage - 1) if args.stage else saved_stage
    start = max(0, min(start, len(CURRICULUM) - 1))

    if args.stage and args.stage - 1 != saved_stage:
        ep_global = 0

    _, map_path, n_agents, max_steps = CURRICULUM[start]
    env_init = build_env(map_path, n_agents, max_steps)
    policy   = build_policy(env_init, args.model, DQNMode.TRAIN)

    print(f"[INFO] device={policy.device}")
    print(f"[INFO] modelo={args.model}")
    print(f"[INFO] stage inicial={start+1}  ep_global={ep_global}")

    for stage_idx in range(start, len(CURRICULUM)):
        ep_global, _ = train_stage(
            stage_idx, policy, ep_global,
            render=args.render, verbose=not args.quiet,
        )

    print(f"\n[DONE] Treino completo. Modelo: {MODEL_PATH}")

if __name__ == "__main__":
    main()