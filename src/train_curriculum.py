import argparse
import csv
import json
from collections import deque
from pathlib import Path
from core.dqn_mode import DQNMode
from environment.environment import Environment
from environment.map_loader import load_map
from policies.dqn_policy import DQNPolicy
from policies.dqn_policy import PrioritizedReplayBuffer
from rendering.renderer_debug import Renderer
from simulation_params import *

# (nome, caminho, n_agents, max_steps, epsilon_decay, eps_start)
CURRICULUM = [
    ("mall_small",              "maps/train/mall_small.txt",              4,  300,   32_500, 1.00),
    ("school_small",            "maps/train/school_small.txt",            4,  300,   45_000, 1.00),
    ("office_wing_small",       "maps/train/office_wing_small.txt",       4,  300,   45_000, 0.90),
    ("library_small",           "maps/train/library_small.txt",           4,  300,   45_000, 0.90),
    ("mall_small_bottom",       "maps/train/mall_small_bottom.txt",       4,  300,   45_000, 0.90),
    ("library_small_bottom",    "maps/train/library_small_bottom.txt",    4,  300,   45_000, 0.90),
    ("library_medium",          "maps/train/library_medium.txt",          6,  400,   85_000, 0.70),
    ("hazard_corridor_small",   "maps/train/hazard_corridor_small.txt",   4,  350,   41_650, 0.70),
    ("hazard_near_exit_small",  "maps/train/hazard_near_exit_small.txt",  6,  350,  150_000, 0.60),
    ("school_floor",            "maps/train/school_floor.txt",            8,  400,  190_400, 0.60),
    ("office_wing_medium",      "maps/train/office_wing_medium.txt",     10,  400,  238_000, 0.50),
    ("hazard_near_exit_medium", "maps/train/hazard_near_exit_medium.txt",10,  420,  249_900, 0.50),
    ("bridge_open_medium",      "maps/train/bridge_open_medium.txt",     11,  400,  290_000, 0.45),
    ("bridge_corridor_medium",  "maps/train/bridge_corridor_medium.txt", 11,  400,  290_000, 0.45),
    ("bridge_hazard_intro",     "maps/train/bridge_hazard_intro.txt",    12,  420,  330_000, 0.40),
    ("bridge_multi_exit",       "maps/train/bridge_multi_exit.txt",      12,  420,  299_880, 0.40),
    ("hazard_bypass_medium",    "maps/train/hazard_bypass_medium.txt",   12,  450,  436_050, 0.35),
    ("mall_medium",             "maps/train/mall_medium.txt",            12,  450,  367_200, 0.35),
    ("hazard_dense_office",     "maps/train/hazard_dense_office.txt",    12,  450,  436_050, 0.30),
    ("library_hard",            "maps/train/library_hard.txt",           12,  450,  367_200, 0.30),
]

PROMOTION_THRESHOLD  = CURRICULUM_PROMOTION_THRESHOLD
EVAL_WINDOW          = CURRICULUM_EVAL_WINDOW
PATIENCE             = CURRICULUM_PATIENCE
SAVE_EVERY           = CURRICULUM_SAVE_EVERY
EARLY_PATIENCE_AFTER = CURRICULUM_EARLY_PATIENCE_AFTER
EARLY_PATIENCE_THR   = CURRICULUM_EARLY_PATIENCE_THRESHOLD

MODEL_DIR  = Path("models")
LOG_DIR    = Path("logs")
MODEL_PATH = MODEL_DIR / "dqn_fsm.pth"
LOG_PATH   = LOG_DIR / "training_log.csv"
STATE_PATH = MODEL_DIR / "curriculum_state.json"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",     type=int,  default=None,  help="Começa no stage N")
    p.add_argument("--render",    action="store_true")
    p.add_argument("--model",     type=str,  default=str(MODEL_PATH))
    p.add_argument("--scale",     type=int, default=1,      help="Fator de escala visual (ex: 2 = janela 2x maior)")
    p.add_argument("--seed",      type=int, default=None,   help="Seed para reproducibilidade na eval (não afeta treino)")
    return p.parse_args()

def ensure_dirs():
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

def init_log():
    LOG_DIR.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow([
                "episode_global", "stage", "map_name",
                "evacuation_rate", "all_evacuated", "mean_evac_time", "steps",
                "mean_emotion_final", "emotion_variance", "mean_speed_ratio",
                "total_reward", "epsilon",
            ])

def load_state():
    if STATE_PATH.exists():
        d = json.loads(STATE_PATH.read_text())
        return d.get("stage", 0), d.get("episode_global", 0)
    return 0, 0

def save_state(stage, ep):
    STATE_PATH.write_text(json.dumps({"stage": stage, "episode_global": ep}))

def log_ep(row):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

def build_env(map_path, n_agents, max_steps, contagion_radius=None):
    return Environment(
        load_map(map_path),
        dt=0.1,
        max_steps=max_steps,
        use_fsm=True,
        num_agents=n_agents,
        contagion_radius=contagion_radius,
    )

def build_policy(model_path, mode):
    return DQNPolicy(
        mode=mode,
        model_path=model_path,
        state_dim=DQN_OBS_DIM,
        action_dim=8,
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
        update_every=DQN_UPDATE_EVERY
    )

def build_renderer(env, title, fps=30, enabled=True, scale=1):
    r = Renderer(
        width=env.map_data.width,
        height=env.map_data.height,
        title=title, 
        fps=fps, 
        draw_grid=True,
        tile_size=env.map_data.tile_size, 
        scale=scale
    )
    r.initialize()
    r.enabled = enabled
    return r

def _stage_contagion_radius(map_path: str, n_agents: int):
    if n_agents < N_CONTAGION_THRESHOLD:
        return None
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            content = f.read()
        total  = sum(1 for c in content if c in ".EHSO")
        hazard = content.count("H")
        if total > 0 and hazard / total >= HAZARD_CONTAGION_PCTG:
            return CONTAGION_RADIUS_HIGH_N
    except OSError:
        pass
    return None

def run_episode(env, policy, renderer):
    obs_list = env.reset()
    prev_obs = {id(a): obs_list[i] for i, a in enumerate(env.agents)}
    ep_reward = 0.0
    done = False
    running = True

    while running and not done:
        running = renderer.poll_events()
        if not running: break

        actions = [
            policy.choose_action(env, a)
            if not a.evacuated else None
            for a in env.agents
        ]

        next_obs, rewards, done, _ = env.step(actions)

        for i, a in enumerate(env.agents):
            if actions[i] is None: continue

            policy.store_transition(
                obs=prev_obs[id(a)],
                action=actions[i],
                reward=rewards[i],
                next_obs=next_obs[i],
                done=done,
            )

        prev_obs = {id(a): next_obs[i] for i, a in enumerate(env.agents)}
        ep_reward += sum(rewards)

        renderer.update_rewards(env, rewards)
        renderer.render(env)

    metrics = env.get_episode_metrics()
    metrics["total_reward"] = ep_reward
    return metrics

def train_stage(stage_idx, policy, ep_global_start, render, prev_promoted, scale=1):
    name, map_path, n_agents, max_steps, stage_decay, eps_start = CURRICULUM[stage_idx]
    label = f"Stage {stage_idx + 1}"

    contagion_radius = _stage_contagion_radius(map_path, n_agents)
    env              = build_env(map_path, n_agents, max_steps, contagion_radius=contagion_radius)
    renderer         = build_renderer(env, f"{label}: {name}", enabled=render, scale=scale)

    if not prev_promoted:
        policy.reset_buffer()
        print(f"  [buffer resetado]")

    policy.reset_for_stage(stage_decay=stage_decay, epsilon_start=eps_start)

    recent    = deque(maxlen=EVAL_WINDOW)
    ep_global = ep_global_start
    stage_ep  = 0
    promoted  = False

    cr_info = f"  contagion={contagion_radius:.0f}px" if contagion_radius else ""
    print(f"\n{'='*60}")
    print(f"{label}/{len(CURRICULUM)}: {name}")
    print(f"  {map_path}")
    print(f"  agents={n_agents}  max_steps={max_steps}  decay={stage_decay:,}  eps_start={eps_start:.2f}{cr_info}")
    print(f"{'='*60}")

    while stage_ep < PATIENCE:
        m = run_episode(env, policy, renderer)
        recent.append(m["evacuation_rate"])
        ep_global += 1
        stage_ep  += 1

        log_ep({
            "episode_global":     ep_global,
            "stage":              stage_idx + 1,
            "map_name":           name,
            "evacuation_rate":    round(m["evacuation_rate"], 4),
            "all_evacuated":      int(m.get("all_evacuated", False)),
            "mean_evac_time":     round(m.get("mean_evacuation_time", max_steps), 2),
            "steps":              m.get("steps", max_steps),
            "mean_emotion_final": round(m.get("mean_emotion_final", 0), 4),
            "emotion_variance":   round(m.get("emotion_variance", 0), 4),
            "mean_speed_ratio":   round(m.get("mean_speed_ratio", 1), 4),
            "total_reward":       round(m["total_reward"], 2),
            "epsilon":            round(policy.current_epsilon(), 4),
        })

        if stage_ep % 10 == 0:
            avg = sum(recent) / len(recent)
            print(f"  [s{stage_idx+1}|ep{stage_ep:4d}|g{ep_global:5d}] "
                  f"evac={m['evacuation_rate']:.2f}  avg{EVAL_WINDOW}={avg:.2f}  "
                  f"reward={m['total_reward']:7.1f}  eps={policy.current_epsilon():.3f}  "
                  f"emotion={m.get('mean_emotion_final', 0):.2f}  "
                  f"var={m.get('emotion_variance', 0):.4f}")

        if stage_ep % SAVE_EVERY == 0:
            policy.save(str(MODEL_DIR / f"ckpt_s{stage_idx+1}_ep{ep_global}.pth"))
            save_state(stage_idx, ep_global)

        if (stage_ep >= EARLY_PATIENCE_AFTER and len(recent) >= EVAL_WINDOW and sum(recent) / len(recent) < EARLY_PATIENCE_THR):
            print(f"\n  [early patience: avg{EVAL_WINDOW}={sum(recent)/len(recent):.2f} após {stage_ep} ep]")
            break

        if (len(recent) >= EVAL_WINDOW and sum(recent) / len(recent) >= PROMOTION_THRESHOLD):
            promoted = True
            print(f"\n  PROMOVIDO! avg{EVAL_WINDOW}={sum(recent)/len(recent):.2f}")
            break

    renderer.close()

    stage_ckpt = MODEL_DIR / f"ckpt_s{stage_idx+1}_final.pth"
    policy.save(str(stage_ckpt))
    policy.save(str(MODEL_PATH))
    save_state(stage_idx + (1 if promoted else 0), ep_global)

    if not promoted:
        avg = sum(recent) / len(recent) if recent else 0
        print(f"\n  Patience esgotada ({stage_ep} ep, avg={avg:.2f}). Avançando.")

    return ep_global, promoted, renderer

def main():
    args = parse_args()
    ensure_dirs()
    init_log()

    saved_stage, ep_global = load_state()
    start = (args.stage - 1) if args.stage else saved_stage
    start = max(0, min(start, len(CURRICULUM) - 1))

    policy = build_policy(args.model, DQNMode.TRAIN)

    print(f"[INFO] device  = {policy.device}")
    print(f"[INFO] modelo  = {args.model}")
    print(f"[INFO] stage   = {start + 1}/{len(CURRICULUM)}  ep_global={ep_global}")

    render = args.render
    prev_promoted = True
    for stage_idx in range(start, len(CURRICULUM)):
        ep_global, prev_promoted, render = train_stage(
            stage_idx,
            policy,
            ep_global,
            render=render,
            prev_promoted=prev_promoted,
            scale=args.scale,
        )

    print(f"\n[DONE] Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()