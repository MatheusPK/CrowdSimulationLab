"""
train_curriculum.py — treino DQN com currículo progressivo.

Estratégia: multi-agent parameter sharing.
  Todos os agentes compartilham a mesma rede e contribuem transições
  independentes para o replay buffer.

Uso:
    python train_curriculum.py                   # treino completo
    python train_curriculum.py --stage 6         # começa no stage 6
    python train_curriculum.py --eval            # avalia nos mapas eval
    python train_curriculum.py --fine-tune       # fine-tuning em di_style
    python train_curriculum.py --render          # com visualização
    python train_curriculum.py --quiet           # sem prints detalhados
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
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY, DQN_UPDATE_EVERY,
    PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES,
    CURRICULUM_PROMOTION_THRESHOLD, CURRICULUM_EVAL_WINDOW,
    CURRICULUM_PATIENCE, CURRICULUM_SAVE_EVERY,
    CURRICULUM_EARLY_PATIENCE_AFTER, CURRICULUM_EARLY_PATIENCE_THRESHOLD,
    CONTAGION_RADIUS_HIGH_N, N_CONTAGION_THRESHOLD, HAZARD_CONTAGION_PCTG,
)

# ── Currículo ─────────────────────────────────────────────────────────────────
# (nome, caminho, n_agents, max_steps, epsilon_decay)
# epsilon_decay = ep_alvo × (N × max_steps / UPDATE_EVERY)
# ep_alvo = 85% do ep_promoção esperado para o stage

CURRICULUM = [
    # Stages 1-4: navegação básica — top e side exits (N=4)
    ("mall_small",              "maps/train/mall_small.txt",              4,  300,   32_500),
    ("school_small",            "maps/train/school_small.txt",            4,  300,   45_000),
    ("office_wing_small",       "maps/train/office_wing_small.txt",       4,  300,   45_000),
    ("library_small",           "maps/train/library_small.txt",           4,  300,   45_000),
    # Stages 5-6: exit EMBAIXO — generalização de direção de saída (N=4)
    # Sem esses stages, o modelo aprende que exits ficam sempre em topo/lateral.
    # mall_small_bottom e library_small_bottom são as mesmas salas com exit
    # na parede sul — forçam o agente a aprender "ir para baixo" também.
    ("mall_small_bottom",       "maps/train/mall_small_bottom.txt",       4,  300,   45_000),
    ("library_small_bottom",    "maps/train/library_small_bottom.txt",    4,  300,   45_000),
    # Stage 7: primeiro contato com dinâmica coletiva (N=6)
    ("library_medium",          "maps/train/library_medium.txt",          6,  400,   85_000),
    # Stage 8: dilema de rota com hazard lateral (N=4)
    ("hazard_corridor_small",   "maps/train/hazard_corridor_small.txt",   4,  350,   41_650),
    # Stage 9: hazard frente ao exit (N=6)
    ("hazard_near_exit_small",  "maps/train/hazard_near_exit_small.txt",  6,  350,  150_000),
    # Stage 10: escala coletiva gradual (N=8)
    ("school_floor",            "maps/train/school_floor.txt",            8,  400,  190_400),
    # Stage 11: base antes de N=12 (N=10)
    ("office_wing_medium",      "maps/train/office_wing_medium.txt",     10,  400,  238_000),
    # Stage 12: hazard no corredor do exit (N=10)
    ("hazard_near_exit_medium", "maps/train/hazard_near_exit_medium.txt",10,  420,  249_900),
    # Stages 13-16: ponte N=11→12
    ("bridge_open_medium",      "maps/train/bridge_open_medium.txt",     11,  400,  290_000),
    ("bridge_corridor_medium",  "maps/train/bridge_corridor_medium.txt", 11,  400,  290_000),
    ("bridge_hazard_intro",     "maps/train/bridge_hazard_intro.txt",    12,  420,  330_000),
    ("bridge_multi_exit",       "maps/train/bridge_multi_exit.txt",      12,  420,  299_880),
    # Stages 17-20: hazard real, alta densidade (N=12)
    ("hazard_bypass_medium",    "maps/train/hazard_bypass_medium.txt",   12,  450,  436_050),
    ("mall_medium",             "maps/train/mall_medium.txt",            12,  450,  367_200),
    ("hazard_dense_office",     "maps/train/hazard_dense_office.txt",    12,  450,  436_050),
    ("library_hard",            "maps/train/library_hard.txt",           12,  450,  367_200),
]

FINE_TUNE_MAP = ("di_style", "maps/train/di_style.txt", 12, 500, 270_000)

EVAL_MAPS = [
    ("library_bottleneck", "maps/eval/library_bottleneck.txt", 12, 450),
    ("office_single_exit", "maps/eval/office_single_exit.txt", 12, 450),
    ("mall_panic",         "maps/eval/mall_panic.txt",         12, 450),
    ("school_evacuation",  "maps/eval/school_evacuation.txt",  12, 450),
    ("di_emergency",       "maps/eval/di_emergency.txt",       12, 500),
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

DQN_CFG = dict(
    hidden_dim=DQN_HIDDEN_DIM, batch_size=DQN_BATCH_SIZE,
    gamma=DQN_GAMMA, lr=DQN_LR,
    buffer_capacity=DQN_BUFFER_CAPACITY, target_update_freq=DQN_TARGET_UPDATE_FREQ,
    train_start_size=DQN_TRAIN_START_SIZE, epsilon_start=DQN_EPSILON_START,
    epsilon_end=DQN_EPSILON_END, epsilon_decay=DQN_EPSILON_DECAY,
    per_alpha=PER_ALPHA, per_beta_start=PER_BETA_START, per_beta_frames=PER_BETA_FRAMES,
    update_every=DQN_UPDATE_EVERY,
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
                "episode_global", "stage", "map_name",
                "evacuation_rate", "all_evacuated", "mean_evac_time", "steps",
                "mean_emotion_final", "emotion_variance", "mean_speed_ratio",
                "total_reward", "epsilon",
            ])

def log_ep(row):
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

def build_env(map_path, n_agents, max_steps, contagion_radius=None):
    kwargs = dict(map_data=load_map(map_path), dt=0.1,
                  max_steps=max_steps, use_fsm=True, num_agents=n_agents)
    if contagion_radius is not None:
        kwargs["contagion_radius"] = contagion_radius
    return Environment(**kwargs)

def _stage_contagion_radius(map_path: str, n_agents: int):
    """Retorna CONTAGION_RADIUS_HIGH_N para stages com N>=12 e hazard >5%."""
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

def build_policy(env, model_path, mode):
    return DQNPolicy(mode=mode, model_path=model_path,
                     state_dim=env.OBS_DIM, action_dim=8, **DQN_CFG)

def build_renderer(env, title, fps=30, enabled=True, scale=1):
    r = Renderer(width=env.map_data.width, height=env.map_data.height,
                 title=title, fps=fps, draw_grid=True,
                 tile_size=env.map_data.tile_size, scale=scale)
    r.initialize()
    r.enabled = enabled
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

        if train:
            for i, a in enumerate(env.agents):
                if actions[i] is None:
                    continue
                policy.store_transition(
                    obs=prev_obs[id(a)],
                    action=actions[i],
                    reward=rewards[i],
                    next_obs=next_obs[i],
                    done=done,
                )

        prev_obs = {id(a): next_obs[i] for i, a in enumerate(env.agents)}
        ep_reward += sum(rewards)

        if renderer is not None:
            renderer.render(env)

    metrics = env.get_episode_metrics()
    metrics["total_reward"] = ep_reward
    return metrics

# ── Treino de um stage ────────────────────────────────────────────────────────

def train_stage(stage_idx, policy, ep_global_start, render=False, verbose=True,
                stage_label=None, prev_renderer=None, prev_promoted=True, scale=1):
    name, map_path, n_agents, max_steps, stage_decay = CURRICULUM[stage_idx]
    label = stage_label or f"Stage {stage_idx + 1}"

    contagion_radius = _stage_contagion_radius(map_path, n_agents)
    env = build_env(map_path, n_agents, max_steps, contagion_radius=contagion_radius)

    render_enabled = prev_renderer.enabled if prev_renderer is not None else True
    renderer = build_renderer(env, f"{label}: {name}", enabled=render_enabled, scale=scale) if render else None

    # prev_promoted=True  → stage anterior foi promovido (normal ou early patience)
    # prev_promoted=False → stage anterior esgotou a patience completa → reseta buffer
    # Early patience retorna is_promoted=False mas sem necessidade de reset (buffer útil).
    # A distinção é feita pelo parâmetro reset_buffer passado pelo chamador.
    if not prev_promoted:
        # Buffer reset apenas quando patience COMPLETA foi esgotada.
        # Early patience NÃO reseta — o buffer contém transições transferíveis.
        if hasattr(policy.buffer, "reset"):
            policy.buffer.reset()
        else:
            from policies.dqn_policy import PrioritizedReplayBuffer
            policy.buffer = PrioritizedReplayBuffer(
                policy.buffer.capacity, alpha=policy.buffer.alpha
            )
        if verbose:
            print(f"  [buffer resetado — patience completa esgotada]")

    policy.reset_for_stage(stage_decay=stage_decay, epsilon_start=1.0)

    recent    = deque(maxlen=EVAL_WINDOW)
    ep_global = ep_global_start
    stage_ep  = 0
    promoted  = False

    if verbose:
        markers = {
            "hazard_corridor_small":   "★",
            "hazard_near_exit_small":  "★★",
            "hazard_near_exit_medium": "★★",
            "hazard_bypass_medium":    "★",
            "hazard_dense_office":     "★",
            "bridge_open_medium":      "bridge",
            "bridge_corridor_medium":  "bridge",
            "bridge_hazard_intro":     "bridge",
            "bridge_multi_exit":       "bridge",
        }
        marker  = f" [{markers[name]}]" if name in markers else ""
        cr_info = f"  contagion={contagion_radius:.0f}px" if contagion_radius else ""
        print(f"\n{'='*60}")
        print(f"{label}/{len(CURRICULUM)}: {name}{marker}")
        print(f"  {map_path}")
        print(f"  agents={n_agents}  max_steps={max_steps}  decay={stage_decay:,}{cr_info}")
        print(f"{'='*60}")

    while stage_ep < PATIENCE:
        m = run_episode(env, policy, renderer, train=True)
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

        if verbose and stage_ep % 10 == 0:
            avg = sum(recent) / len(recent)
            print(f"  [s{stage_idx+1}|ep{stage_ep:4d}|g{ep_global:5d}] "
                  f"evac={m['evacuation_rate']:.2f}  avg{EVAL_WINDOW}={avg:.2f}  "
                  f"reward={m['total_reward']:7.1f}  eps={policy.current_epsilon():.3f}  "
                  f"emotion={m.get('mean_emotion_final', 0):.2f}  "
                  f"var={m.get('emotion_variance', 0):.4f}")

        if stage_ep % SAVE_EVERY == 0:
            policy.save(str(MODEL_DIR / f"ckpt_s{stage_idx+1}_ep{ep_global}.pth"))
            save_state(stage_idx, ep_global)

        # Early patience: avança se avg30 baixo E epsilon já caiu o suficiente
        # para que a política greedy tenha sido testada de verdade.
        # Com epsilon > 0.20, o sinal de avg30 reflete exploração aleatória, não a policy.
        # Sem o guard de epsilon, o early patience dispara cedo demais quando o
        # epsilon_decay foi aumentado (ex: 45k→90k), avançando antes do agente aprender.
        early_patience_ok = (
            stage_ep >= EARLY_PATIENCE_AFTER
            and len(recent) >= EVAL_WINDOW
            and sum(recent) / len(recent) < EARLY_PATIENCE_THR
            and policy.current_epsilon() < 0.20
        )
        if early_patience_ok:
            if verbose:
                print(f"\n  [early patience: avg{EVAL_WINDOW}={sum(recent)/len(recent):.2f} "
                      f"eps={policy.current_epsilon():.3f} após {stage_ep} ep]")
            # Early patience NÃO reseta o buffer — as transições de navegação
            # deste stage são transferíveis para stages seguintes.
            # O reset de buffer ocorre apenas na patience completa (prev_promoted=False).
            promoted = None  # sinaliza "early" para o chamador
            break

        if (len(recent) >= EVAL_WINDOW
                and sum(recent) / len(recent) >= PROMOTION_THRESHOLD):
            promoted = True
            if verbose:
                print(f"\n  ✓ PROMOVIDO! avg{EVAL_WINDOW}={sum(recent)/len(recent):.2f}")
            break

    if renderer:
        renderer.close()

    # promoted=True  → promovido normalmente
    # promoted=None  → early patience (avança sem considerar promovido)
    # promoted=False → patience completa esgotada
    is_promoted = promoted is True
    is_early    = promoted is None

    # Checkpoint de fim de stage — independente de promoção
    stage_ckpt = MODEL_DIR / f"ckpt_s{stage_idx+1}_final.pth"
    policy.save(str(stage_ckpt))
    policy.save(str(MODEL_PATH))
    save_state(stage_idx + (1 if is_promoted else 0), ep_global)

    if verbose and not is_promoted:
        avg = sum(recent) / len(recent) if recent else 0
        if is_early:
            print(f"\n  → Early patience ({stage_ep} ep, avg={avg:.2f}). Avançando sem reset de buffer.")
        else:
            print(f"\n  → Patience esgotada ({stage_ep} ep, avg={avg:.2f}). Avançando.")

    return ep_global, is_promoted, renderer

# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(policy, ep_global_start, render=False, verbose=True, scale=1):
    name, map_path, n_agents, max_steps, stage_decay = FINE_TUNE_MAP
    env = build_env(map_path, n_agents, max_steps)
    renderer = build_renderer(env, f"Fine-tune: {name}", scale=scale) if render else None
    policy.reset_for_stage(stage_decay=stage_decay, epsilon_start=1.0)

    ep_global = ep_global_start
    recent = deque(maxlen=EVAL_WINDOW)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINE-TUNING: {name}  agents={n_agents}  decay={stage_decay:,}")
        print(f"{'='*60}")

    for ft_ep in range(1, PATIENCE + 1):
        m = run_episode(env, policy, renderer, train=True)
        recent.append(m["evacuation_rate"])
        ep_global += 1

        log_ep({
            "episode_global": ep_global, "stage": len(CURRICULUM) + 1, "map_name": name,
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

        if verbose and ft_ep % 10 == 0:
            avg = sum(recent) / len(recent)
            print(f"  [ft|ep{ft_ep:4d}|g{ep_global:5d}] "
                  f"evac={m['evacuation_rate']:.2f}  avg={avg:.2f}  "
                  f"eps={policy.current_epsilon():.3f}")

        if len(recent) >= EVAL_WINDOW and sum(recent) / len(recent) >= PROMOTION_THRESHOLD:
            if verbose:
                print(f"\n  ✓ Fine-tuning convergido!")
            break

    if renderer:
        renderer.close()
    policy.save(str(MODEL_PATH))
    return ep_global

# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate(model_path, n_episodes=20, render=False, scale=1, seed=None):
    # Coleta todos os resultados antes de imprimir a tabela — evita que os
    # prints de carregamento (DQN, Renderer) apareçam no meio da tabela
    if seed is not None:
        import random as _random
        _random.seed(seed)
        import numpy as _np
        _np.random.seed(seed)

    rows = []
    for name, map_path, n_agents, max_steps in EVAL_MAPS:
        env      = build_env(map_path, n_agents, max_steps)
        policy   = build_policy(env, model_path, DQNMode.EVAL)
        renderer = build_renderer(env, name, scale=scale) if render else None
        results  = [run_episode(env, policy, renderer, train=False)
                    for _ in range(n_episodes)]
        if renderer:
            renderer.close()

        N = max(1, n_episodes)
        rows.append((
            name,
            sum(r["evacuation_rate"]                   for r in results) / N,
            sum(r.get("panic_rate", 0)                 for r in results) / N,
            sum(r.get("mean_peak_emotion", 0)          for r in results) / N,
            sum(r.get("hazard_contact_rate", 0)        for r in results) / N,
            sum(r.get("exit_utilization", 0)           for r in results) / N,
            sum(r.get("emotion_variance", 0)           for r in results) / N,
            sum(r.get("mean_evacuation_time", max_steps) for r in results) / N,
        ))

    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO — {model_path}")
    print(f"{'='*60}")
    print(f"{'Mapa':<25} {'evac':>6} {'panic':>6} {'peak_em':>8} "
          f"{'haz_ct':>7} {'r_util':>7} {'var':>7} {'time':>6}")
    print("-" * 75)
    for name, evac, panic, pk_em, haz, rutil, var, time in rows:
        print(f"  {name:<23} {evac:>6.2f} {panic:>6.2f} {pk_em:>8.3f} "
              f"{haz:>7.2f} {rutil:>7.3f} {var:>7.4f} {time:>6.0f}")
    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",     type=int,  default=None,  help="Começa no stage N")
    p.add_argument("--eval",      action="store_true",       help="Avalia nos mapas eval")
    p.add_argument("--fine-tune", action="store_true",       help="Fine-tuning em di_style")
    p.add_argument("--render",    action="store_true")
    p.add_argument("--model",     type=str,  default=str(MODEL_PATH))
    p.add_argument("--episodes",  type=int,  default=20,    help="Episódios para --eval")
    p.add_argument("--scale",     type=int, default=1,
                   help="Fator de escala visual (ex: 2 = janela 2x maior)")
    p.add_argument("--seed",      type=int, default=None,
                   help="Seed para reproducibilidade na eval (não afeta treino)")
    p.add_argument("--quiet",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    init_log()

    if args.eval:
        evaluate(args.model, args.episodes, args.render, args.scale, args.seed)
        return

    saved_stage, ep_global = load_state()
    start = (args.stage - 1) if args.stage else saved_stage
    start = max(0, min(start, len(CURRICULUM) - 1))

    if args.stage and args.stage - 1 != saved_stage:
        ep_global = 0

    _, map_path, n_agents, max_steps, _ = CURRICULUM[start]
    env_init = build_env(map_path, n_agents, max_steps)
    policy   = build_policy(env_init, args.model, DQNMode.TRAIN)

    print(f"[INFO] device  = {policy.device}")
    print(f"[INFO] modelo  = {args.model}")
    print(f"[INFO] stage   = {start + 1}/{len(CURRICULUM)}  ep_global={ep_global}")

    if not args.fine_tune:
        prev_renderer = None
        prev_promoted = True  # True = não reseta buffer no próximo stage
        for stage_idx in range(start, len(CURRICULUM)):
            ep_global, is_promoted, prev_renderer = train_stage(
                stage_idx, policy, ep_global,
                render=args.render,
                verbose=not args.quiet,
                prev_renderer=prev_renderer,
                prev_promoted=prev_promoted,
                scale=args.scale,
            )
            # Early patience (is_promoted=False sem ter esgotado patience completa)
            # passa prev_promoted=True para NÃO resetar o buffer no stage seguinte.
            # Patience completa esgotada passa False para resetar.
            # A distinção é feita dentro de train_stage via is_early vs is_promoted.
            # Aqui simplificamos: o reset é controlado inteiramente por train_stage;
            # prev_promoted=is_promoted propaga apenas a decisão de save_state.
            prev_promoted = is_promoted

    if args.fine_tune:
        ep_global = fine_tune(
            policy, ep_global,
            render=args.render,
            verbose=not args.quiet,
            scale=args.scale,
        )

    print(f"\n[DONE] Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()