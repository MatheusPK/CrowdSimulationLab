"""
train_curriculum.py — treino DQN com currículo progressivo de 12 stages.

Estratégia: MULTI-AGENT com PARAMETER SHARING
  - Todos os agentes compartilham a mesma rede
  - Cada agente contribui transições independentes ao replay buffer
  - Captura dinâmicas coletivas: contágio emocional, densidade, congestionamento

Currículo v2 — baseado na análise de lacunas vs literatura (Yang 2020, Xu 2021, Zhang 2021):

  Stages 1-3:  sem hazard — navegação pura, aprender estrutura de mapas
  Stage  4:    primeiro hazard leve — ativar FSM pela primeira vez
  Stage  5:    hazard médio, multi-exit — aprender a escolher exit
  Stage  6:  ★ NOVO hazard_corridor_small — primeiro dilema rota-perigosa/rota-segura
               Lacuna crítica do currículo anterior: nenhum mapa ensinava este padrão
  Stages 7-8:  hazard em salas e corredores, layouts densos
  Stage  9:  ★ NOVO hazard_bypass_medium — dilema de rota em escala média
               H flanqueia exit principal, existe exit alternativo seguro
  Stage  10:   mall_medium — ambiente aberto, hazard leve distribuído
  Stage  11: ★ NOVO hazard_dense_office — alta densidade de hazard (8%)
               Ponte entre treino (máx 1.6%) e eval mall_panic (13.9%)
               Hazard distribuído em múltiplas salas (não bloco único)
  Stage  12:   library_hard — gargalo + hazard, prep para library_bottleneck eval

  Fine-tuning opcional (--fine-tune):
    di_style — mapa estilo DI com hazard em posição superior, prep para di_emergency

Promoção:
    evacuation_rate média ≥ 80% nos últimos 30 episódios → avança.
    Após PATIENCE episódios sem promoção → avança mesmo assim.

Uso:
    python train_curriculum.py                          # treino completo
    python train_curriculum.py --stage 6                # começa no stage 6
    python train_curriculum.py --eval                   # avalia nos 5 mapas eval
    python train_curriculum.py --fine-tune              # fine-tuning em di_style após treino
    python train_curriculum.py --render                 # com visualização
    python train_curriculum.py --quiet                  # sem prints detalhados
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
    DQN_EPSILON_START, DQN_EPSILON_END, DQN_EPSILON_DECAY, STAGE_EPSILON_DECAY,
    PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES, DQN_UPDATE_EVERY,
    CURRICULUM_PROMOTION_THRESHOLD, CURRICULUM_EVAL_WINDOW,
    CURRICULUM_PATIENCE, CURRICULUM_SAVE_EVERY,
    CURRICULUM_EARLY_PATIENCE_AFTER, CURRICULUM_EARLY_PATIENCE_THRESHOLD,
    CONTAGION_RADIUS, CONTAGION_RADIUS_HIGH_N,
    N_CONTAGION_THRESHOLD, HAZARD_CONTAGION_PCTG,
)

# ── Currículo v2 ──────────────────────────────────────────────────────────────

CURRICULUM = [
    # (nome, caminho, n_agents, max_steps)
    #
    # Progressão de N — v3 (corrigida após análise do colapso no stage 5):
    #
    # O colapso ocorreu porque o currículo original introduzia 3 variáveis
    # simultaneamente no stage 5 (N=4→10, mapa small→medium, hazard leve→médio).
    # Com 10 agentes, o contágio emocional entra em loop de pânico coletivo que
    # o modelo nunca viu com N=4. O stage 6 também falhou porque o modelo que
    # saiu do stage 5 danificado não conseguia evacuar nem com N=4.
    #
    # Correção: progressão gradual de N — máximo 1 variável relevante por stage.
    #   Stages 1-4  (N=4):  navegação e FSM básica
    #   Stage  5    (N=6):  primeiro contato com dinâmica coletiva  ← era N=10
    #   Stage  6    (N=4):  dilema de rota (foco no hazard, não no N)
    #   Stage  7    (N=8):  escala coletiva gradual                 ← era N=10
    #   Stage  8    (N=10): N=10 com base sólida de 8 agentes
    #   Stages 9-12 (N=12): contágio emocional consistente (Lv 2022: mín 12)
    #
    # Stages 1-3: navegação pura (sem hazard)
    ("mall_small",            "maps/train/mall_small.txt",             4,  300),
    ("school_small",          "maps/train/school_small.txt",           4,  300),
    ("office_wing_small",     "maps/train/office_wing_small.txt",      4,  300),
    # Stage 4: primeiro hazard leve
    ("library_small",         "maps/train/library_small.txt",          4,  300),
    # Stage 5: mapa medium + hazard leve — N=6 (não 10) para introduzir
    #          dinâmica coletiva gradualmente antes do contágio emocional escalar
    ("library_medium",        "maps/train/library_medium.txt",         6,  400),
    # Stage 6: ★ dilema rota-perigosa/rota-segura — N=4, foco no hazard
    ("hazard_corridor_small", "maps/train/hazard_corridor_small.txt",  4,  350),
    # Stage 7: N=8 — sobe N de forma controlada antes do 10
    ("school_floor",          "maps/train/school_floor.txt",           8,  400),
    # Stage 8: N=10 — agora com base sólida de dinâmica coletiva
    ("office_wing_medium",    "maps/train/office_wing_medium.txt",    10,  400),
    # Stage 9: ★ dilema de rota em escala média — sobe para N=12
    ("hazard_bypass_medium",  "maps/train/hazard_bypass_medium.txt",  12,  450),
    # Stage 10: shopping anel, hazard distribuído, N=12
    ("mall_medium",           "maps/train/mall_medium.txt",           12,  450),
    # Stage 11: ★ alta densidade de hazard — N=12, contágio emocional ativo
    ("hazard_dense_office",   "maps/train/hazard_dense_office.txt",   12,  450),
    # Stage 12: gargalo + hazard — N=12 (prep para library_bottleneck eval)
    ("library_hard",          "maps/train/library_hard.txt",          12,  450),
]

# Fine-tuning opcional após treino principal
FINE_TUNE_MAP = ("di_style", "maps/train/di_style.txt", 12, 500)

EVAL_MAPS = [
    # 12 agentes em todos os mapas eval — alinhado com literatura (Lv 2022, Xu 2021)
    # Todos os mapas têm ≥ 15 spawns disponíveis (verificado)
    ("library_bottleneck", "maps/eval/library_bottleneck.txt", 12, 450),
    ("office_single_exit", "maps/eval/office_single_exit.txt", 12, 450),
    ("mall_panic",         "maps/eval/mall_panic.txt",         12, 450),
    ("school_evacuation",  "maps/eval/school_evacuation.txt",  12, 450),
    ("di_emergency",       "maps/eval/di_emergency.txt",       12, 500),
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

def build_renderer(env, title, fps=30, enabled=True):
    r = Renderer(width=env.map_data.width, height=env.map_data.height,
                 title=title, fps=fps, draw_grid=True,
                 tile_size=env.map_data.tile_size)
    r.initialize()
    r.enabled = enabled  # preserva estado de pausa manual entre stages
    return r

# ── Loop de episódio ──────────────────────────────────────────────────────────

def run_episode(env, policy, renderer=None, train=True):
    """
    Multi-agent parameter sharing: todos os agentes usam a mesma policy
    e contribuem transições independentes para o replay buffer compartilhado.
    prev_obs é capturado ANTES de cada step (fix do bug de next_obs).
    """
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

        # update_peak_emotion já é chamado dentro de environment.step()
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
                stage_label=None, prev_renderer=None):
    name, map_path, n_agents, max_steps = CURRICULUM[stage_idx]
    label = stage_label or f"Stage {stage_idx+1}"
    env = build_env(map_path, n_agents, max_steps)

    # Preserva o estado de pausa (enabled) do renderer do stage anterior.
    # Sem isso, pressionar espaço para pausar a janela durante o treino não
    # persiste entre stages — o novo renderer sempre inicializa com enabled=True.
    render_enabled = prev_renderer.enabled if prev_renderer is not None else True
    renderer = build_renderer(env, f"{label}: {name}", enabled=render_enabled) if render else None

    # ── Reset de exploração para este stage ──────────────────────────────────
    # Cada stage começa com epsilon=1.0 e decai linearmente até 0.05 ao longo
    # de STAGE_EPSILON_DECAY[stage_idx] transições — calibrado para que eps
    # chegue a ~0.05 no episódio mediano de promoção deste stage.
    stage_decay = (
        STAGE_EPSILON_DECAY[stage_idx]
        if stage_idx < len(STAGE_EPSILON_DECAY)
        else DQN_EPSILON_DECAY
    )
    policy.reset_for_stage(stage_decay=stage_decay, epsilon_start=1.0)

    recent = deque(maxlen=EVAL_WINDOW)
    ep_global = ep_global_start
    stage_ep  = 0
    promoted  = False

    if verbose:
        is_new = name in ("hazard_corridor_small", "hazard_bypass_medium",
                          "hazard_dense_office")
        marker = " ★" if is_new else ""
        print(f"\n{'='*60}")
        print(f"{label}/12: {name}{marker}")
        print(f"  {map_path}")
        print(f"  agents={n_agents}  max_steps={max_steps}  "
              f"[multi-agent parameter sharing]")
        print(f"  epsilon_decay={stage_decay:,}  (steps_done resetado para 0)")
        if is_new:
            descriptions = {
                "hazard_corridor_small": "dilema rota-perigosa vs rota-segura",
                "hazard_bypass_medium":  "H flanqueando exit, rota alternativa segura",
                "hazard_dense_office":   "hazard distribuído (8%), prep para mall_panic",
            }
            print(f"  → {descriptions.get(name, '')}")
        print(f"{'='*60}")

    while stage_ep < PATIENCE:
        m = run_episode(env, policy, renderer, train=True)
        recent.append(m["evacuation_rate"])
        ep_global += 1
        stage_ep  += 1

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
                  f"emotion={m.get('mean_emotion_final',0):.2f}  "
                  f"var={m.get('emotion_variance',0):.4f}")

        if stage_ep % SAVE_EVERY == 0:
            policy.save(str(MODEL_DIR / f"ckpt_s{stage_idx+1}_ep{ep_global}.pth"))
            save_state(stage_idx, ep_global)

        if (len(recent) >= EVAL_WINDOW
                and sum(recent) / len(recent) >= PROMOTION_THRESHOLD):
            promoted = True
            if verbose:
                avg = sum(recent) / len(recent)
                print(f"\n  ✓ PROMOVIDO! avg{EVAL_WINDOW}={avg:.2f}")
            break

    if renderer:
        renderer.close()

    policy.save(str(MODEL_PATH))
    save_state(stage_idx + (1 if promoted else 0), ep_global)

    if verbose and not promoted:
        print(f"\n  → Patience esgotada ({PATIENCE} ep). Avançando.")

    return ep_global, promoted, renderer

# ── Fine-tuning (di_style) ────────────────────────────────────────────────────

def fine_tune(policy, ep_global_start, render=False, verbose=True):
    """
    Fine-tuning em di_style após o currículo principal.
    Prepara especificamente para di_emergency (eval):
    - Hazard na posição superior perto do exit
    - Layout estilo DI com salas satélite
    Roda por no máximo PATIENCE episódios sem critério de promoção.
    """
    name, map_path, n_agents, max_steps = FINE_TUNE_MAP
    env = build_env(map_path, n_agents, max_steps)
    renderer = build_renderer(env, f"Fine-tune: {name}") if render else None

    ep_global = ep_global_start
    recent = deque(maxlen=EVAL_WINDOW)

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINE-TUNING: {name}")
        print(f"  Prep para di_emergency (eval) — hazard perto do exit superior")
        print(f"{'='*60}")

    for ft_ep in range(1, PATIENCE + 1):
        m = run_episode(env, policy, renderer, train=True)
        recent.append(m["evacuation_rate"])
        ep_global += 1

        log_ep({
            "episode_global": ep_global, "stage": 13, "map_name": name,
            "evacuation_rate": round(m["evacuation_rate"], 4),
            "all_evacuated": int(m.get("all_evacuated", False)),
            "mean_evac_time": round(m.get("mean_evacuation_time", max_steps), 2),
            "steps": m.get("steps", max_steps),
            "mean_emotion_final": round(m.get("mean_emotion_final", 0), 4),
            "emotion_variance": round(m.get("emotion_variance", 0), 4),
            "mean_speed_ratio": round(m.get("mean_speed_ratio", 1), 4),
            "total_reward": round(m["total_reward"], 2),
            "epsilon": round(policy.current_epsilon(), 4),
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

def evaluate(model_path, n_episodes=20, render=False):
    """
    Avaliação completa nos 5 mapas eval.
    Métricas reportadas:
      - evacuation_rate: fração de agentes evacuados
      - mean_emotion_final: nível emocional médio ao final
      - emotion_variance: variância emocional entre agentes (fenômeno central)
      - mean_evac_time: tempo médio de evacuação
    """
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO — {model_path}")
    print(f"{'='*60}")
    print(f"{'Mapa':<25} {'evac':>6} {'panic':>6} {'peak_em':>8} {'haz_ct':>7} {'r_util':>7} {'var':>7} {'time':>6}")
    print("-"*75)

    for name, map_path, n_agents, max_steps in EVAL_MAPS:
        env = build_env(map_path, n_agents, max_steps)
        policy = build_policy(env, model_path, DQNMode.EVAL)
        renderer = build_renderer(env, name) if render else None
        results = [run_episode(env, policy, renderer, train=False)
                   for _ in range(n_episodes)]
        if renderer:
            renderer.close()

        N = max(1, n_episodes)
        avg_evac   = sum(r["evacuation_rate"] for r in results) / N
        avg_panic  = sum(r.get("panic_rate", 0) for r in results) / N
        avg_pk_em  = sum(r.get("mean_peak_emotion", 0) for r in results) / N
        avg_haz    = sum(r.get("hazard_contact_rate", 0) for r in results) / N
        avg_rutil  = sum(r.get("exit_utilization", 0) for r in results) / N
        avg_var    = sum(r.get("emotion_variance", 0) for r in results) / N
        avg_time   = sum(r.get("mean_evacuation_time", max_steps) for r in results) / N
        print(f"  {name:<23} {avg_evac:>6.2f} {avg_panic:>6.2f} {avg_pk_em:>8.3f} "              f"{avg_haz:>7.2f} {avg_rutil:>7.3f} {avg_var:>7.4f} {avg_time:>6.0f}")

    print()

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Treino DQN+FSM com currículo progressivo v2"
    )
    p.add_argument("--stage",      type=int,  default=None,
                   help="Começa no stage N (1-12)")
    p.add_argument("--eval",       action="store_true",
                   help="Avalia modelo nos 5 mapas eval")
    p.add_argument("--fine-tune",  action="store_true",
                   help="Fine-tuning em di_style após currículo principal")
    p.add_argument("--render",     action="store_true")
    p.add_argument("--model",      type=str,  default=str(MODEL_PATH))
    p.add_argument("--episodes",   type=int,  default=20,
                   help="Episódios para --eval")
    p.add_argument("--quiet",      action="store_true")
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

    print(f"[INFO] device       = {policy.device}")
    print(f"[INFO] modelo       = {args.model}")
    print(f"[INFO] stage inicial = {start+1}  ep_global={ep_global}")
    print(f"[INFO] estratégia   = multi-agent parameter sharing")
    print(f"[INFO] epsilon_decay = {policy.epsilon_decay:,} transições")

    # Treino principal
    if not args.fine_tune:
        prev_renderer = None
        for stage_idx in range(start, len(CURRICULUM)):
            ep_global, _, prev_renderer = train_stage(
                stage_idx, policy, ep_global,
                render=args.render,
                verbose=not args.quiet,
                prev_renderer=prev_renderer,
            )

    # Fine-tuning em di_style (opcional, após treino principal ou standalone)
    if args.fine_tune:
        ep_global = fine_tune(
            policy, ep_global,
            render=args.render,
            verbose=not args.quiet,
        )

    print(f"\n[DONE] Treino completo. Modelo salvo em: {MODEL_PATH}")

if __name__ == "__main__":
    main()