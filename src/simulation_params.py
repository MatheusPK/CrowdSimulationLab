"""
simulation_params.py — fonte única de todos os parâmetros de simulação.

Organize em grupos lógicos:
  1. MUNDO          — tile size, dimensões físicas
  2. AGENTE         — geometria, velocidade, avoidance base
  3. FSM / EMOÇÃO   — thresholds, deltas emocionais, parâmetros por estado
  4. PERCEPÇÃO      — raios de visão e detecção
  5. A*             — custo de hazard no planner
  6. DQN            — arquitetura e hiperparâmetros de treino
  7. RECOMPENSA     — pesos de cada componente do reward
  8. CURRÍCULO      — thresholds de promoção entre stages

Para calibrar o treino, edite os valores aqui.
environment.py, agent.py e train_curriculum.py importam deste arquivo.
"""

# ══════════════════════════════════════════════════════════════════════
# 1. MUNDO
# ══════════════════════════════════════════════════════════════════════

TILE_SIZE = 8  # px por tile — altera escala de todo o mapa; não mudar sem recriar mapas


# ══════════════════════════════════════════════════════════════════════
# 2. AGENTE — física e movimento
# ══════════════════════════════════════════════════════════════════════

# Geometria
AGENT_RADIUS          = 6.0   # px — raio físico para colisão e detecção de exit
PLANNER_RADIUS_MARGIN = 5.0   # px — margem extra para o A* evitar paredes
                               # planner_radius = AGENT_RADIUS + PLANNER_RADIUS_MARGIN = 11px
                               # Reduzir permite passar por corredores mais estreitos,
                               # mas aumenta risco de colisão durante navegação.
                               # Mínimo recomendado: 2.0 (raio 8px = 1 tile)

# Velocidade base (px/step × dt)
# Velocidade real = base_speed × dt = 50 × 0.1 = 5px/step
AGENT_BASE_SPEED      = 50.0  # px/s — velocidade padrão no estado CALM
                               # Aumentar → evacuação mais rápida (menos steps)
                               # Diminuir → mais tempo para desenvolver emoção

# Avoidance de obstáculos (paredes e bordas)
OBSTACLE_AVOIDANCE_DISTANCE  = 14.0  # px — distância em que a repulsão começa
OBSTACLE_AVOIDANCE_STRENGTH  = 70.0  # magnitude da força repulsiva [0..∞]

# Avoidance de outros agentes
AGENT_AVOIDANCE_DISTANCE     = 12.0  # px — distância de início da repulsão
AGENT_AVOIDANCE_STRENGTH     = 90.0  # magnitude [0..∞]
                                      # Valores altos → agentes muito dispersos
                                      # Valores baixos → aglomeração, mais contágio

# Suavização de velocidade (filtro exponencial)
VELOCITY_SMOOTHING    = 0.18  # [0..1] — quanto da velocidade target é aplicada por step
                               # 0.0 = velocidade nunca muda (inerte)
                               # 1.0 = velocidade muda instantaneamente (sem inércia)
                               # Valores altos → movimento brusco; baixos → movimento fluido


# ══════════════════════════════════════════════════════════════════════
# 3. FSM / EMOÇÃO — modelo emocional
# ══════════════════════════════════════════════════════════════════════

# Deltas de emoção por step (emotion_level ∈ [0, 1])
EMOTION_DECAY            = -0.02  # decaimento passivo por step (tende à calma)
                                   # Tornar mais negativo → calma mais rápida
EMOTION_DELTA_HAZARD_CONTACT  = +0.15  # agente tocando o hazard
EMOTION_DELTA_HAZARD_VISIBLE  = +0.08  # hazard visível mas sem contato
EMOTION_DELTA_CONTAGION       = +0.04  # multiplicador do contágio emocional
                                        # delta_contagion += coeff × avg_emotion_vizinhos
                                        # Aumentar → pânico se propaga mais rápido

# Raio de contágio entre agentes (px)
CONTAGION_RADIUS         = 35.0   # px — vizinhos dentro deste raio influenciam a emoção
                                   # ~4 tiles; aumentar → mais contágio a distância

# Thresholds da FSM com histerese (evita oscilação nos limiares)
# Subida sempre maior que descida para o mesmo limiar
FSM_CALM_TO_EVACUATE    = 0.35   # emotion ≥ este valor → CALM → EVACUATE
FSM_EVACUATE_TO_CALM    = 0.25   # emotion < este valor → EVACUATE → CALM
FSM_EVACUATE_TO_PANIC   = 0.72   # emotion ≥ este valor → EVACUATE → PANIC
FSM_PANIC_TO_EVACUATE   = 0.62   # emotion < este valor → PANIC → EVACUATE

# Parâmetros comportamentais por estado FSM
# Velocidade como múltiplo de AGENT_BASE_SPEED
FSM_SPEED_CALM          = 1.00   # velocidade normal
FSM_SPEED_EVACUATE      = 1.15   # 15% mais rápido — começa a correr
FSM_SPEED_PANIC         = 1.30   # 30% mais rápido — corrida descoordenada

# Speed smoothing (quanto da velocidade target é aplicada por step no update_fsm)
FSM_SPEED_SMOOTHING     = 0.20   # [0..1]

# Avoidance no estado CALM
FSM_CALM_OBS_DIST       = 16.0
FSM_CALM_OBS_STR        = 80.0
FSM_CALM_AGENT_DIST     = 14.0
FSM_CALM_AGENT_STR      = 95.0
FSM_CALM_VEL_SMOOTH     = 0.20

# Avoidance no estado EVACUATE
FSM_EVAC_OBS_DIST       = 12.0
FSM_EVAC_OBS_STR        = 70.0
FSM_EVAC_AGENT_DIST     = 10.0
FSM_EVAC_AGENT_STR      = 85.0
FSM_EVAC_VEL_SMOOTH     = 0.16

# Avoidance no estado PANIC
# Distâncias menores → agente ignora mais o espaço pessoal → aglomeração realista
# Strengths menores → empurra menos → pisoteamento simulado
FSM_PANIC_OBS_DIST      = 8.0
FSM_PANIC_OBS_STR       = 55.0
FSM_PANIC_AGENT_DIST    = 7.0
FSM_PANIC_AGENT_STR     = 70.0
FSM_PANIC_VEL_SMOOTH    = 0.10


# ══════════════════════════════════════════════════════════════════════
# 4. PERCEPÇÃO — raios de visão e detecção
# ══════════════════════════════════════════════════════════════════════

# Raio de visão do hazard (px)
# Representa até onde um humano consegue ver fumaça / sentir calor
# Usado em: update_emotion (+delta), compute_reward (shaping), obs[6] e obs[7]
#
# Recomendações por contexto:
#   40px (5 tiles) — padrão atual, "vê o hazard ao entrar na sala"
#   64px (8 tiles) — "vê de qualquer ponto da sala pequena"
#   80px (10 tiles) — "vê de salas adjacentes"
#
# NÃO igualar ao PLANNER_RADIUS (11px) — eliminaria o gradiente de antecipação,
# que é o mecanismo central que diferencia CALM de EVACUATE.
HAZARD_VISION_RADIUS    = 5 * TILE_SIZE   # 40px = 5 tiles

# Raio de densidade local (px)
# Agentes dentro deste raio contam para a densidade e para o contágio emocional
LOCAL_DENSITY_RADIUS    = 35.0   # px (~4 tiles)


# ══════════════════════════════════════════════════════════════════════
# 5. A* / PLANNER
# ══════════════════════════════════════════════════════════════════════

# Custo extra por tile de hazard no cálculo do caminho A*
# Valores maiores → A* evita mais o hazard (rota mais longa mas mais segura)
# 0.0 → A* ignora hazards completamente
# 8.0 → equivale a ~8 tiles extras de custo (rota alternativa até 8 tiles mais longa)
# 20+ → A* nunca atravessa hazard se houver outra saída
ASTAR_HAZARD_COST       = 25.0

# Número máximo de steps sem mudar de célula antes de replanejar (anti-stuck)
ASTAR_STUCK_THRESHOLD   = 5   # steps


# ══════════════════════════════════════════════════════════════════════
# 6. DQN — arquitetura e hiperparâmetros de treino
# ══════════════════════════════════════════════════════════════════════

# Arquitetura da rede
DQN_HIDDEN_DIM          = 128   # neurônios por camada oculta (2 camadas)
                                  # Aumentar para mapas mais complexos (256)
                                  # Diminuir para treino mais rápido (64)

# Replay buffer
DQN_BUFFER_CAPACITY     = 100_000 # transições armazenadas
                                    # Com N=12 agentes × 450 steps ≈ 3510 trans/ep.
                                    # 100k ≈ 28 episódios de histórico — diversidade adequada.
                                    # Aumentar para 200k se houver RAM disponível.

DQN_TRAIN_START_SIZE    = 2_000   # buffer mínimo antes de começar os updates
                                   # Com multi-agent, atinge esse valor em < 1 episódio.

# Mini-batch
DQN_BATCH_SIZE          = 64      # amostras por update de gradient

# Desconto temporal
DQN_GAMMA               = 0.99    # [0..1] — peso de recompensas futuras
                                   # 0.99 → horizonte longo (~100 steps)
                                   # 0.95 → horizonte mais curto (~20 steps)

# Learning rate do Adam
DQN_LR                  = 1e-3    # Reduzir (1e-4) se o treino for instável
                                   # Aumentar (3e-3) se convergir muito lento

# Target network — frequência de sincronização
DQN_TARGET_UPDATE_FREQ  = 300     # updates de gradient entre cada sync da target net
                                   # Valores menores → target muda mais rápido (instável)
                                   # Valores maiores → target muda mais devagar (conservador)

# Epsilon-greedy — decay LINEAR calibrado por fase do currículo
#
# Estratégia:
#     eps = eps_start + (steps_done / stage_decay) * (eps_end - eps_start)
#
# steps_done é RESETADO a 0 no início de cada stage (train_curriculum.py),
# e stage_decay vem de STAGE_EPSILON_DECAY[stage_idx].
#
# Isso garante que cada stage começa com exploração alta (eps_start) e decai
# até eps_end ao longo das transições esperadas para aquele stage — em vez de
# usar um único decay global que chegaria a eps=0.05 somente no ep ~2051 para
# stages com N=4 (780 trans/ep), causando catastrophic forgetting observado
# no stage 1 (colapso ep 450→500 com eps=0.67).
#
# Por que linear em vez de exponencial (Zhang et al. 2021):
#   O exponencial concentra exploração nos primeiros episódios e chega ao mínimo
#   mais rápido — vantagem em ambientes estacionários. No currículo, cada stage
#   é um ambiente novo; o agente precisa de exploração sustentada durante toda a
#   fase de aprendizado do stage, não apenas no início. O linear distribui a
#   exploração proporcionalmente, mantendo eps > 0.15 nos stages de dilema de
#   rota (6, 9, 11) onde a rota alternativa precisa ser descoberta por tentativa.
DQN_EPSILON_START       = 1.0     # exploração inicial por stage (100% aleatório)
DQN_EPSILON_END         = 0.05    # exploração mínima (5% aleatório)
DQN_EPSILON_DECAY       = 1_600_000  # fallback global — usado se o stage não estiver
                                     # coberto por STAGE_EPSILON_DECAY

# Decay local por stage (índice 0-based, stage 1 = índice 0).
#
# Calculado como: N_agentes × avg_steps_por_ep × ep_mediana_promoção
#   avg_steps = max_steps × 0.65  (fração média de steps antes da evacuação)
#
# Alvo: eps chega a 0.05 aproximadamente no ep mediano de promoção.
# stages de dilema (6, 9, 11) usam valores maiores para manter exploração
# suficiente para descobrir rotas alternativas ao hazard.
# NOTA: valores recalibrados para DQN_UPDATE_EVERY=4.
# Com UPDATE_EVERY=4, steps_done cresce N/4 por env step (não N).
# Dividir por 4 garante que epsilon chega a 0.05 no mesmo episódio mediano —
# o perfil de exploração por episódio é idêntico ao da versão anterior.
# Fórmula: decay = N × avg_steps × ep_mediano / UPDATE_EVERY
STAGE_EPSILON_DECAY = [
     17_500,   # stage 1  mall_small             N=4,  ep_med ~58
     17_500,   # stage 2  school_small            N=4,  ep_med ~58
     19_500,   # stage 3  office_wing_small       N=4,  ep_med ~65
     23_500,   # stage 4  library_small           N=4,  ep_med ~78
     58_500,   # stage 5  library_medium          N=6,  ep_med ~98
     41_000,   # stage 6  hazard_corridor_small   N=4,  ep_med ~117
    156_000,   # stage 7  school_floor            N=8,  ep_med ~195
    195_000,   # stage 8  office_wing_medium      N=10, ep_med ~195
    # ── 4 stages-ponte (v4) ──────────────────────────────────────────────
    214_500,   # stage 9a bridge_open_medium      N=11, ep_med ~195
    214_500,   # stage 9b bridge_corridor_medium  N=11, ep_med ~195
    234_000,   # stage 9c bridge_hazard_intro     N=12, ep_med ~186
    234_000,   # stage 9d bridge_multi_exit       N=12, ep_med ~186
    # ── stages originais (agora 13-16) ───────────────────────────────────
    307_125,   # stage 13 hazard_bypass_medium    N=12, ep_med ~228  ★ dilema
    263_250,   # stage 14 mall_medium             N=12, ep_med ~195
    307_125,   # stage 15 hazard_dense_office     N=12, ep_med ~228  ★ dilema
    263_250,   # stage 16 library_hard            N=12, ep_med ~195
]

# Gradient clipping (estabiliza treino com reward shaping)
DQN_GRAD_CLIP_NORM      = 10.0    # max_norm para clip_grad_norm_

# Prioritized Experience Replay (PER) — Schaul et al. 2015
#
# PER resolve o problema de sinal esparso identificado nos logs de treino:
# transições de evacuação bem-sucedida (+80) eram minoria no buffer de 100k
# e raramente apareciam nos batches de 64. Com PER, transições com TD-error
# alto (evacuações, primeiros contatos com hazard) são amostradas com muito
# mais frequência — exatamente o que o agente mais precisa aprender.
#
# PER_ALPHA:       grau de priorização [0=uniforme, 1=totalmente prioritizado]
#                  0.6 é o valor padrão da literatura (Schaul et al. 2015)
#                  Reduzir → mais uniforme; aumentar → mais agressivo
PER_ALPHA            = 0.6

# PER_BETA_START:  correção de importância inicial (IS weights)
#                  IS weights corrigem o viés introduzido pela amostragem
#                  não-uniforme. Beta cresce linearmente de START até 1.0.
#                  0.4 é o valor padrão da literatura.
PER_BETA_START       = 0.4

# PER_BETA_FRAMES: transições para beta chegar a 1.0
#                  Calibrado para o currículo completo (~3M transições).
#                  Beta=1.0 ao final garante correção total do viés.
PER_BETA_FRAMES      = 3_000_000


# ══════════════════════════════════════════════════════════════════════
# 7. RECOMPENSA — pesos de cada componente
# ══════════════════════════════════════════════════════════════════════
#
# Estrutura em 3 camadas:
#
#   Camada 1 — Navegação (sinal principal)
#     R += REWARD_PROGRESS_SCALE × (prev_dist - new_dist) / diagonal_mapa
#     R += REWARD_EVACUATED       (terminal, se evacuou)
#     R += REWARD_TIME_PENALTY    (por step — urgência)
#     R += REWARD_NO_PROGRESS     (se não avançou — evita ficar parado)
#
#   Camada 2 — Hazard e emoção
#     R += REWARD_HAZARD_CONTACT  (por step dentro do hazard)
#     R += REWARD_HAZARD_VISIBLE_CALM × (1 - emotion)  (bônus por calma)
#     R += REWARD_HAZARD_PANIC    (penalidade por pânico perto do hazard)
#
#   Camada 3 — Interação social
#     R += REWARD_COLLISION       (por colisão com parede ou agente)
#     R += REWARD_DENSITY_SCALE × densidade_norm  (penalidade por aglomeração)

# Camada 1 — navegação
REWARD_PROGRESS_SCALE   = 15.0    # peso do progresso BFS em direção à saída
                                   # Aumentado de 10 → 15: com REWARD_NO_PROGRESS agora
                                   # só em retrocesso real, o sinal de progresso precisa
                                   # ser mais forte para superar o time_penalty de -0.05/step
REWARD_EVACUATED        = 80.0    # recompensa terminal por evacuação bem-sucedida
REWARD_TIME_PENALTY     = -0.05   # por step (urgência suave, não paralisa)
REWARD_NO_PROGRESS      = -1.0    # se progress <= 0 e não evacuou

# Camada 2 — hazard e emoção
REWARD_HAZARD_CONTACT   = -3.0    # por step dentro do hazard (forte e contínuo)
REWARD_HAZARD_VISIBLE_CALM = 0.4  # bônus × (1 - emotion) ao ver hazard sem entrar
REWARD_HAZARD_PANIC     = -0.5    # penalidade por emotion > 0.5 perto do hazard

# Camada 3 — interação social
REWARD_COLLISION        = -0.3    # por colisão
REWARD_DENSITY_SCALE    = -0.1    # × densidade_norm (penaliza aglomeração extrema)


# ══════════════════════════════════════════════════════════════════════
# 8. CURRÍCULO — promoção entre stages do train_curriculum.py
# ══════════════════════════════════════════════════════════════════════

CURRICULUM_PROMOTION_THRESHOLD = 0.80   # evacuation_rate média mínima para promover
CURRICULUM_EVAL_WINDOW         = 30     # episódios na janela de avaliação
CURRICULUM_PATIENCE            = 500    # máx de episódios por stage antes de avançar mesmo sem promoção
                                         # aumentado de 300 → 500: ambiente mais complexo que literatura
                                         # (FSM + contágio + hazard + multi-exit juntos) precisa de mais
                                         # experiência. Xu et al. 2021 usaram ~30k ep num ambiente simples.
CURRICULUM_SAVE_EVERY          = 50     # salva checkpoint a cada N episódios

# Early patience: aborta stage rapidamente se aprendizado claramente parado.
# Evita acumular centenas de episódios de transições ruins no buffer
# (causa raiz do catastrophic forgetting S9→S10-S12 diagnosticado nos logs).
#
# Condição: se avg30 < THRESHOLD após AFTER episódios → avança + reseta buffer.
CURRICULUM_EARLY_PATIENCE_AFTER     = 200   # episódios mínimos antes de verificar
CURRICULUM_EARLY_PATIENCE_THRESHOLD = 0.15  # avg30 mínimo após os N episódios acima

# Raio de contágio reduzido para stages com N≥12 e hazard >5%.
# CONTAGION_RADIUS=35px com N=12 cria cascata emocional irrecuperável
# (diagnosticado no S9: avg chegou a 0.50 ep230 mas não consolidou).
# Com 20px o contágio fica restrito a vizinhos imediatos (~2 tiles).
CONTAGION_RADIUS_HIGH_N  = 20.0   # px — ativado quando N e hazard% atingem threshold
N_CONTAGION_THRESHOLD    = 12     # N mínimo para ativar raio reduzido
HAZARD_CONTAGION_PCTG    = 0.05   # fração de tiles H/total para ativar raio reduzido

# ══════════════════════════════════════════════════════════════════════
# 9. PERFORMANCE — frequência de update do DQN
# ══════════════════════════════════════════════════════════════════════

# UPDATE_EVERY: faz 1 gradient step a cada N transições armazenadas,
# em vez de 1 por agente por step (comportamento anterior = 12x/step).
#
# Problema diagnosticado:
#   Com N=12 agentes, store_transition() chamava _update() 12x por env step.
#   Isso gerava ~5.400 gradient steps/ep (12 × 450 steps), dominando o tempo
#   de treino mesmo sem render. O render representava apenas ~7-20% do total.
#
# UPDATE_EVERY=4 significa: acumula 4 transições, então faz 1 update.
# Equivale a fazer ~1350 updates/ep (5400/4) — 4x menos que antes,
# mas ainda ~3x mais que o padrão DQN single-agent (450/ep).
# Mantém amostragem rica do PER sem saturar o CPU.
#
# Impacto no tempo (estimado):
#   Antes:  4.5–12.6s/ep  (domina tudo, render irrelevante)
#   Depois: ~1.5–4.0s/ep  (redução ~3-4x)
#   render: ~0.9s/ep      (agora sim representa ~20-60% — faz diferença desligar)
#
# Qualidade de aprendizado:
#   UPDATE_EVERY=1: 1 update por env step (padrão DeepMind single-agent)
#   UPDATE_EVERY=4: balança velocidade e amostragem — recomendado para N=12
#   UPDATE_EVERY=12: 1 update por "rodada" de todos os agentes (mínimo aceitável)
DQN_UPDATE_EVERY = 4