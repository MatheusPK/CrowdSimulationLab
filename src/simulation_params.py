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
STAGE_EPSILON_DECAY = [
     70_000,   # stage 1  mall_small            N=4,  ~780 t/ep,  ep_med ~90
     70_000,   # stage 2  school_small           N=4,  ~780 t/ep,  ep_med ~90
     78_000,   # stage 3  office_wing_small      N=4,  ~780 t/ep,  ep_med ~100
     94_000,   # stage 4  library_small          N=4,  ~780 t/ep,  ep_med ~120
    234_000,   # stage 5  library_medium         N=6,  ~1560 t/ep, ep_med ~150
    164_000,   # stage 6  hazard_corridor_small  N=4,  ~910 t/ep,  ep_med ~180
    624_000,   # stage 7  school_floor           N=8,  ~2080 t/ep, ep_med ~300 ← corrigido
    780_000,   # stage 8  office_wing_medium     N=10, ~2600 t/ep, ep_med ~300 ← corrigido
  1_228_500,   # stage 9  hazard_bypass_medium   N=12, ~3510 t/ep, ep_med ~350 ← dilema crítico
  1_053_000,   # stage 10 mall_medium            N=12, ~3510 t/ep, ep_med ~300 ← corrigido
  1_228_500,   # stage 11 hazard_dense_office    N=12, ~3510 t/ep, ep_med ~350 ← dilema
  1_053_000,   # stage 12 library_hard           N=12, ~3510 t/ep, ep_med ~300 ← corrigido
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