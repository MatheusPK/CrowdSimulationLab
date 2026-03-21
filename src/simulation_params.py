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

# Epsilon-greedy (exploração)
DQN_EPSILON_START       = 1.0     # exploração inicial (100% aleatório)
DQN_EPSILON_END         = 0.05    # exploração mínima (5% aleatório)
DQN_EPSILON_DECAY       = 1_600_000 # steps (transições) para decair de start até end
                                     #
                                     # Calibrado para currículo de 12 stages com N=4→12 agentes:
                                     #   Trans/ep (N=12, 450 steps): ~3510
                                     #   Stages 1-4  (N=4,  300 steps):  ~780 trans/ep × 150ep ≈  468k
                                     #   Stages 5-8  (N=10, 400 steps): ~2600 trans/ep × 150ep ≈ 1560k
                                     #   Epsilon = 0.05 atingido ~no stage 10 (mall_medium)
                                     #
                                     # Isso garante ε≈0.15 no stage 9 (hazard_bypass_medium),
                                     # onde o dilema de rota com 12 agentes precisa de exploração
                                     # para descobrir a rota alternativa ao hazard.
                                     # Com valor antigo (200k), epsilon virava 0.05 no stage 2.

# Gradient clipping (estabiliza treino com reward shaping)
DQN_GRAD_CLIP_NORM      = 10.0    # max_norm para clip_grad_norm_


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
REWARD_NO_PROGRESS      = -1.0    # se progress < -1e-4 (retrocesso real) e não evacuou
                                   # NÃO é aplicado em estagnação (progress == 0):
                                   # com VELOCITY_SMOOTHING=0.18 o agente leva ~4 steps
                                   # para mudar de célula BFS — penalizar estagnação
                                   # mascarava o sinal de progresso em ~94% dos steps.

# Camada 2 — hazard e emoção
REWARD_HAZARD_CONTACT   = -3.0    # por step dentro do hazard (forte e contínuo)
REWARD_HAZARD_VISIBLE_CALM = 0.4  # bônus × (1 - emotion) ao ver hazard sem entrar
                                   # ATENÇÃO: incentiva calma perto do hazard, não
                                   # aproximação. Monitorar hazard_contact_rate nos
                                   # stages 4-5; se subir, reduzir para 0.2 ou zerar.
REWARD_HAZARD_PANIC     = -0.5    # penalidade por emotion > 0.5 perto do hazard

# Camada 3 — interação social
REWARD_COLLISION        = -0.3    # por colisão
REWARD_DENSITY_SCALE    = -0.1    # × densidade_norm (penaliza aglomeração extrema)


# ══════════════════════════════════════════════════════════════════════
# 8. CURRÍCULO — promoção entre stages do train_curriculum.py
# ══════════════════════════════════════════════════════════════════════

CURRICULUM_PROMOTION_THRESHOLD = 0.80   # evacuation_rate média mínima para promover
CURRICULUM_EVAL_WINDOW         = 30     # episódios na janela de avaliação
CURRICULUM_PATIENCE            = 500    # episódios máximos por stage
CURRICULUM_SAVE_EVERY          = 50     # salva checkpoint a cada N episódios