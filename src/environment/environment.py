import math

from core.direction import Direction
from core.fsm_state import FSMState
from entities.agent import Agent

# Raio (em pixels) dentro do qual um hazard é considerado "visível" pelo agente
HAZARD_VISION_RADIUS = 5 * 8  # 5 tiles × tile_size

class Environment:
    def __init__(self, map_data, dt=0.1, max_steps=300, use_fsm=False, num_agents=1):
        self.map_data = map_data
        self.dt = dt
        self.max_steps = max_steps
        self.use_fsm = use_fsm
        self.num_agents_requested = num_agents

        self.agents = []
        self.time = 0

        s = math.sqrt(2) / 2
        self.actions = {
            0: (Direction.N,   0.0, -1.0),
            1: (Direction.NE,  s,   -s),
            2: (Direction.E,   1.0,  0.0),
            3: (Direction.SE,  s,    s),
            4: (Direction.S,   0.0,  1.0),
            5: (Direction.SW, -s,    s),
            6: (Direction.W,  -1.0,  0.0),
            7: (Direction.NW, -s,   -s),
        }

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self):
        self.time = 0
        self.agents = []

        available_spawns = list(self.map_data.spawns)
        num_to_create = min(self.num_agents_requested, len(available_spawns))

        if num_to_create <= 0:
            raise ValueError("O mapa não possui nenhum spawn ('S').")

        for row, col in available_spawns[:num_to_create]:
            spawn_x, spawn_y = self.cell_center_to_world(row, col)
            agent = Agent(spawn_x, spawn_y)
            self.agents.append(agent)

        return [self.get_observation(agent) for agent in self.agents]

    def step(self, actions):
        self.time += 1

        if len(actions) != len(self.agents):
            raise ValueError("Número de ações diferente do número de agentes.")

        prev_distances = [
            self.distance_to_nearest_exit(agent) if not agent.evacuated else 0.0
            for agent in self.agents
        ]

        collided_list = [False for _ in self.agents]

        for i, agent in enumerate(self.agents):
            if agent.evacuated:
                continue
            action = actions[i]
            if action is None:
                continue
            collided_list[i] = self.apply_action(agent, action)

        for agent in self.agents:
            if agent.evacuated:
                continue
            self.update_emotion(agent)
            if self.use_fsm:
                self.update_fsm(agent)

        new_distances = [
            self.distance_to_nearest_exit(agent) if not agent.evacuated else 0.0
            for agent in self.agents
        ]

        reward_list = []
        for i, agent in enumerate(self.agents):
            if agent.evacuated and prev_distances[i] == 0.0 and new_distances[i] == 0.0:
                reward_list.append(0.0)
                continue

            reward = self.compute_reward(
                agent=agent,
                prev_dist=prev_distances[i],
                new_dist=new_distances[i],
                collided=collided_list[i],
            )
            reward_list.append(reward)

        done = self.is_done()
        obs_list = [self.get_observation(agent) for agent in self.agents]

        info = {
            "collided": collided_list,
            "evacuated_count": sum(1 for a in self.agents if a.evacuated),
            "active_count": sum(1 for a in self.agents if not a.evacuated),
        }

        return obs_list, reward_list, done, info

    # ------------------------------------------------------------------
    # Física
    # ------------------------------------------------------------------

    def apply_action(self, agent, action):
        direction, dir_x, dir_y = self.actions[action]
        agent.direction = direction
        agent.last_action = action

        desired_vx = dir_x * agent.current_speed
        desired_vy = dir_y * agent.current_speed

        obstacle_avoid_x, obstacle_avoid_y = self.compute_obstacle_avoidance(agent)
        agent_avoid_x, agent_avoid_y = self.compute_agent_avoidance(agent)

        target_vx = desired_vx + obstacle_avoid_x + agent_avoid_x
        target_vy = desired_vy + obstacle_avoid_y + agent_avoid_y

        speed = math.hypot(target_vx, target_vy)
        if speed > agent.current_speed and speed > 0.0:
            scale = agent.current_speed / speed
            target_vx *= scale
            target_vy *= scale

        agent.vx += (target_vx - agent.vx) * agent.velocity_smoothing
        agent.vy += (target_vy - agent.vy) * agent.velocity_smoothing

        total_dx = agent.vx * self.dt
        total_dy = agent.vy * self.dt

        total_dist = math.hypot(total_dx, total_dy)
        max_substep_distance = max(1.0, agent.radius * 0.35)
        substeps = max(1, math.ceil(total_dist / max_substep_distance))

        step_dx = total_dx / substeps
        step_dy = total_dy / substeps

        collided = False
        moved_any = False

        for _ in range(substeps):
            candidate_x = agent.x + step_dx
            candidate_y = agent.y + step_dy

            if not self.check_collision(candidate_x, candidate_y, agent.radius, ignore_agent=agent):
                agent.apply_position(candidate_x, candidate_y)
                moved_any = True
                continue

            collided = True

            moved_x = False
            moved_y = False

            if not self.check_collision(candidate_x, agent.y, agent.radius, ignore_agent=agent):
                agent.apply_position(candidate_x, agent.y)
                moved_any = True
                moved_x = True
            elif not self.check_collision(agent.x, candidate_y, agent.radius, ignore_agent=agent):
                agent.apply_position(agent.x, candidate_y)
                moved_any = True
                moved_y = True

            if not moved_x and not moved_y:
                break

        if not moved_any:
            agent.stop()

        if self.touches_exit(agent):
            agent.evacuated = True
            agent.stop()

        return collided

    def compute_obstacle_avoidance(self, agent):
        fx, fy = 0.0, 0.0
        for obstacle in self.map_data.obstacles:
            closest_x = max(obstacle.x, min(agent.x, obstacle.x + obstacle.width))
            closest_y = max(obstacle.y, min(agent.y, obstacle.y + obstacle.height))
            dx = agent.x - closest_x
            dy = agent.y - closest_y
            dist = math.hypot(dx, dy)
            if dist == 0.0:
                continue
            threshold = agent.radius + agent.obstacle_avoidance_distance
            if dist >= threshold:
                continue
            nx, ny = dx / dist, dy / dist
            strength = agent.obstacle_avoidance_strength * (1.0 - dist / threshold)
            fx += nx * strength
            fy += ny * strength
        return fx, fy

    def compute_agent_avoidance(self, agent):
        fx, fy = 0.0, 0.0
        for other in self.agents:
            if other is agent or other.evacuated:
                continue
            dx = agent.x - other.x
            dy = agent.y - other.y
            dist = math.hypot(dx, dy)
            min_dist = agent.radius + other.radius
            threshold = min_dist + agent.agent_avoidance_distance
            if dist == 0.0 or dist >= threshold:
                continue
            nx, ny = dx / dist, dy / dist
            strength = agent.agent_avoidance_strength * (1.0 - dist / threshold)
            fx += nx * strength
            fy += ny * strength
        return fx, fy

    # ------------------------------------------------------------------
    # Colisões e detecções
    # ------------------------------------------------------------------

    def check_collision_static(self, x, y, radius):
        if x - radius < 0 or x + radius > self.map_data.width:
            return True
        if y - radius < 0 or y + radius > self.map_data.height:
            return True
        for obstacle in self.map_data.obstacles:
            if self.circle_intersects_rect(x, y, radius, obstacle):
                return True
        return False

    def check_collision(self, x, y, radius, ignore_agent=None):
        if self.check_collision_static(x, y, radius):
            return True
        for other in self.agents:
            if other is ignore_agent or other.evacuated:
                continue
            if math.hypot(x - other.x, y - other.y) < (radius + other.radius):
                return True
        return False

    def circle_intersects_rect(self, cx, cy, radius, rect):
        closest_x = max(rect.x, min(cx, rect.x + rect.width))
        closest_y = max(rect.y, min(cy, rect.y + rect.height))
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (radius * radius)

    def touches_exit(self, agent):
        for exit_obj in self.map_data.exits:
            if self.circle_intersects_rect(agent.x, agent.y, agent.radius, exit_obj):
                return True
        return False

    def touches_hazard(self, agent):
        for hazard in self.map_data.hazards:
            if self.circle_intersects_rect(agent.x, agent.y, agent.radius, hazard):
                return True
        return False

    def hazard_visible(self, agent) -> bool:
        """True se qualquer hazard está dentro do raio de visão do agente."""
        for hazard in self.map_data.hazards:
            hcx = hazard.x + hazard.width / 2.0
            hcy = hazard.y + hazard.height / 2.0
            if math.hypot(hcx - agent.x, hcy - agent.y) <= HAZARD_VISION_RADIUS:
                return True
        return False

    def nearest_hazard_dist(self, agent) -> float:
        """Distância ao centro do hazard mais próximo (inf se não houver nenhum)."""
        min_dist = float("inf")
        for hazard in self.map_data.hazards:
            hcx = hazard.x + hazard.width / 2.0
            hcy = hazard.y + hazard.height / 2.0
            d = math.hypot(hcx - agent.x, hcy - agent.y)
            if d < min_dist:
                min_dist = d
        return min_dist

    # ------------------------------------------------------------------
    # Vizinhança e métricas
    # ------------------------------------------------------------------

    def get_nearest_exit(self, agent):
        best, best_dist = None, float("inf")
        for exit_obj in self.map_data.exits:
            cx = exit_obj.x + exit_obj.width / 2.0
            cy = exit_obj.y + exit_obj.height / 2.0
            d = math.hypot(cx - agent.x, cy - agent.y)
            if d < best_dist:
                best_dist, best = d, exit_obj
        return best

    def distance_to_nearest_exit(self, agent):
        exit_obj = self.get_nearest_exit(agent)
        cx = exit_obj.x + exit_obj.width / 2.0
        cy = exit_obj.y + exit_obj.height / 2.0
        return math.hypot(cx - agent.x, cy - agent.y)

    def local_density(self, agent, radius=40.0):
        count = 0
        for other in self.agents:
            if other is agent or other.evacuated:
                continue
            if math.hypot(agent.x - other.x, agent.y - other.y) <= radius:
                count += 1
        return count

    def _nearest_obstacle_dist(self, agent) -> float:
        min_dist = float("inf")
        for obstacle in self.map_data.obstacles:
            closest_x = max(obstacle.x, min(agent.x, obstacle.x + obstacle.width))
            closest_y = max(obstacle.y, min(agent.y, obstacle.y + obstacle.height))
            d = math.hypot(agent.x - closest_x, agent.y - closest_y)
            if d < min_dist:
                min_dist = d
        border_dist = min(
            agent.x,
            agent.y,
            self.map_data.width - agent.x,
            self.map_data.height - agent.y,
        )
        return min(min_dist, border_dist)

    # ------------------------------------------------------------------
    # Emoção e FSM
    # ------------------------------------------------------------------

    def update_emotion(self, agent):
        """
        Regras de emoção:
        - Decai passivamente a -0.02/step (tende à calma)
        - +0.15 se em contato direto com hazard (era +0.10, aumentado)
        - +0.08 se hazard visível mas sem contato — percepção de perigo próximo
        - +0.03 se densidade de vizinhos >= 3 — contágio emocional
        """
        delta = -0.02

        if self.touches_hazard(agent):
            delta += 0.15
        elif self.hazard_visible(agent):
            delta += 0.08

        density = self.local_density(agent, radius=35.0)
        if density >= 3:
            delta += 0.03

        agent.emotion_level = max(0.0, min(1.0, agent.emotion_level + delta))

    def update_fsm(self, agent):
        """
        Transições de estado baseadas em emotion_level.
        Cada estado ajusta os parâmetros comportamentais do agente.
        """
        if agent.emotion_level >= 0.7:
            agent.state = FSMState.PANIC
        elif agent.emotion_level >= 0.3:
            agent.state = FSMState.EVACUATE
        else:
            agent.state = FSMState.CALM

        if agent.state == FSMState.CALM:
            target_speed = agent.base_speed
            agent.obstacle_avoidance_distance = 16.0
            agent.obstacle_avoidance_strength = 80.0
            agent.agent_avoidance_distance = 14.0
            agent.agent_avoidance_strength = 95.0
            agent.velocity_smoothing = 0.20

        elif agent.state == FSMState.EVACUATE:
            target_speed = agent.base_speed * 1.15
            agent.obstacle_avoidance_distance = 12.0
            agent.obstacle_avoidance_strength = 70.0
            agent.agent_avoidance_distance = 10.0
            agent.agent_avoidance_strength = 85.0
            agent.velocity_smoothing = 0.16

        else:  # PANIC
            target_speed = agent.base_speed * 1.30
            agent.obstacle_avoidance_distance = 8.0
            agent.obstacle_avoidance_strength = 55.0
            agent.agent_avoidance_distance = 7.0
            agent.agent_avoidance_strength = 70.0
            agent.velocity_smoothing = 0.10

        agent.current_speed += (target_speed - agent.current_speed) * 0.2

    # ------------------------------------------------------------------
    # Observação (estado para o DQN) — 17 features
    # ------------------------------------------------------------------

    # Dimensão do vetor — use esta constante no config para state_dim
    OBS_DIM = 17

    def get_observation(self, agent) -> list[float]:
        """
        Vetor de 17 features:

        [0]  pos_x normalizada
        [1]  pos_y normalizada
        [2]  dx para a saída (normalizado)
        [3]  dy para a saída (normalizado)
        [4]  distância à saída (normalizada)
        [5]  ângulo para a saída em [-1, 1]
        [6]  hazard visível no raio de visão (0/1)  ← novo
        [7]  distância normalizada ao hazard mais próximo  ← novo
        [8]  em contato direto com hazard (0/1)
        [9]  emotion_level [0, 1]
        [10] estado FSM (0=calm, 1=evacuate, 2=panic) normalizado
        [11] vx normalizado pela velocidade atual
        [12] vy normalizado pela velocidade atual
        [13] densidade local normalizada [0, 1]
        [14] distância ao obstáculo/parede mais próxima (normalizada)
        [15] velocidade atual normalizada pelo base_speed
        [16] evacuado (0/1)
        """
        if agent.evacuated:
            return [0.0] * self.OBS_DIM

        exit_obj = self.get_nearest_exit(agent)
        exit_cx = exit_obj.x + exit_obj.width / 2.0
        exit_cy = exit_obj.y + exit_obj.height / 2.0

        dx_exit = exit_cx - agent.x
        dy_exit = exit_cy - agent.y

        max_dist = math.hypot(self.map_data.width, self.map_data.height)
        max_hazard_dist = HAZARD_VISION_RADIUS * 2.0

        angle_to_exit = math.atan2(dy_exit, dx_exit) / math.pi

        haz_visible = 1.0 if self.hazard_visible(agent) else 0.0
        haz_dist = self.nearest_hazard_dist(agent)
        haz_dist_norm = min(1.0, haz_dist / max(1.0, max_hazard_dist))

        in_hazard = 1.0 if self.touches_hazard(agent) else 0.0
        density = self.local_density(agent, radius=35.0)

        nearest_obs = self._nearest_obstacle_dist(agent)
        nearest_obs_norm = min(1.0, nearest_obs / max(1.0, agent.obstacle_avoidance_distance * 2))

        return [
            agent.x / self.map_data.width,                         # 0
            agent.y / self.map_data.height,                         # 1
            dx_exit / self.map_data.width,                          # 2
            dy_exit / self.map_data.height,                         # 3
            math.hypot(dx_exit, dy_exit) / max_dist,                # 4
            angle_to_exit,                                           # 5
            haz_visible,                                             # 6
            haz_dist_norm,                                           # 7
            in_hazard,                                               # 8
            agent.emotion_level,                                     # 9
            float(agent.state.value) / 2.0,                         # 10  normalizado p/ [0,1]
            agent.vx / max(1.0, agent.current_speed),               # 11
            agent.vy / max(1.0, agent.current_speed),               # 12
            min(1.0, density / 8.0),                                # 13
            nearest_obs_norm,                                        # 14
            agent.current_speed / max(1.0, agent.base_speed),       # 15
            1.0 if agent.evacuated else 0.0,                        # 16
        ]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, agent, prev_dist, new_dist, collided) -> float:
        """
        Reward balanceado para DQN com FSM emocional:

        + 5.0 × progresso em direção à saída  — sinal principal de aprendizado
        + 50.0 se evacuou                      — recompensa terminal
        + 0.5 × (1 - emotion) se hazard visível — incentiva manter calma perto do perigo
        - 0.05 por step                         — urgência suave (não paralisa ação)
        - 2.0 cada step dentro de hazard        — penalidade forte e contínua
        - 0.3 por colisão                       — desincentiva, mas não paralisa
        """
        reward = -0.05  # time penalty suave

        # Progresso em direção à saída
        reward += 5.0 * (prev_dist - new_dist)

        # Evacuou
        if agent.evacuated:
            reward += 50.0

        # Hazard
        if self.touches_hazard(agent):
            reward -= 2.0
        elif self.hazard_visible(agent):
            # Bonus pequeno por manter calma (emotion baixa) quando hazard está visível
            # Incentiva o agente a aprender que ver ≠ entrar
            reward += 0.5 * (1.0 - agent.emotion_level)

        # Colisão com parede ou agente
        if collided:
            reward -= 0.3

        return reward

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def is_done(self):
        if all(agent.evacuated for agent in self.agents):
            return True
        if self.time >= self.max_steps:
            return True
        return False

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def cell_center_to_world(self, row, col):
        x = col * self.map_data.tile_size + self.map_data.tile_size / 2.0
        y = row * self.map_data.tile_size + self.map_data.tile_size / 2.0
        return x, y

    def world_to_cell(self, x, y):
        col = int(x // self.map_data.tile_size)
        row = int(y // self.map_data.tile_size)
        row = max(0, min(row, self.map_data.rows - 1))
        col = max(0, min(col, self.map_data.cols - 1))
        return row, col