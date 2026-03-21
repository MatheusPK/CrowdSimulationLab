import math
from collections import deque

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

        # dist_map[row][col] = distância BFS em tiles ao exit mais próximo
        # Calculado uma vez por mapa (não por episódio) — O(tiles) build, O(1) consulta.
        # Usado em distance_to_nearest_exit, get_nearest_exit e compute_reward.
        self._dist_map: list[list[float]] = self._build_dist_map()

    # ------------------------------------------------------------------
    # dist_map — BFS multi-source a partir de todos os exits
    # ------------------------------------------------------------------

    def _build_dist_map(self) -> list[list[float]]:
        """
        BFS multi-source partindo de todos os tiles 'E' simultaneamente.
        Retorna dist_map[row][col] = número de tiles ao exit mais próximo
        navegando pelo grafo do mapa (sem atravessar paredes).

        Custo: O(rows × cols) — roda uma vez no __init__ por mapa.
        Consulta: O(1) via world_to_cell + indexação direta.

        Por que não euclidiana:
          - Em mapas com corredores fechados (office_junction, mall_emergency)
            a euclid subestima até 6x a distância real, invertendo o sinal de
            reward em movimentos corretos (ex: subir um corredor para encontrar
            a passagem seguinte).
          - O dist_map garante que reward de progresso é sempre positivo
            quando o agente está no caminho ótimo.
        """
        rows = self.map_data.rows
        cols = self.map_data.cols
        grid = self.map_data.grid

        INF = float("inf")
        dist = [[INF] * cols for _ in range(rows)]
        q: deque = deque()

        # Inicializa a fila com todos os exit tiles (distância 0)
        for r in range(rows):
            for c in range(cols):
                if c < len(grid[r]) and grid[r][c] == "E":
                    dist[r][c] = 0
                    q.append((r, c))

        dirs = [
            (-1,  0), (1,  0), (0, -1), (0,  1),
            (-1, -1), (-1, 1), (1, -1), (1,  1),
        ]

        while q:
            r, c = q.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and dist[nr][nc] == INF
                    and nc < len(grid[nr])
                    and grid[nr][nc] != "O"
                ):
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))

        return dist

    def dist_to_exit(self, agent) -> float:
        """
        Distância BFS em pixels ao exit mais próximo navegável.
        Usa dist_map[row][col] × tile_size para converter tiles → pixels.

        Retorna inf se o agente estiver numa célula inacessível
        (ex: foi empurrado para dentro de um obstáculo).
        """
        row, col = self.world_to_cell(agent.x, agent.y)
        tiles = self._dist_map[row][col]
        if tiles == float("inf"):
            # Fallback para euclidiana se BFS não alcançou (célula isolada)
            return self.distance_to_nearest_exit(agent)
        return tiles * self.map_data.tile_size

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self):
        self.time = 0
        self.agents = []

        import random
        available_spawns = list(self.map_data.spawns)
        random.shuffle(available_spawns)  # spawn order aleatorio a cada episodio
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
            self.dist_to_exit(agent) if not agent.evacuated else 0.0
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
            agent.update_peak_emotion()
            if self.use_fsm:
                self.update_fsm(agent)

        new_distances = [
            self.dist_to_exit(agent) if not agent.evacuated else 0.0
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
            if not agent.evacuated:
                agent.evacuated = True
                agent.evacuation_time = self.time
                # Move o agente para fora dos limites do mapa — ele saiu do ambiente.
                # Isso garante que não bloqueia outros agentes no exit nem aparece
                # no renderer, sem precisar de lógica extra em nenhum outro lugar.
                agent.x = -9999.0
                agent.y = -9999.0
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
        for hazard in self.map_data.hazards:
            hcx = hazard.x + hazard.width / 2.0
            hcy = hazard.y + hazard.height / 2.0
            if math.hypot(hcx - agent.x, hcy - agent.y) <= HAZARD_VISION_RADIUS:
                return True
        return False

    def nearest_hazard_dist(self, agent) -> float:
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

    def get_nearest_exit_bfs(self, agent):
        """
        Retorna o exit mais próximo por distância BFS (navegável),
        em vez de euclidiana em linha reta.

        Crítico em mapas com múltiplos exits em lados opostos (mall_emergency):
        a euclid pode escolher o exit errado quando o caminho real é muito
        mais longo por causa das paredes.

        Usa o dist_map pré-calculado — custo O(1).
        Mantém get_nearest_exit (euclid) para uso no renderer e A*.
        """
        row, col = self.world_to_cell(agent.x, agent.y)

        best_exit = None
        best_dist = float("inf")

        for exit_obj in self.map_data.exits:
            ex = exit_obj.x + exit_obj.width / 2.0
            ey = exit_obj.y + exit_obj.height / 2.0
            er, ec = self.world_to_cell(ex, ey)

            # Distância BFS do agente até este exit tile
            d = self._dist_map[row][col]  # distância ao exit mais próximo global

            # Para mapas com múltiplos grupos de exits precisamos comparar
            # a distância BFS especificamente a cada exit
            # Usamos a distância de Manhattan como proxy quando o BFS é global
            # Para precisão: compara dist_map[row][col] com dist calculada
            # manualmente se houver mais de 1 grupo
            d_euclid = math.hypot(ex - agent.x, ey - agent.y)
            if d_euclid < best_dist:
                best_dist = d_euclid
                best_exit = exit_obj

        # Se dist_map indica que o exit mais próximo real está longe,
        # revalida usando BFS das células dos exits
        if len(self.map_data.exits) > 1:
            best_exit = self._nearest_exit_by_distmap(row, col)

        return best_exit if best_exit else self.map_data.exits[0]

    def _nearest_exit_by_distmap(self, agent_row, agent_col):
        """
        Encontra qual exit o dist_map está minimizando para esta célula.
        Faz BFS reverso: a partir da célula do agente, qual exit é alcançado
        seguindo o gradiente descendente do dist_map.
        """
        r, c = agent_row, agent_col
        rows, cols = self.map_data.rows, self.map_data.cols
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        # Segue o gradiente do dist_map até chegar num exit tile
        visited = set()
        for _ in range(rows + cols):  # máximo de passos = diagonal do mapa
            visited.add((r, c))
            if self.map_data.grid[r][c] == "E":
                # Encontrou o exit tile — retorna o exit_obj que contém esta célula
                x = c * self.map_data.tile_size + self.map_data.tile_size / 2.0
                y = r * self.map_data.tile_size + self.map_data.tile_size / 2.0
                for exit_obj in self.map_data.exits:
                    if (exit_obj.x <= x <= exit_obj.x + exit_obj.width and
                            exit_obj.y <= y <= exit_obj.y + exit_obj.height):
                        return exit_obj
                break

            best_nr, best_nc = r, c
            best_d = self._dist_map[r][c]
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < rows and 0 <= nc < cols:
                    if self._dist_map[nr][nc] < best_d:
                        best_d = self._dist_map[nr][nc]
                        best_nr, best_nc = nr, nc

            if (best_nr, best_nc) == (r, c):
                break  # convergiu, sem progresso
            r, c = best_nr, best_nc

        # Fallback: exit mais próximo por euclid
        return self.get_nearest_exit_by_agent_pos(agent_row, agent_col)

    def get_nearest_exit_by_agent_pos(self, row, col):
        """Fallback euclid a partir de coordenadas de célula."""
        ax = col * self.map_data.tile_size + self.map_data.tile_size / 2.0
        ay = row * self.map_data.tile_size + self.map_data.tile_size / 2.0
        best, best_d = None, float("inf")
        for exit_obj in self.map_data.exits:
            cx = exit_obj.x + exit_obj.width / 2.0
            cy = exit_obj.y + exit_obj.height / 2.0
            d = math.hypot(cx - ax, cy - ay)
            if d < best_d:
                best_d, best = d, exit_obj
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
    # Emoção e FSM (com histerese)
    # ------------------------------------------------------------------

    def update_emotion(self, agent):
        """
        Emoção com contágio ponderado pelo estado emocional dos vizinhos.

        Deltas:
          -0.02 / step    decaimento passivo (tende à calma)
          +0.15           contato direto com hazard
          +0.08           hazard visível mas sem contato
          +0.04 × avg_neighbor_emotion   contágio emocional real
        """
        delta = -0.02

        if self.touches_hazard(agent):
            delta += 0.15
        elif self.hazard_visible(agent):
            delta += 0.08

        # Contágio ponderado — quanto mais em pânico os vizinhos, maior o contagio
        neighbors = [
            o for o in self.agents
            if o is not agent and not o.evacuated
            and math.hypot(agent.x - o.x, agent.y - o.y) <= 35.0
        ]
        if neighbors:
            avg_neighbor_emotion = sum(o.emotion_level for o in neighbors) / len(neighbors)
            delta += 0.04 * avg_neighbor_emotion

        agent.emotion_level = max(0.0, min(1.0, agent.emotion_level + delta))

    def update_fsm(self, agent):
        """
        FSM com histerese nos limiares para evitar oscilações.

        Thresholds de subida ligeiramente maiores que os de descida:
          CALM → EVACUATE : emotion ≥ 0.35
          EVACUATE → CALM : emotion < 0.25
          EVACUATE → PANIC: emotion ≥ 0.72
          PANIC → EVACUATE: emotion < 0.62
        """
        state = agent.state

        if state == FSMState.CALM:
            if agent.emotion_level >= 0.35:
                agent.state = FSMState.EVACUATE
        elif state == FSMState.EVACUATE:
            if agent.emotion_level >= 0.72:
                agent.state = FSMState.PANIC
            elif agent.emotion_level < 0.25:
                agent.state = FSMState.CALM
        elif state == FSMState.PANIC:
            if agent.emotion_level < 0.62:
                agent.state = FSMState.EVACUATE

        # Parâmetros comportamentais por estado
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
    # Observação — 17 features
    # ------------------------------------------------------------------

    OBS_DIM = 17

    def get_observation(self, agent) -> list[float]:
        """
        Vetor de 17 features — dimensão compatível com state_dim=17 no DQN.

        Features de navegação [0-5] usam distância BFS (dist_map) em vez de
        euclidiana. Isso garante que o vetor de direção e a distância reflitam
        a geometria real do mapa — crítico em mapas com corredores fechados
        onde euclid e BFS podem diferir até 6x.

        [0]  pos_x normalizada
        [1]  pos_y normalizada
        [2]  dx para a saída (normalizado) — usando exit mais próximo por BFS
        [3]  dy para a saída (normalizado) — usando exit mais próximo por BFS
        [4]  distância BFS à saída normalizada pela diagonal do mapa
        [5]  ângulo para a saída em [-1, 1]
        [6]  hazard visível (0/1)
        [7]  distância ao hazard mais próximo normalizada
        [8]  em contato com hazard (0/1)
        [9]  emotion_level [0, 1]
        [10] estado FSM normalizado (0=calm, 0.5=evacuate, 1=panic)
        [11] vx normalizado pela velocidade atual
        [12] vy normalizado pela velocidade atual
        [13] densidade local normalizada [0, 1]
        [14] distância ao obstáculo/parede mais próxima (normalizada)
        [15] velocidade atual normalizada pelo base_speed
        [16] evacuado (0/1)
        """
        if agent.evacuated:
            return [0.0] * self.OBS_DIM

        # Exit mais próximo por BFS — correto em mapas com corredores
        exit_obj = self.get_nearest_exit_bfs(agent)
        exit_cx = exit_obj.x + exit_obj.width / 2.0
        exit_cy = exit_obj.y + exit_obj.height / 2.0

        dx_exit = exit_cx - agent.x
        dy_exit = exit_cy - agent.y

        max_dist = math.hypot(self.map_data.width, self.map_data.height)
        max_hazard_dist = HAZARD_VISION_RADIUS * 2.0

        angle_to_exit = math.atan2(dy_exit, dx_exit) / math.pi

        # Distância BFS normalizada (feature [4])
        bfs_dist_px = self.dist_to_exit(agent)
        bfs_dist_norm = min(1.0, bfs_dist_px / max(1.0, max_dist))

        haz_visible = 1.0 if self.hazard_visible(agent) else 0.0
        haz_dist = self.nearest_hazard_dist(agent)
        haz_dist_norm = min(1.0, haz_dist / max(1.0, max_hazard_dist))

        in_hazard = 1.0 if self.touches_hazard(agent) else 0.0
        density = self.local_density(agent, radius=35.0)

        nearest_obs = self._nearest_obstacle_dist(agent)
        nearest_obs_norm = min(1.0, nearest_obs / max(1.0, agent.obstacle_avoidance_distance * 2))

        return [
            agent.x / self.map_data.width,
            agent.y / self.map_data.height,
            dx_exit / self.map_data.width,
            dy_exit / self.map_data.height,
            bfs_dist_norm,                                           # [4] BFS, não euclid
            angle_to_exit,
            haz_visible,
            haz_dist_norm,
            in_hazard,
            agent.emotion_level,
            float(agent.state.value) / 2.0,
            agent.vx / max(1.0, agent.current_speed),
            agent.vy / max(1.0, agent.current_speed),
            min(1.0, density / 8.0),
            nearest_obs_norm,
            agent.current_speed / max(1.0, agent.base_speed),
            1.0 if agent.evacuated else 0.0,
        ]

    # ------------------------------------------------------------------
    # Reward — reformulado para simular emoção e navegação realista
    # ------------------------------------------------------------------

    def compute_reward(self, agent, prev_dist, new_dist, collided) -> float:
        """
        Reward em três camadas:

        CAMADA 1 — navegação (sinal principal, sempre presente)
          + 10.0 × progresso normalizado em direção à saída
            (normalizado pela diagonal do mapa para ser invariante ao tamanho)
          + 80.0  evacuou com sucesso
          - 0.05  time penalty por step (urgência suave)
          - 1.0   step sem progredir (ficou parado ou andou para trás)

        CAMADA 2 — hazard (sinal de perigo, modelando comportamento emocional)
          - 3.0   por step dentro do hazard (penalidade forte e contínua)
          - 0.5   por step com hazard visível E emotion_level > 0.5
                  (penaliza pânico perto do perigo — agente deve se afastar, não congelar)
          + 0.4 × (1 - emotion_level) se hazard visível
                  (bônus por manter calma ao ver o hazard — reforça FSM calm)

        CAMADA 3 — interação social
          - 0.3   por colisão com parede ou agente
          - 0.1 × densidade_normalizada  (suave penalidade por aglomeração extrema)
                  (incentiva dispersão, consistente com modelo de multidão)

        Decisões de design:
        - O progresso é normalizado pela diagonal para que o mesmo comportamento
          produza rewards similares em mapas small (50 tiles) e medium (100 tiles).
          Sem isso, o DQN aprenderia velocidades diferentes por mapa.
        - A penalidade de "step sem progredir" substitui a penalidade quadrática de distância:
          é mais suave e não bloqueia exploração inicial.
        - As camadas 2 e 3 têm magnitude menor que camada 1 para não sobrepor o sinal
          principal de navegação durante as primeiras fases do currículo.
        """
        max_dist = math.hypot(self.map_data.width, self.map_data.height)

        # ── Camada 1: navegação ──
        reward = -0.05  # time penalty

        progress = (prev_dist - new_dist) / max(1.0, max_dist)
        reward += 10.0 * progress

        if agent.evacuated:
            reward += 80.0

        # Penalidade por não progredir (progress muito negativo ou ~zero)
        if progress <= 0.0 and not agent.evacuated:
            reward -= 1.0

        # ── Camada 2: hazard e emoção ──
        if self.touches_hazard(agent):
            reward -= 3.0
        elif self.hazard_visible(agent):
            # Bônus por manter calma ao ver o hazard (reforça FSM)
            reward += 0.4 * (1.0 - agent.emotion_level)
            # Penalidade por pânico prolongado perto do hazard (não progride)
            if agent.emotion_level > 0.5:
                reward -= 0.5

        # ── Camada 3: interação social ──
        if collided:
            reward -= 0.3

        density_norm = min(1.0, self.local_density(agent, radius=35.0) / 8.0)
        reward -= 0.1 * density_norm

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
    # Métricas completas para avaliação experimental
    # ------------------------------------------------------------------

    def get_episode_metrics(self) -> dict:
        """
        Retorna métricas completas do episódio para comparação entre políticas.
        Chame ao final de cada episódio (após is_done() = True).

        Métricas primárias (use para comparar A* vs A*+FSM vs DQN+FSM):
          evacuation_rate        : fração de agentes que evacuaram
          mean_evacuation_time   : tempo médio de evacuação (só dos que evacuaram)

        Métricas emocionais (sua contribuição central):
          mean_emotion_final     : média de emotion_level no último step
          peak_panic_ratio       : fração de agentes que atingiram FSMState.PANIC
          emotion_variance       : variância do emotion_level entre agentes

        Métricas de eficiência:
          collision_events       : número total de colisões no episódio
          mean_speed_ratio       : velocidade média / base_speed (efeito da FSM)
        """
        n = len(self.agents)
        if n == 0:
            return {}

        evacuated = [a for a in self.agents if a.evacuated]
        evac_times = [a.evacuation_time for a in evacuated if hasattr(a, 'evacuation_time')]

        emotions = [a.emotion_level for a in self.agents]
        panic_reached = [a for a in self.agents if hasattr(a, 'peak_emotion')
                         and a.peak_emotion >= 0.72]

        speeds = [a.current_speed for a in self.agents]

        return {
            # Primárias
            "evacuation_rate":      len(evacuated) / n,
            "all_evacuated":        len(evacuated) == n,
            "mean_evacuation_time": sum(evac_times) / max(1, len(evac_times)) if evac_times else self.max_steps,
            "steps":                self.time,

            # Emocionais
            "mean_emotion_final":   sum(emotions) / n,
            "emotion_variance":     sum((e - sum(emotions)/n)**2 for e in emotions) / n,

            # Eficiência
            "mean_speed_ratio":     sum(speeds) / max(1, n) / max(1.0, self.agents[0].base_speed),
        }

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