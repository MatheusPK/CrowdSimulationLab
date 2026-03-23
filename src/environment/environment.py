import heapq
import math
from collections import deque

from core.direction import Direction
from core.fsm_state import FSMState
from entities.agent import Agent
from simulation_params import (
    HAZARD_VISION_RADIUS,
    LOCAL_DENSITY_RADIUS,
    CONTAGION_RADIUS,
    ASTAR_HAZARD_COST,
    EMOTION_DECAY,
    EMOTION_DELTA_HAZARD_CONTACT,
    EMOTION_DELTA_HAZARD_VISIBLE,
    EMOTION_DELTA_CONTAGION,
    FSM_CALM_TO_EVACUATE, FSM_EVACUATE_TO_CALM,
    FSM_EVACUATE_TO_PANIC, FSM_PANIC_TO_EVACUATE,
    FSM_SPEED_CALM, FSM_SPEED_EVACUATE, FSM_SPEED_PANIC,
    FSM_SPEED_SMOOTHING,
    FSM_CALM_OBS_DIST, FSM_CALM_OBS_STR, FSM_CALM_AGENT_DIST,
    FSM_CALM_AGENT_STR, FSM_CALM_VEL_SMOOTH,
    FSM_EVAC_OBS_DIST, FSM_EVAC_OBS_STR, FSM_EVAC_AGENT_DIST,
    FSM_EVAC_AGENT_STR, FSM_EVAC_VEL_SMOOTH,
    FSM_PANIC_OBS_DIST, FSM_PANIC_OBS_STR, FSM_PANIC_AGENT_DIST,
    FSM_PANIC_AGENT_STR, FSM_PANIC_VEL_SMOOTH,
    REWARD_PROGRESS_SCALE, REWARD_EVACUATED, REWARD_TIME_PENALTY,
    REWARD_NO_PROGRESS, REWARD_HAZARD_CONTACT, REWARD_HAZARD_VISIBLE_CALM,
    REWARD_HAZARD_PANIC, REWARD_COLLISION, REWARD_DENSITY_SCALE,
)


class Environment:
    def __init__(self, map_data, dt=0.1, max_steps=300, use_fsm=False, num_agents=1,
                 contagion_radius: float | None = None):
        self.map_data = map_data
        self.dt = dt
        self.max_steps = max_steps
        self.use_fsm = use_fsm
        self.num_agents_requested = num_agents
        self.contagion_radius = contagion_radius if contagion_radius is not None else CONTAGION_RADIUS

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

        # dist_map[row][col] = distância BFS em tiles ao exit mais próximo.
        self._dist_map: list[list[float]] = self._build_dist_map()

        # dist_map_safe: mesma lógica, mas penalizando tiles H (para A*+FSM).
        self._dist_map_safe: list[list[float]] = self._build_dist_map_safe()

        # Cache de densidade por step — populado em step(), None fora do loop
        self._density_cache: dict | None = None

        # Conjunto de obstacles adjacentes a exits
        self._exit_adjacent_obstacles: set = self._build_exit_adjacent_set()

        # Centróides dos grupos de exits
        self._exit_group_centroids: list = self._build_exit_group_centroids()

    # ------------------------------------------------------------------
    # dist_map — BFS multi-source a partir de todos os exits
    # ------------------------------------------------------------------

    def _build_dist_map(self) -> list[list[float]]:
        rows = self.map_data.rows
        cols = self.map_data.cols
        grid = self.map_data.grid

        INF = float("inf")
        dist = [[INF] * cols for _ in range(rows)]
        q: deque = deque()

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
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if dist[nr][nc] != INF:
                    continue
                if nc >= len(grid[nr]) or grid[nr][nc] == "O":
                    continue

                if dr != 0 and dc != 0:
                    if (grid[r + dr][c] == "O" or
                            r + dr < 0 or r + dr >= rows or
                            grid[r][c + dc] == "O" or
                            c + dc < 0 or c + dc >= cols):
                        continue
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))

        return dist

    def _build_dist_map_safe(self) -> list[list[float]]:
        rows = self.map_data.rows
        cols = self.map_data.cols
        grid = self.map_data.grid

        INF = float("inf")
        dist = [[INF] * cols for _ in range(rows)]
        heap = []

        for r in range(rows):
            for c in range(cols):
                if c < len(grid[r]) and grid[r][c] == "E":
                    dist[r][c] = 0.0
                    heapq.heappush(heap, (0.0, r, c))

        dirs = [
            (-1,  0), (1,  0), (0, -1), (0,  1),
            (-1, -1), (-1, 1), (1, -1), (1,  1),
        ]

        while heap:
            cost, r, c = heapq.heappop(heap)
            if cost > dist[r][c]:
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if nc >= len(grid[nr]) or grid[nr][nc] == "O":
                    continue

                if dr != 0 and dc != 0:
                    if (grid[r + dr][c] == "O" or
                            r + dr < 0 or r + dr >= rows or
                            grid[r][c + dc] == "O" or
                            c + dc < 0 or c + dc >= cols):
                        continue
                step = ASTAR_HAZARD_COST if grid[nr][nc] == "H" else 1.0
                new_cost = dist[r][c] + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    heapq.heappush(heap, (new_cost, nr, nc))

        return dist

    def dist_to_exit(self, agent) -> float:
        row, col = self.world_to_cell(agent.x, agent.y)
        tiles = self._dist_map[row][col]
        if tiles == float("inf"):
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
        random.shuffle(available_spawns)
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

        self._density_cache = self._compute_density_cache()

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

        if self.touches_hazard(agent):
            agent.touched_hazard = True

        if self.touches_exit(agent):
            if not agent.evacuated:
                agent.evacuated = True
                agent.evacuation_time = self.time
                for exit_obj in self.map_data.exits:
                    if self.circle_intersects_rect(agent.x, agent.y, agent.radius, exit_obj):
                        agent.exit_used = exit_obj
                        break

                agent.x = -9999.0
                agent.y = -9999.0
            agent.stop()

        return collided

    def _build_exit_adjacent_set(self) -> set:
        tile = self.map_data.tile_size
        adjacent = set()
        for obs in self.map_data.obstacles:
            for ex in self.map_data.exits:
                ex_cx = ex.x + ex.width / 2.0
                ex_cy = ex.y + ex.height / 2.0
                closest_x = max(obs.x, min(ex_cx, obs.x + obs.width))
                closest_y = max(obs.y, min(ex_cy, obs.y + obs.height))
                dist = math.hypot(ex_cx - closest_x, ex_cy - closest_y)
                if dist <= 2 * tile:  # 16px — cobre borda do exit + 1 tile extra
                    adjacent.add(id(obs))
                    break
        return adjacent

    def _build_exit_group_centroids(self) -> list[tuple[float, float]]:
        exits = self.map_data.exits
        if not exits:
            return []

        tile = self.map_data.tile_size
        pos_to_exit = {(e.x, e.y): e for e in exits}
        parent = {e: e for e in exits}

        def find(x):
            while parent[x] is not x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        for e in exits:
            for dx, dy in [(tile, 0), (-tile, 0), (0, tile), (0, -tile)]:
                neighbor = pos_to_exit.get((e.x + dx, e.y + dy))
                if neighbor:
                    union(e, neighbor)

        groups: dict = {}
        for e in exits:
            root = find(e)
            if id(root) not in groups:
                groups[id(root)] = []
            groups[id(root)].append((e.x + e.width / 2.0, e.y + e.height / 2.0))

        return [(sum(x for x, y in pts) / len(pts),
                 sum(y for x, y in pts) / len(pts))
                for pts in groups.values()]

    def _bfs_direction_vector(self, agent_row: int, agent_col: int,
                            agent) -> tuple[float, float]:
        dm = self._dist_map
        rows = self.map_data.rows
        cols = self.map_data.cols
        tile = self.map_data.tile_size

        current_dist = dm[agent_row][agent_col]

        if current_dist == float("inf"):
            return self._vector_to_nearest_exit_tile(agent)

        if self.map_data.grid[agent_row][agent_col] == "E":
            return self._vector_to_nearest_exit_tile(agent)

        best_dist = current_dist
        best_r, best_c = agent_row, agent_col

        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for dr, dc in dirs:
            nr, nc = agent_row + dr, agent_col + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if dr != 0 and dc != 0:
                if (self.map_data.grid[agent_row + dr][agent_col] == "O" or
                        self.map_data.grid[agent_row][agent_col + dc] == "O"):
                    continue
            if dm[nr][nc] < best_dist:
                best_dist = dm[nr][nc]
                best_r, best_c = nr, nc

        if (best_r, best_c) == (agent_row, agent_col):
            return self._vector_to_nearest_exit_tile(agent)

        next_cx = best_c * tile + tile / 2.0
        next_cy = best_r * tile + tile / 2.0
        return next_cx - agent.x, next_cy - agent.y

    def compute_obstacle_avoidance(self, agent):
        fx, fy = 0.0, 0.0
        for obstacle in self.map_data.obstacles:
            # Tiles de parede adjacentes ao exit não geram força de repulsão.
            # São a "moldura" da abertura: sem esta supressão, a força cumulativa
            # de toda a borda do mapa satura o clamp de velocidade e impede o
            # agente de entrar na saída mesmo com action=N.
            if id(obstacle) in self._exit_adjacent_obstacles:
                continue
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

        # FIX: mapas com tiles O individuais acumulam forças de dezenas de obstáculos
        # adjacentes (ex: row 0 inteira = 56 tiles × força individual = 1568 px/s).
        # Sem clamping, agentes próximos à borda ficam completamente presos mesmo em
        # espaço livre. Limitamos a força total a current_speed para que a parede
        # repila mas não paralise o agente.
        total = math.hypot(fx, fy)
        if total > agent.current_speed and total > 0.0:
            scale = agent.current_speed / total
            fx *= scale
            fy *= scale

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

        total = math.hypot(fx, fy)
        if total > agent.current_speed and total > 0.0:
            scale = agent.current_speed / total
            fx *= scale
            fy *= scale

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

    def get_nearest_exit_bfs(self, agent, use_safe_map: bool = False):
        row, col = self.world_to_cell(agent.x, agent.y)

        if len(self.map_data.exits) == 1:
            return self.map_data.exits[0]

        return self._nearest_exit_by_distmap(row, col, use_safe_map)

    def _nearest_exit_by_distmap(self, agent_row, agent_col, use_safe_map: bool = False):
        dm = self._dist_map_safe if use_safe_map else self._dist_map
        r, c = agent_row, agent_col
        rows, cols = self.map_data.rows, self.map_data.cols
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        visited = set()
        for _ in range(rows + cols):
            visited.add((r, c))
            if self.map_data.grid[r][c] == "E":
                x = c * self.map_data.tile_size + self.map_data.tile_size / 2.0
                y = r * self.map_data.tile_size + self.map_data.tile_size / 2.0
                for exit_obj in self.map_data.exits:
                    if (exit_obj.x <= x <= exit_obj.x + exit_obj.width and
                            exit_obj.y <= y <= exit_obj.y + exit_obj.height):
                        return exit_obj
                break

            best_nr, best_nc = r, c
            best_d = dm[r][c]
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (nr, nc) in visited or not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                if dr != 0 and dc != 0:
                    if dm[r + dr][c] == float("inf") or dm[r][c + dc] == float("inf"):
                        continue
                if dm[nr][nc] < best_d:
                    best_d = dm[nr][nc]
                    best_nr, best_nc = nr, nc

            if (best_nr, best_nc) == (r, c):
                break
            r, c = best_nr, best_nc

        return self.get_nearest_exit_by_agent_pos(agent_row, agent_col)

    def get_nearest_exit_by_agent_pos(self, row, col):
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

    def _compute_density_cache(self) -> dict:
        active = [a for a in self.agents if not a.evacuated]
        cache = {id(a): 0 for a in self.agents}
        radius = LOCAL_DENSITY_RADIUS
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i == j:
                    continue
                if math.hypot(a.x - b.x, a.y - b.y) <= radius:
                    cache[id(a)] += 1
        return cache

    def local_density(self, agent, radius=None):
        if radius is None and hasattr(self, '_density_cache') and self._density_cache is not None:
            return self._density_cache.get(id(agent), 0)
        radius = radius if radius is not None else LOCAL_DENSITY_RADIUS
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
        delta = EMOTION_DECAY

        if self.touches_hazard(agent):
            delta += EMOTION_DELTA_HAZARD_CONTACT
        elif self.hazard_visible(agent):
            delta += EMOTION_DELTA_HAZARD_VISIBLE

        neighbors = [
            o for o in self.agents
            if o is not agent and not o.evacuated
            and math.hypot(agent.x - o.x, agent.y - o.y) <= self.contagion_radius
        ]
        if neighbors:
            avg_neighbor_emotion = sum(o.emotion_level for o in neighbors) / len(neighbors)
            delta += EMOTION_DELTA_CONTAGION * avg_neighbor_emotion

        agent.emotion_level = max(0.0, min(1.0, agent.emotion_level + delta))

    def update_fsm(self, agent):
        state = agent.state

        if state == FSMState.CALM:
            if agent.emotion_level >= FSM_CALM_TO_EVACUATE:
                agent.state = FSMState.EVACUATE
        elif state == FSMState.EVACUATE:
            if agent.emotion_level >= FSM_EVACUATE_TO_PANIC:
                agent.state = FSMState.PANIC
            elif agent.emotion_level < FSM_EVACUATE_TO_CALM:
                agent.state = FSMState.CALM
        elif state == FSMState.PANIC:
            if agent.emotion_level < FSM_PANIC_TO_EVACUATE:
                agent.state = FSMState.EVACUATE

        if agent.state == FSMState.CALM:
            target_speed = agent.base_speed * FSM_SPEED_CALM
            agent.obstacle_avoidance_distance = FSM_CALM_OBS_DIST
            agent.obstacle_avoidance_strength = FSM_CALM_OBS_STR
            agent.agent_avoidance_distance    = FSM_CALM_AGENT_DIST
            agent.agent_avoidance_strength    = FSM_CALM_AGENT_STR
            agent.velocity_smoothing          = FSM_CALM_VEL_SMOOTH

        elif agent.state == FSMState.EVACUATE:
            target_speed = agent.base_speed * FSM_SPEED_EVACUATE
            agent.obstacle_avoidance_distance = FSM_EVAC_OBS_DIST
            agent.obstacle_avoidance_strength = FSM_EVAC_OBS_STR
            agent.agent_avoidance_distance    = FSM_EVAC_AGENT_DIST
            agent.agent_avoidance_strength    = FSM_EVAC_AGENT_STR
            agent.velocity_smoothing          = FSM_EVAC_VEL_SMOOTH

        else:  # PANIC
            target_speed = agent.base_speed * FSM_SPEED_PANIC
            agent.obstacle_avoidance_distance = FSM_PANIC_OBS_DIST
            agent.obstacle_avoidance_strength = FSM_PANIC_OBS_STR
            agent.agent_avoidance_distance    = FSM_PANIC_AGENT_DIST
            agent.agent_avoidance_strength    = FSM_PANIC_AGENT_STR
            agent.velocity_smoothing          = FSM_PANIC_VEL_SMOOTH

        agent.current_speed += (target_speed - agent.current_speed) * FSM_SPEED_SMOOTHING

    # ------------------------------------------------------------------
    # Observação — 17 features
    # ------------------------------------------------------------------

    OBS_DIM = 17

    def get_observation(self, agent) -> list[float]:
        """
        Vetor de 17 features — dimensão compatível com state_dim=17 no DQN.

        Features de navegação [0-5] usam distância BFS (dist_map)

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
            obs = [0.0] * self.OBS_DIM
            obs[16] = 1.0
            return obs

        row, col = self.world_to_cell(agent.x, agent.y)
        dx_exit, dy_exit = self._bfs_direction_vector(row, col, agent)

        max_dist = math.hypot(self.map_data.width, self.map_data.height)
        max_hazard_dist = HAZARD_VISION_RADIUS * 2.0

        angle_to_exit = math.atan2(dy_exit, dx_exit) / math.pi

        bfs_dist_px = self.dist_to_exit(agent)
        bfs_dist_norm = min(1.0, bfs_dist_px / max(1.0, max_dist))

        haz_visible = 1.0 if self.hazard_visible(agent) else 0.0
        haz_dist = self.nearest_hazard_dist(agent)
        haz_dist_norm = min(1.0, haz_dist / max(1.0, max_hazard_dist))

        in_hazard = 1.0 if self.touches_hazard(agent) else 0.0
        density = self.local_density(agent)

        nearest_obs = self._nearest_obstacle_dist(agent)
        nearest_obs_norm = min(1.0, nearest_obs / max(1.0, agent.obstacle_avoidance_distance * 2))

        return [
            agent.x / self.map_data.width,
            agent.y / self.map_data.height,
            dx_exit / max_dist,   
            dy_exit / max_dist,   
            bfs_dist_norm,        
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
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, agent, prev_dist, new_dist, collided) -> float:
        max_dist = math.hypot(self.map_data.width, self.map_data.height)

        reward = REWARD_TIME_PENALTY

        progress = (prev_dist - new_dist) / max(1.0, max_dist)
        reward += REWARD_PROGRESS_SCALE * progress

        if agent.evacuated:
            reward += REWARD_EVACUATED

        if progress <= 0 and not agent.evacuated:
            reward += REWARD_NO_PROGRESS

        if self.touches_hazard(agent):
            reward += REWARD_HAZARD_CONTACT
        elif self.hazard_visible(agent):
            reward += REWARD_HAZARD_VISIBLE_CALM * (1.0 - agent.emotion_level)
            if agent.emotion_level > 0.5:
                reward += REWARD_HAZARD_PANIC

        if collided:
            reward += REWARD_COLLISION

        density_norm = min(1.0, self.local_density(agent) / 8.0)
        reward += REWARD_DENSITY_SCALE * density_norm

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

        Métricas primárias:
          evacuation_rate        : fração de agentes que evacuaram
          mean_evacuation_time   : tempo médio de evacuação (só dos que evacuaram)

        Métricas emocionais:
          mean_emotion_final     : média de emotion_level no último step
          peak_panic_ratio       : fração de agentes que atingiram FSMState.PANIC
          emotion_variance       : variância do emotion_level entre agentes

        Métricas de eficiência:
          collision_events       : número total de colisões no episódio
          mean_speed_ratio       : velocidade média / base_speed 
        """
        n = len(self.agents)
        if n == 0:
            return {}

        evacuated = [a for a in self.agents if a.evacuated]
        evac_times = [a.evacuation_time for a in evacuated if a.evacuation_time is not None]

        emotions = [a.emotion_level for a in self.agents]
        speeds = [a.current_speed for a in self.agents]

        panic_reached_count = sum(
            1 for a in self.agents
            if hasattr(a, "peak_emotion") and a.peak_emotion >= FSM_EVACUATE_TO_PANIC
        )

        peak_emotions = [a.peak_emotion for a in self.agents if hasattr(a, "peak_emotion")]

        hazard_touched_count = sum(
            1 for a in self.agents if getattr(a, "touched_hazard", False)
        )

        def _group_exit_tiles(exits_list, tile_size):
            if not exits_list:
                return {}
            positions = {(e.x, e.y): e for e in exits_list}
            parent = {e: e for e in exits_list}
            def find(x):
                while parent[x] is not x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                parent[find(a)] = find(b)
            for e in exits_list:
                for dx, dy in [(tile_size, 0), (-tile_size, 0), (0, tile_size), (0, -tile_size)]:
                    neighbor = positions.get((e.x + dx, e.y + dy))
                    if neighbor:
                        union(e, neighbor)
            groups = {}
            for e in exits_list:
                root = find(e)
                groups[id(root)] = groups.get(id(root), 0)
            return {e: id(find(e)) for e in exits_list}

        exit_group_map = _group_exit_tiles(self.map_data.exits, self.map_data.tile_size)
        exit_usage: dict = {}
        for a in evacuated:
            if getattr(a, "exit_used", None) is not None:
                group_id = exit_group_map.get(a.exit_used, id(a.exit_used))
                exit_usage[group_id] = exit_usage.get(group_id, 0) + 1
        n_groups = len(set(exit_group_map.values())) if exit_group_map else 0
        if len(exit_usage) >= 2:
            counts = list(exit_usage.values())
            r_util = min(counts) / max(counts) if max(counts) > 0 else 0.0
        elif len(exit_usage) == 1 and n_groups >= 2:
            r_util = 0.0
        else:
            r_util = 1.0 if exit_usage else 0.0

        mean_emotion_at_evac = (
            sum(peak_emotions) / len(peak_emotions) if peak_emotions else 0.0
        )

        return {
            # M1 — evacuação
            "evacuation_rate":      len(evacuated) / n,
            "all_evacuated":        len(evacuated) == n,
            # M2 — tempo
            "mean_evacuation_time": sum(evac_times) / max(1, len(evac_times)) if evac_times else self.max_steps,
            "steps":                self.time,
            # M4 — emoção final
            "mean_emotion_final":   sum(emotions) / n,
            # M4b — emoção no pico
            "mean_peak_emotion":    sum(peak_emotions) / max(1, len(peak_emotions)),
            # M4c — emoção média no momento da evacuação
            "mean_emotion_at_evac": mean_emotion_at_evac,
            # M5 — variância emocional
            "emotion_variance":     sum((e - sum(emotions)/n)**2 for e in emotions) / n,
            # M6 — taxa de pânico
            "panic_rate":           panic_reached_count / n,
            # M8 — velocidade média
            "mean_speed_ratio":     sum(speeds) / max(1, n) / max(1.0, self.agents[0].base_speed),
            # M9 — contato com hazard
            "hazard_contact_rate":  hazard_touched_count / n,
            # M10 — utilização de exits
            "exit_utilization":     r_util,
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