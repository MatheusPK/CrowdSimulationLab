import math

from environment.astar import AStarPlanner

# Quantos steps consecutivos sem mudar de célula antes de replanear
_STUCK_THRESHOLD = 5


class AStarPolicy:
    def __init__(self, hazard_cost=8.0):
        self.hazard_cost = hazard_cost
        self.planner_cache = {}
        self.path_cache = {}

        self.direction_to_action = {
            (-1,  0): 0,
            (-1,  1): 1,
            ( 0,  1): 2,
            ( 1,  1): 3,
            ( 1,  0): 4,
            ( 1, -1): 5,
            ( 0, -1): 6,
            (-1, -1): 7,
        }

    def _get_planner(self, env, agent):
        planner_radius = round(agent.radius + agent.planner_radius_margin, 3)
        key = (id(env.map_data), planner_radius, self.hazard_cost)

        if key not in self.planner_cache:
            self.planner_cache[key] = AStarPlanner(
                map_data=env.map_data,
                agent_radius=planner_radius,
                hazard_cost=self.hazard_cost,
            )

        return self.planner_cache[key]

    def _is_stuck(self, cached, start_cell):
        """
        Detecta se o agente está preso: não mudou de célula por _STUCK_THRESHOLD steps.
        Atualiza o contador no cache e retorna True quando o limite é atingido.
        """
        if cached is None:
            return False

        if cached.get("last_cell") == start_cell:
            cached["stuck_count"] = cached.get("stuck_count", 0) + 1
        else:
            cached["stuck_count"] = 0
            cached["last_cell"] = start_cell

        return cached["stuck_count"] >= _STUCK_THRESHOLD

    def _find_closest_path_index(self, path, start_cell):
        """
        Encontra o índice do ponto no path mais próximo da célula atual do agente
        usando distância de Manhattan. Isso substitui o while loop frágil do
        código original, que esgotava o índice quando o agente escorregava de tile.
        """
        best_idx = 0
        best_dist = float("inf")
        for i, cell in enumerate(path):
            d = abs(cell[0] - start_cell[0]) + abs(cell[1] - start_cell[1])
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def choose_action(self, env, agent, exit_obj):
        if agent.evacuated:
            return None

        planner = self._get_planner(env, agent)

        start_cell = env.world_to_cell(agent.x, agent.y)
        exit_cx = exit_obj.x + exit_obj.width / 2.0
        exit_cy = exit_obj.y + exit_obj.height / 2.0
        goal_cell = env.world_to_cell(exit_cx, exit_cy)

        agent_key = id(agent)
        cached = self.path_cache.get(agent_key)

        stuck = self._is_stuck(cached, start_cell)

        need_replan = (
            cached is None
            or cached["goal"] != goal_cell
            or stuck
        )

        if need_replan:
            path = planner.find_path(start_cell, goal_cell)

            if not path or len(path) < 2:
                return self.fallback_action_towards_exit(agent, exit_obj)

            self.path_cache[agent_key] = {
                "goal": goal_cell,
                "path": path,
                "last_cell": start_cell,
                "stuck_count": 0,
            }
            cached = self.path_cache[agent_key]

        path = cached["path"]

        # Encontra onde o agente está no path por proximidade (robusto a deslizamento)
        closest_idx = self._find_closest_path_index(path, start_cell)

        # Pega o próximo waypoint à frente
        next_idx = min(closest_idx + 1, len(path) - 1)

        # Se já chegamos no fim do path, usa fallback
        if next_idx == closest_idx:
            return self.fallback_action_towards_exit(agent, exit_obj)

        next_cell = path[next_idx]

        dr = max(-1, min(1, next_cell[0] - start_cell[0]))
        dc = max(-1, min(1, next_cell[1] - start_cell[1]))

        return self.direction_to_action.get(
            (dr, dc),
            self.fallback_action_towards_exit(agent, exit_obj),
        )

    def fallback_action_towards_exit(self, agent, exit_obj):
        exit_cx = exit_obj.x + exit_obj.width / 2.0
        exit_cy = exit_obj.y + exit_obj.height / 2.0

        dx = exit_cx - agent.x
        dy = exit_cy - agent.y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0

        angle = math.atan2(dy, dx)

        directions = [
            (0, -math.pi / 2),
            (1, -math.pi / 4),
            (2,  0.0),
            (3,  math.pi / 4),
            (4,  math.pi / 2),
            (5,  3 * math.pi / 4),
            (6,  math.pi),
            (7, -3 * math.pi / 4),
        ]

        best_action = 0
        best_diff = float("inf")

        for action, direction_angle in directions:
            diff = abs(self.angle_diff(angle, direction_angle))
            if diff < best_diff:
                best_diff = diff
                best_action = action

        return best_action

    def angle_diff(self, a, b):
        d = a - b
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        return d