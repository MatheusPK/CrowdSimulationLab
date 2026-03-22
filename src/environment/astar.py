import heapq
import math

from simulation_params import AGENT_RADIUS as _AGENT_PHYSICAL_RADIUS


class AStarPlanner:
    def __init__(self, map_data, agent_radius, hazard_cost=8.0):
        self.map_data = map_data
        self.rows = map_data.rows
        self.cols = map_data.cols
        self.agent_radius = agent_radius
        self.hazard_cost = hazard_cost

        self.clearance_grid = [
            [self._compute_cell_clearance(r, c) for c in range(self.cols)]
            for r in range(self.rows)
        ]

        # Cache de "célula de approach" por exit tile — calculado uma vez por planner
        self._approach_cache: dict[tuple, tuple | None] = {}

    # ------------------------------------------------------------------
    # Utilitários de mapa
    # ------------------------------------------------------------------

    def is_valid_index(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def cell_type(self, row, col):
        if not self.is_valid_index(row, col):
            return None
        return self.map_data.grid[row][col]

    def cell_center_to_world(self, row, col):
        x = col * self.map_data.tile_size + self.map_data.tile_size / 2.0
        y = row * self.map_data.tile_size + self.map_data.tile_size / 2.0
        return x, y

    def circle_intersects_rect(self, cx, cy, radius, rect):
        closest_x = max(rect.x, min(cx, rect.x + rect.width))
        closest_y = max(rect.y, min(cy, rect.y + rect.height))
        dx = cx - closest_x
        dy = cy - closest_y
        return (dx * dx + dy * dy) <= (radius * radius)

    # ------------------------------------------------------------------
    # Clearance
    # ------------------------------------------------------------------

    def _compute_cell_clearance(self, row, col):
        if not self.is_valid_index(row, col):
            return False
        if self.cell_type(row, col) == "O":
            return False

        x, y = self.cell_center_to_world(row, col)

        if x - self.agent_radius < 0 or x + self.agent_radius > self.map_data.width:
            return False
        if y - self.agent_radius < 0 or y + self.agent_radius > self.map_data.height:
            return False

        for obstacle in self.map_data.obstacles:
            if self.circle_intersects_rect(x, y, self.agent_radius, obstacle):
                return False

        return True

    def has_clearance(self, row, col):
        if not self.is_valid_index(row, col):
            return False
        return self.clearance_grid[row][col]

    # ------------------------------------------------------------------
    # Custo extra de célula
    # ------------------------------------------------------------------

    def extra_cell_cost(self, row, col):
        if self.cell_type(row, col) == "H":
            return self.hazard_cost
        return 0.0

    # ------------------------------------------------------------------
    # Heurística e vizinhos
    # ------------------------------------------------------------------

    def heuristic(self, a, b):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def get_neighbors(self, row, col):
        """
        Retorna vizinhos navegáveis de (row, col) com seus custos de movimento.

        Duas garantias de consistência física:
          1. Clearance: cada vizinho é verificado via has_clearance (baseado no
             agent_radius do planner, incluindo margem de segurança).
          2. Anti corner-cutting diagonal: um movimento diagonal (dr≠0, dc≠0)
             só é permitido se ambas as células ortogonais intermediárias
             (row+dr, col) e (row, col+dc) também tiverem clearance.
             Isso impede que o agente "corte" a quina entre dois obstáculos
             adjacentes — fisicamente impossível para um agente com raio > 0.
        """
        directions = [
            (-1,  0, 1.0),
            (-1,  1, math.sqrt(2)),
            ( 0,  1, 1.0),
            ( 1,  1, math.sqrt(2)),
            ( 1,  0, 1.0),
            ( 1, -1, math.sqrt(2)),
            ( 0, -1, 1.0),
            (-1, -1, math.sqrt(2)),
        ]

        neighbors = []
        for dr, dc, base_cost in directions:
            nr, nc = row + dr, col + dc
            if not self.has_clearance(nr, nc):
                continue
            if dr != 0 and dc != 0:
                if not self.has_clearance(row + dr, col):
                    continue
                if not self.has_clearance(row, col + dc):
                    continue
            neighbors.append((nr, nc, base_cost + self.extra_cell_cost(nr, nc)))

        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # FIX PRINCIPAL: approach cell por exit tile
    # ------------------------------------------------------------------

    def get_exit_approach_cell(self, exit_obj) -> tuple | None:
        """
        Retorna a célula livre mais próxima do exit_obj que efetivamente
        permite ao agente tocar o tile de saída (usando agent_radius do agente
        físico, não do planner — aqui usamos uma margem conservadora).

        Essa célula é usada como goal do A* em vez do centro do exit tile,
        garantindo que o agente realmente evacua ao chegá-la.

        O resultado é cacheado por (exit_obj.x, exit_obj.y).
        """
        key = (exit_obj.x, exit_obj.y)
        if key in self._approach_cache:
            return self._approach_cache[key]

        tile = self.map_data.tile_size

        # Células que formam o exit (pode ter vários tiles E consecutivos)
        exit_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.map_data.grid[r][c] == "E":
                    cx = c * tile + tile / 2.0
                    cy = r * tile + tile / 2.0
                    # Pertence a este exit_obj?
                    if (exit_obj.x <= cx <= exit_obj.x + exit_obj.width and
                            exit_obj.y <= cy <= exit_obj.y + exit_obj.height):
                        exit_cells.append((r, c))

        # Raio físico do agente — importado de simulation_params para manter consistência.
        # Se AGENT_RADIUS mudar, get_exit_approach_cell acompanha automaticamente.
        AGENT_PHYSICAL_RADIUS = _AGENT_PHYSICAL_RADIUS
        best_cell = None
        best_dist = float("inf")

        for (er, ec) in exit_cells:
            for dist in range(0, 8):
                for dr in range(-dist, dist + 1):
                    for dc in range(-dist, dist + 1):
                        if abs(dr) != dist and abs(dc) != dist:
                            continue
                        nr, nc = er + dr, ec + dc
                        if not self.has_clearance(nr, nc):
                            continue
                        # Verifica que o centro desta célula toca o exit_obj
                        cx, cy = self.cell_center_to_world(nr, nc)
                        if self.circle_intersects_rect(
                            cx, cy, AGENT_PHYSICAL_RADIUS, exit_obj
                        ):
                            h = self.heuristic((nr, nc), (er, ec))
                            if h < best_dist:
                                best_dist = h
                                best_cell = (nr, nc)
                if best_cell is not None:
                    break  # encontrou na menor distância possível para este exit tile
            # Não break aqui — pode haver exit tile mais próximo em outro tile E

        self._approach_cache[key] = best_cell
        return best_cell

    # ------------------------------------------------------------------
    # Goal fallback legacy (mantido para compatibilidade)
    # ------------------------------------------------------------------

    def find_nearest_valid_goal(self, goal_cell):
        """
        Fallback legado: usado apenas para o start_cell quando o agente
        é empurrado para fora de uma célula válida.
        NÃO use para exits — use get_exit_approach_cell.
        """
        if self.has_clearance(*goal_cell):
            return goal_cell

        for dist in range(1, 6):
            best = None
            best_h = float("inf")
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    if abs(dr) != dist and abs(dc) != dist:
                        continue
                    candidate = (goal_cell[0] + dr, goal_cell[1] + dc)
                    if self.has_clearance(*candidate):
                        h = self.heuristic(candidate, goal_cell)
                        if h < best_h:
                            best_h = h
                            best = candidate
            if best is not None:
                return best

        return None

    # ------------------------------------------------------------------
    # find_path
    # ------------------------------------------------------------------

    def find_path(self, start_cell, goal_cell, exit_obj=None):
        """
        Planeja caminho de start_cell até goal_cell.

        Se exit_obj for fornecido, usa get_exit_approach_cell para garantir
        que o goal permite ao agente físico tocar o tile de saída.
        Se exit_obj não for fornecido, usa o comportamento legado
        (find_nearest_valid_goal).
        """
        # Resolve o goal real
        if exit_obj is not None:
            adjusted_goal = self.get_exit_approach_cell(exit_obj)
        else:
            adjusted_goal = self.find_nearest_valid_goal(goal_cell)

        if adjusted_goal is None:
            return []
        goal_cell = adjusted_goal

        # Resolve start
        if not self.has_clearance(*start_cell):
            adjusted_start = self.find_nearest_valid_goal(start_cell)
            if adjusted_start is None:
                return []
            start_cell = adjusted_start

        # A* padrão
        open_heap = []
        heapq.heappush(open_heap, (0.0, start_cell))

        came_from = {}
        g_score = {start_cell: 0.0}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            if current == goal_cell:
                return self.reconstruct_path(came_from, current)

            for nr, nc, move_cost in self.get_neighbors(*current):
                neighbor = (nr, nc)
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal_cell)
                    heapq.heappush(open_heap, (f, neighbor))

        return []