import heapq
import math

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

    def extra_cell_cost(self, row, col):
        if self.cell_type(row, col) == "H":
            return self.hazard_cost
        return 0.0

    def heuristic(self, a, b):
        dx = abs(a[1] - b[1])
        dy = abs(a[0] - b[0])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    def get_neighbors(self, row, col):
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
            nr = row + dr
            nc = col + dc

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

    def find_nearest_valid_goal(self, goal_cell):
        """
        Se goal_cell não tem clearance (ex: exit tile na borda do mapa),
        busca em espiral a célula válida mais próxima com até 5 tiles de distância.
        Retorna None se não encontrar nenhuma.
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

    def find_path(self, start_cell, goal_cell):
        # Ajusta goal para célula válida mais próxima (ex: exit tile na borda)
        adjusted_goal = self.find_nearest_valid_goal(goal_cell)
        if adjusted_goal is None:
            return []
        goal_cell = adjusted_goal

        if not self.has_clearance(*start_cell):
            # Tenta recuperar o start também, caso agente tenha sido empurrado
            adjusted_start = self.find_nearest_valid_goal(start_cell)
            if adjusted_start is None:
                return []
            start_cell = adjusted_start

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