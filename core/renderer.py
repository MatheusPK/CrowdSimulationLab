import pygame
import math

class Renderer:
    def __init__(self, world_width_m, world_height_m, agents, obstacles, exits, meters_to_px, fps, draw_grid):
        self.world_width_m = float(world_width_m)
        self.world_height_m = float(world_height_m)

        self.agents = agents
        self.obstacles = obstacles
        self.exits = exits

        self.meters_to_px = int(meters_to_px)
        self.fps = fps
        self.draw_grid = draw_grid

        self.screen = None
        self.clock = pygame.time.Clock()
        self.enabled = True

        self.width_px = int(self.world_width_m * self.meters_to_px)
        self.height_px = int(self.world_height_m * self.meters_to_px)

        self.bg_color = (255, 255, 255)
        self.grid_color = (220, 220, 220)

        self.obstacle_color = (0, 0, 0)
        self.obstacle_border = (80, 80, 80)

        self.agent_color = (0, 0, 255)
        self.agent_border = (255, 255, 255)
        self.agent_done_color = (180, 180, 180)
        self.agent_done_border = (120, 120, 120)

        self.exit_color = (0, 255, 0)

    # MARK: Pygame Lifecycle
    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        pygame.display.set_caption("Crowd Simulation (meters)")

    def close(self):
        pygame.quit()

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.enabled = not self.enabled
                    if not self.enabled:
                        self.screen.fill((0, 0, 0))
                        pygame.display.flip()

    # MARK: Coordinate Conversion
    def _world_to_screen(self, x_m, y_m):
        return int(x_m * self.meters_to_px), int(y_m * self.meters_to_px)

    # MARK: Drawing Functions
    def _draw_grid_lines(self, step_m=1.0):
        step_px = int(step_m * self.meters_to_px)

        x = 0
        while x <= self.width_px:
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height_px), 1)
            x += step_px

        y = 0
        while y <= self.height_px:
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width_px, y), 1)
            y += step_px

    def _draw_obstacles(self):
        for ob in self.obstacles:
            x = int(ob.x * self.meters_to_px)
            y = int(ob.y * self.meters_to_px)
            w = int(ob.width * self.meters_to_px)
            h = int(ob.height * self.meters_to_px)

            pygame.draw.rect(self.screen, self.obstacle_color, (x, y, w, h))
            pygame.draw.rect(self.screen, self.obstacle_border, (x, y, w, h), 1)

    def _draw_exits(self):
        for ex in self.exits:
            x = int(ex.x * self.meters_to_px)
            y = int(ex.y * self.meters_to_px)
            w = int(ex.width * self.meters_to_px)
            h = int(ex.height * self.meters_to_px)

            pygame.draw.rect(self.screen, self.exit_color, (x, y, w, h))
            pygame.draw.rect(self.screen, (0, 120, 0), (x, y, w, h), 2)

    def _draw_agents(self):
        for agent in self.agents:
            cx, cy = self._world_to_screen(agent.x, agent.y)
            r_m = getattr(agent, "radius", 0.20)
            r_px = max(2, int(r_m * self.meters_to_px))

            is_done = getattr(agent, "done", False)

            fill = self.agent_done_color if is_done else self.agent_color
            border = self.agent_done_border if is_done else self.agent_border

            pygame.draw.circle(self.screen, fill, (cx, cy), r_px)
            pygame.draw.circle(self.screen, border, (cx, cy), r_px, 1)

    def render(self):
        self.poll_events()

        if not self.enabled:
            return

        self.screen.fill(self.bg_color)

        self._draw_obstacles()
        self._draw_exits()
        self._draw_agents()

        if self.draw_grid:
            self._draw_grid_lines(step_m=1.0)

        pygame.display.flip()
        self.clock.tick(self.fps)