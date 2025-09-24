import constants as K
import pygame

class Renderer:
    def __init__(self, environment, cell_size=60, fps=30):
        self.map = environment.map
        self.width = environment.map.width
        self.height = environment.map.height
        self.cell_size = cell_size
        self.screen = None
        self.clock = None
        self.fps = fps
        self.colors = {
            K.EMPTY:    (255, 255, 255),
            K.OBSTACLE: (0, 0, 0),
            K.AGENT:    (0, 0, 255),
            K.TARGET:   (0, 255, 0),
        }

    def initialize(self):
        pygame.init()
        width = self.width * self.cell_size
        height = self.height * self.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Environment Renderer")
        self.clock = pygame.time.Clock()

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def render(self):
        self.poll_events()

        self.screen.fill((0, 0, 0))

        for y in range(self.height):
            for x in range(self.width):
                cell = self.map.map[y][x]
                cell_color = self.colors.get(cell.type, (0, 0, 0))
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()