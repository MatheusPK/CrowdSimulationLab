import constants as K
import pygame

class Renderer:
    def __init__(self, environment, cell_size, fps):
        self.environment = environment
        self.cell_size = cell_size
        self.screen = None
        self.clock = pygame.time.Clock()
        self.width = self.environment.map.width * self.cell_size
        self.height = self.environment.map.height * self.cell_size
        self.fps = fps
        self.enabled = True

        self.colors = {
            K.EMPTY:    (255, 255, 255),
            K.OBSTACLE: (0, 0, 0),
            K.AGENT:    (0, 0, 255),
            K.TARGET:   (0, 255, 0),
        }

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

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
                        

    def render(self):
        self.poll_events()

        if not self.enabled:
            return

        self.screen.fill((0, 0, 0))

        for y in range(self.environment.map.height):
            for x in range(self.environment.map.width):
                cell = self.environment.map.map[y][x]
                cell_color = self.colors.get(cell.type, (0, 0, 0))
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        pygame.display.flip()
        self.clock.tick(self.fps)