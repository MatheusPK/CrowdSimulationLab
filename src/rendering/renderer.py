import math
import pygame

from core.direction import DIRECTION_VECTORS
from core.fsm_state import FSMState
from simulation_params import FSM_CALM_TO_EVACUATE, FSM_EVACUATE_TO_PANIC


# Paleta de cores por estado emocional (RGB)
# Interpolação suave entre os três pontos conforme emotion_level vai de 0 → 1
_COLOR_CALM    = (100, 180, 255)   # azul claro  — calmo
_COLOR_EVACUATE = (30,  80, 220)   # azul médio  — alerta
_COLOR_PANIC   = (110,  20, 180)   # roxo escuro — pânico


def _lerp_color(c1, c2, t):
    """Interpolação linear entre duas cores RGB. t ∈ [0, 1]."""
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def emotion_to_color(emotion_level: float) -> tuple[int, int, int]:
    """
    Mapeia emotion_level [0, 1] para uma cor RGB com interpolação suave
    entre os três pontos da FSM:

        0.0                    → CALM     (azul claro)
        FSM_CALM_TO_EVACUATE   → EVACUATE (azul médio)   = 0.35
        FSM_EVACUATE_TO_PANIC  → PANIC    (roxo escuro)  = 0.72
        1.0                    → PANIC    (roxo escuro)
    """
    lo = FSM_CALM_TO_EVACUATE    # 0.35
    hi = FSM_EVACUATE_TO_PANIC   # 0.72
    if emotion_level <= lo:
        t = emotion_level / max(lo, 1e-6)
        return _lerp_color(_COLOR_CALM, _COLOR_EVACUATE, t)
    else:
        t = (emotion_level - lo) / max(hi - lo, 1e-6)
        t = min(t, 1.0)
        return _lerp_color(_COLOR_EVACUATE, _COLOR_PANIC, t)


class Renderer:
    def __init__(
        self,
        width,
        height,
        title="Crowd Simulation",
        fps=30,
        draw_grid=False,
        tile_size=8,
    ):
        self.width = int(width)
        self.height = int(height)
        self.title = title
        self.fps = fps
        self.draw_grid = draw_grid
        self.tile_size = tile_size

        self.screen = None
        self.clock = pygame.time.Clock()
        self.enabled = True

        self.bg_color = (255, 255, 255)
        self.grid_color = (220, 220, 220)

        self.obstacle_color = (30, 30, 30)
        self.exit_color = (0, 200, 80)
        self.hazard_color = (220, 40, 40)

        # A cor do agente agora é calculada por emoção — este atributo
        # serve apenas de fallback se alguém chamar o renderer sem FSM.
        self.agent_dir_color = (255, 255, 255)

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)

    def close(self):
        pygame.quit()

    def poll_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.enabled = not self.enabled
                    if not self.enabled:
                        self.screen.fill((0, 0, 0))
                        pygame.display.flip()
        return True

    def render(self, env):
        if not self.enabled:
            return

        self.screen.fill(self.bg_color)

        if self.draw_grid:
            self._draw_grid_lines()

        self._draw_hazards(env)
        self._draw_exits(env)
        self._draw_obstacles(env)
        self._draw_agents(env)

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Elementos do mapa
    # ------------------------------------------------------------------

    def _draw_grid_lines(self):
        x = 0
        while x <= self.width:
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height), 1)
            x += self.tile_size
        y = 0
        while y <= self.height:
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y), 1)
            y += self.tile_size

    def _draw_obstacles(self, env):
        for obstacle in env.map_data.obstacles:
            pygame.draw.rect(
                self.screen,
                self.obstacle_color,
                pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height),
            )

    def _draw_exits(self, env):
        for exit_obj in env.map_data.exits:
            pygame.draw.rect(
                self.screen,
                self.exit_color,
                pygame.Rect(exit_obj.x, exit_obj.y, exit_obj.width, exit_obj.height),
            )

    def _draw_hazards(self, env):
        for hazard in env.map_data.hazards:
            pygame.draw.rect(
                self.screen,
                self.hazard_color,
                pygame.Rect(hazard.x, hazard.y, hazard.width, hazard.height),
            )

    # ------------------------------------------------------------------
    # Agentes
    # ------------------------------------------------------------------

    def _draw_agents(self, env):
        for agent in env.agents:
            if agent.evacuated:
                continue  # agente saiu do ambiente — não renderizar
            self._draw_agent(agent)

    def _draw_agent(self, agent):
        cx, cy = int(agent.x), int(agent.y)
        radius = int(agent.radius)

        # Cor do corpo baseada no nível de emoção — interpolação contínua
        body_color = emotion_to_color(agent.emotion_level)
        pygame.draw.circle(self.screen, body_color, (cx, cy), radius)

        # Borda fina mais escura para legibilidade sobre fundo claro
        border_color = _lerp_color(body_color, (0, 0, 0), 0.35)
        pygame.draw.circle(self.screen, border_color, (cx, cy), radius, 1)

        # Seta de direção (triângulo branco)
        dx, dy = DIRECTION_VECTORS[agent.direction]
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx /= length
            dy /= length

        tip_x = agent.x + dx * radius
        tip_y = agent.y + dy * radius

        perp_x = -dy
        perp_y = dx

        base_center_x = agent.x + dx * (radius * 0.2)
        base_center_y = agent.y + dy * (radius * 0.2)

        base_left_x  = base_center_x + perp_x * (radius * 0.45)
        base_left_y  = base_center_y + perp_y * (radius * 0.45)
        base_right_x = base_center_x - perp_x * (radius * 0.45)
        base_right_y = base_center_y - perp_y * (radius * 0.45)

        points = [
            (int(tip_x),        int(tip_y)),
            (int(base_left_x),  int(base_left_y)),
            (int(base_right_x), int(base_right_y)),
        ]
        pygame.draw.polygon(self.screen, self.agent_dir_color, points)