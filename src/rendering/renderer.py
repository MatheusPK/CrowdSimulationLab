import math
import os
import pygame

from core.direction import DIRECTION_VECTORS
from core.fsm_state import FSMState
from simulation_params import FSM_CALM_TO_EVACUATE, FSM_EVACUATE_TO_PANIC


# ---------------------------------------------------------------------------
# Paleta de cores (fallback sem assets)
# ---------------------------------------------------------------------------

_COLOR_CALM     = (100, 180, 255)
_COLOR_EVACUATE = ( 30,  80, 220)
_COLOR_PANIC    = (110,  20, 180)


def _lerp_color(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def emotion_to_color(emotion_level: float) -> tuple[int, int, int]:
    """Mapeia emotion_level [0,1] → RGB entre CALM, EVACUATE e PANIC."""
    lo = FSM_CALM_TO_EVACUATE
    hi = FSM_EVACUATE_TO_PANIC
    if emotion_level <= lo:
        return _lerp_color(_COLOR_CALM, _COLOR_EVACUATE, emotion_level / max(lo, 1e-6))
    t = (emotion_level - lo) / max(hi - lo, 1e-6)
    return _lerp_color(_COLOR_EVACUATE, _COLOR_PANIC, min(t, 1.0))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


class Renderer:
    """
    Renderer pygame com suporte a sprites PNG e escala visual.

    O parâmetro `scale` amplia a janela e todos os elementos desenhados
    sem afetar a simulação física — coordenadas internas permanecem em
    pixels de simulação (tile=8px); só o desenho é multiplicado.

    Assets esperados em assets/:
        tile_floor.png, tile_obstacle.png, tile_hazard.png, agent_sprite.png

    Pressionar SPACE pausa/retoma a renderização.
    """

    def __init__(
        self,
        width,
        height,
        title="Crowd Simulation",
        fps=30,
        draw_grid=False,
        tile_size=8,
        scale=1,
    ):
        self.sim_width  = int(width)
        self.sim_height = int(height)
        self.title      = title
        self.fps        = fps
        self.draw_grid  = draw_grid
        self.tile_size  = tile_size
        self.scale      = max(1, int(scale))

        # Dimensões da janela pygame
        self.width  = self.sim_width  * self.scale
        self.height = self.sim_height * self.scale

        self.screen = None
        self.clock  = pygame.time.Clock()
        self.enabled = True

        self.bg_color        = (245, 245, 230)
        self.grid_color      = (210, 210, 200)
        self.obstacle_color  = ( 35,  35,  35)
        self.exit_color      = (  0, 200,  80)
        self.hazard_color    = (220,  40,  40)
        self.agent_dir_color = (255, 255, 255)

        self._tile_floor    = None
        self._tile_obstacle = None
        self._tile_hazard   = None
        self._agent_sprite  = None
        self._use_assets    = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def initialize(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.title)
        self._load_assets()

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

        if self._use_assets:
            self._draw_floor_tiled(env)
        else:
            self.screen.fill(self.bg_color)

        if self.draw_grid and not self._use_assets:
            self._draw_grid_lines()

        self._draw_hazards(env)
        self._draw_exits(env)
        self._draw_obstacles(env)
        self._draw_agents(env)

        pygame.display.flip()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    def _load_assets(self):
        paths = {
            "floor":    os.path.join(_ASSETS_DIR, "tile_floor.png"),
            "obstacle": os.path.join(_ASSETS_DIR, "tile_obstacle.png"),
            "hazard":   os.path.join(_ASSETS_DIR, "tile_hazard.png"),
            "agent":    os.path.join(_ASSETS_DIR, "agent_sprite.png"),
        }
        try:
            dts = self.tile_size * self.scale  # tile size na janela

            floor = pygame.image.load(paths["floor"]).convert()
            self._tile_floor = pygame.transform.scale(floor, (dts, dts))

            obs = pygame.image.load(paths["obstacle"]).convert()
            self._tile_obstacle = pygame.transform.scale(obs, (dts, dts))

            haz = pygame.image.load(paths["hazard"]).convert()
            self._tile_hazard = pygame.transform.scale(haz, (dts, dts))

            agent = pygame.image.load(paths["agent"]).convert_alpha()
            agent_size = max(4, int(6 * 2 * self.scale))  # AGENT_RADIUS × 2 × scale
            self._agent_sprite = pygame.transform.scale(agent, (agent_size, agent_size))

            self._use_assets = True
            print(f"[Renderer] Assets carregados — scale={self.scale}x")

        except (pygame.error, FileNotFoundError) as e:
            self._use_assets = False
            print(f"[Renderer] Assets não encontrados ({e}) — modo primitivos")

    # ------------------------------------------------------------------
    # Fundo e grid
    # ------------------------------------------------------------------

    def _draw_floor_tiled(self, env):
        dts = self.tile_size * self.scale
        for row in range(env.map_data.rows):
            for col in range(env.map_data.cols):
                self.screen.blit(self._tile_floor, (col * dts, row * dts))

    def _draw_grid_lines(self):
        dts = self.tile_size * self.scale
        for x in range(0, self.width + 1, dts):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height + 1, dts):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.width, y), 1)

    # ------------------------------------------------------------------
    # Elementos do mapa
    # ------------------------------------------------------------------

    def _scaled_rect(self, x, y, w, h):
        s = self.scale
        return pygame.Rect(x * s, y * s, w * s, h * s)

    def _draw_obstacles(self, env):
        dts = self.tile_size * self.scale
        for obstacle in env.map_data.obstacles:
            if self._use_assets:
                x0 = obstacle.x // self.tile_size
                y0 = obstacle.y // self.tile_size
                x1 = (obstacle.x + obstacle.width)  // self.tile_size
                y1 = (obstacle.y + obstacle.height) // self.tile_size
                for r in range(y0, y1):
                    for c in range(x0, x1):
                        self.screen.blit(self._tile_obstacle, (c * dts, r * dts))
            else:
                pygame.draw.rect(self.screen, self.obstacle_color,
                                 self._scaled_rect(obstacle.x, obstacle.y,
                                                   obstacle.width, obstacle.height))

    def _draw_exits(self, env):
        for exit_obj in env.map_data.exits:
            pygame.draw.rect(self.screen, self.exit_color,
                             self._scaled_rect(exit_obj.x, exit_obj.y,
                                               exit_obj.width, exit_obj.height))

    def _draw_hazards(self, env):
        dts = self.tile_size * self.scale
        for hazard in env.map_data.hazards:
            if self._use_assets:
                x0 = hazard.x // self.tile_size
                y0 = hazard.y // self.tile_size
                x1 = (hazard.x + hazard.width)  // self.tile_size
                y1 = (hazard.y + hazard.height) // self.tile_size
                for r in range(y0, y1):
                    for c in range(x0, x1):
                        self.screen.blit(self._tile_hazard, (c * dts, r * dts))
            else:
                pygame.draw.rect(self.screen, self.hazard_color,
                                 self._scaled_rect(hazard.x, hazard.y,
                                                   hazard.width, hazard.height))

    # ------------------------------------------------------------------
    # Agentes
    # ------------------------------------------------------------------

    def _draw_agents(self, env):
        for agent in env.agents:
            if not agent.evacuated:
                self._draw_agent(agent)

    def _draw_agent(self, agent):
        s  = self.scale
        cx = int(agent.x * s)
        cy = int(agent.y * s)
        r  = int(agent.radius * s)
        body_color = emotion_to_color(agent.emotion_level)

        if self._use_assets and self._agent_sprite is not None:
            sprite = self._agent_sprite.copy()
            w, h   = sprite.get_size()
            for px in range(w):
                for py in range(h):
                    rv, gv, bv, a = sprite.get_at((px, py))
                    if a > 0:
                        sprite.set_at((px, py), (
                            int(rv * body_color[0] / 255),
                            int(gv * body_color[1] / 255),
                            int(bv * body_color[2] / 255),
                            a,
                        ))
            self.screen.blit(sprite, (cx - w // 2, cy - h // 2))
        else:
            pygame.draw.circle(self.screen, body_color, (cx, cy), r)
            border_color = _lerp_color(body_color, (0, 0, 0), 0.35)
            pygame.draw.circle(self.screen, border_color, (cx, cy), r, 1)

        self._draw_direction_arrow(agent, r, body_color)

    def _draw_direction_arrow(self, agent, radius, body_color):
        s = self.scale
        dx, dy = DIRECTION_VECTORS[agent.direction]
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return
        dx /= length
        dy /= length

        ax, ay = agent.x * s, agent.y * s
        tip_x  = ax + dx * radius
        tip_y  = ay + dy * radius

        perp_x, perp_y = -dy, dx
        bc_x = ax + dx * (radius * 0.2)
        bc_y = ay + dy * (radius * 0.2)

        points = [
            (int(tip_x),                              int(tip_y)),
            (int(bc_x + perp_x * radius * 0.45),     int(bc_y + perp_y * radius * 0.45)),
            (int(bc_x - perp_x * radius * 0.45),     int(bc_y - perp_y * radius * 0.45)),
        ]
        pygame.draw.polygon(self.screen, self.agent_dir_color, points)