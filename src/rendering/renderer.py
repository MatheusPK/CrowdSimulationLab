import math
import os
import pygame

from core.direction import DIRECTION_VECTORS
from core.fsm_state import FSMState
from simulation_params import FSM_CALM_TO_EVACUATE, FSM_EVACUATE_TO_PANIC


# ---------------------------------------------------------------------------
# Paleta de cores (fallback quando assets não estão disponíveis)
# ---------------------------------------------------------------------------

_COLOR_CALM     = (100, 180, 255)   # azul claro  — calmo
_COLOR_EVACUATE = ( 30,  80, 220)   # azul médio  — alerta
_COLOR_PANIC    = (110,  20, 180)   # roxo escuro — pânico


def _lerp_color(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def emotion_to_color(emotion_level: float) -> tuple[int, int, int]:
    """
    Mapeia emotion_level [0, 1] → cor RGB interpolada entre os três
    pontos da FSM: CALM (azul claro) → EVACUATE (azul médio) → PANIC (roxo).
    """
    lo = FSM_CALM_TO_EVACUATE   # 0.35
    hi = FSM_EVACUATE_TO_PANIC  # 0.72
    if emotion_level <= lo:
        t = emotion_level / max(lo, 1e-6)
        return _lerp_color(_COLOR_CALM, _COLOR_EVACUATE, t)
    t = (emotion_level - lo) / max(hi - lo, 1e-6)
    return _lerp_color(_COLOR_EVACUATE, _COLOR_PANIC, min(t, 1.0))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

# Diretório de assets relativo ao arquivo renderer.py
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


class Renderer:
    """
    Renderer pygame com suporte a sprites/assets PNG.

    Assets esperados em assets/ (relativo à raiz do projeto):
        tile_floor.png    — espaço livre (fundo do mapa)
        tile_obstacle.png — paredes e obstáculos
        tile_hazard.png   — tiles de hazard
        agent_sprite.png  — sprite do agente (branco puro = área colorida por emoção)

    Se algum asset não for encontrado, o renderer usa os primitivos pygame
    originais como fallback — sem crash.

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
    ):
        self.width     = int(width)
        self.height    = int(height)
        self.title     = title
        self.fps       = fps
        self.draw_grid = draw_grid
        self.tile_size = tile_size

        self.screen = None
        self.clock  = pygame.time.Clock()
        self.enabled = True

        # Cores de fallback (usadas quando assets não estão disponíveis)
        self.bg_color       = (245, 245, 230)
        self.grid_color     = (210, 210, 200)
        self.obstacle_color = (35,  35,  35)
        self.exit_color     = (0,  200,  80)
        self.hazard_color   = (220, 40,  40)
        self.agent_dir_color = (255, 255, 255)

        # Assets carregados em initialize()
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
    # Carregamento de assets
    # ------------------------------------------------------------------

    def _load_assets(self):
        """
        Tenta carregar os 4 assets PNG.
        Se qualquer um falhar, desativa o modo sprite e usa fallback de cores.
        """
        paths = {
            "floor":    os.path.join(_ASSETS_DIR, "tile_floor.png"),
            "obstacle": os.path.join(_ASSETS_DIR, "tile_obstacle.png"),
            "hazard":   os.path.join(_ASSETS_DIR, "tile_hazard.png"),
            "agent":    os.path.join(_ASSETS_DIR, "agent_sprite.png"),
        }

        try:
            ts = self.tile_size

            floor = pygame.image.load(paths["floor"]).convert()
            self._tile_floor = pygame.transform.scale(floor, (ts, ts))

            obs = pygame.image.load(paths["obstacle"]).convert()
            self._tile_obstacle = pygame.transform.scale(obs, (ts, ts))

            haz = pygame.image.load(paths["hazard"]).convert()
            self._tile_hazard = pygame.transform.scale(haz, (ts, ts))

            agent = pygame.image.load(paths["agent"]).convert_alpha()
            # Escala para diâmetro do agente (raio×2), com margem de 1px
            agent_size = max(4, int(6 * 2))  # AGENT_RADIUS × 2 = 12px
            self._agent_sprite = pygame.transform.scale(agent, (agent_size, agent_size))

            self._use_assets = True
            print("[Renderer] Assets carregados — modo sprite ativo")

        except (pygame.error, FileNotFoundError) as e:
            self._use_assets = False
            print(f"[Renderer] Assets não encontrados ({e}) — modo primitivos")

    # ------------------------------------------------------------------
    # Desenho do fundo com tile de chão
    # ------------------------------------------------------------------

    def _draw_floor_tiled(self, env):
        """Preenche o fundo com o tile de chão repetido."""
        ts = self.tile_size
        for row in range(env.map_data.rows):
            for col in range(env.map_data.cols):
                self.screen.blit(self._tile_floor, (col * ts, row * ts))

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
        ts = self.tile_size
        for obstacle in env.map_data.obstacles:
            if self._use_assets:
                # Preenche a área do obstáculo tile a tile
                x0 = obstacle.x // ts
                y0 = obstacle.y // ts
                x1 = (obstacle.x + obstacle.width)  // ts
                y1 = (obstacle.y + obstacle.height) // ts
                for r in range(y0, y1):
                    for c in range(x0, x1):
                        self.screen.blit(self._tile_obstacle, (c * ts, r * ts))
            else:
                pygame.draw.rect(
                    self.screen, self.obstacle_color,
                    pygame.Rect(obstacle.x, obstacle.y, obstacle.width, obstacle.height),
                )

    def _draw_exits(self, env):
        for exit_obj in env.map_data.exits:
            pygame.draw.rect(
                self.screen, self.exit_color,
                pygame.Rect(exit_obj.x, exit_obj.y, exit_obj.width, exit_obj.height),
            )

    def _draw_hazards(self, env):
        ts = self.tile_size
        for hazard in env.map_data.hazards:
            if self._use_assets:
                x0 = hazard.x // ts
                y0 = hazard.y // ts
                x1 = (hazard.x + hazard.width)  // ts
                y1 = (hazard.y + hazard.height) // ts
                for r in range(y0, y1):
                    for c in range(x0, x1):
                        self.screen.blit(self._tile_hazard, (c * ts, r * ts))
            else:
                pygame.draw.rect(
                    self.screen, self.hazard_color,
                    pygame.Rect(hazard.x, hazard.y, hazard.width, hazard.height),
                )

    # ------------------------------------------------------------------
    # Agentes
    # ------------------------------------------------------------------

    def _draw_agents(self, env):
        for agent in env.agents:
            if agent.evacuated:
                continue
            self._draw_agent(agent)

    def _draw_agent(self, agent):
        cx, cy = int(agent.x), int(agent.y)
        radius  = int(agent.radius)
        body_color = emotion_to_color(agent.emotion_level)

        if self._use_assets and self._agent_sprite is not None:
            # Coloriza o sprite: pixels brancos recebem a cor da emoção
            sprite = self._agent_sprite.copy()
            w, h   = sprite.get_size()
            for px in range(w):
                for py in range(h):
                    r, g, b, a = sprite.get_at((px, py))
                    if a > 0:
                        # Mistura a cor do pixel com a cor de emoção
                        blended = (
                            int(r * body_color[0] / 255),
                            int(g * body_color[1] / 255),
                            int(b * body_color[2] / 255),
                            a,
                        )
                        sprite.set_at((px, py), blended)
            self.screen.blit(sprite, (cx - w // 2, cy - h // 2))
        else:
            # Fallback: primitivos pygame
            pygame.draw.circle(self.screen, body_color, (cx, cy), radius)
            border_color = _lerp_color(body_color, (0, 0, 0), 0.35)
            pygame.draw.circle(self.screen, border_color, (cx, cy), radius, 1)

        # Seta de direção (sempre desenhada, independente do modo)
        self._draw_direction_arrow(agent, radius, body_color)

    def _draw_direction_arrow(self, agent, radius, body_color):
        dx, dy = DIRECTION_VECTORS[agent.direction]
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return
        dx /= length
        dy /= length

        tip_x = agent.x + dx * radius
        tip_y = agent.y + dy * radius

        perp_x, perp_y = -dy, dx
        bc_x = agent.x + dx * (radius * 0.2)
        bc_y = agent.y + dy * (radius * 0.2)

        bl_x = bc_x + perp_x * (radius * 0.45)
        bl_y = bc_y + perp_y * (radius * 0.45)
        br_x = bc_x - perp_x * (radius * 0.45)
        br_y = bc_y - perp_y * (radius * 0.45)

        points = [
            (int(tip_x), int(tip_y)),
            (int(bl_x),  int(bl_y)),
            (int(br_x),  int(br_y)),
        ]
        pygame.draw.polygon(self.screen, self.agent_dir_color, points)
