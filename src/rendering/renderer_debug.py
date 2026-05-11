import math
import os
import pygame

from core.direction import DIRECTION_VECTORS
from core.fsm_state import FSMState
from simulation_params import FSM_CALM_TO_EVACUATE, FSM_EVACUATE_TO_PANIC

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
    lo = FSM_CALM_TO_EVACUATE
    hi = FSM_EVACUATE_TO_PANIC
    if emotion_level <= lo:
        return _lerp_color(_COLOR_CALM, _COLOR_EVACUATE, emotion_level / max(lo, 1e-6))
    t = (emotion_level - lo) / max(hi - lo, 1e-6)
    return _lerp_color(_COLOR_EVACUATE, _COLOR_PANIC, min(t, 1.0))

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

class Renderer:
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
        self.fps        = 60
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

        # ── Debug overlay ──────────────────────────────────────────────
        # Pressione D para ligar/desligar
        # Pressione B para ligar/desligar apenas BFS arrows
        # Pressione I para ligar/desligar info de estado/reward
        self.debug_mode      = False   # overlay completo
        self.debug_bfs       = True    # seta BFS por agente
        self.debug_info      = True    # estado FSM + reward por agente
        self._debug_font     = None    # inicializado no primeiro uso
        self._agent_rewards  = {}      # id(agent) → reward último step

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
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                    print(f"[Debug] overlay={'ON' if self.debug_mode else 'OFF'}")
                elif event.key == pygame.K_b:
                    self.debug_bfs = not self.debug_bfs
                    print(f"[Debug] BFS arrows={'ON' if self.debug_bfs else 'OFF'}")
                elif event.key == pygame.K_i:
                    self.debug_info = not self.debug_info
                    print(f"[Debug] info overlay={'ON' if self.debug_info else 'OFF'}")
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

        if self.debug_mode:
            if self.debug_bfs:
                self._draw_bfs_arrows(env)
            if self.debug_info:
                self._draw_agent_info(env)

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

    def update_rewards(self, env, reward_list):
        """Chame isso após env.step() para atualizar os rewards no debug overlay."""
        for i, agent in enumerate(env.agents):
            if i < len(reward_list):
                self._agent_rewards[id(agent)] = reward_list[i]

    def _get_font(self, size=9):
        if self._debug_font is None:
            try:
                self._debug_font = pygame.font.SysFont("monospace", size * self.scale)
            except Exception:
                self._debug_font = pygame.font.Font(None, size * self.scale + 4)
        return self._debug_font

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

    # Cores distintas por agente (índice) para o path overlay
    _AGENT_PATH_COLORS = [
        (  0, 220,  80),   # verde
        ( 30, 160, 255),   # azul
        (255, 180,   0),   # amarelo
        (220,  60, 255),   # roxo
        (255,  80,  80),   # vermelho
        (  0, 220, 220),   # ciano
        (255, 140,  40),   # laranja
        (180, 255,  80),   # verde-limão
        (255,  80, 180),   # rosa
        (80,  255, 200),   # turquesa
        (200, 200,  60),   # dourado
        (140,  80, 255),   # lilás
    ]

    def _trace_bfs_path(self, env, start_row, start_col):
        """
        Segue o gradiente dist_map de (start_row, start_col) até um tile E.
        Retorna lista de (row, col) representando o caminho ótimo.
        Limite de passos = rows + cols para evitar loop infinito.
        """
        dm   = env._dist_map
        rows = env.map_data.rows
        cols = env.map_data.cols
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        path    = [(start_row, start_col)]
        visited = {(start_row, start_col)}
        r, c    = start_row, start_col

        for _ in range(rows + cols):
            if env.map_data.grid[r][c] == "E":
                break
            if dm[r][c] == float("inf"):
                break

            best_d  = dm[r][c]
            best_nr, best_nc = r, c

            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in visited:
                    continue
                if dr != 0 and dc != 0:
                    if (env.map_data.grid[r+dr][c] == "O" or
                            env.map_data.grid[r][c+dc] == "O"):
                        continue
                if dm[nr][nc] < best_d:
                    best_d  = dm[nr][nc]
                    best_nr, best_nc = nr, nc

            if (best_nr, best_nc) == (r, c):
                break  # mínimo local

            r, c = best_nr, best_nc
            visited.add((r, c))
            path.append((r, c))

        return path

    def _draw_bfs_arrows(self, env):
        """
        Para cada agente ativo:
          1. Overlay semitransparente do caminho BFS completo até o exit
             — tiles coloridos com opacidade decrescente (mais forte perto do agente)
          2. Seta imediata apontando para o próximo tile ótimo
             Verde = gradiente BFS normal  |  Amarelo = fallback
        """
        s    = self.scale
        tile = env.map_data.tile_size
        dts  = tile * s  # tile size na janela

        # Surface semitransparente para o path overlay (criada por frame)
        path_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for agent_idx, agent in enumerate(env.agents):
            if agent.evacuated:
                continue

            row, col = env.world_to_cell(agent.x, agent.y)
            dm   = env._dist_map
            rows = env.map_data.rows
            cols = env.map_data.cols
            ax   = int(agent.x * s)
            ay   = int(agent.y * s)

            # Cor base deste agente
            base_color = self._AGENT_PATH_COLORS[agent_idx % len(self._AGENT_PATH_COLORS)]

            # ── Path overlay ──────────────────────────────────────────
            path = self._trace_bfs_path(env, row, col)
            path_len = max(1, len(path))

            for step_idx, (pr, pc) in enumerate(path):
                # Opacidade: 140 no primeiro tile → 25 no exit
                # Faz o caminho "pulsar" da posição do agente até a saída
                t       = step_idx / path_len
                alpha   = int(140 * (1.0 - t * 0.82))
                r_c     = int(base_color[0] * (1 - t * 0.4))
                g_c     = int(base_color[1] * (1 - t * 0.4))
                b_c     = int(base_color[2] * (1 - t * 0.4))
                px_rect = pygame.Rect(pc * dts, pr * dts, dts, dts)
                pygame.draw.rect(path_surf, (r_c, g_c, b_c, alpha), px_rect)

                # Pequena seta entre tiles consecutivos do caminho
                if step_idx < len(path) - 1:
                    nr2, nc2 = path[step_idx + 1]
                    fx = int((pc * tile + tile / 2.0) * s)
                    fy = int((pr * tile + tile / 2.0) * s)
                    tx2 = int((nc2 * tile + tile / 2.0) * s)
                    ty2 = int((nr2 * tile + tile / 2.0) * s)
                    arrow_alpha = max(40, int(120 * (1.0 - t)))
                    pygame.draw.line(path_surf,
                                     (base_color[0], base_color[1], base_color[2], arrow_alpha),
                                     (fx, fy), (tx2, ty2), max(1, s))

            # ── Seta imediata (próximo tile) ──────────────────────────
            current_dist = dm[row][col]
            is_fallback  = (current_dist == float("inf") or
                            env.map_data.grid[row][col] == "E" or
                            len(path) <= 1)

            if not is_fallback and len(path) >= 2:
                nr2, nc2 = path[1]
                tx2 = int((nc2 * tile + tile / 2.0) * s)
                ty2 = int((nr2 * tile + tile / 2.0) * s)
                arrow_color = base_color
            else:
                is_fallback = True
                # Aponta para exit mais próximo
                best_ex_pos, best_ex_d = None, float("inf")
                for ex in env.map_data.exits:
                    ex_cx = ex.x + ex.width / 2.0
                    ex_cy = ex.y + ex.height / 2.0
                    d = math.hypot(ex_cx - agent.x, ex_cy - agent.y)
                    if d < best_ex_d:
                        best_ex_d   = d
                        best_ex_pos = (ex_cx, ex_cy)
                if best_ex_pos:
                    tx2 = int(best_ex_pos[0] * s)
                    ty2 = int(best_ex_pos[1] * s)
                else:
                    continue
                arrow_color = (255, 220, 0)  # amarelo = fallback

            dx2 = tx2 - ax
            dy2 = ty2 - ay
            length = math.hypot(dx2, dy2)
            if length < 1:
                continue
            ndx, ndy  = dx2 / length, dy2 / length
            arrow_len = int(agent.radius * 2.5 * s)
            tip_x     = ax + int(ndx * arrow_len)
            tip_y     = ay + int(ndy * arrow_len)

            pygame.draw.line(self.screen, arrow_color,
                             (ax, ay), (tip_x, tip_y), max(2, s))
            perp_x, perp_y = -ndy, ndx
            head = max(3, int(arrow_len * 0.35))
            p1 = (tip_x - int(ndx * head + perp_x * head * 0.5),
                  tip_y - int(ndy * head + perp_y * head * 0.5))
            p2 = (tip_x - int(ndx * head - perp_x * head * 0.5),
                  tip_y - int(ndy * head - perp_y * head * 0.5))
            pygame.draw.polygon(self.screen, arrow_color, [(tip_x, tip_y), p1, p2])

        # Aplica o path overlay na tela principal
        self.screen.blit(path_surf, (0, 0))

    def _draw_agent_info(self, env):
        """
        Exibe estado FSM e reward do último step sobre cada agente.
        """
        font = self._get_font(9)
        s    = self.scale
        STATE_LABELS = {0: "C", 1: "E", 2: "P"}  # Calm, Evacuate, Panic
        STATE_COLORS = {0: (120,200,255), 1: (50,120,255), 2: (200,50,255)}

        for agent in env.agents:
            if agent.evacuated:
                continue
            ax = int(agent.x * s)
            ay = int(agent.y * s)
            r  = int(agent.radius * s)

            state_val   = agent.state.value if hasattr(agent.state, "value") else 0
            state_label = STATE_LABELS.get(state_val, "?")
            state_color = STATE_COLORS.get(state_val, (255,255,255))
            reward      = self._agent_rewards.get(id(agent), None)

            # Linha 1: estado FSM + emoção
            em_pct = int(agent.emotion_level * 100)
            line1 = f"{state_label} {em_pct}%"
            surf1 = font.render(line1, True, state_color, (0, 0, 0))
            self.screen.blit(surf1, (ax - surf1.get_width() // 2,
                                     ay - r - surf1.get_height() - 2))

            # Linha 2: reward (se disponível)
            if reward is not None:
                rew_color = (100, 255, 100) if reward >= 0 else (255, 100, 100)
                line2 = f"{reward:+.0f}"
                surf2 = font.render(line2, True, rew_color, (0, 0, 0))
                self.screen.blit(surf2, (ax - surf2.get_width() // 2,
                                         ay - r - surf1.get_height() - surf2.get_height() - 4))

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