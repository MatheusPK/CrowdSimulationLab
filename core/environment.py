import math
import constants as K

class Environment:
    def __init__(self, agents, obstacles, exits, dt):
        self.agents = agents
        self.obstacles = obstacles
        self.exits = exits
        self.dt = dt
        self._spawn_by_id = {a.id: (a.x, a.y) for a in self.agents}

        s = math.sqrt(2) / 2
        self.actions = {
            K.N:  (0.0, -1.0),
            K.S:  (0.0,  1.0),
            K.E:  (1.0,  0.0),
            K.W:  (-1.0, 0.0),
            K.NE: (s,   -s),
            K.NW: (-s,  -s),
            K.SE: (s,    s),
            K.SW: (-s,   s)
        }

        self.A = 100.0
        self.B = 0.08
        self.k = 8.0e4
        self.tau = 0.5

    # Environemnt Functions
    def reset(self):
        for agent in self.agents:
            x0, y0 = self._spawn_by_id.get(agent.id, (agent.x, agent.y))
            agent.x = float(x0)
            agent.y = float(y0)

            agent.vx = 0.0
            agent.vy = 0.0

            # leapfrog internal state
            agent.vx_half = 0.0
            agent.vy_half = 0.0
            agent._lf_initialized = False

            agent.done = False

    def get_microscopic_observation(self, agent):
        return (agent.x, agent.y, agent.vx, agent.vy)

    def step(self, actions_by_id):
        transitions = {}

        for agent in self.agents:
            if agent.done:
                continue

            prev_state = self.get_microscopic_observation(agent)
            action = actions_by_id[agent.id]

            Fx, Fy = self._compute_forces(agent, action)

            ax = Fx / agent.mass
            ay = Fy / agent.mass

            self.leapfrog(agent, ax, ay, self.dt)

            done = self._is_done(agent)
            if done:
                agent.done = True
                agent.vx = 0.0
                agent.vy = 0.0
                agent.vx_half = 0.0
                agent.vy_half = 0.0

            reward = 0.0 if done else -0.1
            
            next_state = self.get_microscopic_observation(agent)

            transitions[agent.id] = (
                prev_state,
                reward,
                next_state,
                done
            )

        return transitions
    
    # MARK: Helper Functions
    def _compute_forces(self, agent, action):
        Fx, Fy = 0.0, 0.0

        f_self_driven = self.self_driven_force(agent, action)
        Fx += f_self_driven[0]
        Fy += f_self_driven[1]

        f_viscous = self.viscous_force(agent)
        Fx += f_viscous[0]
        Fy += f_viscous[1]

        for other_agent in self.agents:
            if other_agent.id == agent.id or other_agent.done:
                continue
            f_aa = self.agent_agent_force(agent, other_agent)
            Fx += f_aa[0]
            Fy += f_aa[1]

        for obstacle in self.obstacles:
            f_ao = self.agent_obstacle_force(agent, obstacle)
            Fx += f_ao[0]
            Fy += f_ao[1]

        return Fx, Fy

    def leapfrog(self, agent, ax, ay, dt):
        if not getattr(agent, "_lf_initialized", False):
            agent.vx_half = agent.vx + 0.5 * ax * dt
            agent.vy_half = agent.vy + 0.5 * ay * dt
            agent._lf_initialized = True
        else:
            agent.vx_half += ax * dt
            agent.vy_half += ay * dt

        agent.x += agent.vx_half * dt
        agent.y += agent.vy_half * dt

        agent.vx = agent.vx_half - 0.5 * ax * dt
        agent.vy = agent.vy_half - 0.5 * ay * dt

    def _closest_point_on_rect(self, px, py, obstacle):
        left   = obstacle.x
        right  = obstacle.x + obstacle.width
        top    = obstacle.y
        bottom = obstacle.y + obstacle.height

        cx = max(left,  min(px, right))
        cy = max(top,   min(py, bottom))
        return cx, cy
    
    def _is_done(self, agent):
        ax = agent.x
        ay = agent.y
        r = agent.radius

        for ex in self.exits:
            cx, cy = self._closest_point_on_rect(ax, ay, ex)
            dx = ax - cx
            dy = ay - cy
            if (dx * dx + dy * dy) <= (r * r):
                return True
        return False

    def g(self, x):
        return max(0.0, x)

    # MARK: Forces
    def self_driven_force(self, agent, action):
        mass = agent.mass
        Vd = agent.v_desired
        direction = self.actions[action]

        Fsfx = mass / self.tau * (Vd * direction[0])
        Fsfy = mass / self.tau * (Vd * direction[1])
        return (Fsfx, Fsfy)
    
    def viscous_force(self, agent):
        mass = agent.mass
        Fvx = -mass / self.tau * agent.vx
        Fvy = -mass / self.tau * agent.vy
        return (Fvx, Fvy)

    # MARK: Agent-Agent Forces
    def agent_agent_force(self, agentA, agentB):
        f_avoidance = self.avoidance_force(agentA, agentB)
        f_compression = self.compression_force(agentA, agentB)
        f_friction = self.friction_force(agentA, agentB)

        fx = f_avoidance[0] + f_compression[0] + f_friction[0]
        fy = f_avoidance[1] + f_compression[1] + f_friction[1]
        
        return (fx, fy)

    def avoidance_force(self, agentA, agentB):
        dij = agentA.radius + agentB.radius
        dx = agentA.x - agentB.x
        dy = agentA.y - agentB.y
        rij = math.hypot(dx, dy)

        if rij == 0.0:
            return (0.0, 0.0)

        rijHatX = dx / rij
        rijHatY = dy / rij

        fx = self.A * math.exp((dij - rij) / self.B) * rijHatX
        fy = self.A * math.exp((dij - rij) / self.B) * rijHatY

        return (fx, fy)
    
    def compression_force(self, agentA, agentB):
        dij = agentA.radius + agentB.radius
        dx = agentA.x - agentB.x
        dy = agentA.y - agentB.y
        rij = math.hypot(dx, dy)

        if rij == 0.0:
            return (0.0, 0.0)

        rijHatX = dx / rij
        rijHatY = dy / rij

        fx = self.k * self.g(dij - rij) * rijHatX
        fy = self.k * self.g(dij - rij) * rijHatY

        return (fx, fy)

    def friction_force(self, agentA, agentB):
        dij = agentA.radius + agentB.radius
        dx = agentA.x - agentB.x
        dy = agentA.y - agentB.y
        rij = math.hypot(dx, dy)

        if rij == 0.0:
            return (0.0, 0.0)

        rijHatX = dx / rij
        rijHatY = dy / rij

        tijHatX = -rijHatY
        tijHatY =  rijHatX

        dvx = agentB.vx - agentA.vx
        dvy = agentB.vy - agentA.vy
        uij = dvx * tijHatX + dvy * tijHatY

        fx = self.k * self.g(dij - rij) * uij * tijHatX
        fy = self.k * self.g(dij - rij) * uij * tijHatY

        return (fx, fy)
    
    # MARK: Agent-Obstacle Forces
    def agent_obstacle_force(self, agent, obstacle):
        f_avoidance = self.obstacle_avoidance_force(agent, obstacle)
        f_compression = self.obstacle_compression_force(agent, obstacle)
        f_friction = self.obstacle_friction_force(agent, obstacle)

        fx = f_avoidance[0] + f_compression[0] + f_friction[0]
        fy = f_avoidance[1] + f_compression[1] + f_friction[1]
        return (fx, fy)

    def obstacle_avoidance_force(self, agent, obstacle):
        diw = agent.radius
        cx, cy = self._closest_point_on_rect(agent.x, agent.y, obstacle)
        dx = agent.x - cx
        dy = agent.y - cy
        riw = math.hypot(dx, dy)

        if riw == 0.0:
            return (0.0, 0.0)

        riwHatX = dx / riw
        riwHatY = dy / riw

        fx = self.A * math.exp((diw - riw) / self.B) * riwHatX
        fy = self.A * math.exp((diw - riw) / self.B) * riwHatY
        return (fx, fy)

    def obstacle_compression_force(self, agent, obstacle):
        diw = agent.radius

        cx, cy = self._closest_point_on_rect(agent.x, agent.y, obstacle)
        dx = agent.x - cx
        dy = agent.y - cy
        riw = math.hypot(dx, dy)

        if riw == 0.0:
            return (0.0, 0.0)

        riwHatX = dx / riw
        riwHatY = dy / riw

        fx = self.k * self.g(diw - riw) * riwHatX
        fy = self.k * self.g(diw - riw) * riwHatY
        return (fx, fy)

    def obstacle_friction_force(self, agent, obstacle):
        diw = agent.radius

        cx, cy = self._closest_point_on_rect(agent.x, agent.y, obstacle)
        dx = agent.x - cx
        dy = agent.y - cy
        riw = math.hypot(dx, dy)

        if riw == 0.0:
            return (0.0, 0.0)

        riwHatX = dx / riw
        riwHatY = dy / riw

        tiwHatX = -riwHatY
        tiwHatY =  riwHatX

        uiw = agent.vx * tiwHatX + agent.vy * tiwHatY

        fx = self.k * self.g(diw - riw) * uiw * tiwHatX
        fy = self.k * self.g(diw - riw) * uiw * tiwHatY
        return (fx, fy)