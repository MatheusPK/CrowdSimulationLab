import constants as K
from map import *
from agent import *

class Environment:
    def __init__(self, agents, map):
        self.agents = agents
        self.map = map
        self.initial_map = self.copy_map(map)
        self.initial_agents = self.copy_agents(agents)

        self.actions = {
            K.NORTH: (0, -1),
            K.SOUTH: (0, 1),
            K.EAST: (1, 0),
            K.WEST: (-1, 0),
            K.NORTHEAST: (1, -1),
            K.NORTHWEST: (-1, -1),
            K.SOUTHEAST: (1, 1),
            K.SOUTHWEST: (-1, 1)
        }

        self.collision_rewards = {
            K.NO_COLLISION: -1,
            K.OBSTACLE_COLLISION: -10,
            K.AGENT_COLLISION: -5,
            K.TARGET_COLLISION: 10
        }

    def reset(self):
        self.map = self.copy_map(self.initial_map)
        self.agents = self.copy_agents(self.initial_agents)
    
    def act(self, agent, action):
        action_vector = self.actions[action]
        current_position = (agent.x, agent.y)
        new_position = (agent.x + action_vector[0], agent.y + action_vector[1])
        collision_type = self.check_collision_type(new_position)

        if collision_type == K.NO_COLLISION:
            agent.x, agent.y = new_position
            self.map.map[agent.y][agent.x].type = K.AGENT
            self.map.map[agent.y][agent.x].entity = agent

            self.map.map[current_position[1]][current_position[0]].type = K.EMPTY
            self.map.map[current_position[1]][current_position[0]].entity = None

        if collision_type == K.TARGET_COLLISION:
            agent.done = True

        reward = self.collision_rewards[collision_type]
        obs = self.get_microscopic_observation(agent)        
        
        return obs, reward, agent.done

    def check_collision_type(self, position):
        x, y = position
        
        if x < 0 or x >= self.map.width or y < 0 or y >= self.map.height:
            return K.OBSTACLE_COLLISION
        
        cellType = self.map.map[y][x].type
        if cellType == K.OBSTACLE: return K.OBSTACLE_COLLISION
        elif cellType == K.AGENT: return K.AGENT_COLLISION
        elif cellType == K.TARGET: return K.TARGET_COLLISION
        else: return K.NO_COLLISION
    
    def get_microscopic_observation(self, agent):
        return [agent.x, agent.y]

    def copy_map(self, map):
        new_map = Map()
        new_map.width = map.width
        new_map.height = map.height
        new_map.map = [[Cell(type=cell.type, entity=cell.entity) for cell in row] for row in map.map]
        return new_map

    def copy_agents(self, agents):
        new_agents = []
        for agent in agents:
            new_agent = Agent(agent_id=agent.id, x=agent.x, y=agent.y, policy=agent.policy)
            new_agent.done = agent.done
            new_agents.append(new_agent)
        return new_agents
        

    
