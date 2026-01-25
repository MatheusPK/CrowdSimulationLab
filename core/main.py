import constants as K
from environment import *
from policy import *
from agentEntity import *
from obstacleEntity import *
from exitEntity import *
from renderer import *
from trainer import *

if __name__ == "__main__":
    policy = Policy()

    agents = [
        AgentEntity(agent_id="Naruto",  x=1.0, y=1.0, policy=policy),
        AgentEntity(agent_id="Sasuke",  x=4.0, y=3.0, policy=policy),
        AgentEntity(agent_id="Sakura",  x=2.5, y=7.5, policy=policy),
        AgentEntity(agent_id="Kakashi", x=8.5, y=5.0, policy=policy),
    ]

    obstacles = [
        ObstacleEntity(x=0.0, y=0.0, width=K.DOOR_X0, height=K.WALL), # parede de CIMA esquerda
        ObstacleEntity(x=K.DOOR_X1, y=0.0, width=K.ROOM_W - K.DOOR_X1, height=K.WALL), # parede de CIMA direita

        ObstacleEntity(x=0.0, y=K.ROOM_H - K.WALL, width=K.DOOR_X0, height=K.WALL), # parede de BAIXO esquerda
        ObstacleEntity(x=K.DOOR_X1, y=K.ROOM_H - K.WALL, width=K.ROOM_W - K.DOOR_X1, height=K.WALL), # parede de BAIXO direita

        ObstacleEntity(x=0.0, y=0.0, width=K.WALL, height=K.ROOM_H), # parede da ESQUERDA
        ObstacleEntity(x=K.ROOM_W - K.WALL, y=0.0, width=K.WALL, height=K.ROOM_H), # parede da DIREITA
    ]

    exits = [
        ExitEntity(x=K.DOOR_X0, y=0.0, width=K.DOOR_W, height=K.EXIT_H),
        ExitEntity(x=K.DOOR_X0, y=K.ROOM_H - K.EXIT_H, width=K.DOOR_W, height=K.EXIT_H),
    ]

    env = Environment(
        agents=agents,
        obstacles=obstacles,
        exits=exits,
        dt=K.DT
    )

    renderer = Renderer(
        world_width_m=10.0,
        world_height_m=10.0,
        agents=agents,
        obstacles=obstacles,
        exits=exits,
        meters_to_px=50,
        fps=120,
        draw_grid=True,
    )

    trainer = Trainer(
        episodes=1000, 
        max_steps=10000,
        environment=env,
        renderer=renderer
    )

    trainer.train()