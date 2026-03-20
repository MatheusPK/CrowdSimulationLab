import os
import random
import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from core.dqn_mode import DQNMode


# ---------------------------------------------------------------------------
# Rede neural
# ---------------------------------------------------------------------------

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Política DQN
# ---------------------------------------------------------------------------

class DQNPolicy:
    """
    Deep Q-Network com:
    - Replay buffer de experiências
    - Target network com atualização periódica
    - Epsilon-greedy com decaimento linear
    - Modos TRAIN e EVAL separados

    Parâmetros importantes (vindos do config["dqn"]):
        mode            : DQNMode.TRAIN ou DQNMode.EVAL
        model_path      : caminho para salvar/carregar o modelo
        state_dim       : dimensão do vetor de observação (deve casar com get_observation)
        action_dim      : número de ações (8 direções)
        hidden_dim      : neurônios por camada oculta
        episodes        : total de episódios de treino (usado só no runner)
        batch_size      : tamanho do mini-batch de treino
        gamma           : fator de desconto
        lr              : learning rate do Adam
        buffer_capacity : capacidade máxima do replay buffer
        target_update_freq  : a cada quantos updates copia pesos para a target net
        train_start_size    : mínimo de amostras no buffer para começar o treino
        epsilon_start   : epsilon inicial (exploração máxima)
        epsilon_end     : epsilon final (exploração mínima)
        epsilon_decay   : número de steps para decair de start até end
    """

    def __init__(
        self,
        mode: DQNMode = DQNMode.TRAIN,
        model_path: str = "models/dqn_model.pth",
        state_dim: int = 17,
        action_dim: int = 8,
        hidden_dim: int = 128,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_capacity: int = 20_000,
        target_update_freq: int = 200,
        train_start_size: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 20_000,
        device: str | None = None,
    ):
        self.mode = mode
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.train_start_size = train_start_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Redes
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.steps_done = 0       # total de steps de treino (epsilon decay)
        self.update_count = 0     # total de updates (target net sync)

        # Carrega modelo se existir (tanto em TRAIN quanto em EVAL)
        if os.path.exists(model_path):
            self._load(model_path)
            print(f"[DQN] Modelo carregado de {model_path}")

        if self.mode == DQNMode.EVAL:
            self.policy_net.eval()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def choose_action(self, env, agent, exit_obj=None) -> int:
        """Seleciona ação via epsilon-greedy (treino) ou greedy (eval)."""
        if agent.evacuated:
            return None

        obs = env.get_observation(agent)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.mode == DQNMode.EVAL:
            return self._greedy_action(state)

        # Epsilon-greedy com decaimento linear
        epsilon = self._current_epsilon()

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        return self._greedy_action(state)

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Armazena transição no replay buffer e dispara um update se pronto.
        Deve ser chamado pelo runner a cada step de treino.
        """
        if self.mode != DQNMode.TRAIN:
            return

        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

        if len(self.buffer) >= self.train_start_size:
            self._update()

    def save(self):
        """Salva os pesos da policy net."""
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.model_path)

    def current_epsilon(self) -> float:
        return self._current_epsilon()

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _greedy_action(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.policy_net(state)
        return int(q_values.argmax(dim=1).item())

    def _current_epsilon(self) -> float:
        """Decaimento linear de epsilon_start até epsilon_end ao longo de epsilon_decay steps."""
        progress = min(1.0, self.steps_done / max(1, self.epsilon_decay))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _update(self):
        """Um passo de gradient descent na Bellman loss."""
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        # Q(s, a) da policy net
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # max Q(s', a') da target net  — sem gradiente
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — estabiliza treino com reward shaping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1

        # Sincroniza target net periodicamente
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)