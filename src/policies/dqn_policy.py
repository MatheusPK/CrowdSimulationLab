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
    - save() aceita path opcional para checkpoints de currículo
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
        buffer_capacity: int = 50_000,
        target_update_freq: int = 300,
        train_start_size: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 30_000,
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

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.steps_done = 0
        self.update_count = 0

        if os.path.exists(model_path):
            self._load(model_path)
            print(f"[DQN] Modelo carregado de {model_path}")

        if self.mode == DQNMode.EVAL:
            self.policy_net.eval()

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def choose_action(self, env, agent, exit_obj=None) -> int:
        if agent.evacuated:
            return None

        obs = env.get_observation(agent)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.mode == DQNMode.EVAL:
            return self._greedy_action(state)

        epsilon = self._current_epsilon()
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        return self._greedy_action(state)

    def store_transition(self, obs, action, reward, next_obs, done):
        if self.mode != DQNMode.TRAIN:
            return

        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

        if len(self.buffer) >= self.train_start_size:
            self._update()

    def save(self, path: str | None = None):
        """
        Salva os pesos da policy net.
        Se path for None, usa self.model_path (modelo principal).
        Aceita path diferente para salvar checkpoints de currículo.
        """
        target = path if path is not None else self.model_path
        os.makedirs(os.path.dirname(target) if os.path.dirname(target) else ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), target)

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
        progress = min(1.0, self.steps_done / max(1, self.epsilon_decay))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _update(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states_t).max(dim=1).values
            target  = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load(self, path: str):
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)