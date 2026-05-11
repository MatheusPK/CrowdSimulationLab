import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.dqn_mode import DQNMode
from simulation_params import DQN_GRAD_CLIP_NORM, PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES, DQN_UPDATE_EVERY


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

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity      = capacity
        self.alpha         = alpha
        self.tree          = np.zeros(2 * capacity - 1)
        self.data          = np.empty(capacity, dtype=object)
        self.size          = 0
        self.ptr           = 0
        self._max_priority = 1.0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def push(self, state, action, reward, next_state, done):
        leaf     = self.ptr + self.capacity - 1
        self.data[self.ptr] = (state, action, reward, next_state, done)
        priority = self._max_priority
        delta    = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.total_priority <= 0 or self.size == 0:
            raise RuntimeError(
                "PER.sample() chamado com buffer vazio ou prioridade zero. "
                "Verifique se len(buffer) >= train_start_size antes de amostrar."
            )
        indices    = np.empty(batch_size, dtype=np.int32)
        is_weights = np.empty(batch_size, dtype=np.float32)
        segment    = self.total_priority / batch_size
        min_prob   = (np.min(self.tree[self.capacity - 1: self.capacity - 1 + self.size])
                      / self.total_priority)
        max_weight = (min_prob * self.size) ** (-beta)

        for i in range(batch_size):
            value         = random.uniform(segment * i, segment * (i + 1))
            leaf          = self._retrieve(0, value)
            prob          = self.tree[leaf] / self.total_priority
            is_weights[i] = ((prob * self.size) ** (-beta)) / max_weight
            indices[i]    = leaf

        batch = [self.data[idx - (self.capacity - 1)] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, indices, is_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority           = (abs(td_error) + 1e-6) ** self.alpha
            self._max_priority = max(self._max_priority, priority)
            delta              = priority - self.tree[idx]
            self.tree[idx]     = priority
            self._propagate(idx, delta)

    def reset(self):
        self.tree          = np.zeros(2 * self.capacity - 1)
        self.data          = np.empty(self.capacity, dtype=object)
        self.size          = 0
        self.ptr           = 0
        self._max_priority = 1.0

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Politica DQN
# ---------------------------------------------------------------------------

class DQNPolicy:
    """
    Deep Q-Network com:
    - Prioritized Experience Replay (PER)
    - Target network com atualizacao periodica
    - Epsilon-greedy com decaimento linear calibrado por stage
    - Modos TRAIN e EVAL separados
    - reset_for_stage() para curriculo progressivo
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
        buffer_capacity: int = 100_000,
        target_update_freq: int = 300,
        train_start_size: int = 2_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1_600_000,
        per_alpha: float = PER_ALPHA,
        per_beta_start: float = PER_BETA_START,
        per_beta_frames: int = PER_BETA_FRAMES,
        update_every: int = DQN_UPDATE_EVERY,
        device: str | None = None,
    ):
        self.mode               = mode
        self.model_path         = model_path
        self.state_dim          = state_dim
        self.action_dim         = action_dim
        self.batch_size         = batch_size
        self.gamma              = gamma
        self.target_update_freq = target_update_freq
        self.train_start_size   = train_start_size
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.per_beta_start     = per_beta_start
        self.per_beta_frames    = per_beta_frames
        self.update_every       = update_every
        self._transition_count  = 0

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

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer     = PrioritizedReplayBuffer(buffer_capacity, alpha=per_alpha)

        self.steps_done          = 0
        self.update_count        = 0
        self._global_transitions = 0

        if os.path.exists(model_path):
            self._load(model_path)
            print(f"[DQN] Modelo carregado de {model_path}")
        elif self.mode == DQNMode.EVAL:
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                f"Verifique o caminho em config.py ou rode train_curriculum.py primeiro."
            )

        if self.mode == DQNMode.EVAL:
            self.policy_net.eval()

    # ------------------------------------------------------------------
    # Interface publica
    # ------------------------------------------------------------------

    def choose_action(self, env, agent) -> int:
        if agent.evacuated:
            return None

        obs   = env.get_observation(agent)
        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.mode == DQNMode.EVAL:
            return self._greedy_action(state)

        if random.random() < self._current_epsilon():
            return random.randint(0, self.action_dim - 1)

        return self._greedy_action(state)

    def store_transition(self, obs, action, reward, next_obs, done):
        if self.mode != DQNMode.TRAIN:
            return
        self.buffer.push(obs, action, reward, next_obs, done)
        if len(self.buffer) >= self.train_start_size:
            self._global_transitions += 1
            self._transition_count += 1
            if self._transition_count % self.update_every == 0:
                self.steps_done += 1
                self._update()

    def save(self, path: str | None = None):
        target = path if path is not None else self.model_path
        os.makedirs(os.path.dirname(target) if os.path.dirname(target) else ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), target)

    def current_epsilon(self) -> float:
        return self._current_epsilon()

    def reset_for_stage(self, stage_decay: int, epsilon_start: float = 1.0):
        self.steps_done        = 0
        self._transition_count = 0
        self.epsilon_decay  = stage_decay
        self.epsilon_start  = epsilon_start

    def reset_buffer(self):
        self.buffer = PrioritizedReplayBuffer(
            self.buffer.capacity, 
            alpha=self.buffer.alpha
        )

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

    def _current_beta(self) -> float:
        progress = min(1.0, self._global_transitions / max(1, self.per_beta_frames))
        return self.per_beta_start + progress * (1.0 - self.per_beta_start)

    def _update(self):
        beta = self._current_beta()
        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.buffer.sample(self.batch_size, beta=beta)

        states_t      = torch.tensor(np.array(states),      dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions, dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,   dtype=torch.float32, device=self.device)
        weights_t     = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q       = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)

            agent_evacuated = next_states_t[:, 16]
            effective_done  = torch.clamp(dones_t + agent_evacuated, 0.0, 1.0)
            target = rewards_t + self.gamma * next_q * (1.0 - effective_done)

        td_errors = (target - q_values).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        loss = (weights_t * nn.functional.smooth_l1_loss(
            q_values, target, reduction="none", beta=10.0
        )).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=DQN_GRAD_CLIP_NORM)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load(self, path: str):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)