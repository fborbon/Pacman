"""
Deep Q-Network (DQN) agent.

Architecture
------------
  Q-network  : Linear(obs) → 256 → 256 → n_actions
  Target net : hard-copy of Q-network, updated every `target_update` steps
  Buffer     : uniform random replay, capacity 100 k
  Loss       : Huber (SmoothL1) between predicted Q and Bellman target
  Exploration: linear epsilon decay from 1.0 → 0.05 over `eps_decay` steps
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── neural network ─────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 256), nn.ReLU(),
            nn.Linear(256,      256), nn.ReLU(),
            nn.Linear(256,      n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── experience replay buffer ───────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch                    = random.sample(self._buf, batch_size)
        s, a, r, ns, d          = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(ns)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self._buf)


# ── agent ──────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with experience replay and a periodically synced target network.

    Parameters
    ----------
    obs_size      : dimensionality of the observation vector
    n_actions     : number of discrete actions
    device        : "cpu" or "cuda"
    lr            : Adam learning rate
    gamma         : discount factor
    eps_start     : initial exploration probability
    eps_end       : final exploration probability
    eps_decay     : number of steps over which eps decays linearly
    buffer_size   : replay buffer capacity
    batch_size    : mini-batch size for each training step
    target_update : hard-copy Q → target every N steps
    """

    def __init__(
        self,
        obs_size:      int,
        n_actions:     int,
        device:        str   = "cpu",
        lr:            float = 1e-4,
        gamma:         float = 0.99,
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 200_000,
        buffer_size:   int   = 100_000,
        batch_size:    int   = 64,
        target_update: int   = 1_000,
    ):
        self.n_actions     = n_actions
        self.gamma         = gamma
        self.eps           = eps_start
        self.eps_end       = eps_end
        self.eps_decay     = eps_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.device        = torch.device(device)
        self.steps_done    = 0

        self.q_net      = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net = QNetwork(obs_size, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)
        self._loss_fn  = nn.SmoothL1Loss()

    # ── action selection ────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy action selection with linear epsilon decay."""
        self.steps_done += 1
        frac      = min(self.steps_done / self.eps_decay, 1.0)
        self.eps  = 1.0 + frac * (self.eps_end - 1.0)

        if random.random() < self.eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())

    # ── training ────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))

    def train_step(self) -> float | None:
        """Sample a mini-batch and perform one gradient update. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = (t.to(self.device)
                          for t in self.buffer.sample(self.batch_size))

        # Current Q-values for chosen actions
        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Bellman target (no gradient through target net)
        with torch.no_grad():
            q_next  = self.target_net(ns).max(1).values
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = self._loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Periodic hard sync of target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "eps":        self.eps,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.eps        = ckpt["eps"]
        print(f"Checkpoint loaded ← {path}  (step {self.steps_done}, ε={self.eps:.3f})")
