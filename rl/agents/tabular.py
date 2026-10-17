from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..env import Obs, PlatoonEnv
from .base import Agent, apply_discrete_action, episode_log as _ep_log, n_discrete_actions, soc_rank_state


class _Tabular(Agent):
    trainable = True

    def __init__(self, n: int, gamma: float = 0.98, eps_start: float = 1.0, eps_end: float = 0.05,
                 seed: int = 0):
        self.n = n
        self.na = n_discrete_actions(n)
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.Q: dict = defaultdict(lambda: np.zeros(self.na))
        self.rng = np.random.default_rng(seed)
        self.eps = eps_start

    def _act_index(self, obs: Obs, greedy: bool) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.na))
        q = self.Q[soc_rank_state(obs)]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def select(self, obs: Obs, env: PlatoonEnv, greedy: bool = True) -> np.ndarray:
        return apply_discrete_action(obs.order, self._act_index(obs, greedy))

    def _eps_at(self, ep: int, n_ep: int) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * max(0.0, 1 - ep / (0.8 * n_ep))

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            for k, v in pickle.load(f).items():
                self.Q[k] = v


class QLearning(_Tabular):
    name = "q_learning"

    def __init__(self, n: int, lr: float = 0.1, **kw):
        super().__init__(n, **kw)
        self.lr = lr

    def train(self, env, scenarios, cfg, rng, log=None):
        hist = []
        n_ep = cfg["episodes"]
        for ep in range(n_ep):
            self.eps = self._eps_at(ep, n_ep)
            obs = env.reset(scenarios[rng.integers(len(scenarios))], rng=rng)
            done = False
            while not done:
                s = soc_rank_state(obs)
                a = self._act_index(obs, greedy=False)
                obs, r, done, _ = env.step(apply_discrete_action(obs.order, a))
                target = r if done else r + self.gamma * self.Q[soc_rank_state(obs)].max()
                self.Q[s][a] += self.lr * (target - self.Q[s][a])
            hist.append(_ep_log(ep, env))
            if log and (ep + 1) % max(1, n_ep // 10) == 0:
                log(hist[-1])
        return hist


class MonteCarloControl(_Tabular):
    name = "monte_carlo"

    def __init__(self, n: int, **kw):
        super().__init__(n, **kw)
        self.counts: dict = defaultdict(lambda: np.zeros(self.na))

    def train(self, env, scenarios, cfg, rng, log=None):
        hist = []
        n_ep = cfg["episodes"]
        for ep in range(n_ep):
            self.eps = self._eps_at(ep, n_ep)
            obs = env.reset(scenarios[rng.integers(len(scenarios))], rng=rng)
            traj, done = [], False
            while not done:
                s = soc_rank_state(obs)
                a = self._act_index(obs, greedy=False)
                obs, r, done, _ = env.step(apply_discrete_action(obs.order, a))
                traj.append((s, a, r))
            g = 0.0
            for s, a, r in reversed(traj):
                g = r + self.gamma * g
                self.counts[s][a] += 1
                self.Q[s][a] += (g - self.Q[s][a]) / self.counts[s][a]
            hist.append(_ep_log(ep, env))
            if log and (ep + 1) % max(1, n_ep // 10) == 0:
                log(hist[-1])
        return hist

