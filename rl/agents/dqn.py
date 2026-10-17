from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env import Obs, PlatoonEnv
from .base import Agent, apply_discrete_action, episode_log, n_discrete_actions


def dqn_features(obs: Obs, n_regimes: int) -> np.ndarray:
    soc_by_pos = obs.soc[obs.order]
    onehot = np.zeros(n_regimes)
    onehot[obs.regime] = 1.0
    return np.concatenate([soc_by_pos, soc_by_pos - soc_by_pos.mean(), onehot]).astype(np.float32)


class DQNAgent(Agent):
    name = "dqn"
    trainable = True

    def __init__(self, n: int, n_regimes: int, cfg: dict, device=None, seed: int = 0):
        self.n, self.G, self.cfg = n, n_regimes, cfg
        self.na = n_discrete_actions(n)
        self.device = device or torch.device("cpu")
        d_in = 2 * n + n_regimes
        h = cfg["hidden"]
        mk = lambda: nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, self.na))
        self.q, self.q_tgt = mk().to(self.device), mk().to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buf: deque = deque(maxlen=cfg["buffer"])
        self.rng = np.random.default_rng(seed)
        self.eps = 1.0

    @torch.no_grad()
    def _q(self, obs: Obs) -> np.ndarray:
        x = torch.as_tensor(dqn_features(obs, self.G), device=self.device).unsqueeze(0)
        return self.q(x)[0].cpu().numpy()

    def _choose(self, obs: Obs, greedy: bool) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.na))
        return int(np.argmax(self._q(obs)))

    def select(self, obs: Obs, env: PlatoonEnv, greedy: bool = True) -> np.ndarray:
        return apply_discrete_action(obs.order, self._choose(obs, greedy))

    def train(self, env, scenarios, cfg, rng, log=None):
        hist, step = [], 0
        n_ep = cfg["episodes"]
        for ep in range(n_ep):
            self.eps = max(self.cfg["eps_end"], 1.0 - ep / (self.cfg["eps_decay_frac"] * n_ep))
            obs = env.reset(scenarios[rng.integers(len(scenarios))], rng=rng)
            done = False
            while not done:
                a = self._choose(obs, greedy=False)
                nxt, r, done, _ = env.step(apply_discrete_action(obs.order, a))
                self.buf.append((dqn_features(obs, self.G), a, r, dqn_features(nxt, self.G), float(done)))
                obs = nxt
                step += 1
                if len(self.buf) >= self.cfg["batch"]:
                    self._learn()
                if step % self.cfg["target_every"] == 0:
                    self.q_tgt.load_state_dict(self.q.state_dict())
            hist.append(episode_log(ep, env))
            if log and (ep + 1) % max(1, n_ep // 10) == 0:
                log(hist[-1])
        return hist

    def _learn(self) -> None:
        idx = self.rng.integers(len(self.buf), size=self.cfg["batch"])
        s, a, r, s2, d = zip(*(self.buf[i] for i in idx))
        t = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt, device=self.device)
        s, a, r, s2, d = t(s), t(a, torch.int64), t(r), t(s2), t(d)
        q = self.q(s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            target = r + self.cfg["gamma"] * (1 - d) * self.q_tgt(s2).max(1).values
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

    def save(self, path: Path) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: Path) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
