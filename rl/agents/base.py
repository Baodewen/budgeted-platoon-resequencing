from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..env import Obs, PlatoonEnv


class Agent(ABC):
    name: str = "agent"
    trainable: bool = False

    @abstractmethod
    def select(self, obs: Obs, env: PlatoonEnv, greedy: bool = True) -> np.ndarray:
        ...

    def train(self, env: PlatoonEnv, scenarios, cfg: dict, rng: np.random.Generator, log=None) -> list[dict]:
        return []

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


def n_discrete_actions(n: int) -> int:
    return 2 * n - 1


def apply_discrete_action(order: np.ndarray, a: int) -> np.ndarray:
    n = len(order)
    out = order.copy()
    if a == 0:
        return out
    if a <= n - 1:
        out[a - 1], out[a] = out[a], out[a - 1]
        return out
    j = a - n
    v = out[j]
    return np.concatenate([np.delete(out, j), [v]])


def soc_rank_state(obs: Obs) -> tuple:
    soc_by_pos = obs.soc[obs.order]
    ranks = tuple(np.argsort(np.argsort(-np.round(soc_by_pos, 6), kind="stable"), kind="stable").tolist())
    return ranks + (obs.regime,)


def episode_log(ep: int, env: PlatoonEnv) -> dict:
    st = env.stats
    return {"episode": ep, "reward": st.reward, "distance": st.distance, "frequency": st.changes,
            "rejected": st.rejected, "decisions": st.decisions, "violations": st.violations,
            "final_sigma": st.final_sigma}
