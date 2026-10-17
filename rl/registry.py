from __future__ import annotations

import numpy as np

from .agents.base import Agent
from .agents.tabular import MonteCarloControl, QLearning
from .energy import EnergyModel
from .env import Obs, PlatoonEnv, RewardConfig

METHODS: dict[str, tuple[str, str]] = {
    "ppo": ("ppo", "filter"),
    "ppo_unconstrained": ("ppo", "none"),
    "ppo_mask": ("ppo", "mask"),
    "ppo_lagrangian": ("ppo", "lagrangian"),
    "a2c": ("a2c", "filter"),
    "trpo": ("trpo", "filter"),
    "dqn": ("dqn", "filter"),
    "q_learning": ("tabular", "filter"),
    "monte_carlo": ("tabular", "filter"),
}


class FixedOrder(Agent):
    name = "fixed_order"

    def select(self, obs: Obs, env: PlatoonEnv, greedy: bool = True) -> np.ndarray:
        return obs.order.copy()


def env_filter(method: str) -> str:
    if method == "fixed_order":
        return "filter"
    return "none" if METHODS[method][1] in ("none", "lagrangian") else "filter"


def make_env(cfg: dict, n: int, method: str) -> PlatoonEnv:
    e = cfg["env"]
    return PlatoonEnv(n, EnergyModel(**cfg["energy"]), np.asarray(cfg["budget"][n]), env_filter(method),
                      e["decision_steps"], reward=RewardConfig(**cfg["reward"]), max_decisions=e["max_decisions"])


def build_agent(method: str, n: int, cfg: dict, seed: int, device=None) -> Agent:
    if method == "fixed_order":
        return FixedOrder()
    family, constraint = METHODS[method]
    G = cfg["n_regimes"]
    if method == "q_learning":
        return QLearning(n, lr=cfg["tabular"]["lr"], gamma=cfg["rl"]["gamma"], seed=seed)
    if method == "monte_carlo":
        return MonteCarloControl(n, gamma=cfg["rl"]["gamma"], seed=seed)
    if family == "dqn":
        from .agents.dqn import DQNAgent

        return DQNAgent(n, G, cfg["dqn"], device, seed)
    from .agents.onpolicy import OnPolicyAgent

    return OnPolicyAgent(n, G, cfg["rl"], algo=family, constraint=constraint, device=device, name=method)


def train_config(method: str, cfg: dict) -> dict:
    family = METHODS[method][0]
    if family == "tabular":
        return {"episodes": cfg["tabular"]["episodes"]}
    if family == "dqn":
        return {"episodes": cfg["dqn"]["episodes"]}
    return {"episodes": cfg["rl"]["episodes"], "episodes_per_update": cfg["rl"]["episodes_per_update"],
            "log_every": cfg["rl"]["log_every"]}


def ckpt_ext(method: str) -> str:
    return ".pkl" if METHODS[method][0] == "tabular" else ".pt"
