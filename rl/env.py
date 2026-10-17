from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .complexity import inversion_count, max_complexity, positions
from .energy import EnergyModel

FILTER_MODES = ("filter", "none")


@dataclass
class Scenario:
    speed: np.ndarray
    regime: np.ndarray
    tag: str = ""

    def at(self, arr: np.ndarray, i: int):
        return arr[min(i, len(arr) - 1)]


@dataclass
class RewardConfig:
    w_mid_soc: float = 1.0
    w_mid_dist: float = 1.0
    w_final: float = 1000.0


@dataclass
class Obs:
    soc: np.ndarray
    order: np.ndarray
    pos: np.ndarray
    regime: int
    budget: int
    k: int


@dataclass
class EpisodeStats:
    reward: float = 0.0
    distance: float = 0.0
    decisions: int = 0
    changes: int = 0
    rejected: int = 0
    violations: int = 0
    final_sigma: float = float("nan")
    final_soc: np.ndarray = field(default_factory=lambda: np.zeros(0))


class PlatoonEnv:
    def __init__(self, n: int, energy: EnergyModel, budget_table: np.ndarray, filter_mode: str = "filter",
                 decision_steps: int = 5, step_minutes: float = 5.0, reward: RewardConfig | None = None,
                 max_decisions: int = 200):
        if filter_mode not in FILTER_MODES:
            raise ValueError(filter_mode)
        self.n = n
        self.energy = energy
        self.budget_table = np.asarray(budget_table, dtype=np.int64)
        self.filter_mode = filter_mode
        self.decision_steps = decision_steps
        self.step_h = step_minutes / 60.0
        self.reward_cfg = reward or RewardConfig()
        self.max_decisions = max_decisions
        self.drop = energy.soc_drop_per_km(n)
        self.e_head = energy.position_coefficients(n)[0]
        self.d_avg = energy.fixed_order_distance(n)
        self.b_max = max_complexity(n)

    def budget(self, regime: int) -> int:
        return int(self.budget_table[regime])

    def interval_distance(self, sc: Scenario, k: int) -> float:
        i0 = k * self.decision_steps
        sp = [sc.at(sc.speed, i) for i in range(i0, i0 + self.decision_steps)]
        return float(np.sum(sp) * self.step_h)

    def d_min(self, soc: np.ndarray) -> float:
        lam = self.energy.usable_kwh
        return lam * float(soc.min()) / (self.energy.delta_kwh_per_km * (1.0 - self.e_head))

    def _obs(self) -> Obs:
        i = self.k * self.decision_steps
        rho = int(self.sc.at(self.sc.regime, i))
        return Obs(self.soc.copy(), self.order.copy(), positions(self.order), rho, self.budget(rho), self.k)

    def reset(self, scenario: Scenario, init_order: np.ndarray | None = None,
              rng: np.random.Generator | None = None) -> Obs:
        self.sc = scenario
        if init_order is None:
            rng = rng or np.random.default_rng()
            init_order = rng.permutation(self.n)
        self.order = np.asarray(init_order, dtype=np.int64).copy()
        self.soc = np.ones(self.n)
        self.k = 0
        self.done = False
        self.stats = EpisodeStats()
        return self._obs()

    def step(self, cand_order: np.ndarray) -> tuple[Obs, float, bool, dict]:
        assert not self.done, "episode finished; call reset()"
        cand_order = np.asarray(cand_order, dtype=np.int64)
        obs = self._obs()
        c = inversion_count(self.order, cand_order)
        within = c <= obs.budget
        rejected = (self.filter_mode == "filter") and not within
        executed = self.order if rejected else cand_order
        changed = not np.array_equal(executed, self.order)
        violated = changed and not within

        sigma_prev = float(self.soc.std())
        pos = positions(executed)
        drops = self.drop[pos]
        d_int = self.interval_distance(self.sc, self.k)
        d_empty = float(np.min(self.soc / drops))
        terminal = d_int >= d_empty
        d = d_empty if terminal else d_int
        self.soc = self.soc - drops * d
        if terminal:
            self.soc[np.argmin(self.soc)] = 0.0
            self.soc = np.maximum(self.soc, 0.0)
        self.order = executed.copy()
        self.k += 1
        if self.k >= self.max_decisions:
            terminal = True
        sigma = float(self.soc.std())

        rc = self.reward_cfg
        r = (sigma_prev - sigma) / rc.w_mid_soc + self.d_min(self.soc) / (rc.w_mid_dist * self.d_avg)
        if terminal:
            r -= rc.w_final * sigma

        st = self.stats
        st.reward += r
        st.distance += d
        st.decisions += 1
        st.changes += int(changed)
        st.rejected += int(rejected)
        st.violations += int(violated)
        if terminal:
            st.final_sigma = sigma
            st.final_soc = self.soc.copy()
        self.done = terminal
        info = {"complexity": c, "budget": obs.budget, "rejected": rejected, "violated": violated,
                "changed": changed, "distance": d, "cost": float(violated)}
        return self._obs(), float(r), terminal, info
