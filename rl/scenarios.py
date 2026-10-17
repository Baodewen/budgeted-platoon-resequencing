from __future__ import annotations

import numpy as np

from .env import Scenario

STEPS_PER_HOUR = 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


def synthetic_traffic(n_days: int, thresholds: list[float], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    T = n_days * STEPS_PER_DAY
    h = (np.arange(T) % STEPS_PER_DAY) / STEPS_PER_HOUR
    weekend = ((np.arange(T) // STEPS_PER_DAY) % 7) >= 5

    def plateau(a, b, k=4.0):
        return 1 / (1 + np.exp(-(h - a) * k)) / (1 + np.exp((h - b) * k))

    c = (0.6 * plateau(7.0, 20.5) + 0.4 * plateau(9.0, 18.5)) * np.where(weekend, 0.6, 1.0)
    e = np.zeros(T)
    for t in range(1, T):
        e[t] = 0.85 * e[t - 1] + rng.normal(0, 1.0)
    speed = np.clip(109.0 - 66.0 * c + e, 20.0, 125.0)
    regime = np.searchsorted(-np.asarray(thresholds), -speed, side="right")
    return speed, regime.astype(np.int64)


def load_traffic(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    sc = cfg["scenario"]
    if sc["source"] == "file":
        z = np.load(sc["file"])
        return z["speed"].astype(np.float64), z["regime"].astype(np.int64)
    return synthetic_traffic(sc["n_days"], sc["speed_thresholds"], sc["seed"])


def _slice(speed, regime, start, length, tag):
    idx = np.clip(np.arange(start, start + length), 0, len(speed) - 1)
    return Scenario(speed[idx], regime[idx], tag)


def build_scenarios(cfg: dict) -> tuple[list[Scenario], list[Scenario]]:
    sc = cfg["scenario"]
    speed, regime = load_traffic(cfg)
    length = sc["episode_steps"]
    test_start = len(speed) - sc["n_test_days"] * STEPS_PER_DAY
    lo, hi = sc["train_hours"]
    train = []
    for s in range(0, test_start - length, sc["train_stride"]):
        hour = (s % STEPS_PER_DAY) / STEPS_PER_HOUR
        if lo <= hour <= hi:
            train.append(_slice(speed, regime, s, length, f"train@{s}"))
    test = []
    for d in range(sc["n_test_days"]):
        for hour in sc["test_start_hours"]:
            s = test_start + d * STEPS_PER_DAY + int(hour * STEPS_PER_HOUR)
            test.append(_slice(speed, regime, s, length, f"day{d}@{hour:02d}h"))
    return train, test
