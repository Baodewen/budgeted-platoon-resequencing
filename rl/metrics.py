from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .agents.base import Agent
from .env import PlatoonEnv, Scenario


def run_episode(agent: Agent, env: PlatoonEnv, sc: Scenario, init_order: np.ndarray | None = None) -> dict:
    obs = env.reset(sc, init_order=np.arange(env.n) if init_order is None else init_order)
    done, t_dec = False, []
    while not done:
        t0 = time.perf_counter()
        cand = agent.select(obs, env, greedy=True)
        t_dec.append(time.perf_counter() - t0)
        obs, _, done, _ = env.step(cand)
    st = env.stats
    return {"scenario": sc.tag, "reward": st.reward, "distance": st.distance,
            "final_sigma": st.final_sigma, "frequency": st.changes, "decisions": st.decisions,
            "invalid_count": st.rejected, "invalid_rate": st.rejected / max(st.decisions, 1),
            "over_rate": st.violations / max(st.changes, 1), "runtime_s": float(np.mean(t_dec))}


def evaluate(agent: Agent, env: PlatoonEnv, scenarios: list[Scenario]) -> pd.DataFrame:
    return pd.DataFrame([run_episode(agent, env, sc) for sc in scenarios])


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    metrics = ["reward", "distance", "final_sigma", "frequency", "invalid_count", "invalid_rate", "over_rate",
               "runtime_s"]
    per_seed = df.groupby(by + ["seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby(by)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def relative_to(df: pd.DataFrame, ref_method: str, key: str = "method", n_col: str = "N") -> pd.DataFrame:
    out = df.copy()
    for n in out[n_col].unique():
        ref = out[(out[key] == ref_method) & (out[n_col] == n)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        m = out[n_col] == n
        out.loc[m, "d_reward"] = (out.loc[m, "reward_mean"] - ref["reward_mean"]) / abs(ref["reward_mean"]) * 100
        out.loc[m, "d_distance"] = (out.loc[m, "distance_mean"] / ref["distance_mean"] - 1) * 100
        out.loc[m, "d_sigma"] = (out.loc[m, "final_sigma_mean"] / ref["final_sigma_mean"] - 1) * 100
    return out
