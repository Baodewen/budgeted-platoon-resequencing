from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..env import Obs, PlatoonEnv
from .base import Agent, episode_log
from .networks import ActorCritic, featurize

ENV_FILTER = {"filter": "filter", "none": "none", "mask": "filter", "lagrangian": "none"}


class OnPolicyAgent(Agent):
    trainable = True

    def __init__(self, n: int, n_regimes: int, cfg: dict, algo: str = "ppo", constraint: str = "filter",
                 device=None, name: str | None = None):
        self.n, self.G, self.cfg = n, n_regimes, cfg
        self.algo, self.constraint = algo, constraint
        self.device = device or torch.device("cpu")
        self.name = name or f"{algo}_{constraint}"
        self.net = ActorCritic(n_regimes, cfg["hidden"], cfg["sage_layers"],
                               shared_encoder=(algo != "trpo")).to(self.device)
        self.lr = cfg["lr_init"]
        params = self.net.critic_parameters() if algo == "trpo" else self.net.parameters()
        self.opt = torch.optim.Adam(params, lr=self.lr)
        self.lam = 0.0
        self.use_mask = constraint == "mask"

    @property
    def env_filter_mode(self) -> str:
        return ENV_FILTER[self.constraint]

    @torch.no_grad()
    def act(self, obs: Obs, greedy: bool = False):
        s = featurize([obs], self.G, self.device)
        seq, logp, _, _ = self.net.policy(s, greedy=greedy, use_budget_mask=self.use_mask)
        return seq[0].cpu().numpy(), float(logp[0]), float(self.net.value(s)[0])

    def select(self, obs: Obs, env: PlatoonEnv, greedy: bool = True) -> np.ndarray:
        self.net.eval()
        return self.act(obs, greedy)[0]

    def train(self, env: PlatoonEnv, scenarios, cfg: dict, rng: np.random.Generator, log=None) -> list[dict]:
        assert env.filter_mode == self.env_filter_mode, "environment filter does not match the constraint"
        hist: list[dict] = []
        n_ep, per = cfg["episodes"], cfg["episodes_per_update"]
        for upd in range(math.ceil(n_ep / per)):
            self.net.train()
            batch = []
            for _ in range(per):
                if len(hist) >= n_ep:
                    break
                obs = env.reset(scenarios[rng.integers(len(scenarios))], rng=rng)
                done, traj = False, []
                while not done:
                    cand, logp, v = self.act(obs)
                    nxt, r, done, info = env.step(cand)
                    traj.append({"obs": obs, "action": cand, "logp": logp, "value": v, "reward": r,
                                 "cost": info["cost"], "done": done})
                    obs = nxt
                batch.append(traj)
                hist.append(episode_log(len(hist), env) | {"lambda": self.lam})
            self._update(batch)
            self.lr = max(self.cfg["lr_min"], self.lr * self.cfg["lr_decay"])
            for g in self.opt.param_groups:
                g["lr"] = self.lr
            if log and (upd + 1) % max(1, cfg.get("log_every", 50)) == 0:
                recent = hist[-per * 10:]
                log({k: float(np.mean([h[k] for h in recent])) for k in ("reward", "distance", "frequency",
                                                                          "rejected", "violations")}
                    | {"episodes": len(hist), "lambda": self.lam})
        return hist

    def _advantages(self, batch):
        gamma, lam_gae = self.cfg["gamma"], self.cfg["gae_lambda"]
        if self.algo == "a2c":
            lam_gae = 1.0
        obs, acts, logps, advs, rets, costs = [], [], [], [], [], []
        for traj in batch:
            r = np.array([t["reward"] - self.lam * t["cost"] for t in traj])
            v = np.array([t["value"] for t in traj] + [0.0])
            adv = np.zeros(len(traj))
            gae = 0.0
            for i in reversed(range(len(traj))):
                delta = r[i] + gamma * v[i + 1] * (1 - traj[i]["done"]) - v[i]
                gae = delta + gamma * lam_gae * (1 - traj[i]["done"]) * gae
                adv[i] = gae
            obs += [t["obs"] for t in traj]
            acts += [t["action"] for t in traj]
            logps += [t["logp"] for t in traj]
            costs += [t["cost"] for t in traj]
            advs.append(adv)
            rets.append(adv + v[:-1])
        adv = np.concatenate(advs)
        ret = np.concatenate(rets)
        return obs, np.stack(acts), np.array(logps), adv, ret, np.array(costs)

    def _update(self, batch) -> None:
        obs, acts, old_logp, adv, ret, costs = self._advantages(batch)
        if self.constraint == "lagrangian":
            self.lam = max(0.0, self.lam + self.cfg["lambda_lr"] * (costs.mean() - self.cfg["cost_limit"]))
        dev = self.device
        s_all = featurize(obs, self.G, dev)
        a_all = torch.as_tensor(acts, device=dev)
        old_all = torch.as_tensor(old_logp, dtype=torch.float32, device=dev)
        adv_all = torch.as_tensor(adv, dtype=torch.float32, device=dev)
        ret_all = torch.as_tensor(ret, dtype=torch.float32, device=dev)
        if self.algo == "trpo":
            self._trpo_step(s_all, a_all, adv_all, ret_all)
            return
        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
        epochs = 1 if self.algo == "a2c" else self.cfg["epochs"]
        mb = len(obs) if self.algo == "a2c" else self.cfg["minibatch"]
        for _ in range(epochs):
            perm = torch.randperm(len(obs), device=dev)
            for i in range(0, len(obs), mb):
                idx = perm[i:i + mb]
                s = {k: v[idx] for k, v in s_all.items()}
                _, logp, ent, _ = self.net.policy(s, actions=a_all[idx], use_budget_mask=self.use_mask)
                v = self.net.value(s)
                if self.algo == "ppo":
                    ratio = torch.exp(logp - old_all[idx])
                    eps = self.cfg["clip"]
                    pl = -torch.min(ratio * adv_all[idx], torch.clamp(ratio, 1 - eps, 1 + eps) * adv_all[idx]).mean()
                else:
                    pl = -(logp * adv_all[idx]).mean()
                loss = pl + self.cfg["vf_coef"] * (ret_all[idx] - v).pow(2).mean() - self.cfg["ent_coef"] * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg["grad_clip"])
                self.opt.step()

    def _trpo_step(self, s, a, adv, ret) -> None:
        cfg = self.cfg
        params = self.net.actor_parameters()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        with torch.no_grad():
            _, old_logp, _, old_steps = self.net.policy(s, actions=a, use_budget_mask=self.use_mask)

        def surrogate():
            _, logp, _, _ = self.net.policy(s, actions=a, use_budget_mask=self.use_mask)
            return (torch.exp(logp - old_logp) * adv).mean()

        def kl():
            _, _, _, steps = self.net.policy(s, actions=a, use_budget_mask=self.use_mask)
            p_old = old_steps.exp()
            return (p_old * (old_steps - steps)).sum(-1).sum(-1).mean()

        flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])
        g = flat(torch.autograd.grad(surrogate(), params))

        def fvp(v):
            grads = flat(torch.autograd.grad(kl(), params, create_graph=True))
            return flat(torch.autograd.grad((grads * v).sum(), params)) + cfg["trpo_damping"] * v

        x = torch.zeros_like(g)
        r_ = g.clone()
        p_ = g.clone()
        rr = r_ @ r_
        for _ in range(cfg["trpo_cg_iters"]):
            Ap = fvp(p_)
            alpha = rr / (p_ @ Ap + 1e-10)
            x += alpha * p_
            r_ -= alpha * Ap
            rr_new = r_ @ r_
            if rr_new < 1e-10:
                break
            p_ = r_ + rr_new / rr * p_
            rr = rr_new
        step = torch.sqrt(2 * cfg["trpo_max_kl"] / (x @ fvp(x) + 1e-10)) * x
        old = flat([p.data for p in params])
        with torch.no_grad():
            base = surrogate()
        for frac in 0.5 ** np.arange(10):
            self._set_flat(params, old + frac * step)
            with torch.no_grad():
                if kl() <= cfg["trpo_max_kl"] * 1.5 and surrogate() > base:
                    break
        else:
            self._set_flat(params, old)
        for _ in range(cfg["epochs"]):
            v = self.net.value(s)
            loss = (ret - v).pow(2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

    @staticmethod
    def _set_flat(params, flat):
        i = 0
        for p in params:
            n = p.numel()
            p.data.copy_(flat[i:i + n].view_as(p))
            i += n

    def save(self, path: Path) -> None:
        torch.save({"net": self.net.state_dict(), "lam": self.lam}, path)

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.lam = ck.get("lam", 0.0)
