from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..env import Obs

NEG = -1e9


def featurize(obs_list: list[Obs], n_regimes: int, device) -> dict[str, torch.Tensor]:
    b = len(obs_list)
    n = len(obs_list[0].soc)
    x = np.zeros((b, n, 2 + n_regimes), np.float32)
    adj = np.zeros((b, n, n), np.float32)
    cur_pos = np.zeros((b, n), np.int64)
    budget = np.zeros(b, np.float32)
    for k, o in enumerate(obs_list):
        x[k, :, 0] = o.soc
        x[k, :, 1] = (o.pos + 1) / n
        x[k, :, 2 + o.regime] = 1.0
        ordr = o.order
        adj[k, ordr, ordr] = 1.0
        adj[k, ordr[1:], ordr[:-1]] = 1.0
        adj[k, ordr[:-1], ordr[1:]] = 1.0
        cur_pos[k] = o.pos
        budget[k] = o.budget
    adj /= adj.sum(-1, keepdims=True)
    t = lambda a: torch.as_tensor(a, device=device)
    return {"x": t(x), "adj": t(adj), "cur_pos": t(cur_pos), "budget": t(budget)}


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.lin_self = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.lin_neigh = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(layers))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for ls, ln in zip(self.lin_self, self.lin_neigh):
            h = F.relu(ls(h) + ln(torch.bmm(adj, h)))
        return h


class PointerDecoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(hidden))
        self.gru = nn.GRUCell(hidden, hidden)
        self.w_ref = nn.Linear(hidden, hidden, bias=False)
        self.w_q = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, emb: torch.Tensor, actions: torch.Tensor | None = None, greedy: bool = False,
                cur_pos: torch.Tensor | None = None, budget: torch.Tensor | None = None):
        b, n, hdim = emb.shape
        ar = torch.arange(b, device=emb.device)
        h = emb.mean(1)
        inp = self.start.unsqueeze(0).expand(b, hdim)
        ref = self.w_ref(emb)
        placed = torch.zeros(b, n, dtype=torch.bool, device=emb.device)
        remaining = budget.clone().float() if budget is not None else None
        ahead = None
        if budget is not None:
            ahead = (cur_pos.unsqueeze(1) < cur_pos.unsqueeze(2)).float()
        seq, logps, ents, step_logp = [], [], [], []
        for j in range(n):
            h = self.gru(inp, h)
            u = self.v(torch.tanh(ref + self.w_q(h).unsqueeze(1))).squeeze(-1)
            mask = placed
            inc = None
            if budget is not None:
                inc = torch.bmm(ahead, (~placed).float().unsqueeze(-1)).squeeze(-1)
                mask = mask | (inc > remaining.unsqueeze(1) + 1e-6)
            logp_all = F.log_softmax(u.masked_fill(mask, NEG), -1)
            if actions is not None:
                a = actions[:, j]
            elif greedy:
                a = logp_all.argmax(-1)
            else:
                a = Categorical(logits=logp_all).sample()
            p_all = logp_all.exp()
            ents.append(-(p_all * logp_all).sum(-1))
            logps.append(logp_all[ar, a])
            step_logp.append(logp_all)
            seq.append(a)
            if inc is not None:
                remaining = remaining - inc[ar, a]
            placed = placed.clone()
            placed[ar, a] = True
            inp = emb[ar, a]
        return (torch.stack(seq, 1), torch.stack(logps, 1).sum(1), torch.stack(ents, 1).sum(1),
                torch.stack(step_logp, 1))


class ActorCritic(nn.Module):
    def __init__(self, n_regimes: int, hidden: int = 128, sage_layers: int = 2, shared_encoder: bool = True):
        super().__init__()
        in_dim = 2 + n_regimes
        self.encoder = GraphSAGEEncoder(in_dim, hidden, sage_layers)
        self.critic_encoder = self.encoder if shared_encoder else GraphSAGEEncoder(in_dim, hidden, sage_layers)
        self.decoder = PointerDecoder(hidden)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.shared = shared_encoder

    def policy(self, s: dict, actions=None, greedy=False, use_budget_mask=False):
        emb = self.encoder(s["x"], s["adj"])
        if use_budget_mask:
            return self.decoder(emb, actions, greedy, s["cur_pos"], s["budget"])
        return self.decoder(emb, actions, greedy)

    def value(self, s: dict) -> torch.Tensor:
        return self.value_head(self.critic_encoder(s["x"], s["adj"]).mean(1)).squeeze(-1)

    def actor_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def critic_parameters(self):
        if self.shared:
            return list(self.value_head.parameters())
        return list(self.critic_encoder.parameters()) + list(self.value_head.parameters())
