from __future__ import annotations

import numpy as np


def positions(order: np.ndarray) -> np.ndarray:
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order)] = np.arange(len(order))
    return pos


def inversion_count(cur_order: np.ndarray, tgt_order: np.ndarray) -> int:
    p = positions(cur_order)
    q = positions(tgt_order)
    dp = p[:, None] - p[None, :]
    dq = q[:, None] - q[None, :]
    return int(np.triu((dp * dq) < 0, k=1).sum())


def max_complexity(n: int) -> int:
    return n * (n - 1) // 2
