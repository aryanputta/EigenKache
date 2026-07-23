"""KV-cache compression policy that ml_agent optimizes.

Contract (do not change): expose `compress(K, V, budget, sink, tail)` returning
a 1-D int array of token indices to KEEP, with len <= budget, no duplicates, all
in [0, S). The benchmark reconstructs attention over only the kept tokens and
measures error against full attention. Your job: pick the budget tokens that
best preserve the attention output. You only see K and V (not the queries), so
use the structure of K/V, not the queries.

Baseline below is a naive recency window: keep the sink tokens and the most
recent (budget - sink) tokens. It ignores the cold middle entirely.
"""

import numpy as np


def compress(K: np.ndarray, V: np.ndarray, budget: int, sink: int = 4, tail: int = 32) -> np.ndarray:
    S = K.shape[0]
    if budget >= S:
        return np.arange(S, dtype=np.int64)
    sink = min(sink, budget)
    keep_recent = budget - sink
    idx = np.concatenate([
        np.arange(0, sink, dtype=np.int64),
        np.arange(S - keep_recent, S, dtype=np.int64),
    ])
    return np.unique(idx)
