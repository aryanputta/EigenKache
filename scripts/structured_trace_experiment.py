"""Structured-trace fidelity experiment for EigenKache.

The proof-of-concept in build/sweep.csv uses an i.i.d. random trace, which has
no redundant structure -- so compression (landmark) cannot beat eviction there.
Real long-context KV is different: the cold region is structurally redundant
(repeated topics, low intrinsic rank). This script builds a *structured* trace
that captures that property and re-runs the matched-budget fidelity comparison.

It uses only the existing public policies/harness; no model or GPU required.
"""

from __future__ import annotations

import numpy as np

from eigenkache.bench import sweep_benchmarks
from eigenkache.types import KVTrace


def make_structured_trace(
    *,
    n: int = 512,
    d: int = 128,
    q: int = 16,
    n_topics: int = 6,
    sink: int = 8,
    tail: int = 32,
    noise: float = 0.15,
    seed: int = 0,
) -> KVTrace:
    """A long-context-like trace with a low-rank, redundant cold middle.

    - sink tokens: distinct anchors (each its own direction)
    - cold middle: drawn from a small set of topic centroids + small noise
      (this is the redundant, compressible structure real traces have)
    - tail tokens: distinct recent tokens
    - values: a fixed linear map of keys plus small noise (K/V correlated)
    - queries: recent queries that mostly attend back into the cold middle
    """
    rng = np.random.default_rng(seed)

    def unit(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    keys = np.zeros((n, d), dtype=np.float32)
    # sink: distinct anchors
    keys[:sink] = unit(rng.standard_normal((sink, d)))
    # tail: distinct recent
    keys[n - tail :] = unit(rng.standard_normal((tail, d)))
    # cold middle: low-rank / clustered (redundant)
    centroids = unit(rng.standard_normal((n_topics, d)))
    mid = np.arange(sink, n - tail)
    assign = rng.integers(0, n_topics, size=mid.size)
    keys[mid] = unit(centroids[assign] + noise * rng.standard_normal((mid.size, d)))

    # values: correlated with keys via a fixed linear map + small noise
    w = rng.standard_normal((d, d)).astype(np.float32) / np.sqrt(d)
    values = (keys @ w + 0.05 * rng.standard_normal((n, d))).astype(np.float32)

    # recent queries: aligned with a few topics so salience is structured
    qtopics = centroids[rng.integers(0, n_topics, size=q)]
    queries = unit(qtopics + 0.3 * rng.standard_normal((q, d))).astype(np.float32)

    return KVTrace(
        keys=keys.astype(np.float32),
        values=values,
        queries=queries,
        source="structured-synthetic",
        metadata={"n_topics": n_topics, "noise": noise, "seed": seed},
    )


def cosine_at_budgets(trace: KVTrace, budgets: list[int], sink: int, tail: int):
    results = sweep_benchmarks(
        trace,
        budgets=budgets,
        policies=["full", "window", "h2o_like", "landmark"],
        sink_tokens=sink,
        tail_tokens=tail,
    )
    # group by budget -> {policy: cosine}
    table: dict[int, dict[str, float]] = {}
    for r in results:
        table.setdefault(r.retained_tokens, {})[r.policy] = r.mean_cosine_similarity
    return table


def main() -> None:
    sink, tail = 8, 32
    trace = make_structured_trace(sink=sink, tail=tail)
    n = trace.token_count
    # matched budgets ~ 25%, 33%, 50% retention
    budgets = [int(round(n * frac)) for frac in (0.25, 1 / 3, 0.5)]

    print(f"structured trace: n={n}, d={trace.dim}, q={trace.queries.shape[0]}, "
          f"sink={sink}, tail={tail}")
    print("cosine similarity to exact attention output (higher better):\n")
    print(f"{'retained':>9} {'comp':>6} {'window':>8} {'h2o_like':>9} {'landmark':>9}  winner")
    table = cosine_at_budgets(trace, budgets, sink, tail)
    for kept in sorted(table):
        row = table[kept]
        comp = n / kept
        win = row.get("window", 0.0)
        h2o = row.get("h2o_like", 0.0)
        lm = row.get("landmark", 0.0)
        best = max(("window", win), ("h2o_like", h2o), ("landmark", lm), key=lambda kv: kv[1])[0]
        print(f"{kept/n*100:8.0f}% {comp:5.1f}x {win:8.3f} {h2o:9.3f} {lm:9.3f}  {best}")


if __name__ == "__main__":
    main()
