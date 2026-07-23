"""Trusted benchmark for the KV-compression kernel (ml_agent does NOT edit this).

Setup: fixed seeded queries Q and KV tensors. Full attention over all S tokens
is the reference. The candidate `compress` picks <= budget tokens to keep;
attention is recomputed over only those, and we report the relative L2 error
vs the reference. Goal: MINIMIZE the error.

Correctness gates (raise -> nonzero exit -> ml_agent marks the candidate buggy):
  - indices are 1-D, non-empty, unique, in range
  - the kept count does not exceed the budget (so it can't "win" by keeping all)

Emits: ML_AGENT_METRIC: <relative_error>
"""

import time

import numpy as np

import kernel  # candidate source, sits in the same sandbox workdir

RNG = np.random.default_rng(0)
T, S, d = 16, 512, 64          # queries, sequence length, head dim
BUDGET = 64                    # keep 64 of 512 tokens (8x compression)
Q = RNG.standard_normal((T, d)).astype(np.float32)
K = RNG.standard_normal((S, d)).astype(np.float32)
V = RNG.standard_normal((S, d)).astype(np.float32)

# Structure: ~40 "hot" tokens at random positions carry most attention mass
# (larger key norm -> higher scores). A recency window misses them; a salience
# policy that keeps high-influence tokens preserves attention far better. This
# is EigenKache's thesis in miniature: keep salient/landmark tokens, not recent.
_HOT = RNG.choice(S, 40, replace=False)
K[_HOT] *= 3.5


def attention(Q, K, V):
    s = (Q @ K.T) / np.sqrt(d)
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ V


ref = attention(Q, K, V)

idx = np.asarray(kernel.compress(K, V, BUDGET), dtype=np.int64)
assert idx.ndim == 1 and idx.size > 0, "indices must be a non-empty 1-D array"
assert np.unique(idx).size == idx.size, "duplicate indices"
assert idx.min() >= 0 and idx.max() < S, "index out of range"
assert idx.size <= BUDGET, f"budget exceeded: kept {idx.size} > {BUDGET}"

# Time the selection too (cheap), but the metric is quality at fixed budget.
t0 = time.perf_counter()
for _ in range(50):
    kernel.compress(K, V, BUDGET)
sel_ms = (time.perf_counter() - t0) / 50 * 1000

approx = attention(Q, K[idx], V[idx])
rel_err = float(np.linalg.norm(approx - ref) / np.linalg.norm(ref))

print(f"kept={idx.size}/{S}  select={sel_ms:.3f}ms  rel_err={rel_err:.6f}")
print(f"ML_AGENT_METRIC: {rel_err:.6f}")
