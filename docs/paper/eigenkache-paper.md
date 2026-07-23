# EigenKache: Attention-Conditioned Landmark Compression for Long-Context KV Caches

**Aryan Putta**
Rutgers University — `aryansputta@gmail.com`

Draft v0.1 (work in progress). This document is the paper half of a paper + project hybrid; the implementation lives in the same repository.

---

## Abstract

Long-context transformer decoding is bottlenecked by the key/value (KV) cache,
whose size grows linearly with sequence length. Existing approaches reduce this
cost by improving allocation (PagedAttention), evicting low-importance tokens
(H2O, SnapKV, StreamingLLM), or budgeting across layers and offload paths
(LayerKV, FlowKV). Eviction permanently discards old context: under a hard
memory budget, long-range evidence that is individually low-salience but
collectively informative is lost.

EigenKache reframes the cold region of the KV cache as a *streaming compression*
target rather than an eviction target. It keeps attention-sink tokens and the
hot decode tail exact, and replaces the cold middle with a small set of
attention-mass-balanced *landmarks*: each landmark is a salience-weighted
summary of an equal-attention-mass segment of the cold region. The
representation is deliberately *structured* (contiguous segments, fixed budget)
so it maps to regular, fusible GPU memory access rather than irregular gather.

We describe the method, a CPU reference implementation that exactly reproduces
the partition-and-summarize math, and a proof-of-concept fidelity study against
eviction baselines. On synthetic random traces, landmark compression is
competitive with importance eviction but does not dominate it — an expected
result, since random KV has no redundant structure to compress. This motivates
the central experiment, in progress: evaluating recall fidelity on real
long-context model traces, where the cold region is structurally redundant, and
measuring end-to-end decode latency and memory against vLLM and SGLang.

---

## 1. Introduction

Autoregressive decoding caches per-token keys and values so that each new token
attends to all previous ones without recomputation. The cache size is
`2 * layers * heads * head_dim * seq_len * dtype_bytes`, which at long context
dominates both HBM capacity and the memory traffic of attention. Serving systems
therefore spend significant effort keeping the KV cache small.

Three families of techniques are common:

1. **Allocation / layout.** PagedAttention removes fragmentation by paging KV
   blocks, improving utilization without changing what is stored.
2. **Eviction.** H2O, SnapKV, and StreamingLLM decide which tokens survive a
   budget using accumulated attention or recency, then drop the rest.
3. **Budget routing.** LayerKV and FlowKV/KVDirect allocate or transport KV
   across layers, offload tiers, and disaggregated workers.

These are complementary and effective, but eviction-based methods share a
structural limit: **once a token is dropped, the evidence it carried is gone.**
Under heavy budget pressure, the cold middle of a long context is exactly where
many individually-unremarkable tokens still collectively matter (e.g.
retrieval-style recall across a long document). Dropping them trades recall for
memory in a way that cannot be recovered later in the decode.

**EigenKache's stance.** Old KV state has a third option besides *exact*,
*evicted*, and *offloaded*: keep a *compact summary aligned with future
attention demand*. Concretely, EigenKache:

1. preserves the first `S` sink tokens exactly,
2. preserves the last `T` hot-tail tokens exactly,
3. scores the cold middle by recent-query attention salience,
4. partitions the middle into equal-attention-mass segments, and
5. replaces each segment with one salience-weighted landmark key/value pair,
   optionally keeping the few highest-salience middle tokens exact.

The result fits a fixed budget while retaining global anchors, near-term decode
locality, and a compressed summary of older evidence.

**Contributions.**

- A KV-cache method that treats the cold region as a structured compression
  target (attention-mass landmarks) rather than a set of survivors.
- A CPU reference implementation and an apples-to-apples benchmark harness that
  measures compression ratio, KV bytes saved, attention-output L2 / cosine
  fidelity, and runtime across `full`, `window`, `h2o_like`, and `landmark`
  policies at matched budgets.
- A proof-of-concept fidelity study and an honest negative finding on synthetic
  random traces, which scopes the conditions under which compression should beat
  eviction and defines the real-model evaluation that follows.

---

## 2. Background and Related Work

**Attention and HBM traffic.** FlashAttention shows that attention is
memory-movement-bound; a useful KV optimization must reduce real HBM traffic, not
only token count on paper. EigenKache's landmark layout is contiguous so the
compress-and-attend path can be tiled and fused.

**Paging.** PagedAttention fixes allocation waste but not the *semantic* cost of
a hard budget; it stores the same tokens, just better.

**Eviction.** H2O keeps "heavy hitter" tokens by accumulated attention; SnapKV
selects tokens via recent-query attention pooling; StreamingLLM keeps sink
tokens plus a recent window. All discard the rest. EigenKache reuses the
sink + tail structure of StreamingLLM and the recent-query salience signal of
SnapKV/H2O, but *summarizes* the middle instead of dropping it.

**Budget routing and transport.** LayerKV budgets across layers and offload;
FlowKV/KVDirect reduce disaggregated transfer cost. A smaller
semantically-useful representation should compound with these, since there is
less state to move.

The gap EigenKache targets: *existing systems improve layout, routing, offload,
or eviction, but still lose long-range evidence when old context must fit a hard
budget.*

---

## 3. Method

Let a single head's cached state be `K, V ∈ R^{n×d}` for `n` cached tokens of
head dimension `d`, with a set of recent query vectors `Q ∈ R^{q×d}` used to
estimate salience.

**Salience.** Define per-token salience as the mean recent-query attention mass

```
A = softmax(Q Kᵀ / sqrt(d))            # q × n
s_j = (1/q) Σ_i A[i, j]                 # salience of cached token j
```

**Region split.** Given a token budget `B`, reserve the `S` sink tokens
`[0, S)` and the `T` tail tokens `[n−T, n)` as exact. The cold middle is
`M = [S, n−T)`. The middle budget is `B_m = max(0, B − S − T)`.

**Exact carve-out.** A small fraction of the middle budget keeps the
highest-salience cold tokens exact (in the reference, `⌊B_m / 4⌋`, clamped to
`[1, |M|−1]`). The remaining `B_m − e` budget is spent on landmarks.

**Equal-attention-mass landmarks.** Let the remaining cold tokens (in original
order) have salience `w_1..w_m` with cumulative mass `C_k = Σ_{j≤k} w_j` and
total `C_m`. To produce `L` landmarks, place `L+1` evenly spaced mass boundaries
`b_0=0 < b_1 < ... < b_L = C_m`, `b_i = i·C_m/L`, and assign cold token `k` to
landmark `i` iff `C_{k−1} < b_i ≤ C_k` (via `searchsorted`). Each landmark is the
salience-weighted mean of its segment:

```
K̂_i = Σ_{k∈seg_i} (w_k / Σ w) K_k
V̂_i = Σ_{k∈seg_i} (w_k / Σ w) V_k
```

This is the key design choice: segments carry *equal attention mass*, not equal
token count, so high-salience regions get finer landmarks and flat regions get
coarse ones. The compressed cache is the concatenation
`[K_sink ; K_exact ; K̂ ; K_tail]` (and likewise for `V`), giving a fixed size of
`S + e + L + T` entries regardless of `n`.

**Why structured.** Landmarks are contiguous spans over the cold region, so the
gather-compress-attend path stays regular. Unstructured per-token sparsity tends
to hurt real GPU throughput because access becomes irregular; EigenKache trades a
little theoretical flexibility for memory-access regularity that a fused kernel
can exploit.

---

## 4. Implementation

The repository contains:

- A **CPU reference** (`src/eigenkache/policies.py`) implementing `full`,
  `window` (StreamingLLM-style sink+window), `h2o_like` (salience eviction), and
  `landmark` (this method) behind a common `run_policy` interface, with a
  benchmark harness (`bench.py`) that records compression ratio, KV bytes saved,
  attention-output L2 / cosine fidelity, and runtime.
- A **CUDA-first design** (`cuda/attention_landmark_kernel.cu`, `docs/cuda-plan.md`)
  targeting fused landmark attention with shared-memory tiling, so compression
  and decode are scheduled as one dataflow rather than separate passes.

The CPU reference is the source of truth for correctness; the CUDA path is for
the latency/throughput claims in the planned evaluation.

---

## 5. Experimental Setup

**Proof-of-concept (done).** A synthetic KV trace (`n = 96`, `d = 128`,
`q = 12`) is compressed to matched budgets by every policy, and we measure the
fidelity of the resulting attention output against the exact (`full`) output.
This validates that the harness measures the right thing and that landmark
compression is numerically well-behaved.

**Real-model evaluation (in progress).** The central experiment, which the
proof-of-concept is designed to motivate, fixes the gaps above:

- **Models:** Qwen2.5-7B and Llama-3.1-8B (real attention traces, GQA).
- **Datasets:** LongBench and RULER for recall-sensitive long-context tasks;
  PG19 for perplexity under budget.
- **Baselines:** vLLM and SGLang with (a) full cache, (b) StreamingLLM window,
  (c) H2O/SnapKV eviction — all at matched KV budgets.
- **Metrics:** task accuracy / perplexity vs KV budget; decode latency and
  peak KV memory; HBM traffic from the fused kernel.

The hypothesis is explicit: on real traces whose cold region is structurally
redundant, attention-mass landmarks preserve recall at a given budget better
than eviction, at competitive latency.

---

## 6. Results

Attention-output cosine similarity to the exact cache at three matched budgets
(higher is better). At a given budget all policies hit identical compression
ratios and KV-byte savings, so the comparison is purely about *fidelity per
byte*. We run two regimes differing only in the structure of the cold region.

**Regime 1 — i.i.d. random trace (n=96, d=128).** Each token is independent
Gaussian noise. Eviction wins; with no redundancy, averaging into a landmark
destroys as much as it preserves.

| Retained / Compression | window | h2o_like | landmark |
|---|---|---|---|
| 25% (4.0×) | 0.558 | **0.657** | 0.598 |
| 33% (3.0×) | 0.653 | **0.804** | 0.683 |
| 50% (2.0×) | 0.858 | **0.872** | 0.857 |

**Regime 2 — structured trace (n=512, d=128).** The cold middle is drawn from a
small set of topic centroids plus small noise — a controlled stand-in for the
low-rank redundancy real long-context KV exhibits. Landmark compression **wins at
every budget**, by a widening margin as the budget tightens. `h2o_like` now does
*worse* than a plain window: salience eviction repeatedly picks near-duplicate
high-salience tokens from the same cluster and loses topic coverage, while
attention-mass landmarks place one summary per topic. Reproduce with
`scripts/structured_trace_experiment.py` (output in `build/structured_sweep.txt`).

| Retained / Compression | window | h2o_like | landmark |
|---|---|---|---|
| 25% (4.0×) | 0.867 | 0.692 | **0.902** |
| 33% (3.0×) | 0.930 | 0.736 | **0.944** |
| 50% (2.0×) | 0.965 | 0.857 | **0.977** |

**Takeaway.** The two regimes isolate the mechanism: landmark compression beats
eviction exactly when the cold region is redundant, and only then. Random noise
favors eviction; structured redundancy favors compression. Real long-context
traces are far closer to Regime 2, which is why the real-model evaluation
(Section 5) is expected to favor EigenKache — but that is settled only by
measuring real traces, not by either synthetic regime alone.

---

## 7. Limitations and Next Steps

- The decisive accuracy/latency claims require the real-model benchmark
  (Section 5); current numbers are a CPU-reference proof of concept on synthetic
  data and should not be read as end-to-end wins.
- The CUDA fused kernel is in design; latency and HBM-traffic numbers depend on
  it landing and being profiled against vLLM/SGLang.
- Salience uses a recent-query window; the sensitivity of fidelity to window
  size and to GQA head grouping is unmeasured.
- The exact-carve-out fraction (`B_m/4`) is a heuristic; it should be swept.

**Immediate next steps:** (1) capture real per-layer KV traces from Qwen/Llama on
LongBench; (2) run the matched-budget fidelity study on those traces; (3) wire
the landmark policy behind the torch hook for end-to-end decode; (4) profile the
CUDA landmark-attention kernel for HBM traffic.

---

## References (to be formatted)

- Vaswani et al., *Attention Is All You Need*, 2017.
- Dao et al., *FlashAttention*, 2022.
- Kwon et al., *PagedAttention / vLLM*, 2023.
- Zhang et al., *H2O: Heavy-Hitter Oracle for KV Cache*, 2023.
- Li et al., *SnapKV*, 2024.
- Xiao et al., *StreamingLLM*, 2023.
- *LayerKV*, *FlowKV*, *KVDirect* (KV budgeting and transport).
- LongBench, RULER, PG19 (evaluation benchmarks).
