# EigenKache First Paper Build Plan

## Working title

**EigenKache: Attention-Conditioned Landmark Compression for Memory-Bounded Long-Context Decoding**

## Narrow research question

At the same retained-token budget, can attention-conditioned landmark compression preserve more useful long-range context than sink+window and salience/eviction baselines, while keeping runtime overhead bounded?

## Minimum publishable contribution

1. A clearly specified compression policy with exact sink and hot-tail retention.
2. A reproducible CPU reference implementation.
3. Comparisons against full, window, and H2O-like policies.
4. Real model traces from a documented long-context workload.
5. Quality, memory, and runtime measurements with confidence intervals or repeated seeds.
6. An honest limitations section covering missing GPU/end-to-end evidence.

Do not claim a CUDA speedup until the kernel runs on real hardware. The current repository has a CPU reference and CUDA sources, but this Mac has no `nvcc`, `nvidia-smi`, or local PyTorch.

## Build sequence

### Phase 1 — Reproduce (July)

- Run the existing tests and example sweep.
- Freeze baseline outputs and environment metadata.
- Write the exact policy equations and budget definitions.
- Add deterministic seed/config files.

### Phase 2 — Validate the question (August)

- Capture real traces from one supported model/workload.
- Run full, window, H2O-like, and landmark policies at equal budgets.
- Produce quality-vs-memory and error-vs-budget plots.
- Add threshold-boundary and multi-head tests.

### Phase 3 — Systems evidence (September)

- Run on a CUDA machine or rented GPU.
- Compare kernel status, runtime, memory traffic proxy, TTFT/TPOT where available.
- Profile before changing kernels; keep a CPU reference oracle.

### Phase 4 — Write and release (October–December)

- Draft abstract, introduction, method, experiments, limitations, and related work.
- Release code, configs, trace-generation instructions, and raw/processed results.
- Ask two technically informed reviewers to reproduce one table.
- Submit to an appropriate workshop or post a reproducible arXiv preprint after advisor/reviewer feedback.

## Weekly research rhythm

- 2 sessions: linear algebra/attention derivation.
- 2 sessions: C/CUDA or memory-system implementation.
- 1 session: experiment and plotting.
- 1 session: paper reading and related-work matrix.
- 1 session: writing one paragraph from measured evidence.

## Paper notebook entry

```text
Claim being tested:
Baseline/config:
Dataset/trace/model:
Budget:
Metric:
Expected result:
Observed result:
Plot/table updated:
Threat to validity:
Next experiment:
```

## Current baseline

`python3 -m pytest -q` currently reports **14 passed, 1 skipped**. The skipped test requires PyTorch for real-trace capture. That is a known environment limitation, not evidence for the paper's claim.
