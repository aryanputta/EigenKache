# Research Foundation

## Core Thesis

Old KV state should not be treated as either:

- fully exact
- fully evicted
- merely offloaded

There is a third option:

- retain a compact summary aligned with future attention demand

## Source Mapping

### `Attention Is All You Need`

Reason to use attention itself as the organizing signal. The compression target is whatever best preserves downstream attention output.

### `FlashAttention`

Reason to care about HBM traffic and kernel shape. A useful compressor must reduce real memory movement, not only token count on paper.

### `PagedAttention`

Reason to separate allocation efficiency from semantic preservation. Paging fixes waste, not the semantic cost of severe budget pressure.

### `LayerKV`

Reason to think in budgets and pressure regimes instead of one global exact-cache assumption.

### `FlowKV` and `KVDirect`

Reason to keep the representation compact before transport. Smaller semantically-useful state should help disaggregated transfer paths too.

### `AdaSkip` and `GateSkip`

Reason to borrow adaptive, importance-aware thinking from compute skipping and apply it to memory-state compression.

## Comparison Claim

`EigenKache` should be compared to three classes of methods:

1. Memory layout methods
2. Token-retention or eviction methods
3. Tiering or transfer methods

The goal is not to replace all three. The goal is to show that representation-aware compression is a missing axis.

## Honest Scope

This repo currently implements:

- trace-based CPU reference benchmarking
- sink/window/H2O-like/landmark comparisons
- CUDA environment readiness checking

This repo does not yet implement:

- direct Hugging Face trace extraction
- fused CUDA kernels
- end-to-end TTFT and TPOT measurement on a live model server
