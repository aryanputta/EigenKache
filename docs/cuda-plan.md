# CUDA Plan

## Goal

Make `EigenKache` a real CUDA-oriented inference optimization project, not only a Python benchmark.

## Kernel Set

### 1. `reduce_weighted_landmarks`

Purpose:

- compress cold KV spans into one landmark key/value pair each
- do this with regular memory access
- avoid host-side gather loops

Mapping:

- one thread block per landmark span
- threads stride over the hidden dimension
- shared memory accumulates partial weighted sums

### 2. `fused_landmark_attention`

Purpose:

- attend over `[sink | landmarks | tail]`
- keep sink and tail exact
- read landmark summaries from a compact structured buffer

Mapping:

- one block per query token
- tiles of keys and values staged in shared memory
- block-level reduction for softmax max and denominator

### 3. `fused_compress_then_attend`

Purpose:

- remove the extra global-memory round-trip between compression and attention
- make cross-operator optimization explicit

This is where the new `cross opertator optimzation` paper matters most.

## Design Rules From The New Papers

### Structured compression over unstructured sparsity

The new Bo Yuan compression material reinforces that random sparsity often destroys throughput because of irregular memory access and load imbalance.

For EigenKache this means:

- no arbitrary token pruning as the main contribution
- prefer regular segments and regular block shapes
- keep kernel launch geometry simple

### Reorder computation to reduce unnecessary transfers

The accelerator paper's key idea is not only speed, but avoiding unnecessary activation movement.

For EigenKache this means:

- compress once
- attend from the compressed buffer directly
- avoid writing large intermediate buffers back to HBM when possible

### Treat compression and attention as one dataflow problem

The cross-operator attention optimization paper argues that operator-local tuning is not enough.

For EigenKache this means:

- the best kernel may not be the best compressor alone
- the best compressor may not be the best attention kernel alone
- the winning design is the pair that minimizes end-to-end HBM traffic

## Current Machine Status

`nvcc`, `nvidia-smi`, and local `torch` are currently unavailable on this machine.

So:

- CUDA source is added
- CPU reference is tested
- CUDA build/run is blocked by environment, not by missing project code
