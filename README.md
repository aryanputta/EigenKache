# EigenKache

`EigenKache` is a KV-cache compression project for long-context transformer inference.

The project takes a different stance from eviction-heavy and tiering-heavy systems:

- keep attention sink tokens exact
- keep the hot decode tail exact
- compress the cold middle into attention-conditioned landmarks instead of discarding it

That makes it a compression problem over attention structure, not only a cache policy problem.

The implementation target is CUDA-first:

- structured compression that keeps memory access regular
- fused kernels that reduce HBM traffic
- shared-memory tiling for landmark attention
- cross-operator scheduling that treats compression and decode as one dataflow problem

## Gap

Current systems and papers do useful but narrower things:

- `PagedAttention` reduces fragmentation and improves allocation efficiency
- `H2O`, `SnapKV`, and `StreamingLLM` decide which tokens survive a budget
- `LayerKV` allocates budget across layers and offload paths
- `FlowKV` and `KVDirect` reduce disaggregated transfer cost
- `MultieLayer-Kache` already explores adaptive tiering, quantization, offload, and eviction

The remaining gap is:

> Existing KV systems improve layout, routing, offload, or eviction, but they still lose useful long-range evidence when old context must fit a hard memory budget. This matters because long-context decode often needs compressed recall, not just fewer tokens.

## Unique Standpoint

`EigenKache` treats old KV state as a streaming compression target.

For each head or trace slice:

1. Preserve the first `S` sink tokens exactly
2. Preserve the last `T` hot-tail tokens exactly
3. Score the middle region using recent-query attention salience
4. Partition the middle into equal-attention-mass segments
5. Replace each segment with one weighted landmark key/value pair

The important hardware constraint is that this summary must stay structured. The Bo Yuan compression papers in your Brain reinforce that unstructured sparsity often hurts real performance because access becomes irregular. So EigenKache prefers:

- landmark blocks over random pruning
- regular segment layouts over arbitrary token dropping
- fused gather-compress-attend paths over isolated passes

This gives a fixed memory budget while preserving:

- global anchors
- near-term decode locality
- a compressed summary of older evidence

## Why This Is Not A Repeat

It is intentionally different from your existing KV work:

- not another routing project like `NimbusMesh-X`
- not another serving scheduler like `LayerKV Serve`
- not another hybrid tier controller like `MultieLayer-Kache`
- not only an eviction benchmark like `kvcache-bench`

The claim here is narrower and sharper:

> Under the same token budget, attention-conditioned landmark compression can preserve more useful context than pure eviction baselines.

## Research Base From Brain

- `[[Source: Attention Is All You Need]]` for the attention mechanism itself
- `[[Source: flastattention]]` for IO-aware attention cost on GPU
- `[[Source: memeory mangement]]` and `[[Source: vLLM paper]]` for KV layout and paging
- `[[Source: LayerKV]]` for memory pressure as a budget allocation problem
- `[[Source: FlowKV]]` and `[[Source: KVDirect]]` for disaggregated serving pressure
- `[[Source: AdaSkip Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference]]`
- `[[Source: What Layers When Learning to Skip Compute in LLMs with Residual Gates]]`
- `[[Source: DNN model copression]]` for structured tensor compression instead of arbitrary sparsity
- `[[Source: cross opertator optimzation]]` for attention dataflow, tiling, buffer management, and exhaustive design-space pruning
- `[[Source: neural network accelerator]]` for computation reordering and avoiding unnecessary transfers
- `[[Source: vison langauge modles]]` for training-free attention-guided inference signals

## Datasets

Use real traces only.

Primary evaluation targets:

- `LongBench` for long-context quality
- `PG19` for long-form streaming perplexity traces

Recommended first slice:

- `qasper`
- `gov_report`
- `narrativeqa`
- `multi_news`

## Baselines

Compare against:

- `full` exact KV
- `window` sink + local window
- `h2o_like` top-salience exact retention
- `landmark` attention-conditioned summary compression

When the runtime environment is ready, extend comparison to:

- `SnapKV`
- `StreamingLLM`
- `LayerKV` budget profiles
- `MultieLayer-Kache` hybrid policy

## Metrics

- `compression_ratio`
- `retained_tokens`
- `token_keep_ratio`
- `token_savings`
- `original_kv_bytes`, `retained_kv_bytes`, `kv_bytes_saved`
- `estimated_attention_bytes_saved` as a proxy for HBM traffic reduction
- `mean_l2_error`
- `mean_cosine_similarity`
- `policy_runtime_ms`
- `kernel_status`
- `TTFT`, `TPOT`, and task metric once real model tracing is enabled

## CUDA Position

This repo now includes actual CUDA kernel sources and a build/test script, but this machine does not currently expose:

- `nvidia-smi`
- `nvcc`
- local `torch`

So the CPU reference path is implemented now, and the CUDA kernel path is present but cannot be compiled or executed here.

## Layout

```text
EigenKache/
├── README.md
├── pyproject.toml
├── data/
├── docs/
├── scripts/
├── src/eigenkache/
└── tests/
```

## Trace Format

Benchmarks use `.npz` traces with:

- `keys`: `(tokens, dim)`
- `values`: `(tokens, dim)`
- `queries`: `(queries, dim)`

These traces should come from real model runs on LongBench or PG19, not synthetic production benchmarks.

## Quick Start

```bash
cd /Users/srini/EigenKache
python3 -m unittest discover -s tests
python3 scripts/check_cuda_env.py
sh scripts/test_cuda_kernels.sh
PYTHONPATH=src python3 -m eigenkache.cli benchmark data/example_trace.npz --budget 32 --tail-tokens 8
PYTHONPATH=src python3 -m eigenkache.cli sweep data/example_trace.npz --budgets 24 32 48 --tail-tokens 8 --json-out build/sweep.json --csv-out build/sweep.csv
```

## Master Build Prompt

```text
Project: EigenKache
Gap: Existing KV systems improve layout, transfer, or eviction, but fail to preserve long-range evidence when old context must fit a fixed decode-time memory budget.
Stack: Python 3.11, NumPy, C++17, CUDA kernels, optional PyTorch trace extraction, LongBench, PG19
Data: Real KV traces extracted from LongBench and PG19 prompts into NPZ files with keys, values, and held-out decode queries

Build:
1. Trace extractor - run a real decoder model on LongBench/PG19 and save KV/query traces; verify by shape checks and reproducible NPZ outputs.
2. Baseline policies - implement full, window, and H2O-like policies; verify by retained-token counts and exact output reconstruction.
3. EigenKache compressor - build attention-conditioned landmark compression; verify that the output shape respects the token budget and preserves sink/tail spans exactly.
4. CUDA path - implement landmark pooling and fused attention over anchors, landmarks, and tail with shared-memory tiling and regular block layouts; verify against the CPU reference within numeric tolerance.
5. Benchmark harness - compare policies on attention-output error, memory ratio, kernel runtime, and later TTFT/TPOT; verify with JSON/CSV summaries.

Benchmark against: window, H2O-like retention, and later SnapKV/StreamingLLM/LayerKV-style budget profiles
Success: At the same retained-token budget, landmark compression improves cosine similarity and lowers output error versus pure eviction while keeping runtime overhead bounded.
```
