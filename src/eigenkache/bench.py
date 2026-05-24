from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .policies import run_policy
from .types import BenchmarkResult, KVTrace, MultiHeadBenchmarkResult, MultiHeadKVTrace


def benchmark_trace(
    trace: KVTrace,
    budget: int,
    policies: list[str] | None = None,
    sink_tokens: int = 4,
    tail_tokens: int = 32,
) -> list[BenchmarkResult]:
    chosen = policies or ["full", "window", "h2o_like", "landmark"]
    return [
        run_policy(
            trace,
            policy_name=name,
            budget=budget,
            sink_tokens=sink_tokens,
            tail_tokens=tail_tokens,
        )
        for name in chosen
    ]


def sweep_benchmarks(
    trace: KVTrace,
    budgets: list[int],
    policies: list[str] | None = None,
    sink_tokens: int = 4,
    tail_tokens: int = 32,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for budget in budgets:
        results.extend(
            benchmark_trace(
                trace,
                budget=budget,
                policies=policies,
                sink_tokens=sink_tokens,
                tail_tokens=tail_tokens,
            )
        )
    return results


def benchmark_multi_head(
    trace: MultiHeadKVTrace,
    budget: int,
    policies: list[str] | None = None,
    sink_tokens: int = 4,
    tail_tokens: int = 32,
) -> list[MultiHeadBenchmarkResult]:
    """Benchmark all heads independently and aggregate per policy.

    Each head is benchmarked with the given token budget. This is the
    realistic usage: real transformers allocate the same KV budget per head,
    and compression quality varies across heads depending on attention entropy.
    """
    chosen = policies or ["full", "window", "h2o_like", "landmark"]
    results: list[MultiHeadBenchmarkResult] = []

    for policy_name in chosen:
        per_head: list[BenchmarkResult] = []
        t0 = perf_counter()
        for h in range(trace.num_heads):
            head_trace = trace.head_trace(h)
            per_head.append(
                run_policy(head_trace, policy_name=policy_name, budget=budget,
                           sink_tokens=sink_tokens, tail_tokens=tail_tokens)
            )
        total_ms = (perf_counter() - t0) * 1000.0

        cosine_vals = [r.mean_cosine_similarity for r in per_head]
        compression_vals = [r.compression_ratio for r in per_head]
        bytes_saved_ratio_vals = [r.kv_bytes_saved_ratio for r in per_head]
        l2_vals = [r.mean_l2_error for r in per_head]

        results.append(MultiHeadBenchmarkResult(
            policy=policy_name,
            policy_family=per_head[0].policy_family if per_head else "",
            num_heads=trace.num_heads,
            original_tokens=trace.token_count,
            budget=budget,
            mean_retained_tokens=sum(r.retained_tokens for r in per_head) / len(per_head),
            mean_compression_ratio=sum(compression_vals) / len(compression_vals),
            mean_kv_bytes_saved_ratio=sum(bytes_saved_ratio_vals) / len(bytes_saved_ratio_vals),
            mean_cosine_similarity=sum(cosine_vals) / len(cosine_vals),
            mean_l2_error=sum(l2_vals) / len(l2_vals),
            total_runtime_ms=total_ms,
            per_head_cosine_similarity=cosine_vals,
            per_head_compression_ratio=compression_vals,
        ))

    return results


def write_json(results: list[BenchmarkResult] | list[MultiHeadBenchmarkResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")


def write_csv(results: list[BenchmarkResult] | list[MultiHeadBenchmarkResult], path: str | Path) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
