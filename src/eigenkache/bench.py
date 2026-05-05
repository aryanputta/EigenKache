from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .policies import run_policy
from .types import BenchmarkResult, KVTrace


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


def write_json(results: list[BenchmarkResult], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")


def write_csv(results: list[BenchmarkResult], path: str | Path) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
