from __future__ import annotations

import argparse
import json

from .bench import benchmark_trace, sweep_benchmarks, write_csv, write_json
from .policies import result_to_dict
from .types import KVTrace


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eigenkache")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Benchmark a trace file")
    bench.add_argument("trace", help="Path to an NPZ trace with keys, values, queries")
    bench.add_argument("--budget", type=int, required=True, help="Retained token budget")
    bench.add_argument("--sink-tokens", type=int, default=4)
    bench.add_argument("--tail-tokens", type=int, default=32)
    bench.add_argument(
        "--policies",
        nargs="+",
        default=["full", "window", "h2o_like", "landmark"],
    )

    sweep = subparsers.add_parser("sweep", help="Benchmark multiple budgets and optionally write reports")
    sweep.add_argument("trace", help="Path to an NPZ trace with keys, values, queries")
    sweep.add_argument("--budgets", nargs="+", type=int, required=True, help="Retained token budgets")
    sweep.add_argument("--sink-tokens", type=int, default=4)
    sweep.add_argument("--tail-tokens", type=int, default=32)
    sweep.add_argument(
        "--policies",
        nargs="+",
        default=["full", "window", "h2o_like", "landmark"],
    )
    sweep.add_argument("--json-out", help="Optional JSON report path")
    sweep.add_argument("--csv-out", help="Optional CSV report path")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        trace = KVTrace.load(args.trace)
        results = benchmark_trace(
            trace,
            budget=args.budget,
            policies=args.policies,
            sink_tokens=args.sink_tokens,
            tail_tokens=args.tail_tokens,
        )
        print(json.dumps([result_to_dict(r) for r in results], indent=2))
    elif args.command == "sweep":
        trace = KVTrace.load(args.trace)
        results = sweep_benchmarks(
            trace,
            budgets=args.budgets,
            policies=args.policies,
            sink_tokens=args.sink_tokens,
            tail_tokens=args.tail_tokens,
        )
        if args.json_out:
            write_json(results, args.json_out)
        if args.csv_out:
            write_csv(results, args.csv_out)
        print(json.dumps([result_to_dict(r) for r in results], indent=2))


if __name__ == "__main__":
    main()
