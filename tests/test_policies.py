from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eigenkache.bench import benchmark_trace
from eigenkache.policies import attention_output, h2o_like_policy, landmark_policy, window_policy
from eigenkache.types import KVTrace


def build_trace(tokens: int = 64, dim: int = 8, queries: int = 6) -> KVTrace:
    rng = np.random.default_rng(7)
    keys = rng.normal(size=(tokens, dim)).astype(np.float32)
    values = rng.normal(size=(tokens, dim)).astype(np.float32)
    q = rng.normal(size=(queries, dim)).astype(np.float32)
    return KVTrace(keys=keys, values=values, queries=q, source="unit-test")


class PolicyTests(unittest.TestCase):
    def test_window_policy_respects_budget(self) -> None:
        trace = build_trace()
        compressed = window_policy(trace, budget=20, sink_tokens=4, tail_tokens=6)
        self.assertEqual(compressed.token_count, 20)

    def test_h2o_policy_respects_budget(self) -> None:
        trace = build_trace()
        compressed = h2o_like_policy(trace, budget=18, sink_tokens=4, tail_tokens=6)
        self.assertEqual(compressed.token_count, 18)

    def test_landmark_policy_respects_budget(self) -> None:
        trace = build_trace()
        compressed = landmark_policy(trace, budget=16, sink_tokens=4, tail_tokens=4)
        self.assertEqual(compressed.token_count, 16)

    def test_full_attention_output_shape(self) -> None:
        trace = build_trace()
        output = attention_output(trace.queries, trace.keys, trace.values)
        self.assertEqual(output.shape, trace.queries.shape)

    def test_benchmark_returns_all_policies(self) -> None:
        trace = build_trace()
        results = benchmark_trace(trace, budget=20)
        self.assertEqual({r.policy for r in results}, {"full", "window", "h2o_like", "landmark"})

    def test_benchmark_reports_token_and_cache_metrics(self) -> None:
        trace = build_trace(tokens=80, dim=8, queries=5)
        result = benchmark_trace(trace, budget=20, policies=["window"], sink_tokens=4, tail_tokens=4)[0]
        self.assertEqual(result.original_tokens, 80)
        self.assertEqual(result.retained_tokens, 20)
        self.assertEqual(result.token_savings, 60)
        self.assertGreater(result.kv_bytes_saved, 0)
        self.assertGreater(result.estimated_attention_bytes_saved, 0)
        self.assertEqual(result.kernel_status, "cpu_reference")
        self.assertEqual(result.policy_family, "evict")

    def test_landmark_policy_marks_compression_family(self) -> None:
        trace = build_trace(tokens=96, dim=8, queries=6)
        result = benchmark_trace(trace, budget=24, policies=["landmark"], sink_tokens=4, tail_tokens=4)[0]
        self.assertEqual(result.policy_family, "compress")
        self.assertIn("landmark_budget", result.metadata)


if __name__ == "__main__":
    unittest.main()
