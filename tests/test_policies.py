from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eigenkache.bench import benchmark_multi_head, benchmark_trace
from eigenkache.policies import attention_output, h2o_like_policy, landmark_policy, window_policy
from eigenkache.types import KVTrace, MultiHeadKVTrace


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


def build_multi_head_trace(num_heads: int = 4, tokens: int = 64, head_dim: int = 8, queries: int = 6) -> MultiHeadKVTrace:
    rng = np.random.default_rng(42)
    keys = rng.normal(size=(num_heads, tokens, head_dim)).astype(np.float32)
    values = rng.normal(size=(num_heads, tokens, head_dim)).astype(np.float32)
    q = rng.normal(size=(num_heads, queries, head_dim)).astype(np.float32)
    return MultiHeadKVTrace(keys=keys, values=values, queries=q, source="unit-test-multihead")


class MultiHeadTests(unittest.TestCase):
    def test_head_trace_extraction(self) -> None:
        mh = build_multi_head_trace(num_heads=4, tokens=32, head_dim=8)
        head = mh.head_trace(2)
        self.assertEqual(head.keys.shape, (32, 8))
        self.assertEqual(head.queries.shape[1], 8)
        self.assertEqual(head.metadata["head_idx"], 2)

    def test_head_trace_out_of_range(self) -> None:
        mh = build_multi_head_trace(num_heads=4)
        with self.assertRaises(IndexError):
            mh.head_trace(4)

    def test_multi_head_benchmark_shape(self) -> None:
        mh = build_multi_head_trace(num_heads=4, tokens=64, head_dim=8)
        results = benchmark_multi_head(mh, budget=20, policies=["window", "landmark"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.num_heads, 4)
            self.assertEqual(r.original_tokens, 64)
            self.assertEqual(len(r.per_head_cosine_similarity), 4)
            self.assertEqual(len(r.per_head_compression_ratio), 4)

    def test_multi_head_cosine_similarity_range(self) -> None:
        mh = build_multi_head_trace(num_heads=2, tokens=32, head_dim=8)
        results = benchmark_multi_head(mh, budget=16, policies=["landmark"])
        r = results[0]
        self.assertGreaterEqual(r.mean_cosine_similarity, -1.01)
        self.assertLessEqual(r.mean_cosine_similarity, 1.01)
        for v in r.per_head_cosine_similarity:
            self.assertGreaterEqual(v, -1.01)
            self.assertLessEqual(v, 1.01)

    def test_full_policy_preserves_all_heads(self) -> None:
        mh = build_multi_head_trace(num_heads=3, tokens=48, head_dim=8)
        results = benchmark_multi_head(mh, budget=48, policies=["full"])
        r = results[0]
        self.assertAlmostEqual(r.mean_cosine_similarity, 1.0, places=4)
        self.assertEqual(r.mean_compression_ratio, 1.0)

    def test_multi_head_variance_across_heads(self) -> None:
        # landmark policy should show different compression quality per head
        # because attention salience patterns vary across heads
        mh = build_multi_head_trace(num_heads=4, tokens=64, head_dim=16, queries=8)
        results = benchmark_multi_head(mh, budget=24, policies=["landmark"])
        r = results[0]
        per_head = r.per_head_cosine_similarity
        # heads with different salience distributions will have different cosine similarity
        # not all heads should collapse to the same value
        self.assertEqual(len(per_head), 4)
        unique_vals = len(set(round(v, 4) for v in per_head))
        self.assertGreater(unique_vals, 1, "all heads have identical similarity; salience is not varying per head")

    def test_multi_head_invalid_shape(self) -> None:
        with self.assertRaises(ValueError):
            MultiHeadKVTrace(
                keys=np.zeros((4, 64, 8), dtype=np.float32),
                values=np.zeros((4, 64, 8), dtype=np.float32),
                queries=np.zeros((3, 6, 8), dtype=np.float32),  # mismatched num_heads
            )


if __name__ == "__main__":
    unittest.main()
