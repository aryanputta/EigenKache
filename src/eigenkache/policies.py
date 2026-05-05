from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

import numpy as np

from .types import BenchmarkResult, CompressedKV, KVTrace


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def _bytes_per_token(dim: int) -> int:
    return dim * 4 * 2


def _estimate_attention_bytes(query_count: int, token_count: int, dim: int) -> int:
    return query_count * token_count * dim * 4 * 2


def attention_output(queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    scale = np.sqrt(keys.shape[1], dtype=np.float32)
    scores = queries @ keys.T / scale
    weights = _softmax(scores.astype(np.float32))
    return weights @ values


def salience_scores(trace: KVTrace) -> np.ndarray:
    scale = np.sqrt(trace.dim, dtype=np.float32)
    scores = trace.queries @ trace.keys.T / scale
    weights = _softmax(scores.astype(np.float32))
    return weights.mean(axis=0)


def _normalize_budget(token_count: int, budget: int, sink_tokens: int, tail_tokens: int) -> tuple[int, int, int]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    sink = min(max(sink_tokens, 0), token_count)
    tail = min(max(tail_tokens, 0), max(token_count - sink, 0))
    required = sink + tail
    if budget < required:
        tail = max(0, budget - sink)
        required = sink + tail
    return budget, sink, required


def full_policy(trace: KVTrace, budget: int | None = None, sink_tokens: int = 4, tail_tokens: int = 32) -> CompressedKV:
    del budget, sink_tokens, tail_tokens
    return CompressedKV(
        name="full",
        keys=trace.keys.copy(),
        values=trace.values.copy(),
        metadata={"policy_family": "exact"},
    )


def window_policy(trace: KVTrace, budget: int, sink_tokens: int = 4, tail_tokens: int = 32) -> CompressedKV:
    budget, sink, required = _normalize_budget(trace.token_count, budget, sink_tokens, tail_tokens)
    if budget >= trace.token_count:
        return full_policy(trace)
    tail = required - sink
    middle_budget = max(0, budget - required)
    start = sink
    end = trace.token_count - tail
    middle_start = max(start, end - middle_budget)
    indices = np.concatenate(
        [
            np.arange(0, sink, dtype=np.int32),
            np.arange(middle_start, end, dtype=np.int32),
            np.arange(end, trace.token_count, dtype=np.int32),
        ]
    )
    return CompressedKV(
        name="window",
        keys=trace.keys[indices],
        values=trace.values[indices],
        metadata={
            "indices": indices.tolist(),
            "policy_family": "evict",
            "sink_tokens": sink,
            "tail_tokens": tail,
            "middle_tokens_retained": middle_budget,
        },
    )


def h2o_like_policy(trace: KVTrace, budget: int, sink_tokens: int = 4, tail_tokens: int = 32) -> CompressedKV:
    budget, sink, required = _normalize_budget(trace.token_count, budget, sink_tokens, tail_tokens)
    if budget >= trace.token_count:
        return full_policy(trace)
    tail = required - sink
    middle_budget = max(0, budget - required)
    end = trace.token_count - tail
    salience = salience_scores(trace)
    middle_indices = np.arange(sink, end, dtype=np.int32)
    if middle_budget > 0 and middle_indices.size > 0:
        middle_scores = salience[middle_indices]
        keep = middle_indices[np.argsort(-middle_scores)[:middle_budget]]
        keep = np.sort(keep)
    else:
        keep = np.array([], dtype=np.int32)
    indices = np.concatenate(
        [
            np.arange(0, sink, dtype=np.int32),
            keep,
            np.arange(end, trace.token_count, dtype=np.int32),
        ]
    )
    return CompressedKV(
        name="h2o_like",
        keys=trace.keys[indices],
        values=trace.values[indices],
        metadata={
            "indices": indices.tolist(),
            "salience_mean": float(salience.mean()),
            "policy_family": "evict",
            "sink_tokens": sink,
            "tail_tokens": tail,
            "middle_tokens_retained": middle_budget,
        },
    )


def landmark_policy(trace: KVTrace, budget: int, sink_tokens: int = 4, tail_tokens: int = 32) -> CompressedKV:
    budget, sink, required = _normalize_budget(trace.token_count, budget, sink_tokens, tail_tokens)
    if budget >= trace.token_count:
        return full_policy(trace)

    tail = required - sink
    middle_budget = max(0, budget - required)
    end = trace.token_count - tail

    prefix_keys = trace.keys[:sink]
    prefix_values = trace.values[:sink]
    suffix_keys = trace.keys[end:]
    suffix_values = trace.values[end:]

    if middle_budget == 0 or end <= sink:
        keys = np.concatenate([prefix_keys, suffix_keys], axis=0)
        values = np.concatenate([prefix_values, suffix_values], axis=0)
        return CompressedKV(
            name="landmark",
            keys=keys,
            values=values,
            metadata={
                "landmarks": [],
                "policy_family": "compress",
                "sink_tokens": sink,
                "tail_tokens": tail,
                "exact_middle_budget": 0,
                "landmark_budget": 0,
            },
        )

    salience = salience_scores(trace)[sink:end]
    salience = np.maximum(salience, 1e-8)
    middle_indices = np.arange(sink, end, dtype=np.int32)

    exact_budget = 0
    if middle_budget >= 4 and middle_indices.size >= 2:
        exact_budget = min(max(middle_budget // 4, 1), middle_indices.size - 1)
    landmark_budget = middle_budget - exact_budget

    exact_indices = np.array([], dtype=np.int32)
    if exact_budget > 0:
        exact_indices = middle_indices[np.argsort(-salience)[:exact_budget]]
        exact_indices = np.sort(exact_indices)

    compress_mask = np.ones(middle_indices.shape[0], dtype=bool)
    if exact_indices.size > 0:
        compress_mask[np.isin(middle_indices, exact_indices)] = False

    compressed_indices = middle_indices[compress_mask]
    compressed_salience = salience[compress_mask] if compressed_indices.size else np.empty((0,), dtype=np.float32)
    compressed_keys = trace.keys[compressed_indices] if compressed_indices.size else np.empty((0, trace.dim), dtype=np.float32)
    compressed_values = trace.values[compressed_indices] if compressed_indices.size else np.empty((0, trace.dim), dtype=np.float32)

    landmark_keys = []
    landmark_values = []
    spans = []
    if landmark_budget > 0 and compressed_indices.size > 0:
        effective_landmarks = min(landmark_budget, compressed_indices.size)
        cumulative = np.cumsum(compressed_salience)
        total = float(cumulative[-1])
        boundaries = np.linspace(0.0, total, num=effective_landmarks + 1, dtype=np.float32)

        start = 0
        for i in range(effective_landmarks):
            upper = boundaries[i + 1]
            stop = int(np.searchsorted(cumulative, upper, side="right"))
            stop = max(stop, start + 1)
            stop = min(stop, compressed_keys.shape[0])
            chunk_k = compressed_keys[start:stop]
            chunk_v = compressed_values[start:stop]
            chunk_w = compressed_salience[start:stop]
            weights = chunk_w / np.sum(chunk_w)
            landmark_keys.append(np.sum(chunk_k * weights[:, None], axis=0))
            landmark_values.append(np.sum(chunk_v * weights[:, None], axis=0))
            spans.append((int(compressed_indices[start]), int(compressed_indices[stop - 1] + 1)))
            start = stop

        if start < compressed_keys.shape[0] and landmark_keys:
            chunk_k = compressed_keys[start:]
            chunk_v = compressed_values[start:]
            chunk_w = compressed_salience[start:]
            weights = chunk_w / np.sum(chunk_w)
            landmark_keys[-1] = landmark_keys[-1] + np.sum(chunk_k * weights[:, None], axis=0)
            landmark_values[-1] = landmark_values[-1] + np.sum(chunk_v * weights[:, None], axis=0)
            spans[-1] = (spans[-1][0], int(compressed_indices[-1] + 1))

    exact_keys = trace.keys[exact_indices] if exact_indices.size else np.empty((0, trace.dim), dtype=np.float32)
    exact_values = trace.values[exact_indices] if exact_indices.size else np.empty((0, trace.dim), dtype=np.float32)

    keys = np.concatenate(
        [prefix_keys, exact_keys, np.asarray(landmark_keys, dtype=np.float32), suffix_keys],
        axis=0,
    )
    values = np.concatenate(
        [prefix_values, exact_values, np.asarray(landmark_values, dtype=np.float32), suffix_values],
        axis=0,
    )
    return CompressedKV(
        name="landmark",
        keys=keys,
        values=values,
        metadata={
            "landmark_spans": spans,
            "landmark_budget": landmark_budget,
            "exact_middle_budget": int(exact_indices.size),
            "exact_middle_indices": exact_indices.tolist(),
            "policy_family": "compress",
            "sink_tokens": sink,
            "tail_tokens": tail,
            "compressed_cold_tokens": int(compressed_indices.size),
        },
    )


def run_policy(
    trace: KVTrace,
    policy_name: str,
    budget: int,
    sink_tokens: int = 4,
    tail_tokens: int = 32,
) -> BenchmarkResult:
    policy_map = {
        "full": full_policy,
        "window": window_policy,
        "h2o_like": h2o_like_policy,
        "landmark": landmark_policy,
    }
    if policy_name not in policy_map:
        raise ValueError(f"unknown policy: {policy_name}")

    started = perf_counter()
    compressed = policy_map[policy_name](trace, budget=budget, sink_tokens=sink_tokens, tail_tokens=tail_tokens)
    approx = attention_output(trace.queries, compressed.keys, compressed.values)
    elapsed_ms = (perf_counter() - started) * 1000.0
    exact = attention_output(trace.queries, trace.keys, trace.values)
    diff = exact - approx
    l2_error = float(np.linalg.norm(diff, axis=1).mean())
    cosine_num = np.sum(exact * approx, axis=1)
    cosine_den = np.linalg.norm(exact, axis=1) * np.linalg.norm(approx, axis=1) + 1e-8
    cosine = float(np.mean(cosine_num / cosine_den))
    original_tokens = trace.token_count
    retained_tokens = compressed.token_count
    original_kv_bytes = original_tokens * _bytes_per_token(trace.dim)
    retained_kv_bytes = retained_tokens * _bytes_per_token(trace.dim)
    estimated_attention_bytes = _estimate_attention_bytes(trace.queries.shape[0], original_tokens, trace.dim)
    retained_attention_bytes = _estimate_attention_bytes(trace.queries.shape[0], retained_tokens, trace.dim)
    return BenchmarkResult(
        policy=policy_name,
        policy_family=str(compressed.metadata.get("policy_family", "unknown")),
        original_tokens=original_tokens,
        retained_tokens=retained_tokens,
        query_count=int(trace.queries.shape[0]),
        token_keep_ratio=float(retained_tokens / max(original_tokens, 1)),
        token_savings=original_tokens - retained_tokens,
        compression_ratio=float(original_tokens / max(retained_tokens, 1)),
        original_kv_bytes=original_kv_bytes,
        retained_kv_bytes=retained_kv_bytes,
        kv_bytes_saved=original_kv_bytes - retained_kv_bytes,
        kv_bytes_saved_ratio=float((original_kv_bytes - retained_kv_bytes) / max(original_kv_bytes, 1)),
        estimated_attention_bytes=estimated_attention_bytes,
        estimated_attention_bytes_saved=estimated_attention_bytes - retained_attention_bytes,
        mean_l2_error=l2_error,
        mean_cosine_similarity=cosine,
        runtime_ms=elapsed_ms,
        kernel_status="cpu_reference",
        metadata=compressed.metadata,
    )


def result_to_dict(result: BenchmarkResult) -> dict[str, object]:
    return asdict(result)
