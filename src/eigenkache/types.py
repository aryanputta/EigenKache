from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class KVTrace:
    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.keys.ndim != 2 or self.values.ndim != 2 or self.queries.ndim != 2:
            raise ValueError("keys, values, and queries must be rank-2 arrays")
        if self.keys.shape != self.values.shape:
            raise ValueError("keys and values must have identical shapes")
        if self.keys.shape[1] != self.queries.shape[1]:
            raise ValueError("query dimension must match key/value dimension")

    @property
    def token_count(self) -> int:
        return int(self.keys.shape[0])

    @property
    def dim(self) -> int:
        return int(self.keys.shape[1])

    @classmethod
    def load(cls, path: str | Path) -> "KVTrace":
        data = np.load(path, allow_pickle=True)
        metadata = {}
        if "metadata" in data.files:
            raw = data["metadata"]
            if isinstance(raw, np.ndarray) and raw.shape == ():
                metadata = dict(raw.item())
        return cls(
            keys=np.asarray(data["keys"], dtype=np.float32),
            values=np.asarray(data["values"], dtype=np.float32),
            queries=np.asarray(data["queries"], dtype=np.float32),
            source=str(path),
            metadata=metadata,
        )


@dataclass
class MultiHeadKVTrace:
    """KV trace for multi-head attention: shape (num_heads, tokens, head_dim)."""

    keys: np.ndarray
    values: np.ndarray
    queries: np.ndarray
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.keys.ndim != 3 or self.values.ndim != 3 or self.queries.ndim != 3:
            raise ValueError("keys, values, and queries must be rank-3 arrays (num_heads, tokens, head_dim)")
        if self.keys.shape != self.values.shape:
            raise ValueError("keys and values must have identical shapes")
        if self.keys.shape[0] != self.queries.shape[0]:
            raise ValueError("num_heads must match across keys, values, and queries")
        if self.keys.shape[2] != self.queries.shape[2]:
            raise ValueError("head_dim must match across keys, values, and queries")

    @property
    def num_heads(self) -> int:
        return int(self.keys.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.keys.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.keys.shape[2])

    def head_trace(self, head_idx: int) -> KVTrace:
        """Extract a single head as a KVTrace for per-head policy application."""
        if not (0 <= head_idx < self.num_heads):
            raise IndexError(f"head_idx {head_idx} out of range [0, {self.num_heads})")
        return KVTrace(
            keys=self.keys[head_idx],
            values=self.values[head_idx],
            queries=self.queries[head_idx],
            source=self.source,
            metadata={**self.metadata, "head_idx": head_idx},
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultiHeadKVTrace":
        data = np.load(path, allow_pickle=True)
        metadata = {}
        if "metadata" in data.files:
            raw = data["metadata"]
            if isinstance(raw, np.ndarray) and raw.shape == ():
                metadata = dict(raw.item())
        return cls(
            keys=np.asarray(data["keys"], dtype=np.float32),
            values=np.asarray(data["values"], dtype=np.float32),
            queries=np.asarray(data["queries"], dtype=np.float32),
            source=str(path),
            metadata=metadata,
        )


@dataclass
class MultiHeadBenchmarkResult:
    """Aggregated benchmark result across all heads for one policy and budget."""

    policy: str
    policy_family: str
    num_heads: int
    original_tokens: int
    budget: int
    mean_retained_tokens: float
    mean_compression_ratio: float
    mean_kv_bytes_saved_ratio: float
    mean_cosine_similarity: float
    mean_l2_error: float
    total_runtime_ms: float
    per_head_cosine_similarity: list[float] = field(default_factory=list)
    per_head_compression_ratio: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressedKV:
    name: str
    keys: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return int(self.keys.shape[0])


@dataclass
class BenchmarkResult:
    policy: str
    policy_family: str
    original_tokens: int
    retained_tokens: int
    query_count: int
    token_keep_ratio: float
    token_savings: int
    compression_ratio: float
    original_kv_bytes: int
    retained_kv_bytes: int
    kv_bytes_saved: int
    kv_bytes_saved_ratio: float
    estimated_attention_bytes: int
    estimated_attention_bytes_saved: int
    mean_l2_error: float
    mean_cosine_similarity: float
    runtime_ms: float
    kernel_status: str
    metadata: dict[str, Any] = field(default_factory=dict)
