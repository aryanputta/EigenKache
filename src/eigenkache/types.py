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
