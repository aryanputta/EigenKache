"""
Transformers integration for EigenKache landmark compression.

Applies landmark_policy to a transformers past_key_values cache operating
on generated tokens only (positions >= prefix_tokens). Prefill KV is kept
exact. This is the bridge between EigenKache numpy benchmarks and real
LLM inference via transformers / Qwen3-VL.

Supports:
    - Legacy tuple-of-tuples cache: ((k0, v0), (k1, v1), ...)
    - DynamicCache (transformers >= 4.38): .to_legacy_cache() / from_legacy_cache()
    - GQA shapes: (batch, num_kv_heads, seq, head_dim)

Shapes after slicing:
    keys/values: (batch, num_kv_heads, retained_tokens, head_dim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .policies import landmark_policy
from .types import KVTrace


@dataclass
class LandmarkDecisionPerLayer:
    layer_idx: int
    original_tokens: int
    retained_tokens: int
    landmark_spans: list[tuple[int, int]]
    exact_middle_tokens: int
    sink_tokens: int
    tail_tokens: int


@dataclass
class LandmarkDecision:
    applied: bool
    prefix_tokens: int
    generated_tokens_before: int
    budget: int
    per_layer: list[LandmarkDecisionPerLayer]

    @property
    def generated_tokens_after(self) -> int:
        if not self.per_layer:
            return 0
        return self.per_layer[0].retained_tokens - self.prefix_tokens


def _to_legacy(past_key_values: Any) -> tuple[Any, type | None]:
    if hasattr(past_key_values, "to_legacy_cache") and callable(past_key_values.to_legacy_cache):
        return past_key_values.to_legacy_cache(), type(past_key_values)
    return past_key_values, None


def _from_legacy(legacy: Any, original_type: type | None) -> Any:
    if original_type is None:
        return legacy
    from_fn = getattr(original_type, "from_legacy_cache", None)
    if callable(from_fn):
        return from_fn(legacy)
    return legacy


def _head_trace(keys: "np.ndarray", values: "np.ndarray", queries: "np.ndarray") -> KVTrace:
    return KVTrace(
        keys=keys.astype(np.float32),
        values=values.astype(np.float32),
        queries=queries.astype(np.float32),
        source="torch_hook",
    )


def apply_landmark_to_past_kv(
    past_key_values: Any,
    prefix_tokens: int,
    budget: int,
    sink_tokens: int = 4,
    tail_tokens: int = 32,
    layers: list[int] | None = None,
) -> tuple[Any, LandmarkDecision]:
    """
    Apply EigenKache landmark compression to generated tokens in a transformer
    KV cache.

    Args:
        past_key_values: transformers DynamicCache or legacy tuple cache.
        prefix_tokens: number of prefill/image tokens to keep exact.
        budget: total generated-token budget (sink + landmarks + tail).
        sink_tokens: sink token count within generated region.
        tail_tokens: hot tail token count within generated region.
        layers: layer indices to compress (None = all layers).

    Returns:
        (compressed_past_key_values, LandmarkDecision)
    """
    import torch

    legacy, original_type = _to_legacy(past_key_values)
    total_tokens = int(legacy[0][0].shape[-2])
    generated_count = total_tokens - prefix_tokens

    if generated_count < 0:
        raise ValueError(
            f"prefix_tokens ({prefix_tokens}) exceeds total cache length ({total_tokens}). "
            "Check that prefix_tokens matches the actual prefill length for this cache."
        )

    if generated_count <= budget:
        per_layer_info = [
            LandmarkDecisionPerLayer(
                layer_idx=i,
                original_tokens=total_tokens,
                retained_tokens=total_tokens,
                landmark_spans=[],
                exact_middle_tokens=0,
                sink_tokens=sink_tokens,
                tail_tokens=tail_tokens,
            )
            for i in range(len(legacy))
        ]
        return past_key_values, LandmarkDecision(
            applied=False,
            prefix_tokens=prefix_tokens,
            generated_tokens_before=generated_count,
            budget=budget,
            per_layer=per_layer_info,
        )

    compress_layers = set(layers) if layers is not None else set(range(len(legacy)))
    new_layers: list[tuple] = []
    per_layer_info: list[LandmarkDecisionPerLayer] = []

    for layer_idx, layer in enumerate(legacy):
        k_tensor, v_tensor = layer[0], layer[1]
        # k_tensor shape: (batch, num_kv_heads, seq, head_dim)
        batch, num_heads, seq_len, head_dim = k_tensor.shape

        if layer_idx not in compress_layers:
            new_layers.append(layer)
            per_layer_info.append(LandmarkDecisionPerLayer(
                layer_idx=layer_idx,
                original_tokens=seq_len,
                retained_tokens=seq_len,
                landmark_spans=[],
                exact_middle_tokens=0,
                sink_tokens=sink_tokens,
                tail_tokens=tail_tokens,
            ))
            continue

        # Process each head independently with landmark_policy
        # Use head 0 queries as proxy for all heads (uniform salience approximation)
        # In full mode, each head gets its own salience from generated keys
        prefix_k = k_tensor[:, :, :prefix_tokens, :]
        gen_k = k_tensor[:, :, prefix_tokens:, :]
        gen_v = v_tensor[:, :, prefix_tokens:, :]

        new_gen_heads_k: list[torch.Tensor] = []
        new_gen_heads_v: list[torch.Tensor] = []
        first_head_meta: dict = {}

        for b in range(batch):
            batch_heads_k: list[torch.Tensor] = []
            batch_heads_v: list[torch.Tensor] = []
            for h in range(num_heads):
                k_np = gen_k[b, h].cpu().float().numpy()
                v_np = gen_v[b, h].cpu().float().numpy()
                # Self-query approximation: use keys as proxy queries for salience scoring.
                # Real salience would require the next-step query vector, which is not
                # available at compression time. This is a known approximation — see
                # Brain/wiki/landmark-decode-policy.md for implications.
                trace = _head_trace(k_np, v_np, k_np)
                compressed = landmark_policy(
                    trace,
                    budget=budget,
                    sink_tokens=sink_tokens,
                    tail_tokens=tail_tokens,
                )
                if b == 0 and h == 0:
                    first_head_meta = compressed.metadata
                ck = torch.from_numpy(compressed.keys).to(k_tensor.device, k_tensor.dtype)
                cv = torch.from_numpy(compressed.values).to(v_tensor.device, v_tensor.dtype)
                batch_heads_k.append(ck)
                batch_heads_v.append(cv)
            # stack heads: (num_heads, retained, head_dim)
            # landmark_policy always returns exactly `budget` tokens (or fewer if
            # sink+tail >= token_count), so all heads have matching retained length.
            head_retained = batch_heads_k[0].shape[0]
            if any(t.shape[0] != head_retained for t in batch_heads_k):
                raise RuntimeError(
                    f"Inconsistent retained lengths across heads in layer {layer_idx}, batch {b}. "
                    "landmark_policy must return the same token count for all heads."
                )
            new_gen_heads_k.append(torch.stack(batch_heads_k, dim=0))
            new_gen_heads_v.append(torch.stack(batch_heads_v, dim=0))

        # stack batch: (batch, num_heads, retained, head_dim)
        new_gen_k = torch.stack(new_gen_heads_k, dim=0)
        new_gen_v = torch.stack(new_gen_heads_v, dim=0)

        prefix_v = v_tensor[:, :, :prefix_tokens, :]
        new_k = torch.cat([prefix_k, new_gen_k], dim=2)
        new_v = torch.cat([prefix_v, new_gen_v], dim=2)

        retained_generated = new_gen_k.shape[2]
        new_layer = (new_k, new_v, *layer[2:])
        new_layers.append(new_layer)
        per_layer_info.append(LandmarkDecisionPerLayer(
            layer_idx=layer_idx,
            original_tokens=seq_len,
            retained_tokens=prefix_tokens + retained_generated,
            landmark_spans=first_head_meta.get("landmark_spans", []),
            exact_middle_tokens=int(first_head_meta.get("exact_middle_budget", 0)),
            sink_tokens=sink_tokens,
            tail_tokens=tail_tokens,
        ))

    compressed_legacy = tuple(new_layers)
    compressed_pkv = _from_legacy(compressed_legacy, original_type)

    return compressed_pkv, LandmarkDecision(
        applied=True,
        prefix_tokens=prefix_tokens,
        generated_tokens_before=generated_count,
        budget=budget,
        per_layer=per_layer_info,
    )
