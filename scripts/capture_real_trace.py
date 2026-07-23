"""Capture a real per-layer KV trace from a HuggingFace decoder model.

Produces the npz format the EigenKache harness expects (see data/README.md):
    keys    (tokens, head_dim)   K vectors of the context, one attention head
    values  (tokens, head_dim)   V vectors of the context, same head
    queries (held_out, head_dim) query vectors of the LAST held_out positions,
                                 for the query head that attends to that KV head

Run inside the RKV-VL-Lab venv (has torch + transformers + MPS):
    ~/RKV-VL-Lab/.venv/bin/python scripts/capture_real_trace.py \
        --text-file data/real/pg19_sample.txt \
        --layer 12 --head 0 \
        --out data/real/qwen05b_L12_H0.npz

The one function Aryan writes himself: extract_layer_trace (see its docstring).
Everything else (model loading, the q_proj forward hook, saving) is plumbing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def extract_layer_trace(
    layer_keys: torch.Tensor,
    layer_values: torch.Tensor,
    q_proj_out: torch.Tensor,
    head: int,
    n_context: int,
    n_queries: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARYAN'S PIECE. Slice one head's trace out of the raw layer tensors.

    You get:
        layer_keys   (1, num_kv_heads, seq, head_dim)  K cache for ONE layer,
                     seq = n_context + n_queries (context then held-out tokens)
        layer_values (1, num_kv_heads, seq, head_dim)  V cache, same layout
        q_proj_out   (1, seq, num_q_heads * head_dim)  raw q_proj output for
                     the same layer, all positions, heads still flattened
        head         which KV head to extract

    You must return three float32 numpy arrays:
        keys    (n_context, head_dim)  -> layer_keys, chosen head, FIRST n_context positions
        values  (n_context, head_dim)  -> same slicing on layer_values
        queries (n_queries, head_dim)  -> from q_proj_out, LAST n_queries positions

    The two things you have to figure out (this is the learning):
      1. q_proj_out has heads flattened in its last dim. Reshape it to
         (seq, num_q_heads, head_dim) before you can pick a head.
      2. GQA: there are more query heads than KV heads. Query heads are grouped;
         group size g = num_q_heads // num_kv_heads, and query heads
         [head*g, head*g + g) all attend to KV head `head`. Pick the FIRST
         query head of the group: q_head = head * g.

    Use .squeeze(0) to drop the batch dim, .float().cpu().numpy() to convert.
    Test yourself with:  ~/RKV-VL-Lab/.venv/bin/python -m pytest tests/test_capture_real_trace.py -q
    """
    raise NotImplementedError("Aryan writes this - see docstring")


def capture(model_name: str, text: str, layer: int, head: int, n_context: int, n_queries: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()

    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_q_heads)

    seq = n_context + n_queries
    ids = tok(text, return_tensors="pt").input_ids[:, :seq]
    if ids.shape[1] < seq:
        raise SystemExit(f"text too short: {ids.shape[1]} tokens, need {seq}")

    captured: dict[str, torch.Tensor] = {}

    def q_hook(_module, _inp, out):
        captured["q"] = out.detach()

    handle = model.model.layers[layer].self_attn.q_proj.register_forward_hook(q_hook)
    try:
        with torch.no_grad():
            out = model(ids.to(device), use_cache=True)
    finally:
        handle.remove()

    past = out.past_key_values
    legacy = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
    layer_keys, layer_values = legacy[layer]

    keys, values, queries = extract_layer_trace(
        layer_keys, layer_values, captured["q"], head,
        n_context, n_queries, num_q_heads, num_kv_heads, head_dim,
    )
    meta = {
        "model": model_name, "layer": layer, "head": head,
        "num_q_heads": num_q_heads, "num_kv_heads": num_kv_heads,
        "n_context": n_context, "n_queries": n_queries, "source": "capture_real_trace",
    }
    return keys, values, queries, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--text-file", required=True, help="real text (PG19 / Gutenberg), NOT synthetic")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--tokens", type=int, default=512, help="context tokens kept as the KV trace")
    p.add_argument("--queries", type=int, default=16, help="held-out query positions")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    text = Path(a.text_file).read_text()
    keys, values, queries, meta = capture(a.model, text, a.layer, a.head, a.tokens, a.queries)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, keys=keys, values=values, queries=queries, metadata=np.array(meta, dtype=object))
    print(f"saved {out}  keys{keys.shape} values{values.shape} queries{queries.shape}")
    print(f"meta: {meta}")


if __name__ == "__main__":
    main()
