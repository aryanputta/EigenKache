"""Tests for extract_layer_trace (Aryan's piece in scripts/capture_real_trace.py).

Run: ~/RKV-VL-Lab/.venv/bin/python -m pytest tests/test_capture_real_trace.py -q
All three tests fail with NotImplementedError until the function is written.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from capture_real_trace import extract_layer_trace  # noqa: E402

# Small GQA layout: 4 query heads share 2 KV heads (group size 2).
NUM_Q, NUM_KV, DIM = 4, 2, 8
N_CTX, N_HELD = 10, 3
SEQ = N_CTX + N_HELD


def make_inputs():
    # Encode position and head into the values so slicing mistakes are visible:
    # keys[0, h, s, :] == h*100 + s everywhere in the vector.
    k = torch.zeros(1, NUM_KV, SEQ, DIM)
    v = torch.zeros(1, NUM_KV, SEQ, DIM)
    for h in range(NUM_KV):
        for s in range(SEQ):
            k[0, h, s, :] = h * 100 + s
            v[0, h, s, :] = h * 100 + s + 0.5
    # q_proj output: flattened heads. q[0, s, qh*DIM:(qh+1)*DIM] == qh*1000 + s
    q = torch.zeros(1, SEQ, NUM_Q * DIM)
    for s in range(SEQ):
        for qh in range(NUM_Q):
            q[0, s, qh * DIM:(qh + 1) * DIM] = qh * 1000 + s
    return k, v, q


def run(head):
    k, v, q = make_inputs()
    return extract_layer_trace(k, v, q, head, N_CTX, N_HELD, NUM_Q, NUM_KV, DIM)


def test_shapes_and_dtype():
    keys, values, queries = run(head=0)
    assert keys.shape == (N_CTX, DIM)
    assert values.shape == (N_CTX, DIM)
    assert queries.shape == (N_HELD, DIM)
    assert keys.dtype == values.dtype == queries.dtype == np.float32


def test_kv_slicing_first_context_positions_of_chosen_head():
    keys, values, _ = run(head=1)
    # head 1, position s -> 100 + s; only the FIRST N_CTX positions.
    assert np.allclose(keys[0], 100.0)
    assert np.allclose(keys[N_CTX - 1], 100.0 + N_CTX - 1)
    assert np.allclose(values[2], 102.5)


def test_queries_are_last_positions_of_gqa_matched_head():
    _, _, queries = run(head=1)
    # group size g = 4//2 = 2, so KV head 1 -> query head 2 (first of its group).
    # last N_HELD positions are s = 10, 11, 12 -> values 2010, 2011, 2012.
    expected = np.array([2010.0, 2011.0, 2012.0])
    assert np.allclose(queries[:, 0], expected)
