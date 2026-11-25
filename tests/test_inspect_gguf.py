"""Tests for the grouping helpers in scripts/inspect_gguf.py."""

from scripts.inspect_gguf import by_tensor_name, summarize

TENSORS = [
    ("token_embd.weight", (100, 10), "Q6_K", 820),
    ("blk.0.ffn_down.weight", (10, 10), "Q4_K", 56),
    ("blk.1.ffn_down.weight", (10, 10), "Q6_K", 82),
    ("blk.0.attn_norm.weight", (10,), "F32", 40),
]


def test_summarize_adds_up_parameters_and_bytes_per_type():
    result = summarize(TENSORS)
    assert result["Q6_K"] == {"count": 2, "params": 1100, "bytes": 902}
    assert result["Q4_K"] == {"count": 1, "params": 100, "bytes": 56}
    assert result["F32"]["params"] == 10


def test_by_tensor_name_merges_layers_and_keeps_the_types_apart():
    result = by_tensor_name(TENSORS)
    assert result["blk.N.ffn_down.weight"] == {"Q4_K": 100, "Q6_K": 100}
    assert result["token_embd.weight"] == {"Q6_K": 1000}
