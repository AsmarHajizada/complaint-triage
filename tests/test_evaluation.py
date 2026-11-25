"""Tests for the parsing and agreement helpers in scripts/evaluation.py."""

from scripts.evaluation import agreement, parse_lenient, parse_strict


def test_strict_accepts_only_a_bare_json_object():
    assert parse_strict('{"a": 1}') == {"a": 1}
    assert parse_strict('  {"a": 1}\n') == {"a": 1}
    assert parse_strict('```json\n{"a": 1}\n```') is None
    assert parse_strict('Here it is: {"a": 1}') is None
    assert parse_strict("[1, 2]") is None


def test_lenient_finds_the_object_in_a_fence_or_a_sentence():
    assert parse_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_lenient('Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    assert parse_lenient("no json here") is None
    assert parse_lenient('{"a": 1') is None


def test_agreement_counts_two_nulls_as_a_match():
    assert agreement(["x", None, "z"], ["x", None, "y"]) == (2, 3)
