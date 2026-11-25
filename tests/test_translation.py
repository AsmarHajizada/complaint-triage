"""Tests for the checks in scripts/translation.py."""

from scripts.translation import count_special_letters, kept_unchanged

SOURCE = "On XX/XX/2024 Chase charged me {$460.00} and XXXX XXXX refused."


def test_kept_unchanged_accepts_a_translation_that_keeps_every_placeholder():
    translation = "XX/XX/2024 tarixində Chase məndən {$460.00} tutdu və XXXX XXXX imtina etdi."
    assert kept_unchanged(SOURCE, translation)


def test_kept_unchanged_rejects_a_lost_amount_or_redaction():
    assert not kept_unchanged(SOURCE, "2024-cü ildə Chase 460 dollar tutdu və XXXX imtina etdi.")


def test_count_special_letters():
    assert count_special_letters("abc") == 0
    assert count_special_letters("ə ı ö ü ç ş ğ") == 7
