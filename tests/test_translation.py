"""Tests for the checks in scripts/translation.py."""

from scripts.translation import count_special_letters, kept_unchanged

SOURCE = "On XX/XX/2024 Chase charged me {$460.00} and XXXX XXXX refused."


def test_kept_unchanged_accepts_a_translation_that_keeps_every_placeholder():
    translation = "XX/XX/2024 tarixində Chase məndən {$460.00} tutdu və XXXX XXXX imtina etdi."
    assert kept_unchanged(SOURCE, translation)


def test_kept_unchanged_accepts_two_redactions_merged_into_one_in_a_long_text():
    source = "XXXX XXXX XXXX XXXX XXXX bank {$50.00} XX/XX/2024"
    assert kept_unchanged(source, "XXXX XXXX XXXX XXXX bank {$50.00} XX/XX/2024")


def test_kept_unchanged_rejects_a_lost_or_added_amount():
    assert not kept_unchanged(SOURCE, "XX/XX/2024 Chase 460 dollar tutdu və XXXX XXXX imtina etdi.")
    assert not kept_unchanged(SOURCE, SOURCE + " {$460.00}")


def test_kept_unchanged_rejects_a_changed_date_or_dropped_redactions():
    assert not kept_unchanged(SOURCE, "Chase məndən {$460.00} tutdu və XXXX XXXX imtina etdi.")
    assert not kept_unchanged(SOURCE, "XX/XX/2024 Chase məndən {$460.00} tutdu və imtina etdi.")


def test_count_special_letters():
    assert count_special_letters("abc") == 0
    assert count_special_letters("ə ı ö ü ç ş ğ") == 7
