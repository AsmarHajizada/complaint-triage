"""Tests for the pure functions in scripts/make_sample.py."""

import pandas as pd

from scripts.make_sample import comparison_text, not_a_copy, split_evenly

CASH_APP = (
    "I am filing a complaint against Cash App ( Block, Inc. ) due to inadequate customer "
    "service and unfair practices, which violate the Consumer Financial Protection Act "
    "( CFPA ). Specifically, Cash App failed to take timely and effective action on my "
    "dispute for {amount} and refused to refund the money."
)
MORTGAGE = (
    "My mortgage servicer applied my June payment late and then charged me a late fee twice, "
    "and nobody will explain the charges on my statement."
)


def test_split_evenly_adds_up_and_differs_by_at_most_one():
    parts = split_evenly(5000, 11)
    assert sum(parts) == 5000
    assert max(parts) - min(parts) == 1
    assert parts[:6] == [455] * 6 and parts[6:] == [454] * 5


def test_comparison_text_removes_amounts_redactions_and_digits():
    text = "I paid {$460.00} on XX/XX/2024 to XXXX XXXX 3 times!"
    assert comparison_text(text) == "i paid on to times"


def test_not_a_copy_drops_the_later_copy_of_a_template():
    narratives = pd.Series(
        [CASH_APP.format(amount="{$40.00}"), CASH_APP.format(amount="{$75.00}"), MORTGAGE]
    )
    assert list(not_a_copy(narratives)) == [True, False, True]


def test_not_a_copy_keeps_unrelated_complaints():
    narratives = pd.Series([CASH_APP.format(amount="{$40.00}"), MORTGAGE])
    assert list(not_a_copy(narratives)) == [True, True]
