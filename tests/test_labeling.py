"""Tests for the prompt and schema in scripts/labeling.py."""

import pytest
from pydantic import ValidationError

from scripts.labeling import WANTS, amount_format_ok, build_schema, build_system, fit_flags

ALLOWED = {
    "Credit card": {"sub_products": ["Store credit card"], "issues": ["Billing disputes"]},
    "Mortgage": {"sub_products": ["FHA mortgage"], "issues": ["Struggling to pay mortgage"]},
}
RECORD = {
    "summary": "x",
    "product": "Credit card",
    "sub_product": None,
    "issue": "Billing disputes",
    "money_lost": None,
    "amount": None,
    "customer_wants": "refund",
    "urgency": None,
}


def test_system_prompt_lists_every_product_issue_and_wants_value():
    system = build_system(ALLOWED)
    for text in ["Credit card", "Mortgage", "Billing disputes", "FHA mortgage", *WANTS]:
        assert text in system
    assert "WANTS_LIST" not in system


def test_schema_accepts_a_record_where_most_fields_are_null():
    assert build_schema(ALLOWED)(**RECORD).customer_wants == "refund"


def test_schema_rejects_values_that_are_not_allowed():
    schema = build_schema(ALLOWED)
    with pytest.raises(ValidationError):
        schema(**{**RECORD, "issue": "Made up issue"})
    with pytest.raises(ValidationError):
        schema(**{**RECORD, "customer_wants": "complain_loudly"})


def test_every_field_is_required_so_structured_outputs_accepts_it():
    json_schema = build_schema(ALLOWED).model_json_schema()
    assert set(json_schema["required"]) == set(json_schema["properties"]) == set(RECORD)


def test_fit_flags_catches_an_issue_from_another_product():
    flags = fit_flags({**RECORD, "issue": "Struggling to pay mortgage"}, ALLOWED)
    assert flags == {"sub_product_fits_product": None, "issue_fits_product": False}


def test_amount_format_ok():
    assert amount_format_ok("460.00 USD") is True
    assert amount_format_ok("1200 AZN") is True
    assert amount_format_ok("460.00") is False
    assert amount_format_ok("$460") is False
    assert amount_format_ok(None) is None
