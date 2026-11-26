"""Tests for the example building and splitting in scripts/build_training_data.py."""

import json

import pandas as pd

from scripts.build_training_data import is_clean, split_dev, to_example
from scripts.evaluation import SHORT_PROMPT, STUDENT_FIELDS

RECORD = {
    "summary": "Müştəri pulu geri istəyir.",
    "product": "Credit card",
    "sub_product": None,
    "issue": "Billing disputes",
    "money_lost": True,
    "amount": "40.00 USD",
    "customer_wants": "refund",
    "urgency": "medium",
}


def test_to_example_drops_urgency_and_keeps_the_field_order():
    example = to_example("1", "text", RECORD)
    system, user, assistant = example["messages"]
    target = json.loads(assistant["content"])
    assert list(target) == STUDENT_FIELDS
    assert "urgency" not in target
    assert system["content"] == SHORT_PROMPT and user["content"] == "text"


def test_to_example_keeps_azerbaijani_letters_readable():
    assert "Müştəri" in to_example("1", "text", RECORD)["messages"][2]["content"]


def test_is_clean_rejects_teacher_slips():
    assert is_clean({})
    assert not is_clean({"issue_fits_product": False})
    assert not is_clean({"sub_product_fits_product": False})
    assert not is_clean({"amount_format_ok": False})
    assert is_clean({"issue_fits_product": None, "amount_format_ok": True})


def test_split_dev_takes_the_same_number_from_each_product_and_never_overlaps():
    frame = pd.DataFrame(
        {"complaint_id": [str(i) for i in range(40)], "product": ["A"] * 20 + ["B"] * 20}
    )
    train, dev = split_dev(frame, 3)
    assert dev["product"].value_counts().to_dict() == {"A": 3, "B": 3}
    assert len(train) == 34
    assert set(train["complaint_id"]).isdisjoint(dev["complaint_id"])
