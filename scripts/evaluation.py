"""Helpers for running a student model and scoring its replies against the teacher's labels."""

import json
import re

FIELDS = ["product", "sub_product", "issue", "money_lost", "amount", "customer_wants", "urgency"]
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_strict(text):
    """Return the reply as a dict if the whole reply is one JSON object, else None."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_lenient(text):
    """Return the JSON object in a reply as a dict, allowing a code fence or text around it.

    The strict version asks whether the model replied with the record and nothing else. This one
    asks whether it produced the record at all.
    """
    text = FENCE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def agreement(student, teacher):
    """Return (matches, comparable) for two lists of values, counting None as a value."""
    pairs = list(zip(student, teacher, strict=True))
    return sum(a == b for a, b in pairs), len(pairs)
