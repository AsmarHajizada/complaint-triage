"""The triage prompt and output schema, shared by the labeling test and the labeling run.

The teacher reads one complaint in Azerbaijani and returns one record. Product, sub-product and
issue use CFPB's exact strings, listed in schema/allowed_values.json.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import create_model

PROMPT_VERSION = "label-v1"  # change this whenever RULES or WANTS change
EFFORT = "low"  # reasoning effort; reasoning tokens are billed as output tokens
ALLOWED_PATH = Path("schema/allowed_values.json")

# What each customer_wants value means. The values come from grouping the 973 asks of the
# exploration run (data/exploration/wants_groups.json), then merging overlapping groups.
WANTS = {
    "refund": "get money back: refund, reimburse, reverse a charge or fee, recover stolen money",
    "fix_credit_report": "remove, correct or add information on a credit report",
    "fix_account_record": (
        "correct a balance, charge or payment posting in the company's own records"
    ),
    "restore_access_or_funds": (
        "reopen, unlock or reinstate an account or card, or release held funds"
    ),
    "debt_relief": "cancel, reduce, settle or restructure a debt, or release a lien",
    "stop_activity": "stop collection, garnishment, foreclosure, unwanted contact or billing",
    "provide_information": "provide records, proof, an explanation or a debt validation",
    "review_or_investigate": "review, reconsider or investigate a decision, dispute or application",
    "fulfil_request": (
        "carry out something promised or owed: pay a bonus, approve or process an application"
    ),
    "close_account": "close or cancel an account, product or service",
    "other": "anything that fits none of the above",
}
URGENCY = ["low", "medium", "high"]
AMOUNT = re.compile(r"\d+(\.\d+)? [A-Z]{3}")  # 460.00 USD

RULES = """\
You turn one consumer complaint about a financial company into a triage record. The complaint \
is written in Azerbaijani. It is a translation of a public CFPB complaint, so personal details \
are redacted as XXXX and dollar amounts appear as {$123.00}.

Use only what the complaint says. When it does not say, use null. Never guess.

Fields:
summary: One sentence in Azerbaijani, at most 25 words, with no names. If the complaint cannot \
be understood, write one short Azerbaijani sentence saying it is unclear and use null for every \
other field.
product: The product the complaint is about, from the list below.
sub_product: One of that product's sub-products, or null if the text does not show it.
issue: One of that product's issues, or null if none fits.
money_lost: true only if the complaint says money left the customer's hands or was charged to \
them because of the company's action. false if the customer lost no money. Frozen or unreachable \
funds count as false. null if the complaint does not say.
amount: The amount at the center of the dispute, copied from the text, written as the number \
followed by a space and the currency code, such as 460.00 USD. A dollar amount like {$460.00} is \
USD, manat is AZN, euros are EUR; use the code of any other named currency. Write the number as \
it appears, without thousands separators. Do not add amounts up or estimate. null if there is no \
amount, if it is redacted, or if the text lists several amounts and singles none out.
customer_wants: The main thing the customer wants done. If the customer states no ask, give the \
remedy that would obviously fix the problem; use null only when no remedy is evident. If there \
are several asks, pick the one that would fix the customer's own situation first. Removing \
something from a credit report is fix_credit_report; closing or cancelling an account is \
close_account. Values:
WANTS_LIST
urgency: high if the complaint states a deadline or imminent harm (a foreclosure date, \
repossession, wage garnishment, no access to money needed for living costs). medium if money was \
lost or the problem is ongoing. low otherwise. null if the complaint is too unclear to judge.

Allowed values, by product:
"""


def load_allowed(path=ALLOWED_PATH):
    """Return {product: {"sub_products": [...], "issues": [...]}} read from the JSON file."""
    return json.loads(Path(path).read_text())


def build_system(allowed):
    """Return the system prompt: the rules, the customer_wants values and the allowed values."""
    wants = "\n".join(f"  {name}: {meaning}" for name, meaning in WANTS.items())
    products = "\n".join(
        f"{product}\n  sub_products: {'; '.join(v['sub_products'])}"
        f"\n  issues: {'; '.join(v['issues'])}"
        for product, v in allowed.items()
    )
    return RULES.replace("WANTS_LIST", wants) + products


def one_of(values):
    return Literal[tuple(values)] | None


def build_schema(allowed):
    """Return the pydantic model for one record.

    Every field is required but may be null, which is how Structured Outputs expresses "optional".
    Field order is the output order: the summary comes first, so the model restates the complaint
    before it classifies. Which sub-products and issues belong to which product is not in the
    schema; fit_flags checks it afterwards.
    """
    sub_products = sorted({s for v in allowed.values() for s in v["sub_products"]})
    issues = sorted({i for v in allowed.values() for i in v["issues"]})
    return create_model(
        "Triage",
        summary=(str, ...),
        product=(one_of(sorted(allowed)), ...),
        sub_product=(one_of(sub_products), ...),
        issue=(one_of(issues), ...),
        money_lost=(bool | None, ...),
        amount=(str | None, ...),
        customer_wants=(one_of(WANTS), ...),
        urgency=(one_of(URGENCY), ...),
    )


def fit_flags(record, allowed):
    """Return whether the record's sub-product and issue belong to its product.

    A value is None when the field itself is null.
    """
    valid = allowed.get(record["product"], {"sub_products": [], "issues": []})
    return {
        "sub_product_fits_product": None
        if record["sub_product"] is None
        else record["sub_product"] in valid["sub_products"],
        "issue_fits_product": None
        if record["issue"] is None
        else record["issue"] in valid["issues"],
    }


def amount_format_ok(amount):
    """Return None for a null amount, else whether it is a number, a space and a currency code."""
    return None if amount is None else AMOUNT.fullmatch(amount) is not None


def label(client, model, system, schema, narrative):
    """Send one complaint to the model and return the API response.

    response.output_parsed holds the record, or None when the model refused or ran out of tokens.
    """
    return client.responses.parse(
        model=model,
        input=[{"role": "system", "content": system}, {"role": "user", "content": narrative}],
        text_format=schema,
        reasoning={"effort": EFFORT},
        max_output_tokens=4000,
    )
