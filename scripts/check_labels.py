"""Check the quality of the teacher's labels. Training rows only, nothing is written.

Reads data/processed/labels.jsonl, the Azerbaijani translations and the sample. Prints how
many records parsed, how often each field is null, how the values are spread, how often the
teacher agrees with CFPB's own labels, whether amounts appear in the text, and whether the
summaries are short Azerbaijani. Held-out complaints are never read.

Run from the project root: uv run python scripts/check_labels.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

FIELDS = ["summary", "product", "sub_product", "issue", "money_lost", "amount", "customer_wants"]
FIELDS += ["urgency"]
AMOUNT_IN_TEXT = re.compile(r"\{\$([\d,]+(?:\.\d+)?)\}")
SPECIAL = re.compile("[əıöüçşğ]")
ENGLISH = re.compile(
    r"\b(the|and|was|were|that|with|have|has|not|for|from|account|payment)\b", re.I
)


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def pct(count, total):
    return f"{count} of {total} ({count / total:.0%})" if total else "0 of 0"


labels = [r for r in read_lines("data/processed/labels.jsonl") if r["record"] is not None]
labels = pd.DataFrame(labels).drop_duplicates("complaint_id", keep="last")
labels = labels[labels["split"] == "train"]
sample = pd.read_parquet("data/processed/sample.parquet")[
    ["complaint_id", "product", "sub_product", "issue"]
].rename(columns={"product": "cfpb_product", "sub_product": "cfpb_sub", "issue": "cfpb_issue"})
translations = pd.DataFrame(read_lines("data/processed/translations.jsonl"))[
    ["complaint_id", "translation"]
].drop_duplicates("complaint_id", keep="last")
df = labels.merge(sample, on="complaint_id").merge(translations, on="complaint_id")
records = pd.DataFrame(list(df["record"]), index=df.index)
n = len(df)
print(f"{n} labeled training complaints\n")

print("how often each field is null:")
for field in FIELDS:
    print(f"  {field:15} {pct(int(records[field].isna().sum()), n)}")

for field in ("customer_wants", "urgency", "money_lost"):
    print(f"\n{field}:")
    for value, count in records[field].value_counts(dropna=False).items():
        print(f"  {str(value):28} {pct(count, n)}")

print("\nagreement with CFPB's own labels (CFPB's are chosen by the customer and can be wrong):")
product_match = records["product"] == df["cfpb_product"]
print(f"  product      {pct(int(product_match.sum()), n)}")
has_sub = records["sub_product"].notna() & df["cfpb_sub"].ne("I do not know")
sub_match = (records["sub_product"] == df["cfpb_sub"])[has_sub]
print(f"  sub-product  {pct(int(sub_match.sum()), int(has_sub.sum()))}, where both exist")
has_issue = records["issue"].notna()
issue_match = (records["issue"] == df["cfpb_issue"])[has_issue]
print(f"  issue        {pct(int(issue_match.sum()), int(has_issue.sum()))}, where given")
wrong = Counter(
    (cfpb[:28], str(mine)[:28])
    for cfpb, mine in zip(df["cfpb_product"], records["product"], strict=True)
    if cfpb != mine
)
print("  most common product disagreements (CFPB -> teacher):", wrong.most_common(4))

invented = missed = with_amount = 0
for text, amount in zip(df["translation"], records["amount"], strict=True):
    in_text = {m.replace(",", "") for m in AMOUNT_IN_TEXT.findall(text)}
    if pd.notna(amount):
        number = amount.split()[0]
        with_amount += 1
        invented += not any(float(number) == float(x) for x in in_text) if in_text else True
    elif len(in_text) == 1:
        missed += 1
print("\namounts:")
print(f"  an amount was given, and it is not in the text: {pct(invented, with_amount)}")
print(f"  the text has exactly one amount but the teacher gave none: {pct(missed, n)}")
print(f"  currency codes: {dict(Counter(a.split()[-1] for a in records['amount'].dropna()))}")

summary = records["summary"]
words = summary.str.split().str.len()
azerbaijani = summary.map(lambda s: bool(SPECIAL.search(s)))
english = summary.map(lambda s: len(ENGLISH.findall(s)) >= 2)
print("\nsummaries:")
print(f"  longer than 25 words: {pct(int((words > 25).sum()), n)} (median {int(words.median())})")
print(f"  with at least one Azerbaijani-only letter: {pct(int(azerbaijani.sum()), n)}")
print(f"  with two or more common English words: {pct(int(english.sum()), n)}")
print(f"  identical to another summary: {pct(int(summary.duplicated(keep=False).sum()), n)}")
