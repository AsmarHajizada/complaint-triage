"""Score a student's replies against the teacher's labels. Nothing is written.

Reads the file written by run_student.py. Prints how often the reply is valid JSON, how often
it is a valid record, how often each field matches the teacher, how often it fills a field the
teacher left null, and how often product, sub-product and issue match CFPB's own labels.

Run from the project root: uv run python scripts/score.py data/eval/trial.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from evaluation import FIELDS, agreement, parse_lenient, parse_strict
from labeling import build_schema, load_allowed
from pydantic import ValidationError

parser = argparse.ArgumentParser()
parser.add_argument("replies", type=Path)
parser.add_argument("--skip", nargs="*", default=[], help="fields to leave out, e.g. urgency")
args = parser.parse_args()
fields = [f for f in FIELDS if f not in args.skip]


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


Triage = build_schema(load_allowed())
replies = pd.DataFrame(read_lines(args.replies))
labels = pd.DataFrame(r for r in read_lines(Path("data/processed/labels.jsonl")) if r["record"])
labels = labels.drop_duplicates("complaint_id", keep="last")[["complaint_id", "split", "record"]]
cfpb = pd.read_parquet("data/processed/sample.parquet")[
    ["complaint_id", "product", "sub_product", "issue"]
].rename(columns=lambda c: c if c == "complaint_id" else f"cfpb_{c}")
df = replies.merge(labels, on="complaint_id").merge(cfpb, on="complaint_id")
n = len(df)
model, prompt, split = df["model"].iloc[0], df["prompt"].iloc[0], df["split"].iloc[0]
print(f"{n} replies from {model} ({prompt} prompt), split {split}")

strict = [parse_strict(r) for r in df["reply"]]
lenient = [parse_lenient(r) for r in df["reply"]]
valid = []
for parsed in lenient:
    try:
        if parsed and "urgency" in args.skip:
            parsed = {"urgency": None, **parsed}  # the fine-tuned model does not write urgency
        valid.append(Triage(**parsed).model_dump() if parsed else None)
    except (ValidationError, TypeError):
        valid.append(None)
count = sum(v is not None for v in valid)
print(f"\nthe whole reply is one JSON object:       {sum(s is not None for s in strict)} of {n}")
print(f"a JSON object can be found in the reply:  {sum(p is not None for p in lenient)} of {n}")
print(f"it is a valid record (all values allowed): {count} of {n}")
cut_off = int((df["finish_reason"] == "length").sum())
print(f"stopped because of the length limit:       {cut_off} of {n}")
print(f"median seconds per reply: {df['seconds'].median():.1f}")

ok = [i for i, v in enumerate(valid) if v is not None]
if ok:
    teacher = [df["record"].iloc[i] for i in ok]
    student = [valid[i] for i in ok]
    print(f"\nagreement with the teacher, over the {len(ok)} valid records:")
    for field in fields:
        mine = [s[field] for s in student]
        theirs = [t[field] for t in teacher]
        matches, total = agreement(mine, theirs)
        extra = sum(m is not None and t is None for m, t in zip(mine, theirs, strict=True))
        missed = sum(m is None and t is not None for m, t in zip(mine, theirs, strict=True))
        print(f"  {field:15} {matches / total:5.0%}   extra {extra:>3}   missed {missed:>3}")
    print("\nagreement with CFPB's own labels:")
    for field in ("product", "sub_product", "issue"):
        cf = [df[f"cfpb_{field}"].iloc[i] for i in ok]
        matches, total = agreement([s[field] for s in student], cf)
        print(f"  {field:15} {matches / total:5.0%}")
