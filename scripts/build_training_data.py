"""Build the fine-tuning data: one chat example per labeled training complaint.

Reads the teacher's labels, the Azerbaijani translations and the sample, and writes train.jsonl
and dev.jsonl under --out-dir. Each line holds {"complaint_id", "messages"}, where messages are
the short system prompt, the translated complaint, and the teacher's record as JSON without
urgency, with the fields in a fixed order.

Only training-split complaints are used. Held-out complaints never appear. A slice of the
training complaints, stratified by CFPB product, is held back as a dev set, so a pilot run can
be scored on complaints it did not train on.

Run from the project root: uv run python scripts/build_training_data.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from evaluation import SHORT_PROMPT, STUDENT_FIELDS
from translation import kept_unchanged

SEED = 20261124


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def is_clean(label):
    """Return True when the teacher's record is internally consistent.

    A record is dropped when its issue or sub-product belongs to another product, or its amount
    is not "number CODE". These are teacher slips, and the student should not copy them.
    """
    return not (
        label.get("issue_fits_product") is False
        or label.get("sub_product_fits_product") is False
        or label.get("amount_format_ok") is False
    )


def to_example(complaint_id, translation, record):
    """Return one chat example: the system prompt, the complaint, and the record as JSON."""
    target = {field: record[field] for field in STUDENT_FIELDS}
    return {
        "complaint_id": complaint_id,
        "messages": [
            {"role": "system", "content": SHORT_PROMPT},
            {"role": "user", "content": translation},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def split_dev(frame, per_product, seed=SEED):
    """Return (train, dev). `per_product` complaints of each CFPB product go to dev."""
    dev = frame.groupby("product", group_keys=False).sample(n=per_product, random_state=seed)
    return frame.drop(index=dev.index), dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--dev-per-product", type=int, default=27)
    args = parser.parse_args()

    sample = pd.read_parquet("data/processed/sample.parquet")[
        ["complaint_id", "split", "product", "narrative"]
    ]
    labels = pd.DataFrame(r for r in read_lines("data/processed/labels.jsonl") if r["record"])
    labels = labels.drop_duplicates("complaint_id", keep="last")
    translations = pd.DataFrame(read_lines("data/processed/translations.jsonl"))
    translations = translations.drop_duplicates("complaint_id", keep="last")

    train_rows = sample[sample["split"] == "train"]
    print(f"{len(train_rows)} training complaints in the sample")
    frame = train_rows.merge(labels.drop(columns=["split"]), on="complaint_id")
    print(f"{len(frame)} have a teacher record")
    frame = frame.merge(translations[["complaint_id", "translation"]], on="complaint_id")
    frame = frame[frame["translation"].map(lambda t: isinstance(t, str))]
    print(f"{len(frame)} have a translation")
    placeholders_ok = [
        kept_unchanged(n, t) for n, t in zip(frame["narrative"], frame["translation"], strict=True)
    ]
    frame = frame[placeholders_ok]
    print(f"{len(frame)} kept every amount, date and most redactions")
    clean = [is_clean(row) for row in frame.to_dict("records")]
    frame = frame[clean]
    print(f"{len(frame)} have a consistent teacher record")

    train, dev = split_dev(frame, args.dev_per_product)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("dev", dev)):
        examples = [
            to_example(row.complaint_id, row.translation, row.record) for row in part.itertuples()
        ]
        path = args.out_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8"
        )
        print(f"wrote {len(examples)} examples to {path}")

    held_out = set(sample.loc[sample["split"] == "eval", "complaint_id"])
    used = set(train["complaint_id"]) | set(dev["complaint_id"])
    assert not held_out & used, "a held-out complaint reached the training data"
    assert not set(train["complaint_id"]) & set(dev["complaint_id"])
    nulls = {f: sum(r[f] is None for r in train["record"]) / len(train) for f in STUDENT_FIELDS}
    print(
        "share of null values in the training targets:", {f: f"{v:.0%}" for f, v in nulls.items()}
    )


if __name__ == "__main__":
    main()
