"""Draw the 5,000-complaint sample and hold out 500 for evaluation.

Reads data/interim/complaints_with_narrative.parquet (built by
build_narrative_set.py) and writes:
  data/processed/sample.parquet   all 5,000 rows, with a `split` column (train or eval)
  splits/heldout_ids.txt          the 500 held-out Complaint IDs, kept in git

The held-out complaints must never be used for training or for choosing
prompts. Run again with the same seed and you get the same sample.

Run from the project root: uv run python scripts/make_sample.py
"""
from pathlib import Path

import pandas as pd

SEED = 20261124
WINDOW = ("2024-01-01", "2025-12-31")  # one period, so product names stay consistent
MIN_CHARS = 100    # shorter narratives hold too little to triage
MAX_CHARS = 3000   # longer ones are dropped, not cut, so labels match the full text
TOTAL = 5000
HELD_OUT = 500

COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
}


def split_evenly(total: int, parts: int) -> list[int]:
    """Split total into parts that differ by at most one, e.g. 5000 over 11 = 454 or 455."""
    base, extra = divmod(total, parts)
    return [base + 1 if i < extra else base for i in range(parts)]


df = pd.read_parquet("data/interim/complaints_with_narrative.parquet", columns=list(COLUMNS))
df = df.rename(columns=COLUMNS)
print(f"complaints with a narrative: {len(df):,}")

dates = pd.to_datetime(df["date_received"])
df = df[(dates >= WINDOW[0]) & (dates <= WINDOW[1])]
print(f"in {WINDOW[0]} to {WINDOW[1]}: {len(df):,}")

length = df["narrative"].str.len()
df = df[(length >= MIN_CHARS) & (length <= MAX_CHARS)]
print(f"between {MIN_CHARS} and {MAX_CHARS} characters: {len(df):,}")

# Copy-pasted complaints are common. Keeping one copy of each stops the same text
# from landing in both the training and the held-out set.
key = df["narrative"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
df = df[~key.duplicated()]
print(f"after dropping repeated narratives: {len(df):,}")

products = sorted(df["product"].unique())
per_product = dict(zip(products, split_evenly(TOTAL, len(products))))
held_per_product = dict(zip(products, split_evenly(HELD_OUT, len(products))))

parts = []
for product in products:
    pool = df[df["product"] == product]
    picked = pool.sample(n=per_product[product], random_state=SEED)
    split = ["eval"] * held_per_product[product] + ["train"] * (len(picked) - held_per_product[product])
    parts.append(picked.assign(split=split))
sample = pd.concat(parts, ignore_index=True)

assert len(sample) == TOTAL and (sample["split"] == "eval").sum() == HELD_OUT
assert sample["complaint_id"].is_unique

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("splits").mkdir(exist_ok=True)
sample.to_parquet("data/processed/sample.parquet", index=False)
held = sorted(sample.loc[sample["split"] == "eval", "complaint_id"], key=int)
Path("splits/heldout_ids.txt").write_text("\n".join(held) + "\n")

print("\nper product (train / eval):")
counts = sample.groupby(["product", "split"]).size().unstack(fill_value=0)
print(counts.to_string())
print(f"\ntotal: {len(sample):,}  train: {(sample['split'] == 'train').sum():,}  eval: {HELD_OUT}")
print("narrative length, chars:", sample["narrative"].str.len().describe().round(0).to_dict())
