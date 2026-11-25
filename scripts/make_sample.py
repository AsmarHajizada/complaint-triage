"""Draw the 5,000-complaint sample and hold out 500 for evaluation.

Reads data/interim/complaints_with_narrative.parquet (built by
build_narrative_set.py) and writes:
  data/processed/sample.parquet   all 5,000 rows, with a `split` column (train or eval)
  splits/heldout_ids.txt          the 500 held-out Complaint IDs, kept in git

Many complaints are one template (a Cash App, Zelle or credit-report dispute
letter) filed again and again with small edits. A copy in the held-out set and
another in the training set would make the student look better than it is, so
near-copies are removed before the split.

The held-out complaints must never be used for training or for choosing
prompts. The same seed gives the same sample.

Run from the project root: uv run python scripts/make_sample.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

SEED = 20261124
WINDOW = ("2024-01-01", "2025-12-31")  # one period, so product names stay consistent
MIN_CHARS = 100  # shorter narratives hold too little to triage
MAX_CHARS = 3000  # longer ones are dropped, not cut, so labels match the full text
TOTAL = 5000
HELD_OUT = 500

# Complaints drawn per product before near-copies are removed. Templates are
# common in some products (money transfer, credit reporting), so this has to be
# well above TOTAL / 11 or a product can run short. The assert below catches that.
CANDIDATES_PER_PRODUCT = 1000

# Two complaints count as copies when at least this share of their three-word
# phrases match (cosine similarity, 1.0 = identical). Chosen by reading pairs:
# from 0.2 up they were the same template, below it they were different
# complaints on the same topic or sharing legal boilerplate.
MAX_SIMILARITY = 0.2

# Rows compared against all rows at once, which caps the similarity matrix at
# BLOCK x number of candidates instead of candidates x candidates.
BLOCK = 1000

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


def comparison_text(narrative: str) -> str:
    """Lower-case a narrative and remove what changes between copies of a template.

    Dollar amounts, XXXX redactions, dates and numbers differ from filer to filer.
    """
    text = narrative.lower()
    text = re.sub(r"\{\$[^}]*\}", " ", text)  # {$460.00}
    text = re.sub(r"\b[x/]*x{2,}[x/]*\b", " ", text)  # XXXX and XX/XX/XXXX
    text = re.sub(r"[^a-z ]", " ", text)  # digits and punctuation
    return " ".join(text.split())


def phrase_vectors(narratives: pd.Series):
    """Return one unit-length row per narrative, marking which three-word phrases it contains."""
    counts = CountVectorizer(ngram_range=(3, 3), binary=True, dtype=np.float32)
    return normalize(counts.fit_transform(narratives.map(comparison_text)))


def not_a_copy(narratives: pd.Series) -> np.ndarray:
    """Return a boolean array that is True for the narratives to keep.

    Narratives are visited in order. One is dropped when it is too similar to an
    earlier narrative that was kept, so the first copy in the order survives.
    """
    vectors = phrase_vectors(narratives)
    keep = np.ones(len(narratives), dtype=bool)
    for start in range(0, len(narratives), BLOCK):
        similarity = (vectors[start : start + BLOCK] @ vectors.T).toarray()
        for offset, row in enumerate(similarity):
            i = start + offset
            if (keep[:i] & (row[:i] >= MAX_SIMILARITY)).any():
                keep[i] = False
    return keep


def main() -> None:
    df = pd.read_parquet("data/interim/complaints_with_narrative.parquet", columns=list(COLUMNS))
    df = df.rename(columns=COLUMNS)
    print(f"complaints with a narrative: {len(df):,}")

    dates = pd.to_datetime(df["date_received"])
    df = df[(dates >= WINDOW[0]) & (dates <= WINDOW[1])]
    print(f"in {WINDOW[0]} to {WINDOW[1]}: {len(df):,}")

    length = df["narrative"].str.len()
    df = df[(length >= MIN_CHARS) & (length <= MAX_CHARS)]
    print(f"between {MIN_CHARS} and {MAX_CHARS} characters: {len(df):,}")

    # Exact repeats go first, on the whole pool. This is cheap and keeps repeats
    # from using up candidate slots.
    key = df["narrative"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[~key.duplicated()]
    print(f"after dropping repeated narratives: {len(df):,}")

    products = sorted(df["product"].unique())
    per_product = dict(zip(products, split_evenly(TOTAL, len(products)), strict=True))
    held_per_product = dict(zip(products, split_evenly(HELD_OUT, len(products)), strict=True))

    candidates = pd.concat(
        [
            df[df["product"] == p].sample(n=CANDIDATES_PER_PRODUCT, random_state=SEED)
            for p in products
        ]
    )
    # Shuffled so that which copy of a template survives is random, not decided by
    # the product's place in the alphabet.
    candidates = candidates.sample(frac=1, random_state=SEED).reset_index(drop=True)
    total_candidates = len(candidates)
    candidates = candidates[not_a_copy(candidates["narrative"])]
    print(
        f"candidates left after dropping near-copies: {len(candidates):,} of {total_candidates:,}"
    )

    enough = candidates["product"].value_counts().reindex(products, fill_value=0) >= pd.Series(
        per_product
    )
    assert enough.all(), f"too few candidates left for: {list(enough.index[~enough])}"

    parts = []
    for product in products:
        picked = candidates[candidates["product"] == product].head(per_product[product])
        n_eval = held_per_product[product]
        split = ["eval"] * n_eval + ["train"] * (per_product[product] - n_eval)
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
    print(sample.groupby(["product", "split"]).size().unstack(fill_value=0).to_string())
    print(
        f"\ntotal: {len(sample):,}  train: {(sample['split'] == 'train').sum():,}  eval: {HELD_OUT}"
    )
    print("narrative length, chars:", sample["narrative"].str.len().describe().round(0).to_dict())

    # Should be below MAX_SIMILARITY. Printed, not asserted, because float rounding
    # can put a pair at exactly 0.2.
    vectors = phrase_vectors(sample["narrative"])
    is_eval = (sample["split"] == "eval").to_numpy()
    cross = (vectors[is_eval] @ vectors[~is_eval].T).max()
    print(f"highest similarity between a held-out and a training complaint: {cross:.3f}")


if __name__ == "__main__":
    main()
