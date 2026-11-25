"""Write schema/allowed_values.json: the CFPB products, sub-products and issues the model may use.

Built from the complaints in the sampling window, so it holds every value that can occur in the
training data. The window must match make_sample.py.

Run from the project root: uv run python scripts/build_allowed_values.py
"""

import json
from pathlib import Path

import pandas as pd

WINDOW = ("2024-01-01", "2025-12-31")
NOT_A_VALUE = "I do not know"  # CFPB's own "unknown" sub-product; the model reports null instead

df = pd.read_parquet(
    "data/interim/complaints_with_narrative.parquet",
    columns=["Date received", "Product", "Sub-product", "Issue"],
)
dates = pd.to_datetime(df["Date received"])
df = df[(dates >= WINDOW[0]) & (dates <= WINDOW[1])]

allowed = {}
for product, rows in df.groupby("Product"):
    allowed[product] = {
        "sub_products": sorted(set(rows["Sub-product"].dropna()) - {NOT_A_VALUE}),
        "issues": sorted(rows["Issue"].dropna().unique()),
    }

Path("schema").mkdir(exist_ok=True)
Path("schema/allowed_values.json").write_text(json.dumps(allowed, indent=2) + "\n")
issues = {i for p in allowed.values() for i in p["issues"]}
subs = {s for p in allowed.values() for s in p["sub_products"]}
print(f"{len(allowed)} products, {len(subs)} sub-products, {len(issues)} distinct issues")
