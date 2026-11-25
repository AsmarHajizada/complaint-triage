"""Download the CFPB narratives archive and keep complaints that have a narrative.

The live bulk CSV has no narrative column. The archive files at
files.consumerfinance.gov hold narratives for complaints published up to a
cutoff date, so this script pulls every archive zip, reads it without
extracting, and writes one parquet file with the narrative rows only.

Run from the project root: uv run python scripts/build_narrative_set.py
"""
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://files.consumerfinance.gov/f/documents"
NAMES = [
    "CCDB_Export_1_December_2011_through_April_2018",
    "CCDB_Export_2_May_2018_through_April_2021",
    "CCDB_Export_3_May_2021_through_October_2022",
    "CCDB_Export_4_November_2022_through_August_2023",
    "CCDB_Export_5_September_2023_through_March_2024",
    "CCDB_Export_6_April_2024_through_July_2024",
    "CCDB_Export_7_August_2024_through_October_2024",
    "CCDB_Export_8_November_2024_through_December_2024",
    "CCDB_Export_9_January_2025_through_February_2025",
    "CCDB_Export_10_March_2025_through_April_2025",
    "CCDB_Export_11_May_2025_through_June_2025",
    "CCDB_Export_12_July_2025_through_August_2025",
    "CCDB_Export_13_September_2025_through_October_2025",
    "CCDB_Export_14_November_2025_through_December_2025",
    "CCDB_Export_15_January_2026_through_February_2026",
    "CCDB_Export_16_March_2026",
    "CCDB_Export_17_April_2026",
    "CCDB_Export_18_May_2026",
    "CCDB_Export_19_June_2026",
    "CCDB_Export_20_July_2026",
    "CCDB_Export_21_August_2026",
]
NARRATIVE = "Consumer complaint narrative"

raw = Path("data/raw/archive")
out = Path("data/interim")
raw.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)

kept = []
for name in NAMES:
    zip_path = raw / f"{name}.zip"
    if not zip_path.exists():
        # The earlier probe saved two files under data/raw/archive-probe.
        probe = Path("data/raw/archive-probe") / f"{name}.zip"
        if probe.exists():
            probe.rename(zip_path)
        else:
            urllib.request.urlretrieve(f"{BASE}/{name}.zip", zip_path)
    df = pd.read_csv(zip_path, dtype=str)
    has = df[NARRATIVE].notna()
    dates = pd.to_datetime(df["Date received"])
    print(f"{name}: {len(df):>9,} rows, {has.sum():>8,} with narrative "
          f"({dates.min().date()} to {dates.max().date()})", flush=True)
    kept.append(df[has])

data = pd.concat(kept, ignore_index=True)
data = data.drop_duplicates("Complaint ID")
data.to_parquet(out / "complaints_with_narrative.parquet", index=False)

months = pd.to_datetime(data["Date received"]).dt.to_period("M").value_counts().sort_index()
print(f"\nTotal with narrative: {len(data):,}")
print("Last 8 months that have narratives:")
print(months.tail(8).to_string())
