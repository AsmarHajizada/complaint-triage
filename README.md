# complaint-triage

Distill a large language model into a small one that turns a bank-customer complaint into a structured triage record (JSON). The small model should run locally on a 16 GB GPU. This is a personal learning project and it is at an early stage.

## Status

Done: the data is downloaded and sampled, and the first exploration script has been tried on 20 complaints.
Not done: the output schema is not final, and no complaint has been labeled, nothing has been trained and nothing has been quantized.

The plan, the decisions made and the open questions are in [CLAUDE.md](CLAUDE.md).

## Data

The complaints come from the CFPB Consumer Complaint Database. The live bulk CSV has no narrative column, so the project uses CFPB's [narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/): 21 zip files covering December 2011 to August 2026, with 3,851,415 complaints that have a narrative. Narratives are redacted by CFPB (shown as XXXX). The data is not stored in this repository.

The sample uses complaints received from 2024-01 to 2025-12, because CFPB renamed products over the years. It holds 5,000 complaints, even across the 11 products, after removing exact repeats and near-copies of the same template letter. 500 of them are held out for evaluation. Their IDs are in `splits/heldout_ids.txt`. Those complaints are never used for training and never read while designing the schema or prompts.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or later.

```
uv sync
cp .env.example .env    # then put your OpenAI API key in .env
```

## Run

1. `uv run python scripts/build_narrative_set.py` downloads the archive (about 1.4 GB into `data/raw/archive/`) and writes `data/interim/complaints_with_narrative.parquet`.
2. `uv run python scripts/make_sample.py` takes about a minute and writes `data/processed/sample.parquet`, with a `split` column that is `train` or `eval`, and `splits/heldout_ids.txt`.
3. `uv run --env-file .env python scripts/explore_wants.py --limit 20` asks an OpenAI model what each of 20 training complaints asks for. The trial cost $0.02. Without `--limit` it covers 1,001 complaints, which is estimated at about $1 from the trial, and it stops at $5.

Tests and lint: `uv run pytest`, `uv run ruff check scripts tests` and `uv run ruff format --check scripts tests`. `explore_wants.py` has no tests, because it needs an API key.

## Layout

- `scripts/`: the pipeline, one script per step
- `tests/`: tests for the sampling functions
- `splits/`: the held-out complaint IDs
- `data/`: downloaded and generated data, ignored by git

## Terms of the labeling model

Labels will come from OpenAI models through the API. OpenAI's Services Agreement (version dated 2026-01-01, section 3.3(e)) forbids using Output to develop competing models, except models primarily intended to categorize or classify data that are not distributed or made commercially available to third parties. Check the current text before sharing a trained model.
