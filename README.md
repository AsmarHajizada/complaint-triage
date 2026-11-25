# complaint-triage

Distill a large language model into a small one that turns a bank-customer complaint, written in Azerbaijani, into a structured triage record (JSON). The training complaints are machine translations of public English CFPB complaints. The small model should run locally on a 16 GB GPU. This is a personal learning project and it is at an early stage.

## Status

Done: 5,000 complaints sampled (500 held out), translated into Azerbaijani and labeled by the teacher model. An untrained Qwen3.5-4B has been scored on 100 training complaints.
Not done: the baseline on the held-out complaints, training, and quantization.

The plan, the decisions made and the open questions are in [CLAUDE.md](CLAUDE.md).

## Data

The complaints come from the CFPB Consumer Complaint Database. The live bulk CSV has no narrative column, so the project uses CFPB's [narratives archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/): 21 zip files covering December 2011 to August 2026, with 3,851,415 complaints that have a narrative. Narratives are redacted by CFPB (shown as XXXX). The data is not stored in this repository.

The sample uses complaints received from 2024-01 to 2025-12, because CFPB renamed products over the years. It holds 5,000 complaints, even across the 11 products, after removing exact repeats and near-copies of the same template letter. 500 of them are held out for evaluation. Their IDs are in `splits/heldout_ids.txt`. Those complaints are never used for training and never read while designing the schema or prompts.

Planned: the narratives are translated into Azerbaijani with an OpenAI model before labeling. Machine-translated text is not real customer text, so the evaluation also uses a small test set of complaints written by hand in Azerbaijani.

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
4. `uv run --env-file .env python scripts/group_wants.py` groups the exploration answers into categories for the `customer_wants` field. One call, about $0.13.
5. `uv run --env-file .env python scripts/translate_narratives.py` translates the 5,000 complaints into Azerbaijani with `gpt-6-luna`. It cost $1.04. Add `--limit 20` for a trial on training rows.
6. `uv run --env-file .env python scripts/label_narratives.py` labels them with `gpt-6.1-terra`. It cost $5.62. Then `uv run python scripts/check_labels.py` reports the quality of the training labels.
7. The baseline: start `llama-server` on a Qwen3.5-4B GGUF file, run `uv run python scripts/run_student.py --model qwen3.5-4b --limit 100 --out data/eval/trial.jsonl`, then `uv run python scripts/score.py data/eval/trial.jsonl --skip urgency`. The student runner works with any OpenAI-compatible server.

`scripts/test_translation.py` and `scripts/test_labeling.py` compare three OpenAI models on 10 complaints each, with the models hidden behind letters.

To see which precision each tensor of any GGUF file uses, and its true bits per weight: `uv run scripts/inspect_gguf.py path/to/model.gguf --tensors`.

Tests and lint: `uv run pytest`, `uv run ruff check scripts tests` and `uv run ruff format --check scripts tests`. The scripts that call an API have no tests, because they need a key.

## Layout

- `scripts/`: the pipeline, one script per step
- `tests/`: tests for the sampling, translation, labeling, evaluation and GGUF helpers
- `splits/`: the held-out complaint IDs
- `data/`: downloaded and generated data, ignored by git

## Terms of the labeling model

Labels and translations come from OpenAI models through the API. OpenAI's Services Agreement (version dated 2026-01-01, section 3.3(e)) forbids using Output to develop competing models, except models primarily intended to categorize or classify data that are not distributed or made commercially available to third parties. For that reason the trained model's weights and the OpenAI-labeled data are not published. This repository holds the code, which anyone can run with their own API key.

## License

The code is released under the [MIT License](LICENSE). It covers the code only: the trained model weights and the OpenAI-labeled data are not published.
