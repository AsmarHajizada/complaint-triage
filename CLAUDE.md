# complaint-triage

Personal learning project: distill a large language model into a small one that turns a raw bank-customer complaint into a structured triage record (JSON). The small model should run locally on a 16 GB GPU.

Status as of 2026-11-24: the narrative data is downloaded (see Data). Nothing else is built yet. Python environment is set up with uv (pandas, pyarrow). Next step: sample 5,000 and hold out 500.

## The task

Input: the free-text narrative of one consumer complaint.
Output: one JSON object. Draft schema (field names and allowed values are not final):

```json
{
  "product": "credit card",
  "issue": "unauthorized charge",
  "summary": "Customer was charged twice and the bank refused a refund.",
  "money_lost": true,
  "amount_usd": 412.50,
  "customer_wants": "refund",
  "urgency": "medium"
}
```

## Data

The CFPB Consumer Complaint Database, published by the US Consumer Financial Protection Bureau at consumerfinance.gov/data-research/consumer-complaints/.

- It is public, and personal details in the narratives are already scrubbed (shown as XXXX).
- Only some complaints have a narrative. Keep those and drop the rest.
- Each complaint already has product and issue categories. Use them to check the teacher model's labels.
- The live bulk CSV (complaints.csv.zip) has no narrative column. CFPB moved narratives to a separate archive: 21 zip files at https://files.consumerfinance.gov/f/documents/CCDB_Export_<n>_<period>.zip, listed on consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/. These download by script without a browser (the search API and the bulk CSV link returned 403 on 2026-11-24).
- `scripts/build_narrative_set.py` downloads the archive to `data/raw/archive/` and writes `data/interim/complaints_with_narrative.parquet` (3,851,415 complaints with a narrative, December 2011 to July 2026; August 2026 has 1). The `data/` folder is ignored by git.
- Narrative text for 2024-2025 has a median of about 660 characters, and 58% of narratives contain XXXX redactions. Product names changed over the years, so sample from one period only: 2024-01 to 2025-12 has 2,036,435 narratives and 11 products, of which credit reporting is 76%. The smallest product, debt or credit management, has 4,198.
- Never use data from the owner's employer (a bank). Public data only.

## Plan

1. Download the CSV in a browser and keep complaints that have a narrative.
2. Sample about 5,000, spread evenly across products. Hold out 500 for evaluation and never train on them.
3. Label them with a teacher model. Read 50 teacher labels by hand before training on any.
4. Baseline: run an untrained 4B model on the 500 held-out complaints.
5. QLoRA fine-tune the 4B model on the teacher's labels with Unsloth, then evaluate again.
6. Quantize the fine-tuned model to GGUF with llama.cpp at several levels and evaluate each.

Metrics: valid-JSON rate and accuracy per field, compared with the teacher. For product and issue, also compare with the CFPB's own categories.

## Decisions made

- Teacher: an open-weights model (around 30B) run on Runpod. Some commercial APIs restrict training other models on their outputs, and an open teacher avoids that.
- Student: a 4B model such as Qwen3-4B or Gemma 3 4B. Check Hugging Face for newer models in that size class before choosing.
- Training tool: Unsloth.

## Still open

- Final schema, including the allowed values for `customer_wants` and `urgency`.
- Which teacher model exactly.
- Where to train: the local GPU or a Runpod pod.

## Hardware

- Development happens on a Mac. It has no NVIDIA GPU, so training does not run here.
- An RTX 5070 Ti (16 GB VRAM) is available, assumed to be in a separate PC. Training on it needs Linux or WSL2 and a PyTorch build with CUDA 12.8 or newer. 16 GB fits QLoRA on models up to about 12B and full fine-tuning up to about 1B.
- A Runpod account is available. Pods cost money: quote the hourly price and get an explicit go before creating one, and stop or delete it when the run ends.
- `uv` is installed for Python environments.

## Working with the owner

- A data scientist, not a software engineer. Fluent in Python, pandas and NumPy. Explain engineering concepts in plain language.
- Wants to learn, not to be handed finished work. Show each code change and explain it before making it, then wait for the owner's reply. Read-only checks can run without asking, but say what they do.
- Never run `git commit`, `git push` or `git add`. Give the owner the exact commands each time (also set in ~/.claude/CLAUDE.md).
- Wants the lowest-setup path: prefer one install command over a list of options.
- Give one recommendation, not a menu of options.
