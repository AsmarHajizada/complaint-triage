# complaint-triage

Personal learning project: distill a large language model into a small one that turns a raw bank-customer complaint into a structured triage record (JSON). The small model should run locally on a 16 GB GPU.

Status as of 2026-11-25: 5,000 complaints are sampled, translated into Azerbaijani (Luna) and labeled (Terra), and `urgency` is dropped from the training target. An untrained Qwen3.5-4B (Mac, Q4_K_M) was scored on 100 training complaints: 99 valid records, agreement with the teacher 69% on product and 34% on issue. Next: the baseline on the held-out complaints, then the training-data builder and a 16-bit LoRA run on Runpod. Nothing is trained.

## The task

Input: the free-text narrative of one consumer complaint, in Azerbaijani. The training narratives are OpenAI machine translations of the English CFPB narratives. CFPB categories, `customer_wants` and `urgency` keep English values. `summary` is in Azerbaijani. Labels are made from the Azerbaijani text.
Output: one JSON object with these fields in this order. The prompt and the schema are in `scripts/labeling.py`, and they are final only after the labeling test. Every field except `summary` can be `null` when the complaint does not say. `product`, `sub_product` and `issue` are CFPB's exact strings (`schema/allowed_values.json`).

```json
{
  "summary": "Musterinin kartindan iki defe pul tutulub, bank ise geri qaytarmaqdan imtina edib.",
  "product": "Credit card",
  "sub_product": "General-purpose credit card or charge card",
  "issue": "Problem with a purchase shown on your statement",
  "money_lost": true,
  "amount": "412.50 USD",
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
- `scripts/make_sample.py` draws 5,000 complaints from 2024-01 to 2025-12, even across the 11 products, after removing exact repeats and near-copies (three-word-phrase similarity of 0.2 or more). The first sample had 42 of 500 held-out complaints with a near-copy in training. Filter to `split == "train"` whenever reading the sample, and never read held-out rows while designing the schema or prompts.
- `scripts/explore_wants.py` asked `gpt-6.1-terra` an open question about 1,001 training complaints (91 per product) to design the `customer_wants` values. It cost $1.21. `scripts/group_wants.py` then grouped the 744 distinct phrasings of the 973 main asks into categories (`data/exploration/wants_groups.json`, $0.13). About half of the main asks were only implied, not stated.
- `scripts/translate_narratives.py` translates all 5,000 sampled complaints into Azerbaijani with `gpt-6-luna` (prompt `translate-v2`, about $1). Translations that change a redaction, amount or date are flagged and must be excluded or redone.
- Public data only. Never use data from any employer or private source.

## Plan

1. Download the narratives archive and keep complaints that have a narrative (`scripts/build_narrative_set.py`).
2. Sample about 5,000, spread evenly across products. Hold out 500 for evaluation and never train on them.
3. Label them with a teacher model. Read 50 teacher labels by hand before training on any.
4. Baseline: run an untrained 4B model on the 500 held-out complaints.
5. Fine-tune Qwen3.5-4B on the teacher's labels with 16-bit LoRA (not QLoRA) using Unsloth, then evaluate again. Unsloth's Qwen3.5 page says QLoRA 4-bit is "not recommended" for this model and gives about 10 GB of GPU memory for 16-bit LoRA on the 4B (their figure, not measured here). Chosen by the owner on 2026-11-25. Unsloth also says Qwen3.5 needs transformers v5.
6. Quantize the fine-tuned model to GGUF with llama.cpp at several levels and evaluate each.

Metrics: valid-JSON rate and accuracy per field, compared with the teacher. For product and issue, also compare with the CFPB's own categories.

## Decisions made

- Teacher: OpenAI models through the API (Luna, Terra or Sol, chosen by tests the owner reads, because the owner is the one who reads Azerbaijani). Decided 2026-11-25: the trained weights and the OpenAI-labeled data are never published or given to anyone, because OpenAI's Services Agreement 3.3(e) allows classifiers only if they are not distributed to third parties. Code can be shared. Publishing weights later would mean relabeling with an open-weights model. Anthropic's Usage Policy forbids training on its outputs without authorization, and Gemini's terms forbid developing competing models, so neither is used.
- Language: Azerbaijani. The owner's contribution is reviewing translations and labels, and writing a test set of 50 to 100 complaints in the style of real Azerbaijani customers (no bank data).
- `customer_wants` values (approved 2026-11-25): `refund`, `fix_credit_report`, `fix_account_record`, `restore_access_or_funds`, `debt_relief`, `stop_activity`, `provide_information`, `review_or_investigate`, `fulfil_request`, `close_account`, `other`, plus `null`. Rule: if the customer states no ask, give the obvious remedy; use `null` only when no remedy is evident.
- `amount` is one string with its currency code, such as `"460.00 USD"`. All training amounts are dollars, because CFPB is US data, so manat amounts are untested until the hand-written test set exists.
- Translation model: Luna, chosen on 2026-11-25 by the owner's blind rating of 10 translations (all three were acceptable, so the cheapest won).
- `urgency` is dropped from the student's training target (decided 2026-11-25): the teacher labeled 89% of 4,431 training complaints `medium`, so it carries no signal. The labels with `urgency` are kept untouched in `data/processed/labels.jsonl` and in a read-only copy, `data/processed/labels_with_urgency.jsonl`. Add it back only with a rule from someone who knows what a bank treats as urgent. The training-file builder and `scripts/score.py --skip urgency` leave it out.
- Student: Qwen3.5-4B (chosen 2026-11-25), trained on Runpod with Unsloth. A first trial of the untrained model runs on the Mac (16 GB, llama.cpp, Q4_K_M GGUF from `unsloth/Qwen3.5-4B-GGUF`). Qwen3.5 thinks by default: turn thinking off with `enable_thinking: false`.
- Labeling model: GPT-6.1 Terra, chosen by the owner on 2026-11-25 without a blind rating (a 10-complaint blind test was run and not read). Read 50 labels by hand before training, as the plan says.
- Training tool: Unsloth.

## Still open

- The student's training format (which fields, which prompt) is built from `labels.jsonl` without `urgency`. Not written yet.
- Whether Qwen3.5-4B handles Azerbaijani well enough. The Mac trial of the untrained model is the first evidence.
- The hand-written Azerbaijani test set.
- Which Runpod GPU and template to train on (price quote and explicit go before any pod).

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
