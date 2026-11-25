"""Label 10 translated complaints with three OpenAI models and lay the records side by side.

Purpose: choose the labeling model by reading its records in the final output format.
Nothing here is training data. The complaints come from the training split only, and only
ones whose translation kept every redaction, amount and date.

Writes, under data/exploration/:
  labeling_test.jsonl      one line per complaint and model: record, tokens, cost, checks
  labeling_test.md         each complaint with three records labeled A, B and C, in a different
                           random order for each complaint
  labeling_test_key.txt    which letter was which model. Open it after reading.

Run from the project root: uv run --env-file .env python scripts/test_labeling.py
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import pandas as pd
from labeling import (
    PROMPT_VERSION,
    amount_format_ok,
    build_schema,
    build_system,
    fit_flags,
    label,
    load_allowed,
)
from translation import PRICES, cost

SEED = 20261124
N_COMPLAINTS = 10
MAX_COST = 1.00  # dollars; no new request starts once the total passes this
OUT = Path("data/exploration")

allowed = load_allowed()
SYSTEM = build_system(allowed)
Triage = build_schema(allowed)
client = openai.OpenAI(max_retries=5)
spent = 0.0


def run(job):
    """Label one complaint with one model and return the record, token counts, cost and checks."""
    global spent
    model, row = job
    result = {
        "complaint_id": row.complaint_id,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "record": None,
        "error": None,
    }
    if spent > MAX_COST:
        result["error"] = "skipped: cost limit reached"
        return result
    try:
        response = label(client, model, SYSTEM, Triage, row.translation)
    except openai.OpenAIError as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    usage = response.usage
    spent += cost(model, usage)
    result.update(
        status=response.status,
        input_tokens=usage.input_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cost=cost(model, usage),
    )
    if response.output_parsed is None:
        result["error"] = f"no parsed record (status {response.status})"
        return result
    result["record"] = response.output_parsed.model_dump()
    result.update(fit_flags(result["record"], allowed))
    result["amount_format_ok"] = amount_format_ok(result["record"]["amount"])
    return result


sample = pd.read_parquet("data/processed/sample.parquet")
translations = pd.DataFrame(
    json.loads(line)
    for line in Path("data/processed/translations.jsonl").read_text(encoding="utf-8").splitlines()
)
pool = translations[(translations["split"] == "train") & translations["placeholders_kept"].eq(True)]
pool = pool[["complaint_id", "translation"]].merge(
    sample[["complaint_id", "product", "sub_product", "issue"]], on="complaint_id"
)

# One complaint per product where the translations allow it, then more at random.
shuffled = pool.sample(frac=1, random_state=SEED)
chosen = shuffled.drop_duplicates("product").head(N_COMPLAINTS)
if len(chosen) < N_COMPLAINTS:
    rest = shuffled.drop(index=chosen.index)
    chosen = pd.concat([chosen, rest.head(N_COMPLAINTS - len(chosen))])

jobs = [(model, row) for row in chosen.itertuples() for model in PRICES]
with ThreadPoolExecutor(6) as pool_:
    results = list(pool_.map(run, jobs))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "labeling_test.jsonl").open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quote(text):
    return "\n".join(f"> {line}" for line in text.splitlines())


rng = random.Random(SEED)
lines = [f"# Labeling test: {len(chosen)} complaints\n"]
key = []
for number, row in enumerate(chosen.itertuples(), 1):
    mine = {r["model"]: r for r in results if r["complaint_id"] == row.complaint_id}
    order = list(PRICES)
    rng.shuffle(order)  # a different order for each complaint, so a letter never means one model
    lines += [
        f"## {number}. Complaint {row.complaint_id}\n",
        quote(row.translation),
        "",
        f"CFPB's own labels: product = {row.product}; sub-product = {row.sub_product}; "
        f"issue = {row.issue}\n",
    ]
    for letter, model in zip("ABC", order, strict=True):
        r = mine[model]
        body = json.dumps(r["record"], ensure_ascii=False, indent=2) if r["record"] else r["error"]
        lines += [f"**Record {letter}**\n", "```json", body, "```", ""]
        if r["record"]:
            fits = {k: v for k, v in r.items() if k.endswith("_fits_product") and v is False}
            if fits:
                lines += [f"Sub-product or issue does not belong to the product: {sorted(fits)}\n"]
            if r.get("amount_format_ok") is False:
                lines += ["The amount is not a number, a space and a currency code.\n"]
    key.append(
        f"{number}: " + ", ".join(f"{letter}={m}" for letter, m in zip("ABC", order, strict=True))
    )
(OUT / "labeling_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "labeling_test_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")

print(f"done: {len(results)} calls, ${spent:.3f}, errors: {sum(1 for r in results if r['error'])}")
