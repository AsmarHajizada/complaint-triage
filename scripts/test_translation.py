"""Translate 10 complaints into Azerbaijani with three OpenAI models, for a blind comparison.

Purpose: choose the translation model by having a native reader judge the output, before
the 5,000 training complaints are translated. Nothing here is training data. The
complaints come from the training split only.

Writes, under data/exploration/:
  translation_test.jsonl      one line per complaint and model: translation, tokens, cost, checks
  translation_test.md         the English text with three translations labeled A, B and C, in a
                              different random order for each complaint
  translation_test_key.txt    which letter was which model. Open it after reading.

Run from the project root: uv run --env-file .env python scripts/test_translation.py
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import pandas as pd
from translation import PRICES, cost, count_special_letters, kept_unchanged, translate

SEED = 20261124
N_COMPLAINTS = 10
MAX_COST = 1.00  # dollars; no new request starts once the total passes this
OUT = Path("data/exploration")

client = openai.OpenAI(max_retries=5)
spent = 0.0


def run(job):
    """Translate one complaint with one model and return the text, token counts, cost and checks."""
    global spent
    model, row = job
    result = {"complaint_id": row.complaint_id, "model": model, "translation": None, "error": None}
    if spent > MAX_COST:
        result["error"] = "skipped: cost limit reached"
        return result
    try:
        response = translate(client, model, row.narrative)
    except openai.OpenAIError as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result
    usage = response.usage
    spent += cost(model, usage)
    result.update(
        status=response.status,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cost=cost(model, usage),
    )
    if response.output_parsed is None:
        result["error"] = f"no parsed translation (status {response.status})"
        return result
    text = response.output_parsed.translation
    result["translation"] = text
    result["placeholders_kept"] = kept_unchanged(row.narrative, text)
    result["special_letters"] = count_special_letters(text)
    return result


sample = pd.read_parquet("data/processed/sample.parquet")
train = sample[sample["split"] == "train"].reset_index(drop=True)

# The shortest, the longest and the one with the most dollar amounts, then complaints from
# different products to fill the rest.
length = train["narrative"].str.len()
amounts = train["narrative"].str.count(r"\{\$[\d,.]+\}")
fixed = list(dict.fromkeys([length.idxmin(), length.idxmax(), amounts.idxmax()]))
others = (
    train.drop(index=fixed)
    .groupby("product")
    .sample(n=1, random_state=SEED)
    .sample(n=N_COMPLAINTS - len(fixed), random_state=SEED)
)
chosen = pd.concat([train.loc[fixed], others])

jobs = [(model, row) for row in chosen.itertuples() for model in PRICES]
with ThreadPoolExecutor(6) as pool:
    results = list(pool.map(run, jobs))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "translation_test.jsonl").open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def quote(text):
    return "\n".join(f"> {line}" for line in text.splitlines())


rng = random.Random(SEED)
lines = [f"# Translation test: {len(chosen)} complaints\n"]
key = []
for number, row in enumerate(chosen.itertuples(), 1):
    mine = {r["model"]: r for r in results if r["complaint_id"] == row.complaint_id}
    order = list(PRICES)
    rng.shuffle(order)  # a different order for each complaint, so a letter never means one model
    lines += [
        f"## {number}. Complaint {row.complaint_id}\n",
        "English original:\n",
        quote(row.narrative),
        "",
    ]
    for letter, model in zip("ABC", order, strict=True):
        r = mine[model]
        lines += [
            f"**Translation {letter}**\n",
            quote(r["translation"] or f"(failed: {r['error']})"),
        ]
        if r["translation"]:
            kept = "yes" if r["placeholders_kept"] else "NO"
            lines.append(
                f"\nAmounts, dates and redactions unchanged: {kept}. "
                f"Azerbaijani letters: {r['special_letters']}."
            )
        lines.append("")
    key.append(
        f"{number}: " + ", ".join(f"{letter}={m}" for letter, m in zip("ABC", order, strict=True))
    )
(OUT / "translation_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "translation_test_key.txt").write_text("\n".join(key) + "\n", encoding="utf-8")

print(f"done: {len(results)} calls, ${spent:.3f}, errors: {sum(1 for r in results if r['error'])}")
for model in PRICES:
    done = [r for r in results if r["model"] == model and r["translation"]]
    print(
        f"{model}: {len(done)} translated, placeholders kept in "
        f"{sum(r['placeholders_kept'] for r in done)}, ${sum(r['cost'] for r in done):.3f}"
    )
