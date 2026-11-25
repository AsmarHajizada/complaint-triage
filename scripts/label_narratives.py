"""Label the translated complaints with the teacher model.

Reads data/processed/translations.jsonl (only translations that kept every redaction, amount
and date) and appends one line per complaint to data/processed/labels.jsonl: the record, token
counts and checks. A rerun skips complaints already labeled, so the script can be stopped and
restarted. Held-out complaints are labeled too, because the evaluation compares the student with
the teacher on them, but nobody reads them while designing prompts.

Run from the project root:
    uv run --env-file .env python scripts/label_narratives.py --limit 20
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from translation import cost

MODEL = "gpt-6.1-terra"
SEED = 20261124
WORKERS = 16  # requests sent at the same time
# Dollars. No new request starts once the total passes this; requests already running
# can overshoot it by a few cents.
MAX_COST = 20.00
SRC = Path("data/processed/translations.jsonl")
OUT = Path("data/processed/labels.jsonl")

allowed = load_allowed()
SYSTEM = build_system(allowed)
Triage = build_schema(allowed)
client = openai.OpenAI(max_retries=5, timeout=60)  # seconds; a stuck request is dropped and retried
spent = 0.0


def run(row):
    """Label one complaint. Returns None when the cost limit is reached."""
    global spent
    if spent > MAX_COST:
        return None
    response = label(client, MODEL, SYSTEM, Triage, row.translation)
    usage = response.usage
    spent += cost(MODEL, usage)
    result = {
        "complaint_id": row.complaint_id,
        "split": row.split,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "status": response.status,
        "input_tokens": usage.input_tokens,
        "cached_tokens": usage.input_tokens_details.cached_tokens,
        "output_tokens": usage.output_tokens,
        "record": None,
    }
    if response.output_parsed is not None:
        result["record"] = response.output_parsed.model_dump()
        result.update(fit_flags(result["record"], allowed))
        result["amount_format_ok"] = amount_format_ok(result["record"]["amount"])
    return result


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="label only this many, for a trial run")
args = parser.parse_args()

translations = pd.DataFrame(read_lines(SRC))
usable = translations[translations["translation"].notna() & translations["placeholders_kept"]]
done = set()
if OUT.exists():
    done = {r["complaint_id"] for r in read_lines(OUT) if r["record"] is not None}
todo = usable[~usable["complaint_id"].isin(done)]
if args.limit:
    # Trial runs use training complaints only, so no held-out complaint is ever read.
    todo = todo[todo["split"] == "train"].sample(n=args.limit, random_state=SEED)
print(
    f"{len(translations)} translated, {len(usable)} usable, {len(done)} already labeled, "
    f"{len(todo)} to do"
)

written = unparsed = misfit = bad_amount = 0
with ThreadPoolExecutor(WORKERS) as pool, OUT.open("a", encoding="utf-8") as f:
    futures = [pool.submit(run, row) for row in todo.itertuples()]
    for n, future in enumerate(as_completed(futures), 1):
        try:
            result = future.result()
        except (openai.BadRequestError, openai.AuthenticationError):
            pool.shutdown(wait=False, cancel_futures=True)
            raise  # the request or the key is wrong, so every call would fail
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            print(f"skipped one ({type(e).__name__}); rerun to retry it")
            continue
        if result is None:
            continue
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        f.flush()
        written += 1
        if result["record"] is None:
            unparsed += 1
        else:
            misfit += (
                result["issue_fits_product"] is False or result["sub_product_fits_product"] is False
            )
            bad_amount += result["amount_format_ok"] is False
        if n % 250 == 0 or n == len(futures):
            print(f"{n}/{len(futures)} done, ${spent:.2f} so far")
print(
    f"wrote {written} lines: {unparsed} without a record, {misfit} with an issue or sub-product "
    f"from another product, {bad_amount} with a malformed amount, ${spent:.2f} total"
    + (" (cost limit reached)" if spent > MAX_COST else "")
)
