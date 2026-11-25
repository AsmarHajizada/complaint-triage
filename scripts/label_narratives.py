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
from pydantic import ValidationError
from translation import cost, kept_unchanged

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

sample = pd.read_parquet("data/processed/sample.parquet")[["complaint_id", "narrative"]]
translations = pd.DataFrame(read_lines(SRC)).merge(sample, on="complaint_id")
# The placeholder check is recomputed here, not read from the file, so that a change to the rule
# applies to translations that were written under the old one.
translations["usable"] = [
    isinstance(text, str) and kept_unchanged(source, text)
    for source, text in zip(translations["narrative"], translations["translation"], strict=True)
]
usable = translations[translations["usable"]]
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


def process(rows, workers):
    """Label the rows and append each result to OUT. Returns the IDs that failed."""
    failed = []
    with ThreadPoolExecutor(workers) as pool, OUT.open("a", encoding="utf-8") as f:
        futures = {pool.submit(run, row): row.complaint_id for row in rows.itertuples()}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except (openai.BadRequestError, openai.AuthenticationError):
                pool.shutdown(wait=False, cancel_futures=True)
                raise  # the request or the key is wrong, so every call would fail
            except (openai.APIStatusError, openai.APIConnectionError) as e:
                failed.append(futures[future])
                print(f"failed {futures[future]} ({type(e).__name__})")
                continue
            except ValidationError:
                failed.append(futures[future])
                print(f"failed {futures[future]} (output cut off or not valid)")
                continue
            if result is None:
                continue
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            if result["record"] is None:
                failed.append(futures[future])
            if n % 250 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} done, ${spent:.2f} so far")
    return failed


failed = process(todo, WORKERS)
if failed:
    # Looked at again once more, with fewer parallel requests, before giving up on them.
    print(f"retrying {len(failed)} failed complaints")
    failed = process(todo[todo["complaint_id"].isin(failed)], 4)
Path("data/processed/label_failures.txt").write_text("".join(f"{i}\n" for i in failed))

records = [r for r in read_lines(OUT) if r["record"] is not None]
misfit = sum(
    r["issue_fits_product"] is False or r["sub_product_fits_product"] is False for r in records
)
bad_amount = sum(r["amount_format_ok"] is False for r in records)
print(
    f"finished: ${spent:.2f} total, {len(failed)} complaints still without a record "
    "(listed in data/processed/label_failures.txt; a rerun tries them again). "
    f"Of {len(records)} labeled: {misfit} with an issue or sub-product from another product, "
    f"{bad_amount} with a malformed amount" + (" (cost limit reached)" if spent > MAX_COST else "")
)
