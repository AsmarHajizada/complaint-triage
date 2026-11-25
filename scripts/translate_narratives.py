"""Translate the 5,000 sampled complaints into Azerbaijani.

Reads data/processed/sample.parquet and appends one line per complaint to
data/processed/translations.jsonl. A rerun skips complaints already translated, so the
script can be stopped and restarted. Held-out complaints are translated too, because the
evaluation needs them, but nobody reads them while designing prompts.

Run from the project root:
    uv run --env-file .env python scripts/translate_narratives.py --limit 20
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
from pydantic import ValidationError
from translation import (
    PROMPT_VERSION,
    cost,
    count_special_letters,
    kept_unchanged,
    translate,
)

MODEL = "gpt-6-luna"
SEED = 20261124
WORKERS = 16  # requests sent at the same time
# Dollars. No new request starts once the total passes this; requests already running
# can overshoot it by a few cents.
MAX_COST = 3.00
OUT = Path("data/processed/translations.jsonl")

client = openai.OpenAI(max_retries=5, timeout=60)  # seconds; a stuck request is dropped and retried
spent = 0.0


def run(row):
    """Translate one complaint. Returns None when the cost limit is reached."""
    global spent
    if spent > MAX_COST:
        return None
    response = translate(client, MODEL, row.narrative)
    usage = response.usage
    spent += cost(MODEL, usage)
    record = {
        "complaint_id": row.complaint_id,
        "split": row.split,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "status": response.status,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "translation": None,
    }
    if response.output_parsed is not None:
        text = response.output_parsed.translation
        record["translation"] = text
        record["placeholders_kept"] = kept_unchanged(row.narrative, text)
        record["special_letters"] = count_special_letters(text)
    return record


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="translate only this many, for a trial run")
args = parser.parse_args()

sample = pd.read_parquet("data/processed/sample.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)
done = set()
if OUT.exists():
    lines = (json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines())
    done = {r["complaint_id"] for r in lines if r["translation"] is not None}
todo = sample[~sample["complaint_id"].isin(done)]
if args.limit:
    # Trial runs use training complaints only, so no held-out complaint is ever read.
    todo = todo[todo["split"] == "train"].sample(n=args.limit, random_state=SEED)
print(f"{len(done)} already translated, {len(todo)} to do")


def process(rows, workers):
    """Translate the rows and append each result to OUT. Returns the IDs that failed."""
    failed = []
    with ThreadPoolExecutor(workers) as pool, OUT.open("a", encoding="utf-8") as f:
        futures = {pool.submit(run, row): row.complaint_id for row in rows.itertuples()}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                record = future.result()
            except (openai.BadRequestError, openai.AuthenticationError):
                pool.shutdown(wait=False, cancel_futures=True)
                raise  # the request or the key is wrong, so every call would fail
            except (openai.APIStatusError, openai.APIConnectionError) as e:
                failed.append(futures[future])
                print(f"failed {futures[future]} ({type(e).__name__})")
                continue
            except ValidationError:
                # The model kept writing until max_output_tokens and the JSON was cut off.
                # This happens with complaints that are mostly XXXX.
                failed.append(futures[future])
                print(f"failed {futures[future]} (output cut off)")
                continue
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if record["translation"] is None:
                failed.append(futures[future])
            if n % 250 == 0 or n == len(futures):
                print(f"{n}/{len(futures)} done, ${spent:.2f} so far")
    return failed


failed = process(todo, WORKERS)
if failed:
    # Looked at again once more, with fewer parallel requests, before giving up on them.
    print(f"retrying {len(failed)} failed complaints")
    failed = process(todo[todo["complaint_id"].isin(failed)], 4)
Path("data/processed/translation_failures.txt").write_text("".join(f"{i}\n" for i in failed))
print(
    f"finished: ${spent:.2f} total, {len(failed)} complaints still without a translation "
    "(listed in data/processed/translation_failures.txt; a rerun tries them again)"
    + (" (cost limit reached)" if spent > MAX_COST else "")
)
