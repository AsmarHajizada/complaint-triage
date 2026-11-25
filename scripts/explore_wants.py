"""Ask a model what each customer wants, in free text, to design the customer_wants values.

Takes 1,001 complaints from the training split of data/processed/sample.parquet
(91 per product, never the held-out rows) and asks an open question about each:
what does this customer ask for? The model is shown no list of categories, so the
answers show what customers actually ask for. Answers are appended to
data/exploration/wants_raw.jsonl, one line per complaint. A rerun skips complaints
already done, so the script can be stopped and restarted.

Run from the project root:
    uv run --env-file .env python scripts/explore_wants.py --limit 20
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import pandas as pd
from pydantic import BaseModel

MODEL = "gpt-6.1-terra"
EFFORT = "low"  # reasoning effort; reasoning tokens are billed as output tokens
PROMPT_VERSION = "wants-v1"  # change this whenever SYSTEM changes
SEED = 20261124
PER_PRODUCT = 91
WORKERS = 4  # requests sent at the same time
# Dollars. No new request starts once the total passes this; requests already
# running can overshoot it by a few cents.
MAX_COST = 5.00
# Dollars per 1M tokens: input, cached input, output (OpenAI pricing page, 2026-11-25).
PRICE_IN, PRICE_CACHED, PRICE_OUT = 1.00, 0.05, 5.00
OUT = Path("data/exploration/wants_raw.jsonl")

SYSTEM = """\
Each message is the text of one consumer complaint about a financial company, \
from the public CFPB complaint database. Personal details are redacted as XXXX.

We are designing the categories for a complaint-triage dataset and want to learn \
from real complaints what customers ask for. List every distinct thing this \
customer asks the company or the regulator to do.

For each ask give:
- ask: a short general phrase of three to eight words that starts with a verb and \
has no names, dates or amounts, such as "refund the money". Word the same kind \
of ask the same way every time.
- quote: the words in the complaint that show it, copied exactly, at most 30 words.
- explicit: true if the customer states the ask, false if it is only implied.
- main: true for the one ask that would most directly fix the customer's own \
problem, false for the rest.

Many complaints describe a problem and ask for nothing. Return an empty list for \
those. Include an implied ask only when the wanted action is obvious from the text."""


class Ask(BaseModel):
    ask: str
    quote: str
    explicit: bool
    main: bool


class Answer(BaseModel):
    asks: list[Ask]


client = openai.OpenAI(max_retries=5)
spent = 0.0


def squash(text: str) -> str:
    """Lower-case and collapse whitespace, so a quote can be matched against the complaint."""
    return " ".join(text.lower().split())


def label(row):
    """Ask the model about one complaint. Returns None when the cost limit is reached."""
    global spent
    if spent > MAX_COST:
        return None
    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"<complaint>\n{row.narrative}\n</complaint>"},
        ],
        text_format=Answer,
        reasoning={"effort": EFFORT},
        max_output_tokens=4000,
    )
    usage = response.usage
    cached = usage.input_tokens_details.cached_tokens
    spent += (
        (usage.input_tokens - cached) * PRICE_IN
        + cached * PRICE_CACHED
        + usage.output_tokens * PRICE_OUT
    ) / 1e6
    record = {
        "complaint_id": row.complaint_id,
        "product": row.product,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "status": response.status,
        "input_tokens": usage.input_tokens,
        "cached_tokens": cached,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.output_tokens_details.reasoning_tokens,
        "asks": None,
    }
    if response.output_parsed is not None:
        asks = [a.model_dump() for a in response.output_parsed.asks]
        for a in asks:
            a["quote_found"] = squash(a["quote"]) in squash(row.narrative)
        record["asks"] = asks
    return record


parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="label only this many, for a trial run")
args = parser.parse_args()

sample = pd.read_parquet("data/processed/sample.parquet")
train = sample[sample["split"] == "train"]
chosen = train.groupby("product").sample(n=PER_PRODUCT, random_state=SEED)
chosen = chosen.sample(frac=1, random_state=SEED)  # shuffled, so a trial run mixes products

OUT.parent.mkdir(parents=True, exist_ok=True)
done = set()
if OUT.exists():
    done = {json.loads(line)["complaint_id"] for line in OUT.read_text().splitlines()}
todo = chosen[~chosen["complaint_id"].isin(done)]
if args.limit:
    todo = todo.head(args.limit)
print(f"{len(done)} already labeled, {len(todo)} to do")

written = 0
unparsed = 0
with ThreadPoolExecutor(WORKERS) as pool, OUT.open("a") as f:
    futures = [pool.submit(label, row) for row in todo.itertuples()]
    for n, future in enumerate(as_completed(futures), 1):
        try:
            record = future.result()
        except (openai.BadRequestError, openai.AuthenticationError):
            pool.shutdown(wait=False, cancel_futures=True)
            raise  # the request or the key is wrong, so every call would fail
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            print(f"skipped one ({type(e).__name__}); rerun to retry it")
            continue
        if record is None:
            continue
        f.write(json.dumps(record) + "\n")
        f.flush()
        written += 1
        if record["asks"] is None:
            unparsed += 1
        if n % 50 == 0 or n == len(futures):
            print(f"{n}/{len(futures)} done, ${spent:.2f} so far")
print(
    f"wrote {written} lines ({unparsed} without a parsed answer), ${spent:.2f} total"
    + (" (cost limit reached)" if spent > MAX_COST else "")
)
