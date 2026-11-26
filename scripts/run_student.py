"""Run a student model on translated complaints and save its raw replies.

Talks to any OpenAI-compatible server: llama-server on this Mac for the trial, vLLM on a
Runpod pod later. Replies are appended to --out, one line per complaint, and a rerun skips
complaints already answered. The default split is train. Use --split eval only for the real
baseline and final evaluation, because held-out complaints must not guide any decision.

Start a local server first, for example:
    llama-server -m Qwen3.5-4B-Q4_K_M.gguf -c 8192 --port 8080 \
        --chat-template-kwargs '{"enable_thinking": false}'
Then run from the project root:
    uv run python scripts/run_student.py --model qwen3.5-4b --limit 100 --out data/eval/trial.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import openai
import pandas as pd
from evaluation import SHORT_PROMPT
from labeling import build_system, load_allowed

SEED = 20261124
# The teacher prompt describes the fields but never asks for JSON, because the API forced the
# shape there. An untrained student has no such help, so the baseline has to be told.
JSON_INSTRUCTION = (
    "\n\nReply with one JSON object and nothing else. Use exactly these keys, in this order: "
    "summary, product, sub_product, issue, money_lost, amount, customer_wants, urgency."
)

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8080/v1")
parser.add_argument("--model", required=True, help="the model name the server expects")
parser.add_argument("--split", default="train", choices=["train", "eval"])
parser.add_argument("--limit", type=int, help="answer only this many complaints")
parser.add_argument("--prompt", default="teacher", choices=["teacher", "short"])
parser.add_argument("--max-tokens", type=int, default=800)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()

system = (
    build_system(load_allowed()) + JSON_INSTRUCTION if args.prompt == "teacher" else SHORT_PROMPT
)
client = openai.OpenAI(base_url=args.base_url, api_key="none", timeout=600, max_retries=1)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


labels = pd.DataFrame(
    r for r in read_lines(Path("data/processed/labels.jsonl")) if r["record"] is not None
).drop_duplicates("complaint_id", keep="last")
translations = pd.DataFrame(read_lines(Path("data/processed/translations.jsonl")))
pool = labels[labels["split"] == args.split][["complaint_id"]].merge(
    translations[["complaint_id", "translation"]], on="complaint_id"
)
args.out.parent.mkdir(parents=True, exist_ok=True)
done = {r["complaint_id"] for r in read_lines(args.out)} if args.out.exists() else set()
todo = pool[~pool["complaint_id"].isin(done)].sample(frac=1, random_state=SEED)
if args.limit:
    todo = todo.head(max(args.limit - len(done), 0))
print(f"{len(done)} already answered, {len(todo)} to do ({args.split} split, {args.prompt} prompt)")

with args.out.open("a", encoding="utf-8") as f:
    for n, row in enumerate(todo.itertuples(), 1):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": row.translation},
                ],
                temperature=0,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except openai.APIConnectionError:
            raise SystemExit(f"cannot reach {args.base_url}; is the server running?") from None
        choice = response.choices[0]
        record = {
            "complaint_id": row.complaint_id,
            "model": args.model,
            "prompt": args.prompt,
            "reply": choice.message.content,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "seconds": round(time.time() - start, 1),
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        if n % 10 == 0 or n == len(todo):
            print(f"{n}/{len(todo)} answered, last took {record['seconds']}s")
