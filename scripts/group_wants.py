"""Group the free-text asks from the exploration run into a short list of categories.

Reads data/exploration/wants_raw.jsonl and sends the distinct main-ask phrasings, with how
many complaints used each, to one model call. The model proposes categories and assigns every
phrasing to one of them. Writes data/exploration/wants_groups.json and prints a table to review.

Run from the project root: uv run --env-file .env python scripts/group_wants.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import openai
from pydantic import BaseModel
from translation import cost

MODEL = "gpt-6.2-sol"
EFFORT = "medium"
SRC = Path("data/exploration/wants_raw.jsonl")
OUT = Path("data/exploration/wants_groups.json")

PROMPT = """\
Each line below is a phrase from a consumer complaint about a bank or lender, with the number \
of complaints that used it. The phrase says what the customer wants done.

Group the phrases into 8 to 12 categories that a bank's complaint-triage team could route on.
- Each category has a short snake_case name and a one-sentence definition.
- Every phrase goes into exactly one category. Copy each phrase exactly as given.
- Categories describe the action the customer wants (remove, refund, restore, stop), not the \
product or the reason for the complaint.
- Include a category named other for asks that fit none of the rest, and use it sparingly.
- Prefer categories that cover at least 3% of the complaints."""


class Category(BaseModel):
    name: str
    definition: str


class Assignment(BaseModel):
    phrase: str
    category: str


class Grouping(BaseModel):
    categories: list[Category]
    assignments: list[Assignment]


def normalize(phrase):
    return phrase.strip().lower().rstrip(".")


rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines()]
counts = Counter(normalize(a["ask"]) for r in rows for a in r["asks"] if a["main"])
total = sum(counts.values())
listing = "\n".join(f"{n}\t{phrase}" for phrase, n in counts.most_common())

response = openai.OpenAI(max_retries=5).responses.parse(
    model=MODEL,
    input=[{"role": "system", "content": PROMPT}, {"role": "user", "content": listing}],
    text_format=Grouping,
    reasoning={"effort": EFFORT},
    max_output_tokens=60000,
)
grouping = response.output_parsed
if grouping is None:
    raise SystemExit(f"no parsed answer (status {response.status})")

names = {c.name for c in grouping.categories}
assigned = {}
for a in grouping.assignments:
    assigned.setdefault(normalize(a.phrase), a.category)
missing = [p for p in counts if p not in assigned]
unknown = sorted({c for c in assigned.values() if c not in names})
invented = [p for p in assigned if p not in counts]

by_category = defaultdict(list)
for phrase, category in assigned.items():
    if phrase in counts:
        by_category[category].append(phrase)

spent = cost(MODEL, response.usage)
OUT.write_text(
    json.dumps(
        {
            "model": MODEL,
            "prompt": PROMPT,
            "categories": [c.model_dump() for c in grouping.categories],
            "assignments": {
                p: {"category": assigned[p], "complaints": counts[p]}
                for p in counts
                if p in assigned
            },
            "missing": missing,
            "unknown_categories": unknown,
            "invented_phrases": invented,
            "cost": spent,
        },
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print(f"{len(counts)} phrasings, {total} main asks, ${spent:.2f}")
print(
    f"not assigned: {len(missing)} | unknown category names: {unknown} "
    f"| invented phrases: {len(invented)}"
)
for c in sorted(grouping.categories, key=lambda c: -sum(counts[p] for p in by_category[c.name])):
    n = sum(counts[p] for p in by_category[c.name])
    examples = sorted(by_category[c.name], key=lambda p: -counts[p])[:5]
    print(
        f"\n{c.name}: {n} complaints ({n / total:.0%})\n  {c.definition}\n  e.g. "
        + "; ".join(examples)
    )
