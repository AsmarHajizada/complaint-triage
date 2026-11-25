"""Shared pieces for translating complaints into Azerbaijani with an OpenAI model."""

import re

from pydantic import BaseModel

EFFORT = "low"  # reasoning effort; reasoning tokens are billed as output tokens

# Dollars per 1M tokens: input, cached input, output (OpenAI pricing page, 2026-11-25).
PRICES = {
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-terra": (1.00, 0.05, 5.00),
    "gpt-6.2-sol": (2.00, 0.10, 10.00),
}

# The Azerbaijani Latin alphabet. SPECIAL matches the letters English lacks (schwa, dotless i,
# o-umlaut, u-umlaut, c-cedilla, s-cedilla, g-breve), which a poor translation drops or replaces.
ALPHABET = "abcçdeəfgğhxıijkqlmnoöprsştuüvyz"
SPECIAL = re.compile("[əıöüçşğ]")
# What a translation must leave unchanged: dollar amounts, dates and redactions.
KEPT = re.compile(r"\{\$[^}]*\}|XX/XX/[\w<>]+|X{2,}")

PROMPT_VERSION = "translate-v2"  # change this whenever SYSTEM changes

SYSTEM = f"""\
You translate consumer complaints from English into Azerbaijani. The result will be read as \
if a bank customer in Azerbaijan had written it in Azerbaijani, so it must sound like real \
Azerbaijani, not like a translation.

How to write it:
- Translate the meaning, not the words. Rewrite each sentence the way a native speaker would \
say it, with natural Azerbaijani word order and everyday vocabulary. Never copy English \
sentence structure, and never translate an idiom word for word.
- Keep the customer's voice and level of emotion: plain, direct, sometimes angry or tired. \
Do not make it more formal, polished or polite than the original.
- Use the words people in Azerbaijan actually use for banking and money. If an English term \
has no everyday Azerbaijani word, keep the English term.
- Use the Azerbaijani Latin alphabet with all its letters: {", ".join(ALPHABET)}.

What must not change:
- Every XXXX redaction, every dollar amount such as {{$460.00}} and every date such as \
XX/XX/2024 stays exactly as written. Company names stay unchanged.
- No fact, number, name or demand is added, removed or changed. Keep the same paragraphs.

Reply with the translation only."""


class Translation(BaseModel):
    translation: str


def cost(model, usage):
    """Return the dollar cost of one response, from its token counts."""
    price_in, price_cached, price_out = PRICES[model]
    cached = usage.input_tokens_details.cached_tokens
    return (
        (usage.input_tokens - cached) * price_in
        + cached * price_cached
        + usage.output_tokens * price_out
    ) / 1e6


def kept_unchanged(source, translation):
    """Return True when the translation has the same redactions, amounts and dates as the source."""
    return sorted(KEPT.findall(translation)) == sorted(KEPT.findall(source))


def count_special_letters(text):
    """Return how many Azerbaijani-only letters the text contains."""
    return len(SPECIAL.findall(text))


def translate(client, model, text):
    """Send one complaint to the model and return the API response.

    response.output_parsed holds the Translation, or None when the model refused or
    ran out of tokens.
    """
    return client.responses.parse(
        model=model,
        input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        text_format=Translation,
        reasoning={"effort": EFFORT},
        max_output_tokens=6000,
    )
