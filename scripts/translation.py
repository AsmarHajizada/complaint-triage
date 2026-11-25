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
AMOUNT = re.compile(r"\{\$[^}]*\}")  # {$460.00}
DATE = re.compile(r"XX/XX/(?:\d{4}|XXXX)")  # XX/XX/2024
REDACTION = re.compile(r"X{2,}")  # XXXX
# Azerbaijani grammar sometimes merges two redactions ("XXXX, XXXX" becomes "XXXX-\u0259"), so
# the redaction count may move. Measured on 1,033 translations: 98% fall inside this range.
REDACTION_RANGE = (0.7, 1.4)

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


def _counts(text):
    """Return the amounts, the number of dates and the number of redactions in a text."""
    redactions = REDACTION.findall(DATE.sub("", AMOUNT.sub("", text)))
    return sorted(AMOUNT.findall(text)), len(DATE.findall(text)), len(redactions)


def kept_unchanged(source, translation):
    """Return True when the translation kept the source's amounts and dates and most redactions.

    Amounts and dates must match exactly. The redaction count may differ by the factor in
    REDACTION_RANGE. A source without redactions must stay without.
    """
    amounts, dates, redactions = _counts(source)
    new_amounts, new_dates, new_redactions = _counts(translation)
    low, high = REDACTION_RANGE
    redactions_ok = (
        new_redactions == 0 if redactions == 0 else low <= new_redactions / redactions <= high
    )
    return amounts == new_amounts and dates == new_dates and redactions_ok


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
        max_output_tokens=3000,
    )
