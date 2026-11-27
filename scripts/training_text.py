"""Turn chat examples into token ids and loss masks for fine-tuning.

The model is trained on the exact text it will see at inference: the prompt rendered with
thinking off, which ends in an empty think block, followed by the teacher's JSON and the
end-of-turn token. Only the JSON and the end-of-turn token count toward the loss.
"""

THINKING_OFF = "<think>\n\n</think>\n\n"


def render_prompt(tokenizer, messages):
    """Return the text the model reads before it answers: system and user turns, thinking off."""
    return tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def render_pair(tokenizer, messages):
    """Return (prompt, completion) text for one {system, user, assistant} example.

    Raises ValueError when the template does not produce what inference will produce: the
    prompt must end with the empty think block, the full text must start with the prompt, and
    the answer must end with the end-of-turn token.
    """
    prompt = render_prompt(tokenizer, messages)
    if not prompt.endswith(THINKING_OFF):
        raise ValueError("the prompt does not end with the empty think block")
    full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError("the training text does not start with the inference prompt")
    completion = full[len(prompt) :].removesuffix("\n")
    if not completion.endswith(tokenizer.eos_token):
        raise ValueError("the answer does not end with the end-of-turn token")
    return prompt, completion


def encode(tokenizer, messages, max_len):
    """Return {"input_ids", "labels"} for one example, or None when it is longer than max_len.

    The prompt and the answer are tokenized separately, as they are at inference. Labels are
    -100 on the prompt, so the loss ignores it.
    """
    prompt, completion = render_pair(tokenizer, messages)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(completion_ids) > max_len:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def pad_batch(batch, pad_id):
    """Pad examples on the right to the longest one. Padding is ignored by the loss."""
    longest = max(len(x["input_ids"]) for x in batch)
    padded = {"input_ids": [], "labels": [], "attention_mask": []}
    for x in batch:
        gap = longest - len(x["input_ids"])
        padded["input_ids"].append(x["input_ids"] + [pad_id] * gap)
        padded["labels"].append(x["labels"] + [-100] * gap)
        padded["attention_mask"].append([1] * len(x["input_ids"]) + [0] * gap)
    return padded
