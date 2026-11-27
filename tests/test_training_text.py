"""Tests for the prompt rendering, loss masking and padding in scripts/training_text.py."""

import pytest

from scripts.training_text import THINKING_OFF, encode, pad_batch, render_pair

MESSAGES = [
    {"role": "system", "content": "Reply with JSON."},
    {"role": "user", "content": "A complaint."},
    {"role": "assistant", "content": '{"a": 1}'},
]


class FakeTokenizer:
    """Mimics the Qwen template: one id per character, one id for the end-of-turn token."""

    eos_token = "<|im_end|>"

    def __init__(self, think_block=THINKING_OFF):
        self.think_block = think_block

    def apply_chat_template(self, messages, tokenize, enable_thinking, add_generation_prompt=False):
        text = ""
        for m in messages:
            body = m["content"]
            if m["role"] == "assistant":
                body = self.think_block + body
            text += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n" + self.think_block
        return text

    def __call__(self, text, add_special_tokens):
        return {
            "input_ids": [
                1 if c == "\x01" else ord(c) for c in text.replace(self.eos_token, "\x01")
            ]
        }


def test_render_pair_splits_after_the_empty_think_block():
    prompt, completion = render_pair(FakeTokenizer(), MESSAGES)
    assert prompt.endswith("<|im_start|>assistant\n" + THINKING_OFF)
    assert completion == '{"a": 1}<|im_end|>'


def test_render_pair_refuses_a_template_that_leaves_thinking_on():
    with pytest.raises(ValueError, match="empty think block"):
        render_pair(FakeTokenizer(think_block="<think>\n"), MESSAGES)


def test_encode_masks_the_prompt_and_trains_on_the_answer_and_end_token():
    tokenizer = FakeTokenizer()
    example = encode(tokenizer, MESSAGES, max_len=500)
    prompt, completion = render_pair(tokenizer, MESSAGES)
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    n_answer = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
    assert example["labels"][:n_prompt] == [-100] * n_prompt
    assert example["labels"][n_prompt:] == example["input_ids"][n_prompt:]
    assert len(example["labels"]) == n_prompt + n_answer
    assert example["labels"][-1] == 1  # the end-of-turn token


def test_encode_returns_none_when_the_example_is_too_long():
    assert encode(FakeTokenizer(), MESSAGES, max_len=10) is None


def test_pad_batch_pads_ids_labels_and_mask_to_the_longest():
    batch = [
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
        {"input_ids": [8], "labels": [8]},
    ]
    padded = pad_batch(batch, pad_id=0)
    assert padded["input_ids"] == [[5, 6, 7], [8, 0, 0]]
    assert padded["labels"] == [[-100, 6, 7], [8, -100, -100]]
    assert padded["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
