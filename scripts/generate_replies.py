"""Write a fine-tuned adapter's replies in the format score.py reads. Run on the GPU machine.

Uses the same prompt text as training, greedy decoding, thinking off. Then copy the output file
to the Mac and run `uv run python scripts/score.py <file> --skip urgency`.

    python scripts/generate_replies.py --adapter outputs/pilot --out data/eval/pilot_dev.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from unsloth import FastLanguageModel  # isort: skip  (Unsloth must be imported before transformers)

import torch
from training_text import render_prompt
from transformers import AutoTokenizer

BASE = "unsloth/Qwen3.5-4B"

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True, type=Path)
parser.add_argument("--examples", type=Path, default=Path("data/train/dev.jsonl"))
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--max-len", type=int, default=2048)
args = parser.parse_args()

model, _ = FastLanguageModel.from_pretrained(
    model_name=str(args.adapter),
    max_seq_length=args.max_len,
    load_in_4bit=False,
    load_in_16bit=True,
)
FastLanguageModel.for_inference(model)
tokenizer = AutoTokenizer.from_pretrained(BASE)
tokenizer.padding_side = "left"  # generation continues from the right edge

examples = [json.loads(line) for line in args.examples.read_text(encoding="utf-8").splitlines()]
prompts = [render_prompt(tokenizer, e["messages"]) for e in examples]
order = sorted(range(len(examples)), key=lambda i: len(prompts[i]))  # similar lengths together
eos = tokenizer.eos_token_id

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as f:
    for start in range(0, len(order), args.batch_size):
        chosen = order[start : start + args.batch_size]
        batch = tokenizer(
            [prompts[i] for i in chosen],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        began = time.time()
        with torch.no_grad():
            generated = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos,
                pad_token_id=tokenizer.pad_token_id,
            )
        seconds = (time.time() - began) / len(chosen)  # the batch time shared by its replies
        for row, i, mask in zip(
            generated[:, batch["input_ids"].shape[1] :],
            chosen,
            batch["attention_mask"],
            strict=True,
        ):
            ids = row.tolist()
            finished = eos in ids
            ids = ids[: ids.index(eos)] if finished else ids
            record = {
                "complaint_id": examples[i]["complaint_id"],
                "model": args.adapter.name,
                "prompt": "short",
                "reply": tokenizer.decode(ids, skip_special_tokens=True),
                "finish_reason": "stop" if finished else "length",
                "prompt_tokens": int(mask.sum()),
                "completion_tokens": len(ids),
                "seconds": round(seconds, 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{min(start + args.batch_size, len(order))}/{len(order)} answered")
