"""Fine-tune Qwen3.5-4B on the teacher's labels with 16-bit LoRA. Run on a GPU machine, not the Mac.

Reads data/train/train.jsonl (written by build_training_data.py) and saves the LoRA adapter to
--out. The model learns to write the JSON record after the short prompt, with thinking off.

On the GPU machine, from the project root:
    python scripts/train_lora.py --probe                      # list the layers, then stop
    python scripts/train_lora.py --limit 1000 --epochs 1 --out outputs/pilot
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from unsloth import FastLanguageModel  # isort: skip  (Unsloth must be imported before transformers)

import torch
from datasets import Dataset
from training_text import encode, pad_batch
from transformers import AutoTokenizer, Trainer, TrainingArguments

BASE = "unsloth/Qwen3.5-4B"
SEED = 3407
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=Path, default=Path("data/train/train.jsonl"))
parser.add_argument("--out", type=Path, default=Path("outputs/lora"))
parser.add_argument("--limit", type=int, help="train on a random sample of this many examples")
parser.add_argument("--epochs", type=float, default=2)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--accumulate", type=int, default=4, help="batch-size x accumulate = 16")
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--max-len", type=int, default=2048)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--targets", default=DEFAULT_TARGETS, help="comma-separated layer names")
parser.add_argument("--probe", action="store_true", help="print the layer names and stop")
args = parser.parse_args()

model, _ = FastLanguageModel.from_pretrained(
    model_name=BASE,
    max_seq_length=args.max_len,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
tokenizer = AutoTokenizer.from_pretrained(BASE)

linear = Counter(
    name.split(".")[-1]
    for name, module in model.named_modules()
    if isinstance(module, torch.nn.Linear) and "visual" not in name
)
print("linear layers in the model (name: how many):", dict(linear))
if args.probe:
    raise SystemExit

model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank,
    target_modules=args.targets.split(","),
    lora_alpha=args.rank,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    max_seq_length=args.max_len,
)
attached = Counter(
    name.split(".")[-1] for name, module in model.named_modules() if hasattr(module, "lora_A")
)
print("LoRA attached to (name: how many):", dict(attached))
model.print_trainable_parameters()

examples = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines()]
random.Random(SEED).shuffle(examples)
examples = examples[: args.limit] if args.limit else examples
features = [encode(tokenizer, e["messages"], args.max_len) for e in examples]
too_long = sum(f is None for f in features)
features = [f for f in features if f is not None]
trained = sum(label != -100 for f in features for label in f["labels"])
total = sum(len(f["input_ids"]) for f in features)
print(f"{len(features)} examples ({too_long} left out as longer than {args.max_len} tokens)")
print(f"{total} tokens in all, {trained} of them in the answers the loss counts")
first = features[0]
n_prompt = first["labels"].count(-100)
print(
    "end of the first prompt:", repr(tokenizer.decode(first["input_ids"][n_prompt - 12 : n_prompt]))
)
print(
    "start of its answer:    ", repr(tokenizer.decode(first["input_ids"][n_prompt : n_prompt + 12]))
)

trainer = Trainer(
    model=model,
    train_dataset=Dataset.from_list(features),
    data_collator=lambda batch: {
        k: torch.tensor(v) for k, v in pad_batch(batch, tokenizer.pad_token_id).items()
    },
    args=TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulate,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_8bit",
        bf16=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    ),
)
trainer.train()
model.save_pretrained(str(args.out))
print("adapter saved to", args.out)
