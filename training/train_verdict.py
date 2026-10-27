"""QLoRA fine-tuning entrypoint for both tasks.

- verdict: model learns to emit {"verdict": "BUG"|"CLEAN"} JSON
- review: model learns to emit the full review JSON (type/severity/comment/location)

Runs on a free Kaggle T4 with defaults (Qwen2.5-Coder-1.5B, NF4, rank 16);
runs on CPU with --smoke-test for the end-to-end pipeline check.

Examples:
    python -m training.train_verdict --task verdict --data-dir data/processed/verdict \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct --epochs 2
    python -m training.train_verdict --task verdict --smoke-test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from training.common import (
    TINY_SMOKE_MODEL,
    build_model_and_tokenizer,
    generate_texts,
    load_jsonl,
    make_lora_config,
    tokenize_chat_example,
)


def train(args) -> dict:
    import torch
    from peft import get_peft_model
    from torch.utils.data import Dataset
    from transformers import Trainer, TrainingArguments

    train_records = load_jsonl(Path(args.data_dir) / "train.jsonl")
    val_records = load_jsonl(Path(args.data_dir) / "val.jsonl")[: args.max_examples or None] if \
        (Path(args.data_dir) / "val.jsonl").exists() else []
    if args.max_examples:
        train_records = train_records[: args.max_examples]
    if not train_records:
        raise SystemExit("no training records found — run mining + build_dataset first")
    print(f"train={len(train_records)} val={len(val_records)} task={args.task}")

    model, tokenizer = build_model_and_tokenizer(
        args.model, quantize=("none" if args.smoke_test else args.quantize)
    )
    peft_model = get_peft_model(model, make_lora_config(args.lora_r, args.lora_alpha))
    peft_model.print_trainable_parameters()

    class ChatDataset(Dataset):
        def __init__(self, records):
            self.items = [
                tokenize_chat_example(tokenizer, r["messages"], args.max_len) for r in records
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids, labels, attention = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(list(b["input_ids"]) + [pad_id] * n)
            labels.append(list(b["labels"]) + [-100] * n)
            attention.append(list(b["attention_mask"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }

    use_cuda = torch.cuda.is_available()
    steps_per_epoch = max(1, len(train_records) // max(1, args.batch * args.grad_accum))
    warmup_steps = max(1, int(steps_per_epoch * args.epochs * 0.05))
    train_args = TrainingArguments(
        output_dir=str(Path(args.out) / "_tmp"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=args.log_steps,
        save_strategy="no",
        report_to=[],
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=ChatDataset(train_records),
        eval_dataset=ChatDataset(val_records) if val_records else None,
        data_collator=collate,
    )

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    eval_loss = None
    if val_records:
        eval_loss = trainer.evaluate().get("eval_loss")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # sanity: the saved adapter must reload and generate
    if args.smoke_test:
        texts = generate_texts(peft_model, tokenizer, [train_records[0]["messages"][:-1]], max_new_tokens=32)
        assert texts and texts[0], "generation sanity check failed"

    history = [
        {k: v for k, v in log.items() if k in ("loss", "epoch", "step")}
        for log in trainer.state.log_history
        if "loss" in log
    ]
    meta = {
        "task": args.task,
        "base_model": args.model,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "final_train_loss": history[-1]["loss"] if history else None,
        "eval_loss": eval_loss,
        "train_seconds": round(train_secs, 1),
        "loss_history": history[-20:],
        "args": {k: v for k, v in vars(args).items()},
    }
    with (out_dir / "run_meta.json").open("w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"saved adapter -> {out_dir}")
    print(f"final loss: {meta['final_train_loss']} eval_loss: {eval_loss} ({train_secs:.0f}s)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["verdict", "review"], default="verdict")
    ap.add_argument("--data-dir", default=None, help="default data/processed/<task>")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="default models/<task>-v1")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--quantize", default="auto", choices=["auto", "nf4", "none"])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny model, few examples, no quantization — pipeline check")
    args = ap.parse_args()

    if args.smoke_test:
        args.model = args.model or TINY_SMOKE_MODEL
        args.max_examples = args.max_examples or 16
        args.epochs = 1
        args.out = "models/smoke"
        args.max_len = 1024
        args.log_steps = 2
        args.grad_accum = 1
    else:
        args.model = args.model or "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    if not args.data_dir:
        args.data_dir = f"data/processed/{args.task}"
    if not args.out:
        args.out = f"models/{args.task}-v1"

    train(args)


if __name__ == "__main__":
    main()
