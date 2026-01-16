from __future__ import annotations
import argparse
import os
import numpy as np


from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="cointegrated/rubert-tiny2")
    ap.add_argument("--output_dir", type=str, default="runs/reranker_tiny")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_ratio", type=float, default=0.02)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    ds = load_dataset("json", data_files={"train": args.train_pairs})
    ds = ds["train"].train_test_split(test_size=args.eval_ratio, seed=args.seed)

    def tokenize_fn(batch):
        tok = tokenizer(
            batch["query"],
            batch["candidate"],
            truncation=True,
            max_length=args.max_length,
        )
        tok["labels"] = batch["label"]
        return tok

    cols_to_remove = [c for c in ds["train"].column_names if c not in ("query", "candidate", "label")]
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    fp16 = False 
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[saved] model -> {args.output_dir}")


if __name__ == "__main__":
    main()
