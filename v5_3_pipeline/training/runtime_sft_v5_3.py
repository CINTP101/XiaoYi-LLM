#!/usr/bin/env python3
"""Train one V5.3 LoRA candidate with the exact current consultation wrapper.

This is intentionally separate from supervised_finetuning.py. The older
trainer uses a legacy template that differs from tcm_chat_v5.py at runtime.
No test or blind-test argument exists in this program.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_prompt_v5_3 import (
    extract_consultation_prompt,
    render_runtime_prompt,
    runtime_contract_metadata,
    sha256_file,
)
from v5_3_contract import ContractError, strict_json_loads, validate_jsonl

IGNORE_INDEX = -100


@dataclass
class EncodedExample:
    input_ids: list[int]
    labels: list[int]


class EncodedDataset(Dataset):
    def __init__(self, examples: list[EncodedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": [1] * len(example.input_ids),
            "labels": example.labels,
        }


class CausalDataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_masks: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime-wrapper-aligned V5.3 SFT candidate trainer")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--peft-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--expected-runtime-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-max-length", type=int, required=True)
    parser.add_argument("--validation-split-percentage", type=int, default=5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--num-train-epochs", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, required=True)
    parser.add_argument("--eval-steps", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional bounded A/B trial; -1 uses epoch plan")
    parser.add_argument("--report-to", default="tensorboard")
    return parser.parse_args()


def non_symlink_file(value: str, field: str, suffix: str | None = None) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ContractError(f"{field} must be one regular non-symlink file")
    resolved = supplied.resolve(strict=True)
    if suffix and resolved.suffix.lower() != suffix:
        raise ContractError(f"{field} must have suffix {suffix}")
    return resolved


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = strict_json_loads(line, context=f"line {line_number}")
            if not isinstance(row, dict):
                raise ContractError(f"line {line_number} must be an object")
            rows.append(row)
    return rows


def encode_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    consultation_prompt: str,
    model_max_length: int,
) -> tuple[list[EncodedExample], dict[str, int]]:
    if tokenizer.eos_token_id is None:
        raise ContractError("tokenizer must define eos_token_id")
    encoded: list[EncodedExample] = []
    maximum_prompt_tokens = 0
    maximum_target_tokens = 0
    maximum_total_tokens = 0
    for index, row in enumerate(rows, 1):
        human = row["conversations"][0]["value"]
        target = row["conversations"][1]["value"]
        prompt = render_runtime_prompt(tokenizer, consultation_prompt, human)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if not target_ids:
            raise ContractError(f"line {index}: target tokenization is empty")
        if target_ids[-1] != tokenizer.eos_token_id:
            target_ids.append(tokenizer.eos_token_id)
        total = len(prompt_ids) + len(target_ids)
        if total > model_max_length:
            raise ContractError(
                f"line {index}: runtime prompt plus target has {total} tokens, exceeding model_max_length={model_max_length}; refusing truncation"
            )
        maximum_prompt_tokens = max(maximum_prompt_tokens, len(prompt_ids))
        maximum_target_tokens = max(maximum_target_tokens, len(target_ids))
        maximum_total_tokens = max(maximum_total_tokens, total)
        encoded.append(EncodedExample(
            input_ids=prompt_ids + target_ids,
            labels=[IGNORE_INDEX] * len(prompt_ids) + target_ids,
        ))
    return encoded, {
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_target_tokens": maximum_target_tokens,
        "maximum_total_tokens": maximum_total_tokens,
    }


def split_examples(examples: list[EncodedExample], validation_percentage: int, seed: int) -> tuple[EncodedDataset, EncodedDataset]:
    if not 1 <= validation_percentage < 100:
        raise ContractError("validation split percentage must be in [1, 99]")
    validation_count = math.ceil(len(examples) * validation_percentage / 100)
    if validation_count >= len(examples):
        raise ContractError("validation split leaves no training example")
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_count])
    train = [example for index, example in enumerate(examples) if index not in validation_indices]
    validation = [example for index, example in enumerate(examples) if index in validation_indices]
    return EncodedDataset(train), EncodedDataset(validation)


def write_curves(output_dir: Path, history: list[dict[str, Any]]) -> None:
    train_rows = [
        {"step": item.get("step"), "train_loss": item.get("loss"), "epoch": item.get("epoch")}
        for item in history
        if "loss" in item and "step" in item
    ]
    eval_rows = [
        {"step": item.get("step"), "eval_loss": item.get("eval_loss"), "epoch": item.get("epoch")}
        for item in history
        if "eval_loss" in item and "step" in item
    ]
    for name, fields, rows in (
        ("train_curve.csv", ["step", "train_loss", "epoch"], train_rows),
        ("eval_curve.csv", ["step", "eval_loss", "epoch"], eval_rows),
    ):
        with (output_dir / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_path = non_symlink_file(args.train_jsonl, "--train-jsonl", ".jsonl")
    runtime_source = non_symlink_file(args.runtime_source, "--runtime-source", ".py")
    base_model_path = Path(args.base_model_path).expanduser().resolve(strict=True)
    peft_path = Path(args.peft_path).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    actual_train_hash = sha256_file(train_path)
    if actual_train_hash != args.expected_train_sha256.strip().lower():
        raise ContractError("train JSONL SHA-256 does not match the accepted value")
    actual_runtime_hash = sha256_file(runtime_source)
    if actual_runtime_hash != args.expected_runtime_source_sha256.strip().lower():
        raise ContractError("runtime source SHA-256 does not match the frozen value")
    contract_validation = validate_jsonl(train_path)
    rows = load_rows(train_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True, local_files_only=True, padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ContractError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    consultation_prompt = extract_consultation_prompt(runtime_source)
    encoded, token_stats = encode_rows(rows, tokenizer, consultation_prompt, args.model_max_length)
    train_dataset, eval_dataset = split_examples(encoded, args.validation_split_percentage, args.data_seed)

    output_dir.mkdir(parents=True, exist_ok=False)
    runtime_metadata = runtime_contract_metadata(runtime_source)
    preflight = {
        "train_jsonl": str(train_path),
        "train_sha256": actual_train_hash,
        "peft_path": str(peft_path),
        "base_model_path": str(base_model_path),
        "contract_validation": contract_validation,
        "runtime_contract": runtime_metadata,
        "token_stats": token_stats,
        "train_examples": len(train_dataset),
        "internal_eval_examples": len(eval_dataset),
        "arguments": vars(args),
    }
    with (output_dir / "v5_3_runtime_training_preflight.json").open("x", encoding="utf-8") as handle:
        json.dump(preflight, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(peft_path), is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        optim="adamw_torch_fused",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        data_seed=args.data_seed,
        report_to=[args.report_to] if args.report_to else [],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    final_eval = trainer.evaluate()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("eval", final_eval)
    trainer.save_metrics("eval", final_eval)
    trainer.save_state()
    write_curves(output_dir, trainer.state.log_history)
    with (output_dir / "v5_3_runtime_training_summary.json").open("x", encoding="utf-8") as handle:
        json.dump({
            "train_metrics": train_result.metrics,
            "final_internal_eval": final_eval,
            "global_step": trainer.state.global_step,
            "log_history_entries": len(trainer.state.log_history),
        }, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
