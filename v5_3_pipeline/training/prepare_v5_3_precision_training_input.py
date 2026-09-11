#!/usr/bin/env python3
"""Validate the accepted precision file in place before candidate D training.

This preparer is intentionally separate from the earlier V5.3 candidate
preparer. It accepts only the GO_D-approved source, checks its hash and
protocol composition, and never copies, renames, or accepts a directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from v5_3_contract import ALLOWED_LABELS, ContractError, validate_jsonl

MIN_ROWS = 1800
SUMMARY_MIN_FRACTION = 0.35
SUMMARY_MAX_FRACTION = 0.45
VALIDATION_PERCENTAGE = 5
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 1
EXPECTED_SOURCE = Path(
    "/home/cyh/Medical_Qwen/artifacts/v5_3_pipeline/data_precision/"
    "train_precision_combined_v5_3.jsonl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {label}: {path}")


def accepted_source(source_arg: str) -> Path:
    supplied = Path(source_arg).expanduser()
    if supplied.is_symlink() or not supplied.is_file() or supplied.suffix.lower() != ".jsonl":
        raise ContractError("--source-train-jsonl must be one regular non-symlink JSONL file")
    source = supplied.resolve(strict=True)
    expected = EXPECTED_SOURCE.resolve(strict=True)
    if source != expected:
        raise ContractError(f"accepted precision source must be exactly {expected}")
    return source


def derive_plan(rows: int, label_counts: dict[str, int]) -> dict[str, int | float | dict[str, int]]:
    if rows < MIN_ROWS:
        raise ContractError(f"precision training requires at least {MIN_ROWS} accepted rows; got {rows}")
    missing = [f"{action}/{stage}" for action, stage in sorted(ALLOWED_LABELS) if not label_counts.get(f"{action}/{stage}")]
    if missing:
        raise ContractError(f"precision training lacks required labels: {missing}")
    summaries = label_counts.get("summarize/summary", 0)
    summary_fraction = summaries / rows
    if not SUMMARY_MIN_FRACTION <= summary_fraction <= SUMMARY_MAX_FRACTION:
        raise ContractError(
            "summary fraction must be within "
            f"[{SUMMARY_MIN_FRACTION:.2f}, {SUMMARY_MAX_FRACTION:.2f}]; got {summary_fraction:.6f}"
        )
    validation_rows = math.ceil(rows * VALIDATION_PERCENTAGE / 100)
    train_rows = rows - validation_rows
    steps_per_epoch = math.ceil(train_rows / GRADIENT_ACCUMULATION_STEPS)
    eval_save_steps = max(1, steps_per_epoch // 4)
    eval_events = steps_per_epoch // eval_save_steps
    if eval_events < 4:
        raise AssertionError("precision plan must schedule at least four in-training evaluations")
    return {
        "input_rows": rows,
        "label_counts": dict(sorted(label_counts.items())),
        "summary_rows": summaries,
        "summary_fraction": summary_fraction,
        "summary_fraction_minimum": SUMMARY_MIN_FRACTION,
        "summary_fraction_maximum": SUMMARY_MAX_FRACTION,
        "validation_split_percentage": VALIDATION_PERCENTAGE,
        "validation_rows_expected": validation_rows,
        "train_rows_expected": train_rows,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "learning_rate": 1e-5,
        "model_max_length": 768,
        "seed": 42,
        "data_seed": 42,
        "total_optimization_steps_expected": steps_per_epoch,
        "logging_steps": 1,
        "eval_steps": eval_save_steps,
        "save_steps": eval_save_steps,
        "training_eval_events_expected": eval_events,
        "minimum_loss_records_required": steps_per_epoch,
        "minimum_internal_eval_records_required": 4,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one accepted precision V5.3 input")
    parser.add_argument("--source-train-jsonl", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = accepted_source(args.source_train_jsonl)
    manifest_path = Path(args.manifest).expanduser().resolve(strict=False)
    plan_path = Path(args.plan).expanduser().resolve(strict=False)
    for path, label in ((manifest_path, "input manifest"), (plan_path, "training plan")):
        require_new(path, label)
    expected_hash = args.expected_sha256.strip().lower()
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
        raise ContractError("--expected-sha256 must be exactly 64 lowercase hexadecimal characters")
    source_hash = sha256_file(source)
    if source_hash != expected_hash:
        raise ContractError(f"accepted source SHA-256 mismatch: expected {expected_hash}, got {source_hash}")
    stats = validate_jsonl(source)
    plan = derive_plan(stats["rows"], stats["label_counts"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_train_jsonl": str(source),
        "source_sha256": source_hash,
        "training_input_mode": "in_place_source_only_no_copy_or_rename",
        "sole_training_input": {
            "path": str(source),
            "sha256": source_hash,
            "size_bytes": source.stat().st_size,
            "jsonl_rows_validated": stats["rows"],
        },
        "contract_validation": stats,
        "training_plan": plan,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
