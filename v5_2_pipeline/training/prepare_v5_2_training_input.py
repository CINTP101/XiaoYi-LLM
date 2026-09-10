#!/usr/bin/env python3
"""Create a clean-only, auditable input directory for V5.2 training.

This program is deliberately file-oriented: it accepts exactly one final clean
JSONL source file and creates a fresh directory containing exactly one copied
JSONL file. It never accepts a directory as a source, so it cannot recursively
pick up rejected or held-out data by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


TARGET_OPTIMIZATION_STEPS = 25
MAX_TRAIN_EPOCHS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sharegpt_jsonl(path: Path) -> int:
    valid_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"line {line_number}: blank rows are not allowed")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: row must be a JSON object")
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or not conversations:
                raise ValueError(f"line {line_number}: missing or empty conversations")
            has_human = False
            has_gpt = False
            for message_index, message in enumerate(conversations, start=1):
                if not isinstance(message, dict):
                    raise ValueError(f"line {line_number}, message {message_index}: must be an object")
                role = message.get("from")
                value = message.get("value")
                if role not in {"system", "human", "gpt"}:
                    raise ValueError(f"line {line_number}, message {message_index}: invalid role {role!r}")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"line {line_number}, message {message_index}: empty non-string value")
                has_human = has_human or role == "human"
                has_gpt = has_gpt or role == "gpt"
            if not has_human or not has_gpt:
                raise ValueError(f"line {line_number}: requires at least one human and one gpt message")
            valid_rows += 1
    if valid_rows == 0:
        raise ValueError("clean JSONL has zero valid rows")
    return valid_rows


def derive_plan(row_count: int) -> dict[str, int]:
    holdout_rows = math.ceil(row_count * 0.05)
    train_rows = row_count - holdout_rows
    if train_rows < 1:
        raise ValueError(
            f"row_count={row_count} leaves train_rows={train_rows} after the required 5% validation split"
        )

    # Keep the V5.1 upper bound of eight accumulated micro-batches while
    # selecting the largest value that can still yield the required update
    # count within three complete epochs.
    gradient_accumulation_steps = min(
        8,
        max(1, (MAX_TRAIN_EPOCHS * train_rows) // TARGET_OPTIMIZATION_STEPS),
    )
    steps_per_epoch = math.ceil(train_rows / gradient_accumulation_steps)
    epochs = math.ceil(TARGET_OPTIMIZATION_STEPS / steps_per_epoch)
    if epochs > MAX_TRAIN_EPOCHS:
        raise ValueError(
            "accepted clean data is too small for the V5.2 gate: "
            f"train_rows={train_rows} can produce fewer than {TARGET_OPTIMIZATION_STEPS} "
            f"optimizer steps in at most {MAX_TRAIN_EPOCHS} epochs"
        )
    total_optimization_steps = steps_per_epoch * epochs
    if total_optimization_steps < TARGET_OPTIMIZATION_STEPS:
        raise AssertionError("derived plan violates the minimum optimizer-step gate")
    logging_steps = 1
    eval_and_save_steps = max(1, total_optimization_steps // 4)
    training_eval_events = total_optimization_steps // eval_and_save_steps
    if training_eval_events < 4:
        raise AssertionError("derived plan violates the minimum in-training evaluation gate")
    return {
        "input_rows": row_count,
        "validation_rows_expected": holdout_rows,
        "train_rows_expected": train_rows,
        "target_optimization_steps": TARGET_OPTIMIZATION_STEPS,
        "max_train_epochs": MAX_TRAIN_EPOCHS,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_epoch_expected": steps_per_epoch,
        "num_train_epochs": epochs,
        "total_optimization_steps_expected": total_optimization_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_and_save_steps,
        "save_steps": eval_and_save_steps,
        "training_eval_events_expected": training_eval_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare V5.2 clean-only training input")
    parser.add_argument("--source-clean-jsonl", required=True, help="Accepted final clean JSONL file")
    parser.add_argument("--expected-sha256", required=True, help="SHA-256 supplied with the accepted clean file")
    parser.add_argument("--isolated-dir", required=True, help="Must not exist; becomes the one-file training directory")
    parser.add_argument("--manifest", required=True, help="Output JSON manifest; must not exist")
    parser.add_argument("--plan", required=True, help="Output derived training-plan JSON; must not exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_clean_jsonl).expanduser().resolve(strict=True)
    isolated_dir = Path(args.isolated_dir).expanduser().resolve(strict=False)
    manifest_path = Path(args.manifest).expanduser().resolve(strict=False)
    plan_path = Path(args.plan).expanduser().resolve(strict=False)
    expected_hash = args.expected_sha256.strip().lower()

    if len(expected_hash) != 64 or any(ch not in "0123456789abcdef" for ch in expected_hash):
        raise ValueError("--expected-sha256 must be exactly 64 lowercase/uppercase hexadecimal characters")
    if not source.is_file() or source.is_symlink():
        raise ValueError("--source-clean-jsonl must be one regular JSONL file, not a directory or symlink")
    if source.suffix.lower() != ".jsonl":
        raise ValueError("--source-clean-jsonl must have the .jsonl extension")
    if isolated_dir.exists():
        raise FileExistsError(f"isolated directory already exists: {isolated_dir}")
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    if plan_path.exists():
        raise FileExistsError(f"training plan already exists: {plan_path}")

    source_hash = sha256_file(source)
    if source_hash != expected_hash:
        raise ValueError(f"source SHA-256 mismatch: expected {expected_hash}, got {source_hash}")
    row_count = validate_sharegpt_jsonl(source)
    plan = derive_plan(row_count)

    isolated_dir.mkdir(parents=True, exist_ok=False)
    destination = isolated_dir / "clean_train.jsonl"
    try:
        shutil.copy2(source, destination)
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise RuntimeError("copied training input SHA-256 differs from the accepted source")
        files = sorted(path for path in isolated_dir.iterdir())
        if files != [destination] or not destination.is_file() or destination.is_symlink():
            raise RuntimeError("isolated training directory does not contain exactly one regular JSONL file")

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_clean_jsonl": str(source),
            "source_sha256": source_hash,
            "isolated_directory": str(isolated_dir),
            "isolated_files": [
                {
                    "name": destination.name,
                    "sha256": destination_hash,
                    "size_bytes": destination.stat().st_size,
                    "jsonl_rows_validated": row_count,
                }
            ],
            "training_plan": plan,
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception:
        # Keep any created files for manual audit; a failed isolation attempt
        # must never be silently deleted and then reused as training input.
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
