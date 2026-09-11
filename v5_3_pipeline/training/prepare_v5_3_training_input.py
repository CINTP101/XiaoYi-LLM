#!/usr/bin/env python3
"""Create the V5.3 one-file training input only after data acceptance.

This tool accepts one source JSONL, verifies its supplied SHA-256, enforces
the closed V5.3 contract, and creates a fresh directory containing exactly
one copied clean_train.jsonl. It never accepts a source directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from v5_3_contract import ALLOWED_LABELS, ContractError, validate_jsonl

MIN_CLEAN_ROWS = 74
TARGET_OPTIMIZATION_STEPS = 25
MAX_TRAIN_EPOCHS = 3
VALIDATION_PERCENTAGE = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_plan(row_count: int) -> dict[str, int]:
    if row_count < MIN_CLEAN_ROWS:
        raise ContractError(
            f"V5.3 requires more than the V5.2 clean-set size of 73; got {row_count} rows"
        )
    validation_rows = math.ceil(row_count * VALIDATION_PERCENTAGE / 100)
    train_rows = row_count - validation_rows
    if train_rows < 1:
        raise ContractError("validation split leaves no training rows")
    gradient_accumulation_steps = min(
        8,
        max(1, (MAX_TRAIN_EPOCHS * train_rows) // TARGET_OPTIMIZATION_STEPS),
    )
    steps_per_epoch = math.ceil(train_rows / gradient_accumulation_steps)
    epochs = math.ceil(TARGET_OPTIMIZATION_STEPS / steps_per_epoch)
    if epochs > MAX_TRAIN_EPOCHS:
        raise ContractError(
            f"{train_rows} train rows cannot yield {TARGET_OPTIMIZATION_STEPS} optimizer steps within {MAX_TRAIN_EPOCHS} epochs"
        )
    total_steps = steps_per_epoch * epochs
    if total_steps < TARGET_OPTIMIZATION_STEPS:
        raise AssertionError("optimizer-step gate violated")
    logging_steps = 1
    eval_save_steps = max(1, total_steps // 4)
    eval_events = total_steps // eval_save_steps
    if eval_events < 4:
        raise AssertionError("internal-evaluation gate violated")
    return {
        "input_rows": row_count,
        "validation_split_percentage": VALIDATION_PERCENTAGE,
        "validation_rows_expected": validation_rows,
        "train_rows_expected": train_rows,
        "target_optimization_steps": TARGET_OPTIMIZATION_STEPS,
        "max_train_epochs": MAX_TRAIN_EPOCHS,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_epoch_expected": steps_per_epoch,
        "num_train_epochs": epochs,
        "total_optimization_steps_expected": total_steps,
        "logging_steps": logging_steps,
        "eval_steps": eval_save_steps,
        "save_steps": eval_save_steps,
        "training_eval_events_expected": eval_events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one isolated V5.3 clean training input")
    parser.add_argument("--source-clean-jsonl", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--isolated-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    return parser.parse_args()


def require_new_path(path: Path, name: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {name}: {path}")


def main() -> None:
    args = parse_args()
    supplied_source = Path(args.source_clean_jsonl).expanduser()
    if supplied_source.is_symlink() or not supplied_source.is_file() or supplied_source.suffix.lower() != ".jsonl":
        raise ContractError("--source-clean-jsonl must be one regular, non-symlink .jsonl file")
    source = supplied_source.resolve(strict=True)
    isolated_dir = Path(args.isolated_dir).expanduser().resolve(strict=False)
    manifest_path = Path(args.manifest).expanduser().resolve(strict=False)
    plan_path = Path(args.plan).expanduser().resolve(strict=False)
    require_new_path(isolated_dir, "isolated directory")
    require_new_path(manifest_path, "input manifest")
    require_new_path(plan_path, "training plan")

    expected_hash = args.expected_sha256.strip().lower()
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
        raise ContractError("--expected-sha256 must be exactly 64 hexadecimal characters")
    source_hash = sha256_file(source)
    if source_hash != expected_hash:
        raise ContractError(f"source SHA-256 mismatch: expected {expected_hash}, got {source_hash}")

    data_stats = validate_jsonl(source)
    missing_labels = [f"{action}/{stage}" for action, stage in sorted(ALLOWED_LABELS) if not data_stats["label_counts"].get(f"{action}/{stage}")]
    if missing_labels:
        raise ContractError(f"accepted V5.3 training data lacks required label coverage: {missing_labels}")
    plan = derive_plan(data_stats["rows"])

    isolated_dir.mkdir(parents=True, exist_ok=False)
    destination = isolated_dir / "clean_train.jsonl"
    try:
        shutil.copy2(source, destination)
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise RuntimeError("copied isolated data hash differs from accepted source")
        files = sorted(isolated_dir.iterdir())
        if files != [destination] or destination.is_symlink() or not destination.is_file():
            raise RuntimeError("isolated directory must contain exactly one regular clean_train.jsonl")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_clean_jsonl": str(source),
            "source_sha256": source_hash,
            "isolated_directory": str(isolated_dir),
            "isolated_files": [{
                "name": destination.name,
                "sha256": destination_hash,
                "size_bytes": destination.stat().st_size,
                "jsonl_rows_validated": data_stats["rows"],
            }],
            "contract_validation": data_stats,
            "training_plan": plan,
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with plan_path.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception:
        # Preserve a failed attempt for audit; never silently delete and reuse it.
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
