#!/usr/bin/env python3
"""Validate one parent-accepted Candidate E training JSONL in place."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from v5_3_contract import ALLOWED_LABELS, ContractError, strict_json_loads, validate_jsonl


ACCEPTED_DATA_ROOT = Path(
    "/home/cyh/Medical_Qwen/artifacts/v5_3_pipeline/data_candidate_e"
)
MIN_ROWS = 1800
SUMMARY_MIN_FRACTION = 0.35
SUMMARY_MAX_FRACTION = 0.45
VALIDATION_PERCENTAGE = 5
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 3e-6
WARMUP_RATIO = 0.03


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_new(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing {label}: {path}")


def accepted_source(value: str) -> Path:
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        raise ContractError("--source-train-jsonl must be an absolute path")
    if supplied.is_symlink() or not supplied.is_file() or supplied.suffix.lower() != ".jsonl":
        raise ContractError("--source-train-jsonl must be one regular non-symlink JSONL file")
    source = supplied.resolve(strict=True)
    root = ACCEPTED_DATA_ROOT.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"accepted Candidate E source must be stored below {root}") from exc
    return source


def require_unique_human_inputs(path: Path) -> int:
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            row = strict_json_loads(raw_line, context=f"line {line_number}")
            human = row["conversations"][0]["value"]
            digest = hashlib.sha256(human.encode("utf-8")).hexdigest()
            if digest in seen:
                raise ContractError(f"duplicate human input at line {line_number}")
            seen.add(digest)
    return len(seen)


def derive_plan(rows: int, label_counts: dict[str, int]) -> dict[str, object]:
    if rows < MIN_ROWS:
        raise ContractError(f"Candidate E requires at least {MIN_ROWS} accepted rows; got {rows}")
    missing = [
        f"{action}/{stage}"
        for action, stage in sorted(ALLOWED_LABELS)
        if not label_counts.get(f"{action}/{stage}")
    ]
    if missing:
        raise ContractError(f"Candidate E input lacks required labels: {missing}")
    summaries = label_counts.get("summarize/summary", 0)
    summary_fraction = summaries / rows
    if not SUMMARY_MIN_FRACTION <= summary_fraction <= SUMMARY_MAX_FRACTION:
        raise ContractError(
            "summary fraction must remain within "
            f"[{SUMMARY_MIN_FRACTION:.2f}, {SUMMARY_MAX_FRACTION:.2f}]; "
            f"got {summary_fraction:.6f}"
        )
    validation_rows = math.ceil(rows * VALIDATION_PERCENTAGE / 100)
    train_rows = rows - validation_rows
    optimizer_steps = math.ceil(train_rows / GRADIENT_ACCUMULATION_STEPS)
    eval_save_steps = max(1, optimizer_steps // 4)
    eval_events = optimizer_steps // eval_save_steps
    if eval_events < 4:
        raise ContractError("Candidate E plan requires at least four internal evaluations")
    return {
        "candidate": "minimal_correction_from_candidate_d",
        "base_adapter": (
            "/home/cyh/Medical_Qwen/output/"
            "tcm-qwen-1.5b-v5-3-candidate-precision-from-c"
        ),
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
        "learning_rate": LEARNING_RATE,
        "warmup_ratio": WARMUP_RATIO,
        "model_max_length": 768,
        "seed": 42,
        "data_seed": 42,
        "logging_steps": 1,
        "eval_steps": eval_save_steps,
        "save_steps": eval_save_steps,
        "training_eval_events_expected": eval_events,
        "total_optimization_steps_expected": optimizer_steps,
        "minimum_loss_records_required": optimizer_steps,
        "minimum_internal_eval_records_required": 4,
        "resume_from_checkpoint": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one accepted Candidate E input")
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
    require_new(manifest_path, "input manifest")
    require_new(plan_path, "training plan")
    expected_hash = args.expected_sha256.strip()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ContractError("--expected-sha256 must be exactly 64 lowercase hexadecimal characters")
    actual_hash = sha256_file(source)
    if actual_hash != expected_hash:
        raise ContractError(f"accepted source SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    stats = validate_jsonl(source)
    unique_human_inputs = require_unique_human_inputs(source)
    plan = derive_plan(stats["rows"], stats["label_counts"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_train_jsonl": str(source),
        "source_sha256": actual_hash,
        "size_bytes": source.stat().st_size,
        "jsonl_rows_validated": stats["rows"],
        "unique_human_inputs": unique_human_inputs,
        "training_input_mode": "parent_accepted_in_place_source_only_no_copy_or_rename",
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
