#!/usr/bin/env python3
"""Verify loss/evaluation evidence after the one-epoch precision candidate run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from v5_3_contract import ContractError


def regular_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be one regular non-symlink file")
    return path.resolve(strict=True)


def regular_dir(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} must be one regular non-symlink directory")
    return path.resolve(strict=True)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def step(value: str | None, label: str) -> int:
    try:
        parsed = int(float(value or ""))
    except ValueError as exc:
        raise ContractError(f"{label} has an invalid step") from exc
    if parsed < 1:
        raise ContractError(f"{label} step must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify precision candidate loss/evaluation evidence")
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_output = regular_dir(args.training_output, "--training-output")
    plan_path = regular_file(args.plan, "--plan")
    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {output}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary_path = regular_file(str(training_output / "v5_3_runtime_training_summary.json"), "training summary")
    train_curve = rows(regular_file(str(training_output / "train_curve.csv"), "train curve"))
    eval_curve = rows(regular_file(str(training_output / "eval_curve.csv"), "evaluation curve"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_steps = int(plan["total_optimization_steps_expected"])
    actual_steps = int(summary["global_step"])
    if actual_steps != expected_steps:
        raise ContractError(f"global_step {actual_steps} differs from planned {expected_steps}")
    loss_by_step: dict[int, float] = {}
    for row in train_curve:
        if row.get("train_loss") in (None, ""):
            continue
        loss_by_step[step(row.get("step"), "train curve")] = float(row["train_loss"])
    expected_loss_steps = set(range(1, actual_steps + 1))
    if set(loss_by_step) != expected_loss_steps:
        missing = sorted(expected_loss_steps - set(loss_by_step))
        raise ContractError(f"per-step loss record gate failed; missing steps: {missing[:10]}")
    in_training_eval_steps = sorted({step(row.get("step"), "evaluation curve") for row in eval_curve if row.get("eval_loss") not in (None, "") and step(row.get("step"), "evaluation curve") <= actual_steps})
    if len(in_training_eval_steps) < int(plan["minimum_internal_eval_records_required"]):
        raise ContractError("fewer than four in-training evaluation records")
    losses = [loss_by_step[index] for index in sorted(loss_by_step)]
    window = min(10, len(losses))
    initial_mean = sum(losses[:window]) / window
    final_mean = sum(losses[-window:]) / window
    if final_mean >= initial_mean:
        raise ContractError(f"loss decline gate failed: final {final_mean} >= initial {initial_mean}")
    evidence = {
        "training_output": str(training_output),
        "plan": str(plan_path),
        "expected_global_step": expected_steps,
        "actual_global_step": actual_steps,
        "per_step_loss_records": len(loss_by_step),
        "in_training_eval_records": len(in_training_eval_steps),
        "in_training_eval_steps": in_training_eval_steps,
        "loss_window_size": window,
        "initial_loss_window_mean": initial_mean,
        "final_loss_window_mean": final_mean,
        "loss_decline_verified": True,
        "status": "PASS",
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
