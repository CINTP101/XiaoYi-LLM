#!/usr/bin/env python3
"""Create a reproducible loss-convergence summary for the completed V5.2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def finite_float(value: str, field: str, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {field}={value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"row {row_number}: non-finite {field}={value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V5.2 training-curve convergence")
    parser.add_argument("--train-curve", required=True)
    parser.add_argument("--eval-curve", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_curve = Path(args.train_curve).resolve(strict=True)
    eval_curve = Path(args.eval_curve).resolve(strict=True)
    output_path = Path(args.output).resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite convergence summary: {output_path}")

    train_rows = read_csv(train_curve)
    eval_rows = read_csv(eval_curve)
    if "step" not in train_rows[0] or "train_loss" not in train_rows[0]:
        raise ValueError("train curve must contain step and train_loss columns")
    if "step" not in eval_rows[0] or "eval_loss" not in eval_rows[0]:
        raise ValueError("eval curve must contain step and eval_loss columns")

    losses = [finite_float(row["train_loss"], "train_loss", index) for index, row in enumerate(train_rows, 1)]
    if len(losses) < 25:
        raise ValueError(f"expected at least 25 logged loss points, found {len(losses)}")
    first_window = losses[:5]
    last_window = losses[-5:]
    first_median = statistics.median(first_window)
    last_median = statistics.median(last_window)
    minimum = min(losses)
    final_eval = eval_rows[-1]
    final_eval_loss = finite_float(final_eval["eval_loss"], "eval_loss", len(eval_rows))

    summary: dict[str, Any] = {
        "inputs": {
            "train_curve": str(train_curve),
            "train_curve_sha256": sha256_file(train_curve),
            "eval_curve": str(eval_curve),
            "eval_curve_sha256": sha256_file(eval_curve),
        },
        "counts": {
            "logged_loss_points": len(losses),
            "internal_evaluation_records": len(eval_rows),
        },
        "loss_change": {
            "first_step": int(train_rows[0]["step"]),
            "first_loss": losses[0],
            "last_step": int(train_rows[-1]["step"]),
            "last_loss": losses[-1],
            "minimum_loss": minimum,
            "minimum_loss_step": int(train_rows[losses.index(minimum)]["step"]),
            "first_5_median": first_median,
            "last_5_median": last_median,
            "last_vs_first_5_median_change_percent": (last_median / first_median - 1.0) * 100.0,
        },
        "final_internal_evaluation": {
            "step": int(final_eval["step"]),
            "eval_loss": final_eval_loss,
            "eval_bleu1": finite_float(final_eval["eval_bleu1"], "eval_bleu1", len(eval_rows)),
            "eval_bleu2": finite_float(final_eval["eval_bleu2"], "eval_bleu2", len(eval_rows)),
            "eval_bleu3": finite_float(final_eval["eval_bleu3"], "eval_bleu3", len(eval_rows)),
            "eval_bleu4": finite_float(final_eval["eval_bleu4"], "eval_bleu4", len(eval_rows)),
        },
        "acceptance": {
            "minimum_logged_loss_points": 25,
            "loss_points_requirement_met": len(losses) >= 25,
            "last_5_median_at_most_95_percent_of_first_5_median": last_median <= first_median * 0.95,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
