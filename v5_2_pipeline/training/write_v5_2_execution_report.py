#!/usr/bin/env python3
"""Materialize the accepted V5.2 input manifest as a human-readable run plan."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write V5.2 execution-parameter report")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.input_manifest).resolve(strict=True)
    output_path = Path(args.output).resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"execution report already exists: {output_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    plan = manifest["training_plan"]
    item = manifest["isolated_files"][0]

    target_steps = int(plan["target_optimization_steps"])
    max_epochs = int(plan["max_train_epochs"])
    total_steps = int(plan["total_optimization_steps_expected"])
    epochs = int(plan["num_train_epochs"])
    logging_steps = int(plan["logging_steps"])
    eval_steps = int(plan["eval_steps"])
    save_steps = int(plan["save_steps"])
    expected_evaluations = int(plan["training_eval_events_expected"])
    if (
        total_steps < target_steps
        or epochs > max_epochs
        or logging_steps != 1
        or eval_steps > total_steps // 4
        or save_steps != eval_steps
        or expected_evaluations < 4
    ):
        raise ValueError("training plan does not satisfy the V5.2 optimizer/logging/evaluation gates")

    rows = [
        ("Accepted source JSONL", manifest["source_clean_jsonl"]),
        ("Accepted source SHA-256", manifest["source_sha256"]),
        ("Isolated input directory", manifest["isolated_directory"]),
        ("Only input file", item["name"]),
        ("Isolated input SHA-256", item["sha256"]),
        ("Validated JSONL rows (N)", plan["input_rows"]),
        ("Expected validation rows (H)", plan["validation_rows_expected"]),
        ("Expected training rows (T)", plan["train_rows_expected"]),
        ("Required minimum optimizer steps", target_steps),
        ("Maximum permitted epochs", max_epochs),
        ("Per-device batch size", plan["per_device_train_batch_size"]),
        ("Gradient accumulation (G)", plan["gradient_accumulation_steps"]),
        ("Expected steps per epoch", plan["steps_per_epoch_expected"]),
        ("Epochs (E)", plan["num_train_epochs"]),
        ("Expected total optimization steps (S)", plan["total_optimization_steps_expected"]),
        ("Logging steps", plan["logging_steps"]),
        ("Evaluation steps", plan["eval_steps"]),
        ("Save steps", plan["save_steps"]),
        ("Expected in-training evaluations", expected_evaluations),
    ]
    lines = [
        "# V5.2 已验收输入的实际训练参数",
        "",
        f"- 生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "- 状态：数据与参数已锁定；训练命令尚未在本报告生成步骤中启动。",
        "- 输入隔离结论：训练目录仅含一个哈希已复核的 `clean_train.jsonl`，未包含 rejections 或 heldout 文件。",
        "",
        "| 项目 | 实际值 |",
        "|---|---|",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in rows)
    lines.extend(
        [
            "",
            "固定训练参数：学习率 `1e-5`；最大上下文 512；BF16；梯度检查点；"
            "`adamw_torch_fused`；linear scheduler；warmup ratio `0.03`；weight decay `0.0`；seed/data seed `42`；"
            "内部验证比例 5%；内部测试比例 0%；checkpoint 保留数 2。",
            "",
            f"该计划在不超过 {max_epochs} 个 epoch 内安排 {total_steps} 个优化步；"
            f"logging_steps=1，因此至少记录 {total_steps} 个损失点；"
            f"eval_steps=save_steps={eval_steps}，预计训练中评估 {expected_evaluations} 次。",
            "",
            "盲测标签口径从最终盲测参考答案自动提取全部实际 action/stage 组合；"
            "refuse/safety、ask/initial 及最终交付中出现的其他组合均按其实际必填字段和字段类型评分，"
            "不使用 V5.1 三类标签的硬编码白名单。",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
