#!/usr/bin/env python3
"""Verify blind-evaluation prompts exactly match V5.2 training preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_sft_qwen import build_eval_sample_from_conversation, build_prompt_from_history
from supervised_finetuning import inject_empty_think_block
from template import get_conv_template


DEFAULT_STYLE_CONSTRAINT = "请用纯中文短段落回答，不要用 Markdown。"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify V5.2 training-template evaluation prompt parity")
    parser.add_argument("--blind-test-jsonl", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blind_path = Path(args.blind_test_jsonl).resolve(strict=True)
    output_path = Path(args.output).resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite prompt-alignment manifest: {output_path}")
    actual_hash = sha256_file(blind_path)
    if actual_hash != args.expected_sha256.lower():
        raise ValueError(f"blind SHA-256 mismatch: expected {args.expected_sha256}, got {actual_hash}")

    template = get_conv_template("qwen")
    cases = []
    with blind_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            row = json.loads(raw_line)
            parsed = build_eval_sample_from_conversation(row.get("conversations", []))
            if parsed is None:
                raise ValueError(f"blind source line {line_number}: cannot parse conversation")
            original_system, history_pairs, target_answer = parsed
            aligned_system = (original_system.strip() + "\n\n" + DEFAULT_STYLE_CONSTRAINT).strip() if original_system else DEFAULT_STYLE_CONSTRAINT
            full_pairs = history_pairs[:-1] + [[history_pairs[-1][0], target_answer]]
            training_dialog = template.get_dialog(full_pairs, system_prompt=aligned_system)
            training_prompt = inject_empty_think_block(training_dialog[-2], "qwen", True)
            evaluation_prompt = build_prompt_from_history(
                prompt_template=template,
                history_pairs=history_pairs,
                system_prompt=aligned_system,
                tokenizer=None,
                disable_thinking=True,
            )
            cases.append(
                {
                    "source_line": line_number,
                    "training_prompt_sha256": sha256_text(training_prompt),
                    "evaluation_prompt_sha256": sha256_text(evaluation_prompt),
                    "exact_match": training_prompt == evaluation_prompt,
                }
            )
    if not cases or not all(case["exact_match"] for case in cases):
        raise AssertionError("training/evaluation prompt mismatch")
    manifest = {
        "blind_test_jsonl": str(blind_path),
        "blind_test_sha256": actual_hash,
        "template_name": "qwen",
        "disable_thinking": True,
        "training_style_constraint": DEFAULT_STYLE_CONSTRAINT,
        "prompt_construction": "legacy template path with training-style system constraint; tokenizer.apply_chat_template disabled.",
        "total_cases": len(cases),
        "exact_prompt_matches": sum(case["exact_match"] for case in cases),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_cases": len(cases), "exact_prompt_matches": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
