#!/usr/bin/env python3
"""Validate an isolated prompt-only Candidate F runtime source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


class ValidationError(RuntimeError):
    pass


REQUIRED_MARKER_GROUPS = {
    # A prompt may express default denial either literally or by making the
    # first safety boundary mandatory for every ask/initial response.
    "default_policy": (
        "默认",
        "一律用ask/initial",
        "questions[0] 固定为",
        "questions[0]必须固定为",
        "不得省略该安全边界",
    ),
    "explicit_refusal": ("拒绝", "不能"),
    "broad_medical_scope": ("医疗", "诊疗"),
    "diagnosis": ("诊断",),
    "prescription": ("开方", "方剂"),
    "medicine": ("药物", "药材", "中药"),
    "dose": ("剂量", "用量"),
    "treatment": ("治疗", "调理"),
    "strict_json": ("JSON",),
    "action_key": ('"action"',),
    "stage_key": ('"stage"',),
    "ask_action": ("ask",),
    "initial_stage": ("initial",),
    "summarize_action": ("summarize",),
    "summary_stage": ("summary",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Candidate F prompt-only runtime source")
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.runtime_source).expanduser()
    if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".py":
        raise ValidationError("runtime source must be one regular non-symlink .py file")
    source = source.resolve(strict=True)
    expected = args.expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValidationError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual = sha256_file(source)
    if actual != expected:
        raise ValidationError(f"runtime source SHA-256 mismatch: expected {expected}, got {actual}")

    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    executable = list(module.body)
    if executable and isinstance(executable[0], ast.Expr) and isinstance(executable[0].value, ast.Constant) and isinstance(executable[0].value.value, str):
        executable = executable[1:]
    if len(executable) != 1 or not isinstance(executable[0], (ast.Assign, ast.AnnAssign)):
        raise ValidationError("prompt-only runtime may contain only a module docstring and one CONSULTATION_PROMPT assignment")
    statement = executable[0]
    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
    if len(targets) != 1 or not isinstance(targets[0], ast.Name) or targets[0].id != "CONSULTATION_PROMPT":
        raise ValidationError("the only assignment must target CONSULTATION_PROMPT")
    try:
        prompt = ast.literal_eval(statement.value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("CONSULTATION_PROMPT must be a literal string") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("CONSULTATION_PROMPT must be a non-empty literal string")

    marker_results = {
        name: {"alternatives": list(alternatives), "matched": [value for value in alternatives if value in prompt]}
        for name, alternatives in REQUIRED_MARKER_GROUPS.items()
    }
    missing = [name for name, result in marker_results.items() if not result["matched"]]
    if missing:
        raise ValidationError("default-refusal prompt gate missing concepts: " + ", ".join(missing))

    output = Path(args.output).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite validation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS",
        "runtime_source": str(source),
        "runtime_source_sha256": actual,
        "consultation_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "consultation_prompt_characters": len(prompt),
        "top_level_contract": "optional module docstring plus one literal CONSULTATION_PROMPT assignment",
        "default_refusal_marker_gate": marker_results,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "runtime_source_sha256": actual, "prompt_characters": len(prompt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
