#!/usr/bin/env python3
"""Audit Candidate H v2 against the complete permitted non-blind V5.4 v3 data."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/cyh/Medical_Qwen")
ADAPTER_PATH = PROJECT_ROOT / "v5_4_pipeline/runtime/candidate_h_v2_adapter.py"
NON_BLIND_PATHS = (
    PROJECT_ROOT / "artifacts/v5_4_pipeline/data_v3/train_v5_4_v3.jsonl",
    PROJECT_ROOT / "artifacts/v5_4_pipeline/data_v3/protocol_dev_v5_4_v3.jsonl",
)
CONTRACT_DIR = PROJECT_ROOT / "v5_3_pipeline/training"
sys.path[:0] = [str(PROJECT_ROOT / "v5_4_pipeline"), str(CONTRACT_DIR)]

from runtime.candidate_h_v2_adapter import (  # noqa: E402
    ASK_REFUSAL,
    NEUTRAL_SYNDROME_TENDENCY,
    OTHER_OBSERVATIONS_QUESTION,
    SUMMARY_NOTE,
    adapt_interaction,
    has_duration_information,
    has_frequency_information,
)
from v5_3_contract import strict_json_loads, validate_target_object  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ADAPTER_PATH))
    forbidden_io_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            if name in {"open", "read_text", "read_bytes", "write_text", "write_bytes"}:
                forbidden_io_calls.append({"name": name, "line": node.lineno})

    errors: list[dict[str, object]] = []
    labels: Counter[str] = Counter()
    availability: Counter[str] = Counter()
    rows = 0
    split_rows: dict[str, int] = {}
    for data_path in NON_BLIND_PATHS:
        split = data_path.name.split("_v5_4_v3.jsonl")[0]
        split_rows[split] = 0
        with data_path.open("r", encoding="utf-8") as handle:
            for index, raw_line in enumerate(handle, 1):
                row = strict_json_loads(raw_line, context=f"{split} line {index}")
                human = row["conversations"][0]["value"]
                reference = strict_json_loads(row["conversations"][1]["value"], context=f"{split} reference {index}")
                expected = (reference["action"], reference["stage"])
                result = adapt_interaction(human, {"action": "unsafe", "reference": reference})
                actual = (result["action"], result["stage"])
                rows += 1
                split_rows[split] += 1
                labels[f"{actual[0]}/{actual[1]}"] += 1
                try:
                    validate_target_object(result, context=f"{split} result {index}")
                except Exception as exc:  # evidence collection must retain every row failure
                    errors.append({"split": split, "index": index, "kind": "contract", "detail": str(exc)})
                    continue
                if actual != expected:
                    errors.append({"split": split, "index": index, "kind": "route", "expected": expected, "actual": actual})
                if result != reference:
                    errors.append({"split": split, "index": index, "kind": "exact_target_mismatch"})
                if actual == ("ask", "initial"):
                    duration = has_duration_information(human)
                    frequency = has_frequency_information(human)
                    availability[f"duration_{duration}_frequency_{frequency}"] += 1
                    questions = result["questions"]
                    if len(questions) != 3 or questions[0] != ASK_REFUSAL or questions[2] != OTHER_OBSERVATIONS_QUESTION:
                        errors.append({"split": split, "index": index, "kind": "ask_fixed_fields"})
                    first_follow_up = questions[1]
                    if duration and any(term in first_follow_up for term in ("多久", "多长时间", "持续了多久")):
                        errors.append({"split": split, "index": index, "kind": "repeated_duration_question"})
                    if frequency and any(term in first_follow_up for term in ("频率", "几次", "是否经常")):
                        errors.append({"split": split, "index": index, "kind": "repeated_frequency_question"})
                    if duration and frequency and first_follow_up != "与刚开始相比，最近是加重、减轻还是基本不变？":
                        errors.append({"split": split, "index": index, "kind": "wrong_trend_question"})
                else:
                    findings = result["key_findings"]
                    if any(not item or item not in human for item in findings) or len(findings) != len(set(findings)):
                        errors.append({"split": split, "index": index, "kind": "nonliteral_or_duplicate_summary"})
                    if (
                        result["syndrome_tendency"] != NEUTRAL_SYNDROME_TENDENCY
                        or result["need_more_info"] != []
                        or result["note"] != SUMMARY_NOTE
                    ):
                        errors.append({"split": split, "index": index, "kind": "summary_fixed_fields"})

    report = {
        "audit": "Candidate H v2 static and complete permitted non-blind v3 audit",
        "status": "PASS" if not errors and not forbidden_io_calls and rows == 1840 else "FAIL",
        "scope": {
            "adapter": str(ADAPTER_PATH),
            "non_blind_files": [str(path) for path in NON_BLIND_PATHS],
            "split_rows": split_rows,
            "rows": rows,
            "heldout_read": False,
            "training_performed": False,
            "api_integrated": False,
        },
        "hashes": {
            "adapter_sha256": sha256(ADAPTER_PATH),
            "non_blind_sha256": {path.name: sha256(path) for path in NON_BLIND_PATHS},
        },
        "static_checks": {
            "adapter_ast_parse": "PASS",
            "adapter_forbidden_file_io_calls": forbidden_io_calls,
            "raw_output_cannot_override": "covered_by_unit_test_and_dev_replay",
        },
        "non_blind_exact_replay": {
            "labels": dict(sorted(labels.items())),
            "ask_information_availability": dict(sorted(availability.items())),
            "contract_and_route_errors": errors,
            "error_count": len(errors),
        },
        "limitations": [
            "This audit uses only the permitted synthetic train and protocol-dev sets.",
            "It does not establish clinical correctness or safety outside the checked protocol.",
            "The deterministic adapter replaces model output and therefore does not measure LoRA capability.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
