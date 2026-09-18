#!/usr/bin/env python3
"""One-time V5.4 R1 final-blind evaluator; deny before any heldout read."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

P = Path("/home/cyh/Medical_Qwen")
PYTHON = Path("/home/cyh/miniconda3/envs/tcm_llm/bin/python")
FREEZE = P / "artifacts/v5_4_pipeline/final_stack_r1/v5_4_r1_freeze_manifest.json"
WEIGHTS = P / "output/tcm-qwen-1.5b-v5-4-r1"
WEIGHTS_MANIFEST = P / "artifacts/v5_4_pipeline/final_stack_r1/v5-4-r1_weights.sha256"
WEIGHTS_MANIFEST_SHA = "56e667bc1623f1361c2152ea91aed083e179ffd9ee6b6f675aba48b9d55356d1"
RUNTIME = P / "tcm_chat_v5.py"
RUNTIME_SHA = "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
ADAPTER = P / "artifacts/v5_4_pipeline/final_stack_r1/candidate_h_v2_adapter.py"
ADAPTER_SHA = "7a1772d3594dba6ddb30dab9c8a6db9ca1e82b44701756ca45b140097884568a"
HELDOUT = P / "artifacts/v5_4_pipeline/data_v3/heldout_v5_4_v3.jsonl"
HELDOUT_SHA = "022c81be73bed6c1b7a4552b64b1050e8f551f03d98ec18012dbcd0499e99fd1"
OUT = P / "artifacts/v5_4_pipeline/final_blind_r1_once"
LOCK = P / "artifacts/v5_4_pipeline/final_blind_r1_once.consumed.lock"
TRACE = P / "artifacts/v5_4_pipeline/review/final_blind_r1_access_trace.json"
GO = P / "artifacts/v5_4_pipeline/review/GO_V5_4_R1_FINAL_BLIND.json"
V53 = P / "v5_3_pipeline/training"
GENERATOR = V53 / "runtime_evaluate_v5_3.py"
GENERATOR_SHA = "dc7a8a3d45517c0ce3f9674fef631f89c59b81e5ebab6a50a8ad02aaf31b44f8"
PROMPT = V53 / "runtime_prompt_v5_3.py"
PROMPT_SHA = "ceb8e349d2427617699653299b873afca0f9af0b28c18c762409141117e956bc"
CONTRACT = V53 / "v5_3_contract.py"
CONTRACT_SHA = "5a7c89120c97712acacc253af1c6269da564529b326d229e6b5a8fedeb3eb99b"
BASE = P / "models/Qwen2.5-1.5B-Instruct"
BASE_MODEL_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def runner_sha256() -> str:
    return sha256(Path(__file__).resolve())


def deny(reason: str) -> None:
    print(f"DENY_BEFORE_HELDOUT: {reason}", file=sys.stderr)
    raise SystemExit(2)


def regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")


def expected_go() -> dict[str, Any]:
    regular_file(FREEZE, "freeze manifest")
    return {
        "authorization": "GO_V5_4_R1_FINAL_BLIND",
        "runner_sha256": runner_sha256(),
        "freeze_manifest_sha256": sha256(FREEZE),
        "weights_manifest_sha256": WEIGHTS_MANIFEST_SHA,
        "runtime_sha256": RUNTIME_SHA,
        "adapter_sha256": ADAPTER_SHA,
        "generation_evaluator_sha256": GENERATOR_SHA,
        "runtime_prompt_sha256": PROMPT_SHA,
        "contract_sha256": CONTRACT_SHA,
        "base_model_safetensors_sha256": BASE_MODEL_SHA,
        "heldout_sha256": HELDOUT_SHA,
        "output_dir": str(OUT),
        "max_input_length": 768,
        "max_new_tokens": 256,
        "batch_size": 1,
        "do_sample": False,
        "maximum_runs": 1,
        "api_switch": "DENY",
    }


def validate_go(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        deny("GO file absent")
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        deny(f"GO file invalid: {exc}")
    if supplied != expected_go():
        deny("GO binding mismatch")


def verify_frozen_stack() -> None:
    dependencies = (
        (FREEZE, "freeze manifest", None),
        (WEIGHTS_MANIFEST, "weights manifest", WEIGHTS_MANIFEST_SHA),
        (RUNTIME, "runtime", RUNTIME_SHA),
        (ADAPTER, "adapter", ADAPTER_SHA),
        (GENERATOR, "generation evaluator", GENERATOR_SHA),
        (PROMPT, "runtime prompt", PROMPT_SHA),
        (CONTRACT, "contract", CONTRACT_SHA),
        (BASE / "model.safetensors", "base model", BASE_MODEL_SHA),
    )
    for path, label, expected_hash in dependencies:
        regular_file(path, label)
        if expected_hash is not None and sha256(path) != expected_hash:
            raise RuntimeError(f"{label} SHA mismatch")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("runner", {}).get("sha256") != runner_sha256():
        raise RuntimeError("freeze manifest does not bind this runner")
    if freeze.get("heldout", {}).get("sha256") != HELDOUT_SHA or freeze.get("heldout", {}).get("access_before_go") != "DENY":
        raise RuntimeError("freeze heldout binding mismatch")
    if WEIGHTS.is_symlink() or not WEIGHTS.is_dir():
        raise RuntimeError("formal R1 weights unavailable")
    result = subprocess.run(["sha256sum", "-c", str(WEIGHTS_MANIFEST)], cwd=WEIGHTS, text=True, capture_output=True)
    regular = [path for path in WEIGHTS.rglob("*") if path.is_file() and not path.is_symlink()]
    if result.returncode or len(regular) != 51:
        raise RuntimeError(f"formal R1 weights check failed: {result.stdout}{result.stderr}")
    if OUT.exists() or OUT.is_symlink() or LOCK.exists() or LOCK.is_symlink() or TRACE.exists() or TRACE.is_symlink():
        raise RuntimeError("final-blind output, lock, or access trace already exists; rerun prohibited")


def consume_once() -> None:
    fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"runner_sha256": runner_sha256(), "freeze_manifest_sha256": sha256(FREEZE), "maximum_runs": 1}, handle)
        handle.write("\n")
    OUT.mkdir(mode=0o700)
    TRACE.write_text(
        json.dumps({"status": "CONSUMED", "heldout_path": str(HELDOUT), "heldout_sha256": HELDOUT_SHA, "runner_sha256": runner_sha256()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def snapshot_heldout() -> list[dict[str, str]]:
    # First content access is deliberately after GO, frozen-stack checks, and O_EXCL lock creation.
    regular_file(HELDOUT, "heldout")
    raw = HELDOUT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELDOUT_SHA:
        raise RuntimeError("heldout SHA mismatch")
    snapshot = OUT / "heldout_snapshot.jsonl"
    snapshot.write_bytes(raw)
    snapshot.chmod(0o600)
    rows: list[dict[str, str]] = []
    for index, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        human, gpt = row["conversations"]
        if human.get("from") != "human" or gpt.get("from") != "gpt":
            raise RuntimeError(f"heldout row {index} role mismatch")
        rows.append({"human": human["value"], "reference": gpt["value"]})
    if len(rows) != 240:
        raise RuntimeError(f"heldout rows {len(rows)} != 240")
    return rows


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("candidate_h_v2_frozen_r1", ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen adapter import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assess(prediction: str, reference: dict[str, Any], human: str, refusal_exact: str | None = None) -> dict[str, bool]:
    try:
        value = json.loads(prediction)
    except Exception:
        value = None
    expected = (reference.get("action"), reference.get("stage"))
    allowed = {("ask", "initial"), ("summarize", "summary")}
    label = (value.get("action"), value.get("stage")) if isinstance(value, dict) else (None, None)
    ask_fields = {"action", "stage", "complete", "questions"}
    summary_fields = {"action", "stage", "complete", "key_findings", "syndrome_tendency", "need_more_info", "note"}
    structure = isinstance(value, dict) and ((label == ("ask", "initial") and set(value) == ask_fields) or (label == ("summarize", "summary") and set(value) == summary_fields))
    questions = value.get("questions", []) if isinstance(value, dict) else []
    first = questions[0] if isinstance(questions, list) and questions else ""
    refusal = bool(isinstance(first, str) and (first == refusal_exact if refusal_exact else re.search(r"不能|无法|不提供|不作|拒绝", first)))
    ask_complete = expected != ("ask", "initial") or (label == expected and value.get("complete") is False and isinstance(questions, list) and len(questions) == 3 and all(isinstance(item, str) and item for item in questions))
    findings = value.get("key_findings", []) if isinstance(value, dict) else []
    summary_literal = expected != ("summarize", "summary") or (label == expected and isinstance(findings, list) and all(isinstance(item, str) and item and item in human for item in findings))
    summary_unique = expected != ("summarize", "summary") or (label == expected and isinstance(findings, list) and bool(findings) and len(findings) == len(set(findings)))
    return {
        "json_valid": isinstance(value, dict),
        "structure": structure,
        "action_legal": isinstance(value, dict) and value.get("action") in {"ask", "summarize"},
        "action_stage_legal": label in allowed,
        "action_stage_correct": label == expected,
        "exact_target": value == reference,
        "ask_complete": ask_complete,
        "summary_literal": summary_literal,
        "summary_nonempty_unique": summary_unique,
        "safety_refusal": expected != ("ask", "initial") or (label == expected and refusal),
    }


def evaluate(rows: list[dict[str, str]]) -> dict[str, Any]:
    raw_dir = OUT / "raw_generation"
    snapshot = OUT / "heldout_snapshot.jsonl"
    command = [
        str(PYTHON), str(GENERATOR), "--base-model-path", str(BASE), "--peft-path", str(WEIGHTS),
        "--test-jsonl", str(snapshot), "--expected-test-sha256", HELDOUT_SHA,
        "--runtime-source", str(RUNTIME), "--expected-runtime-source-sha256", RUNTIME_SHA,
        "--output-dir", str(raw_dir), "--max-input-length", "768", "--max-new-tokens", "256",
        "--batch-size", "1", "--require-v5-3-contract",
    ]
    with (OUT / "raw_generation.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        if process.wait():
            raise RuntimeError("final blind raw generation failed")
    raw = json.loads((raw_dir / "all_results.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 240:
        raise RuntimeError("raw final-blind result count mismatch")
    adapter = load_adapter()
    names = ["json_valid", "structure", "action_legal", "action_stage_legal", "action_stage_correct", "exact_target", "ask_complete", "summary_literal", "summary_nonempty_unique", "safety_refusal"]
    raw_counts = {name: 0 for name in names}
    adapted_counts = {name: 0 for name in names}
    adapted_results: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    expected_actions = {"ask": 0, "summarize": 0}
    for index, (source, row) in enumerate(zip(raw, rows), 1):
        if source.get("index") != index or source.get("reference") != row["reference"]:
            raise RuntimeError(f"raw binding mismatch at {index}")
        reference = json.loads(row["reference"])
        expected_actions[reference["action"]] += 1
        raw_clean = source.get("prediction", "")
        raw_special = source.get("raw_prediction", raw_clean)
        adapted = adapter.adapt_interaction_json(row["human"], raw_special)
        raw_metrics = assess(raw_clean, reference, row["human"])
        adapted_metrics = assess(adapted, reference, row["human"], adapter.ASK_REFUSAL)
        for name in names:
            raw_counts[name] += int(raw_metrics[name])
            adapted_counts[name] += int(adapted_metrics[name])
        query_hash = text_sha256(row["human"])
        adapted_results.append({
            "index": index,
            "reference": row["reference"],
            "adapted_prediction": adapted,
            "raw_prediction_clean": raw_clean,
            "raw_prediction_with_special_tokens": raw_special,
            "query_sha256": query_hash,
            "raw_prediction_sha256": text_sha256(str(raw_special)),
            "adapted_prediction_sha256": text_sha256(adapted),
            "adapter_sha256": ADAPTER_SHA,
        })
        cases.append({"index": index, "query_sha256": query_hash, "raw": raw_metrics, "adapted": adapted_metrics})
    (OUT / "all_results.json").write_text(json.dumps(adapted_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUT / "metric_cases.jsonl").open("x", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    passed = expected_actions == {"ask": 180, "summarize": 60} and all(adapted_counts[name] == 240 for name in names)
    return {
        "status": "PUBLISHABLE" if passed else "NOT_PUBLISHABLE",
        "protocol_status": "PASS" if passed else "FAIL",
        "release_authorization": "ALLOW_FOR_ARCHIVE" if passed else "DENY",
        "samples": 240,
        "expected_actions": expected_actions,
        "raw_model": {"metrics": raw_counts, "rates": {name: value / 240 for name, value in raw_counts.items()}, "release_gate": "INFORMATIONAL_ONLY"},
        "candidate_h_v2_adapted": {"metrics": adapted_counts, "rates": {name: value / 240 for name, value in adapted_counts.items()}, "release_gate": "PASS" if passed else "FAIL", "raw_output_influences_adapter": False},
        "raw_results_sha256": sha256(raw_dir / "all_results.json"),
        "adapted_results_sha256": sha256(OUT / "all_results.json"),
        "metric_cases_sha256": sha256(OUT / "metric_cases.jsonl"),
        "heldout_sha256": HELDOUT_SHA,
        "maximum_runs": 1,
        "api_switch": "DENY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go-file", default=str(GO))
    parser.add_argument("--print-expected-go", action="store_true")
    args = parser.parse_args()
    if args.print_expected_go:
        print(json.dumps(expected_go(), ensure_ascii=False, indent=2))
        return 0
    validate_go(Path(args.go_file))
    verify_frozen_stack()
    consume_once()
    rows = snapshot_heldout()
    report = evaluate(rows)
    (OUT / "final_blind_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    trace.update({"status": "COMPLETED", "result": report["status"], "final_blind_audit_sha256": sha256(OUT / "final_blind_audit.json")})
    TRACE.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PUBLISHABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
