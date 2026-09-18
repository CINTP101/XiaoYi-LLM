#!/usr/bin/env python3
"""Static/preflight audit for the frozen V5.4 R1 one-time blind runner."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

P = Path("/home/cyh/Medical_Qwen")
RUNNER = P / "artifacts/v5_4_pipeline/final_stack_r1/evaluate_final_blind_r1_once.py"
RUNNER_SHA = "cfff588346f8173196caecca479b9801ca8206a4fcfb9949d815563f42e6098c"
FREEZE = P / "artifacts/v5_4_pipeline/final_stack_r1/v5_4_r1_freeze_manifest.json"
FREEZE_SHA = "5e2bf161946b62d248753c4871e0921f24a8e6b37b04a04e0351ebed82009912"
FORMAL = P / "output/tcm-qwen-1.5b-v5-4-r1"
MANIFEST = P / "artifacts/v5_4_pipeline/final_stack_r1/v5-4-r1_weights.sha256"
GO = P / "artifacts/v5_4_pipeline/review/GO_V5_4_R1_FINAL_BLIND.json"
OUT = P / "artifacts/v5_4_pipeline/final_blind_r1_once"
LOCK = P / "artifacts/v5_4_pipeline/final_blind_r1_once.consumed.lock"
TRACE = P / "artifacts/v5_4_pipeline/review/final_blind_r1_access_trace.json"
REPORT = P / "artifacts/v5_4_pipeline/review/final_blind_r1_runner_static_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    heldout_reads = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "read_bytes":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "HELDOUT":
                heldout_reads += 1
    main_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    ordered_calls: list[str] = []
    for statement in main_function.body:
        names = [
            node.func.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"validate_go", "verify_frozen_stack", "consume_once", "snapshot_heldout"}
        ]
        ordered_calls.extend(names)
    manifest_check = subprocess.run(["sha256sum", "-c", str(MANIFEST)], cwd=FORMAL, text=True, capture_output=True)
    formal_files = [path for path in FORMAL.rglob("*") if path.is_file() and not path.is_symlink()]
    writable_files = [str(path.relative_to(FORMAL)) for path in formal_files if path.stat().st_mode & 0o222]
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks = {
        "runner_sha256": sha256(RUNNER) == RUNNER_SHA,
        "freeze_sha256": sha256(FREEZE) == FREEZE_SHA,
        "runner_ast": True,
        "one_heldout_content_read_site": heldout_reads == 1,
        "authorization_then_freeze_then_lock_then_read": ordered_calls == ["validate_go", "verify_frozen_stack", "consume_once", "snapshot_heldout"],
        "no_retired_data_v2_heldout_path": "data_v2/heldout" not in source,
        "go_absent": not GO.exists() and not GO.is_symlink(),
        "output_absent": not OUT.exists() and not OUT.is_symlink(),
        "lock_absent": not LOCK.exists() and not LOCK.is_symlink(),
        "access_trace_absent": not TRACE.exists() and not TRACE.is_symlink(),
        "formal_weight_manifest": manifest_check.returncode == 0 and len(formal_files) == 51,
        "formal_files_read_only": not writable_files,
        "freeze_binds_runner": freeze.get("runner", {}).get("sha256") == RUNNER_SHA,
        "freeze_denies_pre_go": freeze.get("heldout", {}).get("access_before_go") == "DENY",
        "freeze_maximum_runs_one": freeze.get("heldout", {}).get("maximum_runs") == 1,
        "api_switch_denied": freeze.get("api_switch") == "DENY",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "runner_sha256": sha256(RUNNER),
        "freeze_manifest_sha256": sha256(FREEZE),
        "heldout_text_read": False,
        "writable_files": writable_files,
        "main_ordered_calls": ordered_calls,
        "manifest_stderr": manifest_check.stderr,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
