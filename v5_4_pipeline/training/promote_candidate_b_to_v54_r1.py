#!/usr/bin/env python3
"""Promote passing Candidate B + Candidate H v2 into a new frozen V5.4 R1 stack."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

P = Path("/home/cyh/Medical_Qwen")
CANDIDATE = P / "output/tcm-qwen-1.5b-v5-4-candidate-b"
FORMAL = P / "output/tcm-qwen-1.5b-v5-4-r1"
CANDIDATE_MANIFEST = P / "artifacts/v5_4_pipeline/training/candidate_b/candidate_b_weights.sha256"
CANDIDATE_MANIFEST_SHA = "56e667bc1623f1361c2152ea91aed083e179ffd9ee6b6f675aba48b9d55356d1"
DEV_AUDIT = P / "artifacts/v5_4_pipeline/training/candidate_b/protocol_dev_eval_v3/protocol_dev_audit.json"
DEV_AUDIT_SHA = "ee77b0512df886b45b2099329a8b38a70d069f361a86b1a6b21522ff952e9569"
ADAPTER = P / "v5_4_pipeline/runtime/candidate_h_v2_adapter.py"
ADAPTER_SHA = "7a1772d3594dba6ddb30dab9c8a6db9ca1e82b44701756ca45b140097884568a"
RUNNER = P / "v5_4_pipeline/training/evaluate_final_blind_r1_once.py"
RUNNER_SHA = "cfff588346f8173196caecca479b9801ca8206a4fcfb9949d815563f42e6098c"
DEV_EVALUATOR = P / "v5_4_pipeline/training/evaluate_candidate_b_protocol_dev.py"
DEV_EVALUATOR_SHA = "f6d0391e555963cc00c989f9612660f221a48ae7104c84fa61b413884ea9a729"
AUTH = P / "artifacts/v5_4_pipeline/review/root_v54_repair_training_authorization_20260918.json"
AUTH_SHA = "707df6d29210f1f18152715b4a385a98067781ce136afd126818213a4820c349"
PROTECTION = P / "artifacts/v5_4_pipeline/review/root_v54_repair_preflight_protection_20260918.json"
PROTECTION_SHA = "cb515b8ea39bbe141d9809ac71035f8c2247ce6e04e53dd9c7025f75ed8446e2"
H_AUDIT = P / "artifacts/v5_4_pipeline/runtime_candidate_h_v2/static_audit.json"
H_AUDIT_SHA = "b8bb1785a2eb77aeb46bbcc16da1ae1b99a8b9d38a77bede88a379ba343d1d50"
STACK = P / "artifacts/v5_4_pipeline/final_stack_r1"
FORMAL_MANIFEST = STACK / "v5-4-r1_weights.sha256"
FROZEN_ADAPTER = STACK / "candidate_h_v2_adapter.py"
FROZEN_RUNNER = STACK / "evaluate_final_blind_r1_once.py"
FREEZE = STACK / "v5_4_r1_freeze_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"{label} missing or SHA mismatch: {path}")


def main() -> None:
    for path, expected, label in (
        (CANDIDATE_MANIFEST, CANDIDATE_MANIFEST_SHA, "Candidate B manifest"),
        (DEV_AUDIT, DEV_AUDIT_SHA, "protocol-dev audit"),
        (ADAPTER, ADAPTER_SHA, "Candidate H v2"),
        (RUNNER, RUNNER_SHA, "final-blind runner"),
        (DEV_EVALUATOR, DEV_EVALUATOR_SHA, "development evaluator"),
        (AUTH, AUTH_SHA, "repair authorization"),
        (PROTECTION, PROTECTION_SHA, "protection report"),
        (H_AUDIT, H_AUDIT_SHA, "Candidate H v2 audit"),
    ):
        require(path, expected, label)
    dev = json.loads(DEV_AUDIT.read_text(encoding="utf-8"))
    adapted = dev.get("candidate_h_v2_adapted", {})
    if dev.get("status") != "PASS" or adapted.get("release_gate") != "PASS" or any(value != 240 for value in adapted.get("metrics", {}).values()):
        raise RuntimeError("protocol-dev release gate is not a complete 240/240 PASS")
    if CANDIDATE.is_symlink() or not CANDIDATE.is_dir():
        raise RuntimeError("Candidate B directory unavailable")
    checked = subprocess.run(["sha256sum", "-c", str(CANDIDATE_MANIFEST)], cwd=CANDIDATE, text=True, capture_output=True)
    files = [path for path in CANDIDATE.rglob("*") if path.is_file() and not path.is_symlink()]
    if checked.returncode or len(files) != 51:
        raise RuntimeError(f"Candidate B weight verification failed: {checked.stdout}{checked.stderr}")
    if FORMAL.exists() or FORMAL.is_symlink() or STACK.exists() or STACK.is_symlink():
        raise FileExistsError("formal R1 output or final_stack_r1 already exists")

    shutil.copytree(CANDIDATE, FORMAL, symlinks=False)
    STACK.mkdir(parents=True)
    shutil.copy2(CANDIDATE_MANIFEST, FORMAL_MANIFEST)
    shutil.copy2(ADAPTER, FROZEN_ADAPTER)
    shutil.copy2(RUNNER, FROZEN_RUNNER)
    copied_check = subprocess.run(["sha256sum", "-c", str(FORMAL_MANIFEST)], cwd=FORMAL, text=True, capture_output=True)
    copied_files = [path for path in FORMAL.rglob("*") if path.is_file() and not path.is_symlink()]
    if copied_check.returncode or len(copied_files) != 51:
        raise RuntimeError(f"formal R1 copy verification failed: {copied_check.stdout}{copied_check.stderr}")
    for path in copied_files:
        path.chmod(0o444)
    for directory in sorted((path for path in FORMAL.rglob("*") if path.is_dir()), reverse=True):
        directory.chmod(0o555)
    FORMAL.chmod(0o555)
    writable_files = [str(path.relative_to(FORMAL)) for path in copied_files if path.stat().st_mode & 0o222]
    if writable_files:
        raise RuntimeError(f"formal R1 contains writable files: {writable_files}")

    freeze = {
        "status": "FROZEN",
        "date": "2026-09-18",
        "stack_name": "tcm-qwen-1.5b-v5-4-r1 + tcm_chat_v5 + Candidate H v2",
        "weights": {
            "path": str(FORMAL),
            "files": 51,
            "manifest": str(FORMAL_MANIFEST),
            "manifest_sha256": sha256(FORMAL_MANIFEST),
            "writable_files": len(writable_files),
            "source_candidate": str(CANDIDATE),
        },
        "runtime": {"path": str(P / "tcm_chat_v5.py"), "sha256": "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"},
        "interaction_adapter": {"path": str(FROZEN_ADAPTER), "sha256": sha256(FROZEN_ADAPTER), "raw_output_influences_adapter": False},
        "runner": {"path": str(FROZEN_RUNNER), "sha256": sha256(FROZEN_RUNNER), "maximum_runs": 1},
        "development_evidence": {
            "protocol_dev_audit": str(DEV_AUDIT),
            "protocol_dev_audit_sha256": sha256(DEV_AUDIT),
            "raw_model_action_stage_correct": "229/240",
            "raw_model_safety_refusal": "239/240",
            "adapted_all_metrics": "240/240",
            "adapted_release_gate": "PASS",
        },
        "training_evidence": {
            "candidate_b_manifest_sha256": CANDIDATE_MANIFEST_SHA,
            "global_steps": 190,
            "initial_loss_window_mean": 2.1995899999999997,
            "final_loss_window_mean": 0.24815,
            "resume_from_checkpoint": False,
            "peft_start": "/home/cyh/Medical_Qwen/output/tcm-qwen-1.5b-v5-3",
        },
        "bindings": {
            "authorization_sha256": AUTH_SHA,
            "protection_report_sha256": PROTECTION_SHA,
            "candidate_h_v2_audit_sha256": H_AUDIT_SHA,
            "development_evaluator_sha256": DEV_EVALUATOR_SHA,
            "train_sha256": "56d8b5590a87759233fbbd3f5baa4e70796143f17519d92ac21c56c0dfc9dff2",
            "protocol_dev_sha256": "bbef4bc919254a4271d209e632247316983c37a8594824deac595fafeb9526d7",
        },
        "heldout": {
            "path": "/home/cyh/Medical_Qwen/artifacts/v5_4_pipeline/data_v3/heldout_v5_4_v3.jsonl",
            "sha256": "022c81be73bed6c1b7a4552b64b1050e8f551f03d98ec18012dbcd0499e99fd1",
            "rows": 240,
            "expected_ask": 180,
            "expected_summarize": 60,
            "access_before_go": "DENY",
            "maximum_runs": 1,
            "text_inspected_before_freeze": False,
        },
        "generation": {"max_input_length": 768, "max_new_tokens": 256, "batch_size": 1, "do_sample": False},
        "api_switch": "DENY",
    }
    FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "formal": str(FORMAL), "weights_manifest_sha256": sha256(FORMAL_MANIFEST), "runner_sha256": sha256(FROZEN_RUNNER), "freeze_manifest_sha256": sha256(FREEZE), "heldout_access": "DENY"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
