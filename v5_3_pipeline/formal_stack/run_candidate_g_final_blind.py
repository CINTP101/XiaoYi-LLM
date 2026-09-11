#!/usr/bin/env python3
"""One-time Candidate G final-blind generation; default execution denies."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


if not (
    os.environ.get("CANDIDATE_G_FORMAL_STACK_ACCEPTED") == "YES"
    and os.environ.get("V5_3_FINAL_BLIND_EVAL_AUTHORIZED") == "YES"
    and os.environ.get("FINAL_BLIND_ONE_TIME_TOKEN") == "CANDIDATE_G_FINAL_BLIND_180"
):
    print(
        "Refusing Candidate G final blind: formal-stack acceptance, explicit "
        "authorization, and one-time token are required.",
        file=sys.stderr,
    )
    raise SystemExit(2)


PROJECT = Path("/home/cyh/Medical_Qwen")
PYTHON = Path("/home/cyh/miniconda3/envs/tcm_llm/bin/python")
BASE = PROJECT / "models/Qwen2.5-1.5B-Instruct"
FORMAL_WEIGHTS = PROJECT / "output/tcm-qwen-1.5b-v5-3"
RUNTIME = PROJECT / "tcm_chat_v5.py"
ADAPTER = PROJECT / "v5_3_pipeline/runtime/safety_contract_adapter_candidate_g.py"
FORMAL_CODE = PROJECT / "v5_3_pipeline/formal_stack"
TRAINING_CODE = PROJECT / "v5_3_pipeline/training"
FORMAL_ARTIFACTS = PROJECT / "artifacts/v5_3_pipeline/formal_stack"
EVAL = PROJECT / "artifacts/v5_3_pipeline/final_blind_candidate_g"

# Frozen literals.  This file is not inspected until after the authorization
# gate above and all stack/GPU preflight checks below.
FINAL_BLIND = PROJECT / "artifacts/v5_3_pipeline/data/heldout_v5_3.jsonl"
FINAL_BLIND_SHA256 = "9bc09b04589a8027035524736eb9291954fb0d861d6f734f781ccbb3a35f489b"

EXPECTED_FILES = {
    RUNTIME: "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe",
    ADAPTER: "acba0785f49f53451c2c02add382509fd605a77a89cc35f69df0f96134af4581",
    FORMAL_ARTIFACTS / "base_model.sha256": "dd65a3e92c82c672f153fc53c4dd0f9d92bb1fec0d2bcad07348f1a78373e91a",
    FORMAL_ARTIFACTS / "v5-3_formal_weights.sha256": "e0ec0c5fd90a032660099cb23a694189d9ab56e4edd779edb6913a72afec5c43",
    TRAINING_CODE / "runtime_evaluate_v5_3.py": "dc7a8a3d45517c0ce3f9674fef631f89c59b81e5ebab6a50a8ad02aaf31b44f8",
    TRAINING_CODE / "runtime_prompt_v5_3.py": "ceb8e349d2427617699653299b873afca0f9af0b28c18c762409141117e956bc",
    TRAINING_CODE / "verify_v5_3_prompt_parity.py": "408ea957f03ccc24dcc37dfcdea03be063c3d3135edc72cd30460c691d113c50",
    TRAINING_CODE / "evaluate_v5_3_contract.py": "5484f0cf187a2a2b2be69ce4925d9788c0b333c8b084136155aff02d28038094",
    TRAINING_CODE / "v5_3_contract.py": "5a7c89120c97712acacc253af1c6269da564529b326d229e6b5a8fedeb3eb99b",
    FORMAL_CODE / "apply_candidate_g_results.py": "010cd86c1fdfe457b50fb92c7524369d88585db8370d895d4255f4334df665da",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"frozen file verification failed: {path}")


def verify_manifest(root: Path, manifest: Path, count: int) -> list[str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"invalid manifest root: {root}")
    lines = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = raw.split(maxsplit=1)
        relative = relative.strip()
        path = root / relative
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"manifest mismatch: {path}")
        lines.append(f"{relative}: OK")
    if len(lines) != count:
        raise RuntimeError(f"manifest count mismatch for {root}: {len(lines)} != {count}")
    return lines


def run_logged(args: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


if EVAL.exists() or EVAL.is_symlink():
    raise RuntimeError(f"refusing to reuse final-blind output: {EVAL}")
for path, expected in EXPECTED_FILES.items():
    verify_file(path, expected)

manifest_jobs = {
    "base_model_preblind_hash_check.log": (BASE, FORMAL_ARTIFACTS / "base_model.sha256", 11),
    "v5-3_preblind_hash_check.log": (FORMAL_WEIGHTS, FORMAL_ARTIFACTS / "v5-3_formal_weights.sha256", 51),
    "v5-1_preblind_hash_check.log": (PROJECT / "output/tcm-qwen-1.5b-v5-1", PROJECT / "artifacts/v5_2_pipeline/v5-1_baseline.sha256", 23),
    "v5-2_preblind_hash_check.log": (PROJECT / "output/tcm-qwen-1.5b-v5-2", PROJECT / "artifacts/v5_3_pipeline/stage0/v5-2_baseline.sha256", 41),
    "candidate-c_preblind_hash_check.log": (PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2", PROJECT / "artifacts/v5_3_pipeline/training/candidates/hardened_from_v5_2/candidate_weights.sha256", 51),
    "candidate-d_preblind_hash_check.log": (PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-precision-from-c", PROJECT / "artifacts/v5_3_pipeline/training/candidates/precision_from_c/candidate-d_weights.sha256", 51),
    "candidate-e_preblind_hash_check.log": (PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-e-minimal-from-d", PROJECT / "artifacts/v5_3_pipeline/training/candidates/minimal_correction_from_d/candidate-e_weights.sha256", 51),
}
verified_logs = {name: verify_manifest(*job) for name, job in manifest_jobs.items()}

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if gpu.returncode:
    raise RuntimeError("nvidia-smi failed before blind-set access")
first_gpu = gpu.stdout.splitlines()[0].split(",")
if len(first_gpu) < 5 or int(first_gpu[4].strip()) < 5800:
    raise RuntimeError("GPU free memory is below 5800 MiB; blind set remains unopened")

EVAL.mkdir(parents=True)
for name, lines in verified_logs.items():
    (EVAL / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
(EVAL / "gpu_preblind_snapshot.csv").write_text(gpu.stdout, encoding="utf-8")

# First heldout access in this program occurs here, after explicit authorization,
# stack integrity, output freshness, and GPU availability all passed.
if FINAL_BLIND.is_symlink() or not FINAL_BLIND.is_file():
    raise RuntimeError("frozen final blind input is not one regular file")
if sha256(FINAL_BLIND) != FINAL_BLIND_SHA256:
    raise RuntimeError("frozen final blind SHA-256 mismatch")
if sum(1 for _ in FINAL_BLIND.open("r", encoding="utf-8")) != 180:
    raise RuntimeError("frozen final blind row count is not 180")
(EVAL / "final_blind_input.sha256").write_text(
    f"{FINAL_BLIND_SHA256}  {FINAL_BLIND}\n", encoding="utf-8"
)

run_logged([
    str(PYTHON), str(TRAINING_CODE / "verify_v5_3_prompt_parity.py"),
    "--jsonl", str(FINAL_BLIND), "--expected-sha256", FINAL_BLIND_SHA256,
    "--base-model-path", str(BASE), "--runtime-source", str(RUNTIME),
    "--expected-runtime-source-sha256", EXPECTED_FILES[RUNTIME],
    "--output", str(EVAL / "final_blind_runtime_prompt_parity.json"),
], EVAL / "prompt_parity.log")

generation_env = dict(os.environ)
generation_env["CUDA_VISIBLE_DEVICES"] = "0"
run_logged([
    str(PYTHON), str(TRAINING_CODE / "runtime_evaluate_v5_3.py"),
    "--base-model-path", str(BASE), "--peft-path", str(FORMAL_WEIGHTS),
    "--test-jsonl", str(FINAL_BLIND), "--expected-test-sha256", FINAL_BLIND_SHA256,
    "--runtime-source", str(RUNTIME), "--expected-runtime-source-sha256", EXPECTED_FILES[RUNTIME],
    "--output-dir", str(EVAL / "raw_generation"), "--max-input-length", "768",
    "--max-new-tokens", "256", "--batch-size", "1", "--require-v5-3-contract",
], EVAL / "raw_generation.log", generation_env)

run_logged([
    str(PYTHON), str(FORMAL_CODE / "apply_candidate_g_results.py"),
    "--authorization-token", "CANDIDATE_G_FINAL_BLIND_AUTHORIZED",
    "--input-jsonl", str(FINAL_BLIND), "--expected-input-sha256", FINAL_BLIND_SHA256,
    "--raw-generation-results", str(EVAL / "raw_generation/all_results.json"),
    "--adapter-source", str(ADAPTER), "--expected-adapter-sha256", EXPECTED_FILES[ADAPTER],
    "--output-dir", str(EVAL / "adapted_generation"),
], EVAL / "adapter_application.log")

run_logged([
    str(PYTHON), str(TRAINING_CODE / "evaluate_v5_3_contract.py"),
    "--blind-test-jsonl", str(FINAL_BLIND), "--expected-blind-sha256", FINAL_BLIND_SHA256,
    "--generation-results", str(EVAL / "adapted_generation/all_results.json"),
    "--output-dir", str(EVAL / "adapted_generation"), "--minimum-samples", "180",
    "--template-only", "--review-template-output", str(EVAL / "manual_safety_review_template.json"),
], EVAL / "review_template.log")

for name, job in manifest_jobs.items():
    post_name = name.replace("preblind", "postblind")
    (EVAL / post_name).write_text("\n".join(verify_manifest(*job)) + "\n", encoding="utf-8")
(EVAL / "frozen_files_postblind_hash_check.log").write_text(
    "\n".join(
        f"{path}: OK"
        for path, expected in EXPECTED_FILES.items()
        if not verify_file(path, expected)
    ) + "\n",
    encoding="utf-8",
)
for directory in (EVAL / "raw_generation", EVAL / "adapted_generation"):
    for path in directory.rglob("*"):
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    directory.chmod(directory.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)

(EVAL / "generation_status.json").write_text(json.dumps({
    "status": "AWAITING_BOUND_MANUAL_REVIEW",
    "samples": 180,
    "formal_weights": str(FORMAL_WEIGHTS),
    "runtime_sha256": EXPECTED_FILES[RUNTIME],
    "adapter_sha256": EXPECTED_FILES[ADAPTER],
    "api_switched": False,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Candidate G final-blind generation completed once; manual review is required before audit.")
