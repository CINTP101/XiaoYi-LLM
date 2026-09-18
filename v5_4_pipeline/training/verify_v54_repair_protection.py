#!/usr/bin/env python3
"""Verify every protected model and recovery archive before V5.4 repair training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path("/home/cyh/Medical_Qwen")
OUT = PROJECT / "artifacts/v5_4_pipeline/review/root_v54_repair_preflight_protection_20260918.json"

MODELS = (
    ("v5_1", PROJECT / "output/tcm-qwen-1.5b-v5-1", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/v5_1.sha256"),
    ("v5_2", PROJECT / "output/tcm-qwen-1.5b-v5-2", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/v5_2.sha256"),
    ("v5_3", PROJECT / "output/tcm-qwen-1.5b-v5-3", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/v5_3.sha256"),
    ("candidate_c", PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/candidate_c.sha256"),
    ("candidate_d", PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-precision-from-c", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/candidate_d.sha256"),
    ("candidate_e", PROJECT / "output/tcm-qwen-1.5b-v5-3-candidate-e-minimal-from-d", PROJECT / "artifacts/v5_4_pipeline/review/release_inventory_manifests_20260918/candidate_e.sha256"),
    ("old_candidate_a", PROJECT / "output/tcm-qwen-1.5b-v5-4-candidate-a", PROJECT / "artifacts/v5_4_pipeline/training/candidate_a/candidate_a_weights.sha256"),
    ("old_failed_v5_4", PROJECT / "output/tcm-qwen-1.5b-v5-4", PROJECT / "artifacts/v5_4_pipeline/final_stack/v5-4_weights.sha256"),
)

ARCHIVES = (
    (
        "original_2026_09_09",
        Path("/mnt/c/Users/cyh/Desktop/小医ai问诊模型/Medical_Qwen_backup_2026-09-09.tar"),
        5017067520,
        "f88d537122d0bb07965186667f9b403adfd8a667b42ea217d86af19336583556",
    ),
    (
        "v5_3_not_publishable_2026_09_11",
        Path("/mnt/c/Users/cyh/Desktop/小医ai问诊模型/Medical_Qwen_V5.3_artifacts_2026-09-11_NOT_PUBLISHABLE.tar"),
        354928640,
        "b880f89129e2c25447d9858ceeb078560aecf81309f131eda4ad08cac5b8d1a5",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        digest, relative = raw.split(maxsplit=1)
        relative = relative.lstrip("*").removeprefix("./")
        entries[relative] = digest.lower()
    return entries


def verify_model(name: str, root: Path, manifest: Path) -> dict[str, object]:
    expected = load_manifest(manifest)
    actual_files = sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    missing = sorted(set(expected) - set(actual_files))
    extra = sorted(set(actual_files) - set(expected))
    mismatched = []
    for relative in sorted(set(expected) & set(actual_files)):
        actual_hash = sha256(root / relative)
        if actual_hash != expected[relative]:
            mismatched.append({"path": relative, "expected": expected[relative], "actual": actual_hash})
    status = "PASS" if root.is_dir() and not missing and not extra and not mismatched else "FAIL"
    return {
        "name": name,
        "path": str(root),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "expected_files": len(expected),
        "actual_files": len(actual_files),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "status": status,
    }


def main() -> None:
    models = [verify_model(*item) for item in MODELS]
    archives = []
    for name, path, expected_bytes, expected_hash in ARCHIVES:
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        archives.append(
            {
                "name": name,
                "path": str(path),
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "status": "PASS" if actual_bytes == expected_bytes and actual_hash == expected_hash else "FAIL",
            }
        )
    report = {
        "status": "PASS" if all(item["status"] == "PASS" for item in [*models, *archives]) else "FAIL",
        "purpose": "V5.4 repair cycle training preflight; read-only verification",
        "models": models,
        "archives": archives,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "models": {item["name"]: item["status"] for item in models}, "archives": {item["name"]: item["status"] for item in archives}}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
