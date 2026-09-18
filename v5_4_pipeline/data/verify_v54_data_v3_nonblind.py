#!/usr/bin/env python3
"""Verify V5.4 v3 train/dev structure and provenance without reading heldout text."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path("/home/cyh/Medical_Qwen")
DATA = PROJECT / "artifacts/v5_4_pipeline/data_v3"
OUT = PROJECT / "artifacts/v5_4_pipeline/review/root_data_v3_nonblind_verification_20260918.json"
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
from v5_3_contract import validate_target_text  # noqa: E402


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    trace: dict[str, dict[str, object]] = {}
    trace_by_split: dict[str, list[dict[str, object]]] = {"train": [], "protocol_dev": []}
    trace_nonblind = 0
    with (DATA / "source_trace_v5_4_v3.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item["split"] in {"train", "protocol_dev"}:
                trace_nonblind += 1
                trace[item["id"]] = item
                trace_by_split[item["split"]].append(item)

    errors: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    categories: Counter[str] = Counter()
    states: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sampled: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()

    for split, expected_count in (("train", 1600), ("protocol_dev", 240)):
        path = DATA / f"{split}_v5_4_v3.jsonl"
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                count += 1
                provenance = trace_by_split[split][line_number - 1] if line_number <= len(trace_by_split[split]) else None
                row_id = provenance.get("id") if provenance is not None else None
                if row_id in seen_ids:
                    errors.append({"split": split, "line": line_number, "kind": "duplicate_trace_id", "id": row_id})
                seen_ids.add(row_id)
                conversations = row.get("conversations", [])
                if len(conversations) != 2 or [entry.get("from") for entry in conversations] != ["human", "gpt"]:
                    errors.append({"split": split, "line": line_number, "kind": "conversation_shape"})
                    continue
                human = conversations[0].get("value", "")
                target = conversations[1].get("value", "")
                try:
                    parsed, _ = validate_target_text(target, context=f"{split}:{line_number}")
                except Exception as exc:
                    errors.append({"split": split, "line": line_number, "kind": "target_contract", "detail": str(exc)})
                    continue
                if provenance is None:
                    errors.append({"split": split, "line": line_number, "kind": "missing_source_trace", "id": row_id})
                    continue
                checks = {
                    "split": provenance.get("split") == split,
                    "user_text_sha256": provenance.get("user_text_sha256") == digest(human),
                    "action": provenance.get("generated_action") == parsed.get("action"),
                    "stage": provenance.get("generated_stage") == parsed.get("stage"),
                    "source_response_reused": provenance.get("source_response_reused") is False,
                }
                if not all(checks.values()):
                    errors.append({"split": split, "line": line_number, "kind": "trace_binding", "checks": checks})
                category = str(provenance["category"])
                state = str(provenance["information_state"])
                categories[f"{split}:{category}"] += 1
                if parsed["action"] == "ask":
                    states[f"{split}:{state}"] += 1
                key = (split, category)
                if key not in sampled:
                    sampled.add(key)
                    samples.append({"split": split, "category": category, "information_state": state, "human": human})
        split_counts[split] = count
        if count != expected_count:
            errors.append({"split": split, "kind": "row_count", "expected": expected_count, "actual": count})

    if trace_nonblind != 1840 or len(trace) != 1840:
        errors.append({"kind": "source_trace_nonblind_count", "lines": trace_nonblind, "unique_ids": len(trace)})

    report = {
        "status": "PASS" if not errors else "FAIL",
        "scope": "train and protocol_dev only; heldout text was not read",
        "split_counts": split_counts,
        "unique_ids": len(seen_ids),
        "source_trace_nonblind_rows": trace_nonblind,
        "category_counts": dict(sorted(categories.items())),
        "ask_information_state_counts": dict(sorted(states.items())),
        "representative_nonblind_samples": samples,
        "checks": {
            "sharegpt_shape": "PASS" if not any(e.get("kind") == "conversation_shape" for e in errors) else "FAIL",
            "strict_contract": "PASS" if not any(e.get("kind") == "target_contract" for e in errors) else "FAIL",
            "source_trace_binding": "PASS" if not any(e.get("kind") in {"missing_source_trace", "trace_binding"} for e in errors) else "FAIL",
            "heldout_text_read": False,
        },
        "errors": errors,
        "error_count": len(errors),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "split_counts", "unique_ids", "source_trace_nonblind_rows", "checks", "error_count")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
