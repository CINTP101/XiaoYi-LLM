#!/usr/bin/env python3
"""Apply the frozen Candidate G adapter to one authorized generation result set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


AUTHORIZATION_TOKEN = "CANDIDATE_G_FINAL_BLIND_AUTHORIZED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(value: str, field: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{field} must be one regular non-symlink file")
    return path.resolve(strict=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--raw-generation-results", required=True)
    parser.add_argument("--adapter-source", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # This check occurs before any supplied path is inspected or opened.
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise SystemExit("refusing Candidate G adaptation without final-blind authorization token")

    input_path = regular_file(args.input_jsonl, "--input-jsonl")
    generation_path = regular_file(args.raw_generation_results, "--raw-generation-results")
    adapter_path = regular_file(args.adapter_source, "--adapter-source")
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    input_sha256 = sha256_file(input_path)
    if input_sha256 != args.expected_input_sha256.strip().lower():
        raise RuntimeError("input JSONL SHA-256 mismatch")
    adapter_sha256 = sha256_file(adapter_path)
    if adapter_sha256 != args.expected_adapter_sha256.strip().lower():
        raise RuntimeError("Candidate G adapter SHA-256 mismatch")

    spec = importlib.util.spec_from_file_location("candidate_g_frozen_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load Candidate G adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows: list[dict[str, str]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            value = json.loads(line)
            conversations = value.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise RuntimeError(f"input row {index} must contain one human/gpt pair")
            human, reference = conversations
            if human.get("from") != "human" or reference.get("from") != "gpt":
                raise RuntimeError(f"input row {index} role order mismatch")
            rows.append({"human": human["value"], "reference": reference["value"]})

    generations: Any = json.loads(generation_path.read_text(encoding="utf-8"))
    if not isinstance(generations, list) or len(generations) != len(rows):
        raise RuntimeError("input/generation count mismatch")

    outputs = []
    evidence = []
    for index, (row, generation) in enumerate(zip(rows, generations), 1):
        if not isinstance(generation, dict) or generation.get("index") != index:
            raise RuntimeError(f"generation index mismatch at {index}")
        raw_prediction = generation.get("raw_prediction", generation.get("prediction", ""))
        adapted = module.adapt_prediction(row["human"], raw_prediction)
        replay = module.adapt_prediction(row["human"], raw_prediction)
        if adapted != replay:
            raise RuntimeError(f"non-deterministic adapter result at {index}")
        parsed = json.loads(adapted)
        outputs.append({
            "index": index,
            "reference": row["reference"],
            "prediction": adapted,
            "raw_prediction": adapted,
            "query_sha256": hashlib.sha256(row["human"].encode("utf-8")).hexdigest(),
            "source_prediction_sha256": hashlib.sha256(str(raw_prediction).encode("utf-8")).hexdigest(),
            "adapter_sha256": adapter_sha256,
        })
        evidence.append({
            "index": index,
            "query_sha256": hashlib.sha256(row["human"].encode("utf-8")).hexdigest(),
            "source_prediction_sha256": hashlib.sha256(str(raw_prediction).encode("utf-8")).hexdigest(),
            "adapted_prediction_sha256": hashlib.sha256(adapted.encode("utf-8")).hexdigest(),
            "action": parsed.get("action"),
            "stage": parsed.get("stage"),
            "reference_passed_to_adapter": False,
        })

    output_dir.mkdir(parents=True)
    results_path = output_dir / "all_results.json"
    evidence_path = output_dir / "adapter_application_cases.jsonl"
    results_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in evidence),
        encoding="utf-8",
    )
    preflight = {
        "status": "PASS",
        "input_jsonl": str(input_path),
        "input_sha256": input_sha256,
        "raw_generation_results": str(generation_path),
        "raw_generation_results_sha256": sha256_file(generation_path),
        "adapter_source": str(adapter_path),
        "adapter_sha256": adapter_sha256,
        "samples": len(outputs),
        "deterministic_replay": len(outputs),
        "reference_passed_to_adapter": False,
        "results_sha256": sha256_file(results_path),
        "application_cases_sha256": sha256_file(evidence_path),
    }
    (output_dir / "adapter_application_preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
