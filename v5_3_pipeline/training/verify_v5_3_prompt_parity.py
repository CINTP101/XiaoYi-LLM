#!/usr/bin/env python3
"""Prove V5.3 training prompts equal the current tcm_chat_v5 runtime wrapper.

This loads only the tokenizer. It checks prompt text and token IDs for every
one-turn sample, while recording the exact runtime-source and system-prompt
hashes used for the proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_prompt_v5_3 import (
    extract_consultation_prompt,
    render_runtime_prompt,
    runtime_contract_metadata,
    runtime_messages,
    sha256_file,
)
from v5_3_contract import ContractError, strict_json_loads, validate_jsonl


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify exact V5.3 train/runtime prompt parity")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--expected-runtime-source-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.jsonl).expanduser().resolve(strict=True)
    runtime_source = Path(args.runtime_source).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite prompt-parity manifest: {output_path}")
    expected_data_hash = args.expected_sha256.strip().lower()
    actual_data_hash = sha256_file(data_path)
    if actual_data_hash != expected_data_hash:
        raise ContractError(f"input SHA-256 mismatch: expected {expected_data_hash}, got {actual_data_hash}")
    expected_runtime_hash = args.expected_runtime_source_sha256.strip().lower()
    actual_runtime_hash = sha256_file(runtime_source)
    if actual_runtime_hash != expected_runtime_hash:
        raise ContractError(
            f"runtime source SHA-256 mismatch: expected {expected_runtime_hash}, got {actual_runtime_hash}"
        )
    validation = validate_jsonl(data_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True, local_files_only=True, padding_side="left"
    )
    consultation_prompt = extract_consultation_prompt(runtime_source)
    cases = []
    with data_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            row = strict_json_loads(raw_line, context=f"line {index}")
            human = row["conversations"][0]["value"]
            # Training path imports the shared V5.3 runtime renderer.
            training_prompt = render_runtime_prompt(tokenizer, consultation_prompt, human)
            # Runtime path is intentionally written as tcm_chat_v5.generate() does.
            runtime_prompt = tokenizer.apply_chat_template(
                runtime_messages(consultation_prompt, human),
                tokenize=False,
                add_generation_prompt=True,
            )
            train_ids = tokenizer(training_prompt, add_special_tokens=True)["input_ids"]
            runtime_ids = tokenizer(runtime_prompt, add_special_tokens=True)["input_ids"]
            cases.append({
                "index": index,
                "training_prompt_sha256": sha256_text(training_prompt),
                "runtime_prompt_sha256": sha256_text(runtime_prompt),
                "prompt_exact_match": training_prompt == runtime_prompt,
                "training_token_ids_sha256": sha256_text(json.dumps(train_ids, separators=(",", ":"))),
                "runtime_token_ids_sha256": sha256_text(json.dumps(runtime_ids, separators=(",", ":"))),
                "token_ids_exact_match": train_ids == runtime_ids,
            })
    exact_prompt_count = sum(case["prompt_exact_match"] for case in cases)
    exact_token_count = sum(case["token_ids_exact_match"] for case in cases)
    metadata = runtime_contract_metadata(runtime_source)
    manifest = {
        "input_jsonl": str(data_path),
        "input_sha256": actual_data_hash,
        "contract_validation": validation,
        "base_model_tokenizer": str(Path(args.base_model_path).resolve(strict=True)),
        "runtime_contract": metadata,
        "total": len(cases),
        "exact_prompt_matches": exact_prompt_count,
        "exact_token_id_matches": exact_token_count,
        "all_exact": exact_prompt_count == len(cases) and exact_token_count == len(cases),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not manifest["all_exact"]:
        raise SystemExit("V5.3 training/runtime prompt parity failed")
    print(json.dumps({key: manifest[key] for key in ("total", "exact_prompt_matches", "exact_token_id_matches", "all_exact")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
