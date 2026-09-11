#!/usr/bin/env python3
"""Generate V5.3 evaluation outputs with the current tcm_chat_v5 wrapper.

The prompt construction and decode call match tcm_chat_v5.generate().
This program performs evaluation only and does not write model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_prompt_v5_3 import (
    extract_consultation_prompt,
    render_runtime_prompt,
    runtime_contract_metadata,
    sha256_file,
)
from v5_3_contract import ContractError, strict_json_loads, validate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime-wrapper-aligned V5.3 generation evaluator")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--peft-path", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--expected-test-sha256", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--expected-runtime-source-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-input-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--require-v5-3-contract", action="store_true")
    return parser.parse_args()


def regular_file(value: str, field: str, suffix: str | None = None) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise ContractError(f"{field} must be one regular non-symlink file")
    path = supplied.resolve(strict=True)
    if suffix and path.suffix.lower() != suffix:
        raise ContractError(f"{field} must use {suffix}")
    return path


def load_generic_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            row = strict_json_loads(raw_line, context=f"line {index}")
            if not isinstance(row, dict):
                raise ContractError(f"line {index}: row must be an object")
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise ContractError(f"line {index}: evaluation rows must contain exactly one human/gpt pair")
            human, gpt = conversations
            if not isinstance(human, dict) or not isinstance(gpt, dict):
                raise ContractError(f"line {index}: conversation messages must be objects")
            if human.get("from") != "human" or gpt.get("from") != "gpt":
                raise ContractError(f"line {index}: roles must be human then gpt")
            if not isinstance(human.get("value"), str) or not human["value"].strip():
                raise ContractError(f"line {index}: human value must be non-empty text")
            if not isinstance(gpt.get("value"), str) or not gpt["value"].strip():
                raise ContractError(f"line {index}: gpt value must be non-empty text")
            rows.append({"human": human["value"], "reference": gpt["value"]})
    if not rows:
        raise ContractError("evaluation JSONL has zero rows")
    return rows


def input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def main() -> None:
    args = parse_args()
    test_path = regular_file(args.test_jsonl, "--test-jsonl", ".jsonl")
    runtime_source = regular_file(args.runtime_source, "--runtime-source", ".py")
    base_model_path = Path(args.base_model_path).expanduser().resolve(strict=True)
    peft_path = Path(args.peft_path).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation output directory: {output_dir}")
    actual_test_hash = sha256_file(test_path)
    if actual_test_hash != args.expected_test_sha256.strip().lower():
        raise ContractError("test JSONL SHA-256 does not match the frozen value")
    actual_runtime_hash = sha256_file(runtime_source)
    if actual_runtime_hash != args.expected_runtime_source_sha256.strip().lower():
        raise ContractError("runtime source SHA-256 does not match the frozen value")
    if args.require_v5_3_contract:
        contract_validation: dict[str, Any] | None = validate_jsonl(test_path)
    else:
        contract_validation = None
    rows = load_generic_rows(test_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ContractError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    consultation_prompt = extract_consultation_prompt(runtime_source)
    prompts = [render_runtime_prompt(tokenizer, consultation_prompt, row["human"]) for row in rows]
    prompt_lengths = [len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) for prompt in prompts]
    if max(prompt_lengths) > args.max_input_length:
        raise ContractError(
            f"runtime prompt requires {max(prompt_lengths)} tokens, exceeding --max-input-length={args.max_input_length}; refusing truncation"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    preflight = {
        "test_jsonl": str(test_path),
        "test_sha256": actual_test_hash,
        "base_model_path": str(base_model_path),
        "peft_path": str(peft_path),
        "runtime_contract": runtime_contract_metadata(runtime_source),
        "contract_validation": contract_validation,
        "sample_count": len(rows),
        "maximum_prompt_tokens": max(prompt_lengths),
        "arguments": vars(args),
    }
    with (output_dir / "generation_preflight.json").open("x", encoding="utf-8") as handle:
        json.dump(preflight, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, str(peft_path))
    model.eval()
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    device = input_device(model)
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        prompt_batch = prompts[start:start + args.batch_size]
        row_batch = rows[start:start + args.batch_size]
        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            truncation=False,
            padding=True,
            add_special_tokens=True,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**encoded, generation_config=config)
        for offset, row in enumerate(row_batch):
            token_ids = generated[offset][prompt_width:]
            raw_prediction = tokenizer.decode(token_ids, skip_special_tokens=False).strip()
            prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            outputs.append({
                "index": start + offset + 1,
                "reference": row["reference"],
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "prompt_sha256": sha256_file_from_text(prompts[start + offset]),
            })
    with (output_dir / "all_results.json").open("x", encoding="utf-8") as handle:
        json.dump(outputs, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"generated": len(outputs), "output_dir": str(output_dir)}, ensure_ascii=False))


def sha256_file_from_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
