#!/usr/bin/env python3
"""Read and reproduce the current tcm_chat_v5.py consultation wrapper.

The module is read-only: it never imports tcm_chat_v5.py, so it does not
load model weights or initialize the CLI. It extracts only its literal
CONSULTATION_PROMPT through Python AST.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

from v5_3_contract import ContractError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_consultation_prompt(runtime_source: Path) -> str:
    if not runtime_source.is_file() or runtime_source.is_symlink():
        raise ContractError(f"runtime source must be a regular file: {runtime_source}")
    try:
        module = ast.parse(runtime_source.read_text(encoding="utf-8"), filename=str(runtime_source))
    except SyntaxError as exc:
        raise ContractError(f"cannot parse runtime source: {exc.msg}") from exc
    for statement in module.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets: list[Any] = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if any(isinstance(target, ast.Name) and target.id == "CONSULTATION_PROMPT" for target in targets):
                try:
                    value = ast.literal_eval(statement.value)
                except ValueError as exc:
                    raise ContractError("CONSULTATION_PROMPT must remain a literal string for reproducible V5.3 training") from exc
                if not isinstance(value, str) or not value.strip():
                    raise ContractError("CONSULTATION_PROMPT must be a non-empty string")
                return value
    raise ContractError("CONSULTATION_PROMPT assignment not found in runtime source")


def runtime_messages(consultation_prompt: str, human_text: str) -> list[dict[str, str]]:
    if not isinstance(human_text, str) or not human_text.strip():
        raise ContractError("human text must be a non-empty string")
    return [
        {"role": "system", "content": consultation_prompt},
        {"role": "user", "content": human_text},
    ]


def render_runtime_prompt(tokenizer: Any, consultation_prompt: str, human_text: str) -> str:
    # This call exactly matches tcm_chat_v5.generate(). Do not add style
    # constraints, enable_thinking, or another chat-template option here.
    prompt = tokenizer.apply_chat_template(
        runtime_messages(consultation_prompt, human_text),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ContractError("tokenizer.apply_chat_template returned an empty non-string prompt")
    return prompt


def runtime_contract_metadata(runtime_source: Path) -> dict[str, str]:
    prompt = extract_consultation_prompt(runtime_source)
    return {
        "runtime_source": str(runtime_source.resolve(strict=True)),
        "runtime_source_sha256": sha256_file(runtime_source),
        "consultation_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "wrapper": "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
    }
