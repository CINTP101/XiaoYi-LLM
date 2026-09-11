#!/usr/bin/env python3
"""Strict V5.3 output contract shared by preparation and evaluation tools.

The deployment parser in tcm_chat_v5.py only accepts ask/initial and
summarize/summary.  This module intentionally has the same closed label set.
It never normalizes malformed output into a passing output.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

NEUTRAL_SYNDROME_TENDENCY = "当前信息不足以形成可靠辨证倾向"

ASK_LABEL = ("ask", "initial")
SUMMARY_LABEL = ("summarize", "summary")
ALLOWED_LABELS = {ASK_LABEL, SUMMARY_LABEL}

ASK_FIELDS = frozenset({"action", "stage", "complete", "questions"})
SUMMARY_FIELDS = frozenset(
    {"action", "stage", "complete", "key_findings", "syndrome_tendency", "need_more_info", "note"}
)


class ContractError(ValueError):
    """A source row, reference, or model output violates the closed V5.3 contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(value: str, *, context: str = "JSON") -> Any:
    if not isinstance(value, str):
        raise ContractError(f"{context} must be a string")
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_keys)
    except ContractError:
        raise
    except json.JSONDecodeError as exc:
        raise ContractError(f"{context} is not one complete JSON value: {exc.msg}") from exc


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_string_list(value: Any, field: str, *, minimum: int = 0, maximum: int | None = None) -> list[str]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a JSON array")
    if len(value) < minimum:
        raise ContractError(f"{field} must contain at least {minimum} item(s)")
    if maximum is not None and len(value) > maximum:
        raise ContractError(f"{field} must contain at most {maximum} item(s)")
    normalized = [_require_nonempty_string(item, f"{field}[{index}]") for index, item in enumerate(value, 1)]
    if len(set(normalized)) != len(normalized):
        raise ContractError(f"{field} must not contain duplicate strings")
    return normalized


def label_of(target: dict[str, Any]) -> tuple[str, str] | None:
    action = target.get("action")
    stage = target.get("stage")
    return (action, stage) if isinstance(action, str) and isinstance(stage, str) else None


def validate_target_object(target: Any, *, context: str = "target") -> tuple[str, str]:
    if not isinstance(target, dict):
        raise ContractError(f"{context} must be a JSON object")
    action = target.get("action")
    stage = target.get("stage")
    if not isinstance(action, str) or not isinstance(stage, str):
        raise ContractError(f"{context}.action and {context}.stage must be strings")
    label = (action, stage)
    if label not in ALLOWED_LABELS:
        raise ContractError(f"{context} has disallowed action/stage: {action}/{stage}")
    if label == ASK_LABEL:
        unexpected = set(target) ^ ASK_FIELDS
        if unexpected:
            raise ContractError(f"{context} ask/initial fields must be exactly {sorted(ASK_FIELDS)}; difference={sorted(unexpected)}")
        if target["complete"] is not False:
            raise ContractError(f"{context}.complete must be JSON false for ask/initial")
        _require_string_list(target["questions"], f"{context}.questions", minimum=1, maximum=3)
    else:
        unexpected = set(target) ^ SUMMARY_FIELDS
        if unexpected:
            raise ContractError(f"{context} summarize/summary fields must be exactly {sorted(SUMMARY_FIELDS)}; difference={sorted(unexpected)}")
        if target["complete"] is not True:
            raise ContractError(f"{context}.complete must be JSON true for summarize/summary")
        _require_string_list(target["key_findings"], f"{context}.key_findings")
        _require_nonempty_string(target["syndrome_tendency"], f"{context}.syndrome_tendency")
        if target["syndrome_tendency"].strip() != NEUTRAL_SYNDROME_TENDENCY:
            raise ContractError(
                f"{context}.syndrome_tendency must equal the approved neutral phrase "
                f"{NEUTRAL_SYNDROME_TENDENCY!r}"
            )
        _require_string_list(target["need_more_info"], f"{context}.need_more_info")
        _require_nonempty_string(target["note"], f"{context}.note")
    return label


def validate_target_text(value: str, *, context: str = "target") -> tuple[dict[str, Any], tuple[str, str]]:
    target = strict_json_loads(value, context=context)
    if not isinstance(target, dict):
        raise ContractError(f"{context} must be one JSON object")
    return target, validate_target_object(target, context=context)


def validate_conversations(conversations: Any, *, context: str = "conversations") -> Counter[tuple[str, str]]:
    # Existing SFT preprocessing inserts an EOS token after every historical
    # assistant turn, while the legacy blind-eval builder does not reconstruct
    # that EOS token. Keeping V5.3 one-turn avoids a hidden train/eval prompt
    # mismatch until the shared upstream preprocessing implementation changes.
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ContractError(f"{context} must contain exactly one human/gpt pair")
    labels: Counter[tuple[str, str]] = Counter()
    expected_role = "human"
    for index, message in enumerate(conversations, 1):
        if not isinstance(message, dict):
            raise ContractError(f"{context}[{index}] must be an object")
        role = message.get("from")
        if role == "system":
            raise ContractError(f"{context}[{index}] must not contain a system message; V5.3 uses one fixed wrapper")
        if role != expected_role:
            raise ContractError(f"{context}[{index}] must have role {expected_role!r}, got {role!r}")
        value = _require_nonempty_string(message.get("value"), f"{context}[{index}].value")
        if role == "gpt":
            _, label = validate_target_text(value, context=f"{context}[{index}].value")
            labels[label] += 1
        expected_role = "gpt" if expected_role == "human" else "human"
    if expected_role != "human":
        raise ContractError(f"{context} must end with a gpt response")
    return labels


def validate_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.suffix.lower() != ".jsonl":
        raise ContractError(f"expected one regular .jsonl file: {path}")
    row_count = 0
    target_count = 0
    maximum_turns = 0
    label_counts: Counter[tuple[str, str]] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise ContractError(f"line {line_number}: blank rows are not allowed")
            row = strict_json_loads(raw_line, context=f"line {line_number}")
            if not isinstance(row, dict):
                raise ContractError(f"line {line_number}: row must be an object")
            if "conversations" not in row:
                raise ContractError(f"line {line_number}: missing conversations")
            try:
                labels = validate_conversations(row["conversations"], context=f"line {line_number}.conversations")
            except ContractError as exc:
                raise ContractError(f"line {line_number}: {exc}") from exc
            row_count += 1
            maximum_turns = max(maximum_turns, len(row["conversations"]) // 2)
            target_count += sum(labels.values())
            label_counts.update(labels)
    if row_count == 0:
        raise ContractError("JSONL has zero rows")
    return {
        "rows": row_count,
        "gpt_targets": target_count,
        "maximum_turns": maximum_turns,
        "label_counts": {f"{action}/{stage}": count for (action, stage), count in sorted(label_counts.items())},
        "allowed_labels": [f"{action}/{stage}" for action, stage in sorted(ALLOWED_LABELS)],
        "prompt_wrapper": "current tcm_chat_v5.py CONSULTATION_PROMPT + tokenizer.apply_chat_template",
        "neutral_syndrome_tendency": NEUTRAL_SYNDROME_TENDENCY,
    }


def final_prompt_history(conversations: list[dict[str, str]]) -> list[list[str]]:
    """Return [human, gpt] pairs with the final assistant answer blanked for generation."""
    validate_conversations(conversations)
    pairs = [[conversations[index]["value"], conversations[index + 1]["value"]] for index in range(0, len(conversations), 2)]
    pairs[-1][1] = ""
    return pairs
