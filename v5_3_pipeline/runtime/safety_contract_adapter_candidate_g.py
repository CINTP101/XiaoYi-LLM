"""Deterministic Candidate G safety adapter.

The module is pure: it performs no file, network, model, GPU, logging, clock,
randomness, or environment access.  Public functions accept values and return
values only.
"""

from __future__ import annotations

import json
import re
from typing import Any


FIXED_SAFETY_BOUNDARY = "仅凭您提供的信息，我不能为您判断疾病或证型、作出诊断、开方或制定治疗方案。"
FIXED_TENDENCY = "当前信息不足以形成可靠辨证倾向"
FIXED_NOTE = "仅复述您提供的信息，不构成诊断或处方。"

_ACTION_PATTERN = re.compile(
    r"^\s*(?:```(?:json)?\s*)?\{\s*[\"“]action[\"”]\s*:\s*[\"“](ask|summarize)(?:[\"”]|(?=\s*[,}]))",
    re.IGNORECASE,
)
_FINDINGS_ARRAY_PATTERN = re.compile(r"[\"“]key_findings[\"”]\s*:\s*(\[[\s\S]*?\])", re.IGNORECASE)
_JSON_STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"')
_TERMINAL_TOKEN_PATTERN = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")

_DOSAGE_PATTERN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:克|毫克|微克|千克|mg|g|ml|mL|片|丸|袋|粒|支|滴|毫升|次|付|剂)",
    re.IGNORECASE,
)
_PRESCRIPTION_OR_TREATMENT_PATTERN = re.compile(
    r"(?:怎么治|如何治|如何治疗|怎样治疗|治疗方案|治疗计划|治法|调理方案|"
    r"吃什么药|用什么药|开什么药|开个方|开方|处方|方剂|药方|用药|服药|服用|"
    r"药物|药材|中药|西药|中成药|汤剂|煎服|冲服|针灸|艾灸|按摩|食疗)"
)
_COMMON_MEDICINE_OR_FORMULA_PATTERN = re.compile(
    r"(?:人参|黄芪|当归|白术|茯苓|甘草|桂枝|麻黄|柴胡|黄连|黄芩|半夏|附子|"
    r"地黄|枸杞|阿胶|金银花|板蓝根|连花清瘟|六味地黄丸|逍遥丸|补中益气丸|"
    r"小柴胡汤|桂枝汤|麻黄汤|四物汤|四君子汤)"
)


def _ask_object() -> dict[str, Any]:
    return {
        "action": "ask",
        "stage": "initial",
        "complete": False,
        "questions": [FIXED_SAFETY_BOUNDARY],
    }


def _summary_object(findings: list[str]) -> dict[str, Any]:
    return {
        "action": "summarize",
        "stage": "summary",
        "complete": True,
        "key_findings": findings,
        "syndrome_tendency": FIXED_TENDENCY,
        "need_more_info": [],
        "note": FIXED_NOTE,
    }


def _clean_raw(raw_prediction: Any) -> str:
    if not isinstance(raw_prediction, str):
        return ""
    value = raw_prediction.strip()
    while True:
        cleaned = _TERMINAL_TOKEN_PATTERN.sub("", value).strip()
        if cleaned == value:
            return value
        value = cleaned


def detect_closed_action(raw_prediction: Any) -> str | None:
    """Return ask/summarize only when the raw output identifies one reliably."""
    raw = _clean_raw(raw_prediction)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and parsed.get("action") in {"ask", "summarize"}:
        return str(parsed["action"])
    match = _ACTION_PATTERN.search(raw)
    return match.group(1).lower() if match else None


def _parsed_object(raw_prediction: Any) -> dict[str, Any] | None:
    raw = _clean_raw(raw_prediction)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_raw_findings(raw_prediction: Any) -> list[str]:
    parsed = _parsed_object(raw_prediction)
    if parsed is not None and isinstance(parsed.get("key_findings"), list):
        return [value for value in parsed["key_findings"] if isinstance(value, str)]
    raw = _clean_raw(raw_prediction)
    match = _FINDINGS_ARRAY_PATTERN.search(raw)
    if not match:
        return []
    array_text = match.group(1)
    try:
        values = json.loads(array_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        values = []
        for token in _JSON_STRING_PATTERN.findall(array_text):
            try:
                value = json.loads(token)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(value, str):
                values.append(value)
    return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []


def contains_prescription_or_treatment_risk(user_text: Any) -> bool:
    """Conservatively detect source text that must not be repeated as a summary."""
    if not isinstance(user_text, str):
        return False
    return bool(
        _DOSAGE_PATTERN.search(user_text)
        or _PRESCRIPTION_OR_TREATMENT_PATTERN.search(user_text)
        or _COMMON_MEDICINE_OR_FORMULA_PATTERN.search(user_text)
    )


def _literal_stable_unique_findings(user_text: str, raw_prediction: Any) -> list[str]:
    findings: list[str] = []
    seen: set[str] = set()
    for raw_finding in _extract_raw_findings(raw_prediction):
        finding = raw_finding.strip()
        if not finding or finding in seen:
            continue
        if finding not in user_text:
            continue
        seen.add(finding)
        findings.append(finding)
    return findings


def adapt_prediction_object(user_text: Any, raw_prediction: Any) -> dict[str, Any]:
    """Return a new strict-schema object without retaining unsafe model prose."""
    if not isinstance(user_text, str) or not user_text.strip():
        return _ask_object()
    if detect_closed_action(raw_prediction) != "summarize":
        return _ask_object()
    if contains_prescription_or_treatment_risk(user_text):
        return _ask_object()
    findings = _literal_stable_unique_findings(user_text, raw_prediction)
    if not findings:
        findings = [user_text]
    return _summary_object(findings)


def adapt_prediction(user_text: Any, raw_prediction: Any) -> str:
    """Return one deterministic, compact, valid JSON object as a string."""
    return json.dumps(
        adapt_prediction_object(user_text, raw_prediction),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
