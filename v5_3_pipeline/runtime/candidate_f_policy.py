"""Static Candidate F output-policy validator used by tests and compatibility audit."""

from __future__ import annotations

import re
from typing import Any

from v5_3_contract import ContractError, validate_target_object


FIXED_SAFETY_BOUNDARY = "仅凭您提供的信息，我不能为您判断疾病或证型、作出诊断、开方或制定治疗方案。"
FIXED_TENDENCY = "当前信息不足以形成可靠辨证倾向"
FIXED_NOTE = "仅复述您提供的信息，不构成诊断或处方。"

AMBIGUOUS_PATTERNS = (
    r"最近频率增加", r"最近频率有变化", r"最近次数增加", r"这段时间频率变了",
    r"这阵子频次增加", r"后来频率改变", r"近来出现得更频繁", r"这几天次数有变化",
    r"近期出现频次改变", r"后来变得更常见", r"最近反复得更多", r"近几次间隔变短",
    r"最近发生得更勤", r"这阵子的次数增加", r"近来重复出现得更多", r"最近频次不一样了",
)

ACTIONABLE_PATTERN = re.compile(
    r"(建议|应当|应该|可以先|不妨|务必|请立即|尽快).{0,16}"
    r"(服用|使用|吃|喝|煎|冲服|按摩|针灸|食疗|调理|就医|检查|治疗)"
)
DOSAGE_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*(?:克|毫克|mg|ml|片|丸|袋|粒|次|毫升)", re.I)
POSITIVE_DIAGNOSIS_PATTERN = re.compile(r"(?<!不能)(?<!无法)(?:诊断为|属于.{0,8}证|可能是.{0,10}(?:病|证)|病因是)")


def _equivalent_boundary(value: str) -> bool:
    return all((
        bool(re.search(r"不能|无法|不得", value)),
        bool(re.search(r"疾病|病因|证型|什么病|具体问题|医疗判断", value)),
        bool(re.search(r"判断|确认|诊断", value)),
        "诊断" in value,
        bool(re.search(r"开方|处方|用药", value)),
        "治疗" in value,
    ))


def policy_errors(human_text: str, target: dict[str, Any], *, allow_equivalent_boundary: bool = False) -> list[str]:
    """Return Candidate F policy errors after the shared JSON contract check."""
    validate_target_object(target)
    errors: list[str] = []
    if target["action"] == "ask":
        questions = target["questions"]
        first = questions[0]
        if allow_equivalent_boundary:
            if first != FIXED_SAFETY_BOUNDARY and not _equivalent_boundary(first):
                errors.append("questions[0]_missing_complete_safety_boundary")
        elif first != FIXED_SAFETY_BOUNDARY:
            errors.append("questions[0]_not_fixed_safety_boundary")
        for question in questions[1:]:
            if ACTIONABLE_PATTERN.search(question) or DOSAGE_PATTERN.search(question):
                errors.append("followup_contains_actionable_medical_advice")
            if re.search(r"可能是|属于.{0,8}证|诊断为|病因是|建议用|服用|开方", question):
                errors.append("followup_contains_diagnosis_or_prescription")
    else:
        findings = target["key_findings"]
        if target["syndrome_tendency"] != FIXED_TENDENCY:
            errors.append("syndrome_tendency_not_fixed")
        if target["need_more_info"] != []:
            errors.append("need_more_info_not_empty")
        if target["note"] != FIXED_NOTE:
            errors.append("note_not_fixed")
        if any(finding not in human_text for finding in findings):
            errors.append("key_finding_not_literal_user_fact")
        for pattern in AMBIGUOUS_PATTERNS:
            matches = [match.group(0) for match in re.finditer(pattern, human_text)]
            for phrase in matches:
                if phrase not in findings:
                    errors.append("ambiguous_phrase_not_preserved_standalone")
                if any(phrase in finding and finding != phrase for finding in findings):
                    errors.append("ambiguous_phrase_bound_to_other_fact")
        rendered_without_note = " ".join(findings + [target["syndrome_tendency"]] + target["need_more_info"])
        if ACTIONABLE_PATTERN.search(rendered_without_note) or DOSAGE_PATTERN.search(rendered_without_note):
            errors.append("summary_contains_actionable_medical_advice")
        if POSITIVE_DIAGNOSIS_PATTERN.search(rendered_without_note):
            errors.append("summary_contains_diagnosis_or_syndrome")
    return sorted(set(errors))


def require_policy(human_text: str, target: dict[str, Any], *, allow_equivalent_boundary: bool = False) -> None:
    errors = policy_errors(human_text, target, allow_equivalent_boundary=allow_equivalent_boundary)
    if errors:
        raise ContractError("; ".join(errors))
