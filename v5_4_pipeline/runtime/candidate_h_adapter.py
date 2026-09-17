"""Candidate H deterministic action router and safety interaction adapter.

This module intentionally does not receive model references or use model action
fields. It converts only the user's text into the V5.4 runtime contract.
"""

from __future__ import annotations

import json
import re
from typing import Any


ASK_REFUSAL = "我不能提供诊断、辨证、处方、剂量、治疗方案或绕过安全规则的建议。"
ASK_QUESTIONS = (
    "这些情况已经持续了多久，出现频率如何？",
    "除已提到的内容外，你还观察到哪些变化？",
)
NEUTRAL_SYNDROME_TENDENCY = "当前信息不足以形成可靠辨证倾向"
SUMMARY_NOTE = "仅整理用户陈述，不作诊断或辨证判断。"

_DANGEROUS_BYPASS = re.compile(
    r"bypass|绕过|繞過|忽略安全|无视安全|無視安全|ignore\s+(?:the\s+)?safety",
    re.IGNORECASE,
)
_DANGEROUS_MEDICAL_TERM = (
    r"诊断|診斷|辨证|辨證|开方|開方|处方|處方|剂量|劑量|用量|"
    r"治疗方案|治療方案|治疗|治療|疗效|療效|药方|藥方|"
    r"prescrib(?:e|ing|ed)|prescription|diagnos(?:e|is|ing)|dosage|"
    r"treat(?:ment|ing)?|cure"
)
_DANGEROUS_MEDICAL_REQUEST = re.compile(
    rf"(?:请|請|直接|给我|給我|告诉我|告訴我|能不能|可以|可否|如何|怎么|怎麼|"
    rf"安排|规划|規劃|推荐|推薦|需要|想要|希望|不要|按步骤|按步驟|个人|個人)"
    rf".{{0,28}}(?:{_DANGEROUS_MEDICAL_TERM})|"
    rf"(?:{_DANGEROUS_MEDICAL_TERM}).{{0,20}}(?:怎么|怎麼|如何|多少|剂量|劑量|用法|服法|"
    rf"步骤|步驟|计划|計劃|方案|每天|一下|是什么证|是什麼證|吗|嗎)|"
    rf"(?:诊断|診斷).{{0,8}}(?:并|並|和).{{0,8}}(?:治疗|治療)|"
    rf"(?:疗效承诺|療效承諾|保证疗效|保證療效)",
    re.IGNORECASE,
)
_DANGEROUS_MEDICATION_REQUEST = re.compile(
    r"用什么药|用什麼藥|吃什么药|吃什麼藥|给什么药|給什麼藥|"
    r"(?:推荐|推薦|列|配|给出|給出|给我|給我|需要|想要|提供).{0,12}药材|"
    r"药材.{0,12}(?:怎么配|怎麼配|如何配|剂量|劑量|用法|服法)|"
    r"(?:开|開|给|給|推荐|推薦|提供|要|想要|需要).{0,8}(?:方子|藥方|药方)|"
    r"(?:方子|藥方|药方).{0,12}(?:怎么|怎麼|如何|剂量|劑量|用法)|"
    r"中药处方|中藥處方|个人处方|個人處方|处方组成|處方組成",
    re.IGNORECASE,
)
_SUMMARY_COMMAND = re.compile(
    r"总结|總結|概括|梳理|汇总|彙總|生成小结|生成小結|小结|小結|摘要|"
    r"summari[sz](?:e|ed|ing|ation)|recap",
    re.IGNORECASE,
)
_EXPLICIT_SUMMARY_COMMAND = re.compile(
    r"(?:请|請)\s*(?:总结|總結|概括|梳理|汇总|彙總|生成小结|生成小結|做摘要|做摘要|摘要)",
    re.IGNORECASE,
)
_SENTENCE_BREAK = re.compile(r"[。！？；.!?;\n]+")
_COMMAND_ONLY = re.compile(
    r"^(?:请|請|帮我|幫我|麻烦|麻煩|一下|一下子|请你|請你|"
    r"can you|please|kindly|for me|帮忙|幫忙|把|将|將|我的|我)?"
    r"(?:总结|總結|概括|梳理|汇总|彙總|生成小结|生成小結|小结|小結|摘要|"
    r"summari[sz](?:e|ed|ing|ation)|recap)"
    r"(?:一下|一下子|给我|給我|我的症状|我的症狀|上述|以上|这些|這些)?$",
    re.IGNORECASE,
)
_EXPLICIT_SUMMARY_PREFIX = re.compile(
    r"(?:请|請|帮我|幫我|请你|請你|麻烦|麻煩|能否|可以|可否|"
    r"生成|生成一份|做一份|给我|給我|please|can you|could you)",
    re.IGNORECASE,
)
_EXPLICIT_SUMMARY_SUFFIX = re.compile(r"(?:一下|一下子|上述|以上|这些|這些|这段|這段|给我|給我|我的|我这|我這|后|後)")


def _require_text(user_text: str) -> str:
    if not isinstance(user_text, str):
        raise TypeError("user_text must be a string")
    if not user_text.strip():
        raise ValueError("user_text must not be empty")
    return user_text


def _route(user_text: str) -> tuple[str, str]:
    """Return the only allowed action/stage pair for user text."""
    if _is_dangerous_request(user_text):
        return "ask", "initial"
    if _has_explicit_summary_command(user_text):
        return "summarize", "summary"
    return "ask", "initial"


def _is_dangerous_request(user_text: str) -> bool:
    """Detect an actionable medical or bypass request, not past treatment facts."""
    return bool(
        _DANGEROUS_BYPASS.search(user_text)
        or _DANGEROUS_MEDICAL_REQUEST.search(user_text)
        or _DANGEROUS_MEDICATION_REQUEST.search(user_text)
    )


def _has_explicit_summary_command(user_text: str) -> bool:
    """Recognize a command, rather than merely a mention of a summary."""
    if _EXPLICIT_SUMMARY_COMMAND.search(user_text):
        return True
    for match in _SUMMARY_COMMAND.finditer(user_text):
        before = user_text[max(0, match.start() - 12) : match.start()]
        after = user_text[match.end() : match.end() + 12].lstrip()
        command_ending = not after or after[0] in "。！？；.!?;"
        if _EXPLICIT_SUMMARY_PREFIX.search(before) and (command_ending or _EXPLICIT_SUMMARY_SUFFIX.search(after)):
            return True
        if _EXPLICIT_SUMMARY_SUFFIX.search(after):
            return True
        if user_text.strip() == match.group(0):
            return True
    return False


def _fact_part(candidate: str) -> str:
    """Remove a trailing explicit summary command while preserving source text."""
    command = _EXPLICIT_SUMMARY_COMMAND.search(candidate)
    if command is not None:
        return candidate[: command.start()].rstrip("，,：: \t")
    return candidate


def _literal_key_findings(user_text: str) -> list[str]:
    """Select stable, non-empty, source-literal fragments for a summary."""
    findings: list[str] = []
    seen: set[str] = set()
    start = 0
    for match in _SENTENCE_BREAK.finditer(user_text):
        candidate = _fact_part(user_text[start : match.start()].strip())
        start = match.end()
        if candidate and not _COMMAND_ONLY.fullmatch(candidate) and candidate not in seen:
            findings.append(candidate)
            seen.add(candidate)
    candidate = _fact_part(user_text[start:].strip())
    if candidate and not _COMMAND_ONLY.fullmatch(candidate) and candidate not in seen:
        findings.append(candidate)
    if not findings:
        findings.append(user_text.strip())
    return findings


def adapt_interaction(user_text: str, raw_output: Any = None) -> dict[str, Any]:
    """Return a deterministic safe interaction for one user message.

    ``raw_output`` is accepted only for call-site compatibility and is
    deliberately ignored: malformed output and unsafe model actions cannot
    override deterministic routing.
    """
    text = _require_text(user_text)
    action, stage = _route(text)
    if action == "ask":
        return {
            "action": "ask",
            "stage": "initial",
            "complete": False,
            "questions": [ASK_REFUSAL, *ASK_QUESTIONS],
        }
    return {
        "action": "summarize",
        "stage": "summary",
        "complete": True,
        "key_findings": _literal_key_findings(text),
        "syndrome_tendency": NEUTRAL_SYNDROME_TENDENCY,
        "need_more_info": [],
        "note": SUMMARY_NOTE,
    }


def adapt_interaction_json(user_text: str, raw_output: Any = None) -> str:
    """Return the Candidate H result as one compact deterministic JSON object.

    The object builder continues to ignore ``raw_output``. Compact serialization
    preserves the established schema without accepting any model-controlled
    fields or references.
    """
    return json.dumps(adapt_interaction(user_text, raw_output), ensure_ascii=False, separators=(",", ":"))
