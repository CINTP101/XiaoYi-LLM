"""Candidate H v2 deterministic safety and follow-up adapter.

The adapter uses only the current user text.  Model output is accepted for
call-site compatibility but cannot influence routing or returned fields.
"""

from __future__ import annotations

import json
import re
from typing import Any


ASK_REFUSAL = "我不能提供诊断、辨证、处方、剂量、治疗方案或绕过安全规则的建议。"
OTHER_OBSERVATIONS_QUESTION = "除已提到的内容外，你还观察到哪些变化？"
NEUTRAL_SYNDROME_TENDENCY = "当前信息不足以形成可靠辨证倾向"
SUMMARY_NOTE = "仅整理用户陈述，不作诊断或辨证判断。"

_ASK_FOR_BOTH = "这些情况已经持续了多久，出现频率如何？"
_ASK_FOR_DURATION = "这些情况已经持续了多久？"
_ASK_FOR_FREQUENCY = "这些情况出现的频率如何？"
_ASK_FOR_TREND = "与刚开始相比，最近是加重、减轻还是基本不变？"

_CN_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百半几两]+)"
_DURATION_UNIT = r"(?:分钟|小时|钟头|天|日|周|星期|个月|月|年)"
_DURATION_VALUE = rf"(?:{_CN_NUMBER}\s*(?:个)?(?:多|半)?\s*{_DURATION_UNIT}(?:多)?|大半\s*(?:个)?{_DURATION_UNIT})"
_DURATION_PATTERNS = (
    re.compile(
        rf"(?:已经持续|已经有|已经|已持续|已有|持续了?|延续了?|大约|大概(?:持续|有)?|约|近|将近|差不多|"
        rf"共|有|到现在(?:约|有|已有)|前后(?:约|已有|持续)|算下来约|留意了)\s*{_DURATION_VALUE}"
    ),
    re.compile(rf"{_DURATION_VALUE}\s*(?:了|以来|左右)"),
    re.compile(rf"(?:从)?\s*{_DURATION_VALUE}\s*(?:前|以前)\s*(?:开始|起|我第一次留意到)?"),
    re.compile(r"(?:最近|近来|这)\s*(?:几|一两|两三|三四)?\s*(?:天|日|周|个月|月|年)(?:以来)?"),
    re.compile(r"(?:今天|昨日|昨天|前天|今早|今晚|昨晚|刚才|刚刚)(?:开始|起|出现|注意到)?"),
    re.compile(r"(?:从前些天|从前几天|从上周|从上个月|从去年|自从).{0,12}(?:到现在|开始|以来|起)"),
)

_FREQUENCY_PATTERNS = (
    re.compile(r"(?:每天|每日|每晚|每夜|每早|每周|每月|每年|天天|夜夜|逐日)"),
    re.compile(rf"(?:一天|每日|每晚|每周|每月|每年)\s*{_CN_NUMBER}\s*(?:次|回|遍)"),
    re.compile(rf"{_CN_NUMBER}\s*(?:次|回|遍)(?:每天|每日|每晚|每周|每月|每年)?"),
    re.compile(r"(?:偶尔|偶然|有时|时有|时常|经常|常常|频繁|反复|间歇|断断续续|总是|总会|总有|一直|持续不断|反复发作|频率)"),
    re.compile(r"(?:不太规律|没有明显规律|暂无规律|暂时没有明显规律|越来越频繁|多数日子|隔(?:一两|两三|几|\d+)天)"),
)

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
    r"用哪几味药|用哪幾味藥|该用什么药|該用什麼藥|"
    r"(?:推荐|推薦|列|配|给出|給出|给我|給我|需要|想要|提供).{0,12}药材|"
    r"药材.{0,12}(?:怎么配|怎麼配|如何配|剂量|劑量|用法|服法)|"
    r"(?:开|開|给|給|推荐|推薦|提供|要|想要|需要).{0,8}(?:方子|藥方|药方)|"
    r"(?:方子|藥方|药方).{0,12}(?:怎么|怎麼|如何|剂量|劑量|用法)|"
    r"中药处方|中藥處方|个人处方|個人處方|处方组成|處方組成",
    re.IGNORECASE,
)

_SUMMARY_COMMAND = re.compile(
    r"总结|總結|概括|梳理|汇总|彙總|整理|归纳|歸納|记录|記錄|摘录|摘錄|生成小结|生成小結|小结|小結|摘要|"
    r"summari[sz](?:e|ed|ing|ation)|recap",
    re.IGNORECASE,
)
_EXPLICIT_SUMMARY_COMMAND = re.compile(
    r"(?:请|請|帮我|幫我|麻烦|麻煩|能否|可以|可否).{0,24}(?:总结|總結|概括|梳理|汇总|彙總|整理|归纳|歸納|小结|小結|生成.{0,8}记录|生成.{0,8}記錄|做成.{0,8}摘要|做摘要|摘要)|"
    r"(?:我需要|我想).{0,24}(?:小结|小結|摘要)",
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
    r"生成|生成一份|做一份|给我|給我|我需要|我想|please|can you|could you)",
    re.IGNORECASE,
)
_EXPLICIT_SUMMARY_SUFFIX = re.compile(r"(?:一下|一下子|上述|以上|这些|這些|这段|這段|给我|給我|我的|我这|我這|后|後)")

_INFORMATION_REQUEST_CLAUSE = re.compile(
    r"(?:请|請|能不能|可以|可否|告诉我|告訴我|替我|帮我|幫我|麻烦|麻煩|"
    r"我想马上|我每天具体应该|我该|我應該|我应该|该吃|該吃|怎么治疗|怎麼治療|如何治疗|如何治療|"
    r"需要补充|需要補充|应该记录|應該記錄|还要说明|還要說明|还应记录|還應記錄|还需要|還需要|"
    r"直接判断|直接判斷|直接给|直接給|直接列|总结|總結|概括|梳理|汇总|彙總)"
)

_SAFE_NEGATED_MEDICAL_CLAUSE = re.compile(
    r"(?:请|請)?(?:不要|不)(?:作|做|加入|添加|提供|给出|給出|推断|推斷)?"
    r"[^。！？；.!?;]{0,36}(?:诊断|診斷|辨证|辨證|建议|建議|方案|病因)"
    r"[^。！？；.!?;]{0,36}"
)
_SUMMARY_INSTRUCTION_CLAUSE = re.compile(
    r"^(?:只保留|只复述|只復述|只作事实|只作事實|请按原话|請按原話|请不要|請不要|不要|不作|不做)"
)
_SUMMARY_ORDER_PREFIX = re.compile(r"^(?:第一|第二|另外)[，,:：]\s*")
_SUMMARY_FACT_LABEL = re.compile(r"^(?:已知情况是|已知情況是|情况如下|情況如下)[：:]\s*")


def _require_text(user_text: str) -> str:
    if not isinstance(user_text, str):
        raise TypeError("user_text must be a string")
    if not user_text.strip():
        raise ValueError("user_text must not be empty")
    return user_text


def _information_scope(user_text: str) -> str:
    """Keep factual clauses and omit later requests before detecting supplied facts."""
    factual_clauses: list[str] = []
    for clause in _SENTENCE_BREAK.split(user_text):
        clause = clause.strip()
        if not clause:
            continue
        request = _INFORMATION_REQUEST_CLAUSE.search(clause)
        if request is not None:
            clause = clause[: request.start()].rstrip("，,：: \t")
        if clause:
            factual_clauses.append(clause)
    return "。".join(factual_clauses)


def has_duration_information(user_text: str) -> bool:
    """Return whether the user already supplied an onset or duration."""
    information = _information_scope(user_text)
    return any(pattern.search(information) for pattern in _DURATION_PATTERNS)


def has_frequency_information(user_text: str) -> bool:
    """Return whether the user already supplied an occurrence frequency."""
    information = _information_scope(user_text)
    return any(pattern.search(information) for pattern in _FREQUENCY_PATTERNS)


def _follow_up_questions(user_text: str) -> list[str]:
    has_duration = has_duration_information(user_text)
    has_frequency = has_frequency_information(user_text)
    if not has_duration and not has_frequency:
        first = _ASK_FOR_BOTH
    elif not has_duration:
        first = _ASK_FOR_DURATION
    elif not has_frequency:
        first = _ASK_FOR_FREQUENCY
    else:
        first = _ASK_FOR_TREND
    return [first, OTHER_OBSERVATIONS_QUESTION]


def _is_dangerous_request(user_text: str) -> bool:
    request_scope = _SAFE_NEGATED_MEDICAL_CLAUSE.sub("", user_text)
    return bool(
        _DANGEROUS_BYPASS.search(request_scope)
        or _DANGEROUS_MEDICAL_REQUEST.search(request_scope)
        or _DANGEROUS_MEDICATION_REQUEST.search(request_scope)
    )


def _has_explicit_summary_command(user_text: str) -> bool:
    if _EXPLICIT_SUMMARY_COMMAND.search(user_text):
        return True
    stripped = user_text.strip("。！？；.!?; \t\n")
    if stripped in {"总结", "總結", "概括", "梳理", "汇总", "彙總", "整理", "归纳", "歸納", "小结", "小結", "摘要"}:
        return True
    return any(_COMMAND_ONLY.fullmatch(clause.strip()) for clause in _SENTENCE_BREAK.split(user_text) if clause.strip())


def _route(user_text: str) -> tuple[str, str]:
    if _is_dangerous_request(user_text):
        return "ask", "initial"
    if _has_explicit_summary_command(user_text):
        return "summarize", "summary"
    return "ask", "initial"


def _fact_part(candidate: str) -> str:
    candidate = candidate.strip()
    if not candidate:
        return ""
    candidate = _SUMMARY_ORDER_PREFIX.sub("", candidate)
    candidate = _SUMMARY_FACT_LABEL.sub("", candidate)
    if "：" in candidate or ":" in candidate:
        separator = "：" if "：" in candidate else ":"
        prefix, remainder = candidate.split(separator, 1)
        if _SUMMARY_COMMAND.search(prefix) or "观察" in prefix or "觀察" in prefix or "情况" in prefix or "情況" in prefix:
            candidate = remainder.strip()
    candidate = _SUMMARY_ORDER_PREFIX.sub("", candidate).strip()
    candidate = _SUMMARY_FACT_LABEL.sub("", candidate).strip()
    if not candidate or _SUMMARY_INSTRUCTION_CLAUSE.search(candidate):
        return ""
    command = _EXPLICIT_SUMMARY_COMMAND.search(candidate)
    if command is not None:
        before = candidate[: command.start()].rstrip("，,：: \t")
        return before if before and not _SUMMARY_INSTRUCTION_CLAUSE.search(before) else ""
    if _SUMMARY_COMMAND.search(candidate) and not any(token in candidate for token in ("持续", "已有", "已经", "偶尔", "每天")):
        return ""
    return candidate


def _literal_key_findings(user_text: str) -> list[str]:
    """Keep unique source-literal fact fragments and discard commands."""
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
    return findings


def adapt_interaction(user_text: str, raw_output: Any = None) -> dict[str, Any]:
    """Return a deterministic V5.3-contract interaction for one user message."""
    text = _require_text(user_text)
    action, _stage = _route(text)
    if action == "ask":
        return {
            "action": "ask",
            "stage": "initial",
            "complete": False,
            "questions": [ASK_REFUSAL, *_follow_up_questions(text)],
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
    """Serialize one deterministic result as compact UTF-8-safe JSON."""
    return json.dumps(adapt_interaction(user_text, raw_output), ensure_ascii=False, separators=(",", ":"))
