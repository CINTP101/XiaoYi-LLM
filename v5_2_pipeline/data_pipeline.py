#!/usr/bin/env python3
"""Build the V5.2 ShenNong safety-only train/held-out artifacts.

The source corpus is synthetic and has no per-record authority citation.  The
pipeline therefore uses a deliberately high-precision policy: only responses
that are clearly a refusal, a request for missing context, or a referral to a
professional are eligible.  Responses containing medical claims, treatments,
herbs/formulas, or doses are retained only in the audit rejection log.

No third-party packages are required.  The implementation is deterministic
given the config seed and writes only to the V5.2 artifact directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT = Path("/home/cyh/Medical_Qwen")
RAW_PATH = PROJECT / "data/shennong/ChatMed_TCM-v0.2.json"
V51_CLEAN_PATH = PROJECT / "data/shennong/shennong_clean_v5_1.jsonl"
V51_TRAIN_PATH = PROJECT / "data/sft_v5_1/train.jsonl"
ARTIFACT_DIR = PROJECT / "artifacts/v5_2_pipeline/data"
PIPELINE_DIR = PROJECT / "v5_2_pipeline"

CONFIG_PATH = PIPELINE_DIR / "config.json"
RULES_PATH = PIPELINE_DIR / "filter_rules_v5_2.md"

CLEAN_PATH = ARTIFACT_DIR / "shennong_clean_v5_2.jsonl"
HELDOUT_PATH = ARTIFACT_DIR / "shennong_heldout_v5_2.jsonl"
REJECTION_PATH = ARTIFACT_DIR / "shennong_rejections_v5_2.jsonl"
SAMPLE_PATH = ARTIFACT_DIR / "stratified_samples_v5_2.jsonl"
STATS_PATH = ARTIFACT_DIR / "statistics_v5_2.json"
DEDUP_PATH = ARTIFACT_DIR / "dedup_report_v5_2.json"
OVERLAP_PATH = ARTIFACT_DIR / "overlap_report_v5_2.json"
RETAINED_VERIFY_PATH = ARTIFACT_DIR / "retention_verification_v5_2.json"
EVIDENCE_PATH = ARTIFACT_DIR / "run_evidence_v5_2.json"
MANIFEST_PATH = ARTIFACT_DIR / "sha256sums_v5_2.txt"
ARTIFACT_RULES_PATH = ARTIFACT_DIR / "filter_rules_v5_2.md"

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 20260910,
    "heldout_fraction": 0.20,
    "min_query_chars": 4,
    "max_query_chars": 200,
    "min_response_chars": 15,
    "max_response_chars": 650,
    "shingle_n": 3,
    "self_query_jaccard_threshold": 0.95,
    "self_pair_jaccard_threshold": 0.90,
    "v51_query_jaccard_threshold": 0.90,
    "v51_pair_jaccard_threshold": 0.90,
    "spot_samples_per_rejection_stratum": 3,
    "spot_samples_per_acceptance_class": 5,
    "retention_samples_per_class": 10,
    "medical_request_requires_explicit_refusal": True,
    "medical_request_disallows_clarification": True,
    "documentary_query_context_clarification_allowed": True,
    "reject_response_promises_and_external_operations": True,
}


# These markers identify answers that do not expose medical facts or treatment
# instructions.  They are intentionally broad; the clause-level checks below
# prevent a factual paragraph from passing merely because it ends with "see a
# doctor".
REFUSAL_MARKERS: Tuple[str, ...] = (
    "无法为您提供",
    "不能为您提供",
    "不能提供",
    "无法提供",
    "不能进行医学诊断",
    "无法进行医学诊断",
    "无法诊断",
    "不能诊断",
    "不具备诊断",
    "不建议自行",
    "没有资格",
    "没有能力",
    "没有足够的信息",
    "没有权限",
    "无权限",
    "不太明白",
    "不理解",
    "不清楚",
    "无法回答",
    "不能回答",
    "不能判断",
    "无法判断",
    "没有资质",
    "不具备",
    "不能解析",
    "无法解析",
    "不能分析",
    "无法分析",
    "不能确定",
    "无法确定",
    "不能实时",
    "无法实时",
    "没有资格或能力",
)

CLARIFICATION_MARKERS: Tuple[str, ...] = (
    "请提供",
    "能否提供",
    "请补充",
    "请描述",
    "请告诉我",
    "您能否",
    "您可以告诉我",
    "请问您",
    "还需要了解",
    "需要更多信息",
    "什么样的帮助",
    "提供更多背景",
    "提供更多上下文",
    "提供更多信息",
    "告诉我您的",
    "请问能否",
    "能否告诉",
    "更多的背景",
    "更多的上下文",
    "更多信息",
)

CARE_MARKERS: Tuple[str, ...] = (
    "建议您咨询医生",
    "建议你咨询医生",
    "建议咨询医生",
    "请咨询医生",
    "建议您就医",
    "建议你就医",
    "及时就医",
    "立即就医",
    "请前往医院",
    "前往医院",
    "寻求专业医生",
    "咨询专业医生",
    "咨询医生或",
    "咨询专业医疗机构",
    "寻求医疗帮助",
    "寻求帮助",
    "请向专业的医生咨询",
    "请向医生咨询",
    "咨询专业的医疗机构",
    "寻求专业医疗人士",
    "拨打当地的急救电话",
    "前往当地医院",
    "医院就诊",
    "专业医疗人员",
)

# These terms make a response actionable or factual.  The source has no
# record-level authoritative citation, so the conservative policy excludes
# such responses instead of attempting to guess whether each claim is true.
HARD_BLOCK_TERMS: Tuple[str, ...] = (
    "中药",
    "方剂",
    "中成药",
    "草药",
    "药材",
    "药物",
    "药方",
    "配方",
    "用药",
    "服用",
    "煎服",
    "水煎",
    "剂量",
    "克",
    "毫克",
    "mg",
    "处方",
    "开方",
    "治疗",
    "治疗方案",
    "治愈",
    "治好",
    "疗效",
    "功效",
    "改善",
    "缓解",
    "推荐",
    "适用于",
    "通常",
    "可能是",
    "可能由",
    "例如",
    "包括",
    "症状包括",
    "主要症状",
    "症状表现",
    "病因",
    "证候",
    "疾病",
    "病症",
    "感染",
    "炎症",
    "体温",
    "药理",
    "成分",
    "作用",
    "使用方法",
    "用法",
    "用量",
    "疗法",
    "食疗",
    "针灸",
    "按摩",
    "多喝水",
    "休息",
    "保持",
    "避免",
    "可以尝试",
    "应该",
    "预防",
    "风险",
    "病情",
    "病史",
    "体质",
    "脉象",
    "舌苔",
    "脉搏",
    "检查结果",
    "医疗建议",
    "回答您的问题",
    "回答这个问题",
)

# Meta words are safe only when they occur in a refusal or information
# request.  They are removed before hard-block matching for that clause.
REFUSAL_META_ALLOWED: Tuple[str, ...] = (
    "医学诊断",
    "医学建议",
    "医疗建议",
    "治疗建议",
    "疾病诊断",
    "诊断",
    "疾病",
    "病情",
    "症状",
    "身体状况",
    "健康状况",
    "身体",
    "健康",
    "病史",
    "中医",
)
ASK_META_ALLOWED: Tuple[str, ...] = (
    "症状",
    "病情",
    "病史",
    "背景",
    "上下文",
    "信息",
    "年龄",
    "性别",
    "具体",
    "详情",
    "原文",
    "标题",
    "文章",
    "文献",
    "名称",
    "作者",
    "问题",
    "情况",
    "内容",
    "细节",
    "描述",
)
CARE_ALLOWED: Tuple[str, ...] = (
    "就医",
    "医生",
    "医院",
    "医疗",
    "专业",
    "专家",
    "急救",
    "急诊",
    "电话",
    "帮助",
    "意见",
    "咨询",
    "建议",
    "检查",
    "诊断",
    "机构",
    "人员",
    "前往",
    "寻求",
    "及时",
    "立即",
    "拨打",
    "当地",
)

CHAIN_MARKERS: Tuple[str, ...] = (
    "推理过程",
    "思考过程",
    "详细推理",
    "一步步的推理",
    "逐步推理",
)

DIRECT_RESPONSE_TERMS: Tuple[str, ...] = (
    "中药",
    "方剂",
    "中成药",
    "草药",
    "药材",
    "药物",
    "药方",
    "处方",
    "配方",
    "开方",
    "服用",
    "用药",
    "剂量",
    "克",
    "毫克",
    "mg",
    "水煎",
    "煎服",
    "每日",
    "每次",
    "治疗",
    "治愈",
    "处方如下",
    "开方如下",
)

UNVERIFIED_RESPONSE_TERMS: Tuple[str, ...] = (
    "可能",
    "通常",
    "例如",
    "包括",
    "症状",
    "分类",
    "分为",
    "病机",
    "病因",
    "原因",
    "证候",
    "疾病",
    "病症",
    "感染",
    "炎症",
    "功效",
    "作用",
    "疗效",
    "改善",
    "缓解",
    "适用于",
    "药理",
    "成分",
    "治疗",
    "诊断",
    "分析",
    "判断",
    "确定",
    "找到",
    "尽力帮助",
    "尽力帮",
    "我会尽力",
    "我将尽力",
    "我会帮助",
    "我可以帮助",
    "我可以帮",
    "可以帮您",
    "可以帮你",
    "提供准确的建议",
    "提供更准确的建议",
    "提供有用的建议",
    "提供有针对性的建议",
    "给出建议",
    "提供建议",
    "提供帮助",
    "更好地帮助",
    "以便我能够",
    "以便我可以",
    "我会根据",
    "我将根据",
    "我会为您",
    "我将为您",
    "我会根据这些信息进行推理",
    "我将根据这些信息进行推理",
    "进行推理",
    "帮您预约",
    "帮助您预约",
    "并预约",
    "查找附近",
    "做出诊断",
    "不是一个合适的",
    "存在的方法",
)

# A prompt requesting a clinical decision or an actionable medical
# intervention must never be retained as an information-gathering question.
# It is eligible only when the answer explicitly refuses the medical service.
# Documentary tasks (for example, editing a paper or identifying a title) are
# intentionally outside this gate and can retain a pure context question.
MEDICAL_REQUEST_TERMS: Tuple[str, ...] = (
    "诊断",
    "确诊",
    "辨证",
    "辨证论治",
    "证候",
    "病因",
    "病机",
    "病情",
    "病症",
    "问诊",
    "症状",
    "体征",
    "临床表现",
    "身体不适",
    "健康问题",
    "病人",
    "患者",
    "治疗",
    "疗法",
    "治愈",
    "治病",
    "医疗建议",
    "医学建议",
    "中药",
    "方剂",
    "方子",
    "药材",
    "草药",
    "药物",
    "药方",
    "处方",
    "配方",
    "开方",
    "用药",
    "服用",
    "剂量",
    "药量",
    "针灸",
    "按摩",
    "食疗",
)

# Short or incomplete clinical prompts often omit an explicit verb (for
# example, "痫, 舌强, 慢惊").  Outside a documentary context they are still
# medical requests and must not be answered with an information-gathering
# question.  Documentary prompts are exempted by ``is_documentary_query``.
MEDICAL_CONTEXT_TERMS: Tuple[str, ...] = (
    "症状",
    "病情",
    "病症",
    "证候",
    "病因",
    "病机",
    "问诊",
    "体征",
    "临床表现",
    "健康问题",
    "身体不适",
    "患者",
    "病人",
    "舌",
    "脉",
    "腹",
    "胸",
    "头痛",
    "头疼",
    "咳",
    "惊",
    "疮",
    "痛",
    "热",
    "寒",
    "血",
    "孕",
    "胎",
    "耳",
    "鼻",
    "麻",
    "痒",
    "吐",
    "呕",
    "泻",
    "便",
    "经",
    "腰",
    "心悸",
    "失眠",
    "发热",
)

DOCUMENTARY_QUERY_TERMS: Tuple[str, ...] = (
    "文献",
    "论文",
    "文章",
    "标题",
    "作者",
    "摘要",
    "原文",
    "书籍",
    "经文",
    "编辑",
    "修改",
    "检索",
    "查找",
    "解析",
)

EXPLICIT_MEDICAL_REFUSAL_TERMS: Tuple[str, ...] = (
    "不能提供医疗建议",
    "无法提供医疗建议",
    "不能为您提供医疗建议",
    "无法为您提供医疗建议",
    "不能提供医学建议",
    "无法提供医学建议",
    "不能提供任何医疗建议",
    "无法提供任何医疗建议",
    "不能进行医学诊断",
    "无法进行医学诊断",
    "没有资格进行医学诊断",
    "没有能力进行医学诊断",
    "没有资质进行医学诊断",
    "无法进行医疗诊断",
    "不能进行医疗诊断",
    "无法诊断",
    "不能诊断",
    "不具备诊断",
    "不能提供诊断",
    "无法提供诊断",
    "不能提供治疗建议",
    "无法提供治疗建议",
    "不能提供医疗诊断",
    "无法提供医疗诊断",
    "不能提供医疗诊断或治疗建议",
    "无法提供医疗诊断或治疗建议",
    "不能提供医学诊断或治疗建议",
    "无法提供医学诊断或治疗建议",
    "没有资格进行医学诊断或提供治疗建议",
    "没有能力进行医学诊断或提供治疗建议",
    "没有资质进行医学诊断或提供治疗建议",
    "不能提供任何医学上的诊断或治疗建议",
    "无法提供任何医学上的诊断或治疗建议",
    "不能提供诊断和治疗建议",
    "无法提供诊断和治疗建议",
    "不能提供医疗建议或诊断",
    "无法提供医疗建议或诊断",
    "不能提供诊疗建议",
    "无法提供诊疗建议",
    "无法回答关于身体健康和医学问题",
    "不能为您提供确切的医疗建议",
    "不能为您提供任何医疗建议",
)

PROMISE_OR_OPERATION_TERMS: Tuple[str, ...] = (
    "我可以帮",
    "可以帮您",
    "可以帮你",
    "我可以帮助",
    "我可以为您",
    "我可以为你",
    "我会帮助",
    "我将帮助",
    "我会尽力",
    "我将尽力",
    "尽力帮助",
    "尽力帮",
    "更好地帮助",
    "以便我能够",
    "以便我可以",
    "我会根据",
    "我将根据",
    "我会为您",
    "我将为您",
    "我会为你",
    "我将为你",
    "我能为您",
    "我能为你",
    "能够为您",
    "能够为你",
    "才能为您",
    "才能为你",
    "我可以回答",
    "我会回答",
    "我将回答",
    "更好地回答",
    "才能回答",
    "为您服务",
    "为你服务",
    "为您解答",
    "为你解答",
    "帮助您",
    "帮助你",
    "让我更好地",
    "让我能够",
    "让我可以",
    "以便我更好地",
    "以便更好地",
    "更好地了解",
    "更好地理解",
    "才能了解",
    "让我了解",
    "才能帮助",
    "才能提供",
    "提供准确的建议",
    "提供更准确的建议",
    "提供有用的建议",
    "提供有针对性的建议",
    "给出建议",
    "提供建议",
    "提供准确的分析",
    "进行推理",
    "根据这些信息进行推理",
    "帮您预约",
    "帮助您预约",
    "查找附近",
    "并预约",
    "为您进行编辑",
    "为您编辑",
    "为您修改",
    "为您检索",
    "为您查找",
)


def load_config() -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config.update(loaded)
    return config


def clean_space(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"\s+", " ", text)


def compact(text: Any) -> str:
    """Canonical text for exact matching; punctuation/whitespace are noise."""
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def split_clauses(text: str) -> List[str]:
    return [
        part.strip(" \t，,：:；;")
        for part in re.split(r"[。！？!?\n]+", text)
        if part.strip(" \t，,：:；;")
    ]


def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def is_documentary_query(query: str) -> bool:
    """Return whether a prompt is clearly about documentary work.

    The source mixes clinical requests with requests to edit or identify
    papers/books.  A documentary prompt may retain a context question, but a
    prompt that also asks for a clinical decision remains a medical request.
    """
    query = clean_space(query)
    if not contains_any(query, DOCUMENTARY_QUERY_TERMS):
        return False
    clinical_action_terms = (
        "请诊断",
        "诊断一下",
        "是否患有",
        "推荐",
        "治疗",
        "治愈",
        "辨证",
        "辨证论治",
        "开方",
        "处方",
        "用药",
        "服用",
        "剂量",
        "药方",
        "方剂",
        "中药",
        "药材",
        "药物",
        "病因分析",
        "病机分析",
        "医疗建议",
        "医学建议",
    )
    return not contains_any(query, clinical_action_terms)


def is_medical_request(query: str) -> bool:
    """Identify prompts that require an explicit safety refusal.

    This is a deliberately deterministic gate.  It blocks the common unsafe
    pattern of answering a clinical request with "please provide symptoms"
    and implicitly promising a later diagnosis, recommendation, or formula.
    """
    query = clean_space(query)
    if not query or is_documentary_query(query):
        return False
    if contains_any(query, MEDICAL_REQUEST_TERMS):
        return True
    if contains_any(query, MEDICAL_CONTEXT_TERMS):
        return True
    return False


def has_explicit_medical_refusal(response: str) -> bool:
    """Require a refusal directed at diagnosis/treatment/medical advice."""
    response = clean_space(response)
    if contains_any(response, EXPLICIT_MEDICAL_REFUSAL_TERMS):
        return True
    refusal_bases = ("不能", "无法", "没有资格", "没有能力", "没有资质", "不具备")
    medical_targets = (
        "医学诊断",
        "医疗诊断",
        "诊断",
        "治疗建议",
        "医疗建议",
        "医学建议",
    "医疗诊疗",
    "诊疗建议",
    "医学影像",
    "影像",
    "CT",
    )
    for clause in split_clauses(response):
        if contains_any(clause, refusal_bases) and contains_any(clause, medical_targets):
            return True
    return False


def residual_for_clause(clause: str, *, refusal: bool, asking: bool, care: bool) -> str:
    residual = clause
    if refusal:
        for term in REFUSAL_META_ALLOWED:
            residual = residual.replace(term, "")
    if asking:
        for term in ASK_META_ALLOWED:
            residual = residual.replace(term, "")
    if care:
        for term in CARE_ALLOWED:
            residual = residual.replace(term, "")
    return residual


def classify_safe_response(response: str, config: Dict[str, Any], query: str = "") -> Optional[str]:
    """Return a safe response class or None.

    Every sentence must be a refusal, a referral, a request for missing
    context, or a short courtesy sentence.  Removing only context/meta terms
    before hard-block matching prevents a response such as "I cannot diagnose,
    but here are five diagnoses" from passing.  A clinical request is gated
    separately: it can pass only with an explicit medical safety refusal, and
    never with an information-gathering question.
    """
    response = clean_space(response)
    if not (config["min_response_chars"] <= len(response) <= config["max_response_chars"]):
        return None
    if re.search(r"\d", response):
        return None
    has_refusal = contains_any(response, REFUSAL_MARKERS)
    has_asking = contains_any(response, CLARIFICATION_MARKERS)
    if not (has_refusal or has_asking):
        return None
    medical_query = is_medical_request(query)

    clauses = split_clauses(response)
    if not clauses:
        return None
    for clause in clauses:
        is_refusal = contains_any(clause, REFUSAL_MARKERS)
        is_asking = contains_any(clause, CLARIFICATION_MARKERS)
        is_care = contains_any(clause, CARE_MARKERS)
        is_courtesy = contains_any(
            clause,
            (
                "抱歉",
                "对不起",
                "谢谢理解",
                "谢谢",
                "很好",
                "好的",
                "当然可以",
                "我会尽力帮助",
                "希望能帮到",
            ),
        )
        if not (is_refusal or is_asking or is_care or is_courtesy):
            return None
        # A clarification may request missing context but cannot promise a
        # later service or an external operation.  In a safety refusal, only
        # an explicit medical refusal may contain a phrase that otherwise
        # looks like a service promise.
        if config.get("reject_response_promises_and_external_operations", True) and contains_any(
            clause, PROMISE_OR_OPERATION_TERMS
        ) and not (is_refusal and has_explicit_medical_refusal(clause)):
            return None
        residual = residual_for_clause(
            clause,
            refusal=is_refusal,
            asking=is_asking,
            care=is_care,
        )
        # Context words such as "症状" are allowed in a pure question, but
        # a factual linker turns that same word into an assertion (for
        # example, "X是一种症状" or "症状可以分为...").
        factual_linker = (
            "是一种",
            "是指",
            "属于",
            "可以分为",
            "可分为",
            "归为",
            "归属于",
            "表示",
            "说明",
            "意味着",
            "通常",
            "可能",
            "因此",
            "因为",
            "由于",
            "根据中医",
            "从中医",
        )
        if contains_any(residual, HARD_BLOCK_TERMS) or contains_any(residual, UNVERIFIED_RESPONSE_TERMS):
            return None
        if contains_any(clause, factual_linker) and contains_any(
            clause,
            (
                "症状",
                "病因",
                "病机",
                "证候",
                "疾病",
                "原因",
                "分类",
                "分为",
                "治疗",
                "药物",
                "中药",
            ),
        ):
            return None
    if medical_query:
        # Requests for diagnosis, treatment, medicines, formulas, herbs, or
        # dose must not be retained as "please provide symptoms" questions.
        if (
            config.get("medical_request_disallows_clarification", True)
            and has_asking
        ) or (
            config.get("medical_request_requires_explicit_refusal", True)
            and (not has_refusal or not has_explicit_medical_refusal(response))
        ):
            return None
        return "safety_refusal"
    # A generic "I don't understand" without an actual context request is
    # not a safety refusal and is excluded.  If it asks for context, emit it
    # as a clarification so it cannot masquerade as a medical refusal.
    if has_asking:
        return "clarification"
    if has_refusal and has_explicit_medical_refusal(response):
        return "safety_refusal"
    return None


def filter_reason(response: str, query: str = "") -> str:
    response = clean_space(response)
    if contains_any(response, CHAIN_MARKERS):
        return "reasoning_chain_or_hidden_cot"
    if contains_any(response, DIRECT_RESPONSE_TERMS):
        return "direct_prescription_or_dose"
    if is_medical_request(query) and not has_explicit_medical_refusal(response):
        return "medical_request_without_explicit_safety_refusal"
    if contains_any(response, UNVERIFIED_RESPONSE_TERMS):
        return "unverified_or_conflicting_knowledge"
    return "no_safe_refusal_or_clarification"


def jsonl_records(path: Path) -> Iterable[Tuple[int, Any, Optional[str]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            try:
                yield line_number, json.loads(line), None
            except Exception as exc:  # pragma: no cover - exercised on bad input
                yield line_number, None, repr(exc)


def answer_from_conversations(item: Any) -> Tuple[str, str]:
    if not isinstance(item, dict):
        return "", ""
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return "", ""
    human = ""
    assistant = ""
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        sender = turn.get("from")
        value = clean_space(turn.get("value", ""))
        if sender == "human" and not human:
            human = value
        elif sender == "gpt" and not assistant:
            assistant = value
    answer = assistant
    try:
        parsed = json.loads(assistant)
        if isinstance(parsed, dict):
            answer = clean_space(parsed.get("answer") or parsed.get("message") or parsed.get("recommendation") or assistant)
    except Exception:
        pass
    return human, answer


def pair_text(query: str, response: str) -> str:
    return compact(query) + "\u241f" + compact(response)


def shingles(text: str, n: int) -> frozenset[str]:
    value = compact(text)
    if not value:
        return frozenset()
    if len(value) <= n:
        return frozenset((value,))
    return frozenset(value[i : i + n] for i in range(len(value) - n + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def output_row(candidate: Dict[str, Any]) -> Dict[str, Any]:
    if candidate["class"] == "safety_refusal":
        assistant = {
            "action": "refuse",
            "stage": "safety",
            "complete": True,
            "message": candidate["response"],
        }
    else:
        assistant = {
            "action": "ask",
            "stage": "initial",
            "complete": False,
            "questions": [candidate["response"]],
        }
    return {
        "conversations": [
            {"from": "human", "value": candidate["query"]},
            {"from": "gpt", "value": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def verify_retained_outputs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run the safety predicate over every retained output row.

    This is deliberately independent of the in-memory candidate list: it
    parses the files that will be handed to training/evaluation, extracts the
    actual assistant message/question, and re-applies the same clause-level
    predicate.  The report includes one compact verification record per row
    and a deterministic stratified sample for human review.
    """
    scan: List[Dict[str, Any]] = []
    expected_classes = (
        (CLEAN_PATH, {"refuse": "safety_refusal", "ask": "clarification"}),
        (HELDOUT_PATH, {"refuse": "safety_refusal", "ask": "clarification"}),
    )
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path, action_map in expected_classes:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                record: Dict[str, Any] = {
                    "output_path": str(path.resolve()),
                    "output_line": line_number,
                    "classification_match": False,
                    "safe_predicate_pass": False,
                    "unapproved_hard_block_terms": [],
                    "unapproved_unverified_terms": [],
                    "unapproved_promise_or_operation_terms": [],
                    "medical_request": False,
                    "explicit_medical_refusal": False,
                }
                try:
                    item = json.loads(raw_line)
                    conversations = item["conversations"]
                    query = clean_space(conversations[0]["value"])
                    assistant = json.loads(conversations[1]["value"])
                    action = assistant.get("action")
                    response = clean_space(assistant.get("message") or (assistant.get("questions") or [""])[0])
                    expected_class = action_map[action]
                    recomputed_class = classify_safe_response(response, config, query=query)
                    record["query_sha256"] = text_hash(query)
                    record["action"] = action
                    record["expected_class"] = expected_class
                    record["recomputed_class"] = recomputed_class
                    record["classification_match"] = recomputed_class == expected_class
                    for clause in split_clauses(response):
                        is_refusal = contains_any(clause, REFUSAL_MARKERS)
                        is_asking = contains_any(clause, CLARIFICATION_MARKERS)
                        is_care = contains_any(clause, CARE_MARKERS)
                        residual = residual_for_clause(
                            clause,
                            refusal=is_refusal,
                            asking=is_asking,
                            care=is_care,
                        )
                        for term in HARD_BLOCK_TERMS:
                            if term in residual and term not in record["unapproved_hard_block_terms"]:
                                record["unapproved_hard_block_terms"].append(term)
                        for term in UNVERIFIED_RESPONSE_TERMS:
                            if term in residual and term not in record["unapproved_unverified_terms"]:
                                record["unapproved_unverified_terms"].append(term)
                        for term in PROMISE_OR_OPERATION_TERMS:
                            if term in clause and term not in record["unapproved_promise_or_operation_terms"]:
                                record["unapproved_promise_or_operation_terms"].append(term)
                    record["medical_request"] = is_medical_request(query)
                    record["explicit_medical_refusal"] = has_explicit_medical_refusal(response)
                    record["safe_predicate_pass"] = bool(
                        record["classification_match"]
                        and (
                            not record["medical_request"]
                            or not config.get("medical_request_requires_explicit_refusal", True)
                            or record["explicit_medical_refusal"]
                        )
                        and (
                            not record["medical_request"]
                            or not config.get("medical_request_disallows_clarification", True)
                            or record.get("action") != "ask"
                        )
                        and not record["unapproved_hard_block_terms"]
                        and not record["unapproved_unverified_terms"]
                        and (
                            not config.get("reject_response_promises_and_external_operations", True)
                            or not record["unapproved_promise_or_operation_terms"]
                        )
                    )
                except Exception as exc:
                    record["verification_error"] = repr(exc)
                scan.append(record)
                by_class.setdefault(record.get("expected_class", "invalid"), []).append(record)

    sample_rng = random.Random(int(config["seed"]))
    stratified_sample: List[Dict[str, Any]] = []
    per_class = int(config.get("retention_samples_per_class", 10))
    for response_class in sorted(by_class):
        rows = list(by_class[response_class])
        sample_rng.shuffle(rows)
        stratified_sample.extend(rows[:per_class])
    full_pass = sum(1 for row in scan if row.get("safe_predicate_pass"))
    report = {
        "verification_scope": "all retained rows in clean and heldout JSONL files",
        "clean_path": str(CLEAN_PATH.resolve()),
        "heldout_path": str(HELDOUT_PATH.resolve()),
        "total_retained_rows": len(scan),
        "full_scan_pass_rows": full_pass,
        "full_scan_fail_rows": len(scan) - full_pass,
        "full_scan_pass_rate": (full_pass / len(scan)) if scan else 1.0,
        "direct_prescription_unapproved_rows": sum(bool(row.get("unapproved_hard_block_terms")) for row in scan),
        "unverified_claim_unapproved_rows": sum(bool(row.get("unapproved_unverified_terms")) for row in scan),
        "promise_or_operation_unapproved_rows": sum(
            bool(row.get("unapproved_promise_or_operation_terms")) for row in scan
        ),
        "medical_request_rows": sum(bool(row.get("medical_request")) for row in scan),
        "medical_request_without_explicit_refusal_rows": sum(
            bool(row.get("medical_request")) and not bool(row.get("explicit_medical_refusal"))
            for row in scan
        ),
        "classification_fail_rows": sum(not row.get("classification_match", False) for row in scan),
        "stratified_sample": {
            "seed": int(config["seed"]),
            "rows_per_class": per_class,
            "sample_count": len(stratified_sample),
            "sample_pass_count": sum(1 for row in stratified_sample if row.get("safe_predicate_pass")),
            "sample_pass_rate": (
                sum(1 for row in stratified_sample if row.get("safe_predicate_pass")) / len(stratified_sample)
                if stratified_sample
                else 1.0
            ),
        },
        "full_scan_records": scan,
        "stratified_sample_records": stratified_sample,
        "conclusion": "PASS: every retained row re-passed the error/unknown-claim and direct-prescription predicate"
        if full_pass == len(scan)
        else "FAIL: retained output contains rows that do not re-pass the predicate",
    }
    write_json(RETAINED_VERIFY_PATH, report)
    return report


def rejection_record(
    line_number: int,
    query: str,
    response: str,
    reasons: Sequence[str],
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "audit_only": True,
        "source_path": str(RAW_PATH.resolve()),
        "source_line": line_number,
        "query": query,
        "response": response,
        "query_sha256": text_hash(query),
        "response_sha256": text_hash(response),
        "decision": "reject",
        "stage": stage,
        "reason_codes": list(dict.fromkeys(reasons)),
        "primary_reason": list(dict.fromkeys(reasons))[0] if reasons else "unspecified",
    }
    if extra:
        record.update(extra)
    return record


def main() -> int:
    config = load_config()
    seed = int(config["seed"])
    rng = random.Random(seed)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

    required = [RAW_PATH, V51_CLEAN_PATH, V51_TRAIN_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    # Keep one rejection object per source line so every raw item receives an
    # explicit reason, including candidates removed at a later dedup/leakage
    # stage.  This is an audit artifact only and is never loaded as training
    # or evaluation data.
    rejected: Dict[int, Dict[str, Any]] = {}
    safe_candidates: List[Dict[str, Any]] = []
    raw_counts = Counter()

    for line_number, item, parse_error in jsonl_records(RAW_PATH):
        raw_counts["input_lines"] += 1
        if parse_error is not None:
            raw_counts["parse_error"] += 1
            rejected[line_number] = rejection_record(
                line_number,
                "",
                "",
                ["parse_error"],
                "safety_filter",
                {"parse_error": parse_error},
            )
            continue
        if not isinstance(item, dict):
            raw_counts["invalid_schema"] += 1
            rejected[line_number] = rejection_record(
                line_number,
                "",
                "",
                ["invalid_schema"],
                "safety_filter",
            )
            continue

        query = clean_space(item.get("query", ""))
        response = clean_space(item.get("response", ""))
        reasons: List[str] = []
        if not query or not response:
            reasons.append("empty_or_missing_field")
        if not (config["min_query_chars"] <= len(query) <= config["max_query_chars"]):
            reasons.append("query_length_out_of_range")
        if not (config["min_response_chars"] <= len(response) <= config["max_response_chars"]):
            reasons.append("response_length_out_of_range")
        if contains_any(query, CHAIN_MARKERS):
            # A prompt asking for reasoning is not itself unsafe, but if it is
            # paired with a response that exposes reasoning, retain this audit
            # label as a secondary reason.
            raw_counts["query_chain_request"] += 1
        response_class = classify_safe_response(response, config, query=query) if not reasons else None
        if response_class is None:
            reasons.append(filter_reason(response, query=query))
        if reasons:
            raw_counts["safety_rejected"] += 1
            for reason in reasons:
                raw_counts[reason] += 1
            rejected[line_number] = rejection_record(
                line_number,
                query,
                response,
                reasons,
                "safety_filter",
            )
            continue
        safe_candidates.append(
            {
                "source_line": line_number,
                "query": query,
                "response": response,
                "class": response_class,
                "query_norm": compact(query),
                "pair_norm": pair_text(query, response),
                "query_grams": shingles(query, int(config["shingle_n"])),
                "pair_grams": shingles(pair_text(query, response), int(config["shingle_n"])),
            }
        )
    raw_counts["safe_candidates"] = len(safe_candidates)

    # Load the two baseline views.  The mixed V5.1 training file is the
    # authoritative leakage reference; the clean file is retained separately
    # in the report so the paths and counts are independently auditable.
    def load_reference(path: Path) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for line_number, item, parse_error in jsonl_records(path):
            if parse_error is not None:
                continue
            query, response = answer_from_conversations(item)
            if not query:
                continue
            refs.append(
                {
                    "line": line_number,
                    "query": query,
                    "response": response,
                    "query_norm": compact(query),
                    "pair_norm": pair_text(query, response),
                    "query_grams": shingles(query, int(config["shingle_n"])),
                    "pair_grams": shingles(pair_text(query, response), int(config["shingle_n"])),
                }
            )
        return refs

    v51_clean_refs = load_reference(V51_CLEAN_PATH)
    v51_train_refs = load_reference(V51_TRAIN_PATH)

    # Text-level then semantic near-duplicate removal within the source pool.
    unique_candidates: List[Dict[str, Any]] = []
    exact_query_lines: Dict[str, int] = {}
    exact_pair_lines: Dict[str, int] = {}
    self_dedup_counts = Counter()
    for candidate in safe_candidates:
        duplicate_line: Optional[int] = None
        if candidate["query_norm"] in exact_query_lines:
            duplicate_line = exact_query_lines[candidate["query_norm"]]
        elif candidate["pair_norm"] in exact_pair_lines:
            duplicate_line = exact_pair_lines[candidate["pair_norm"]]
        if duplicate_line is not None:
            self_dedup_counts["text_duplicate_removed"] += 1
            rejected[candidate["source_line"]] = rejection_record(
                candidate["source_line"],
                candidate["query"],
                candidate["response"],
                ["text_duplicate"],
                "text_dedup",
                {"duplicate_of_source_line": duplicate_line},
            )
            continue

        semantic_match: Optional[Dict[str, Any]] = None
        for prior in unique_candidates:
            query_score = jaccard(candidate["query_grams"], prior["query_grams"])
            pair_score = jaccard(candidate["pair_grams"], prior["pair_grams"])
            if (
                query_score >= float(config["self_query_jaccard_threshold"])
                or pair_score >= float(config["self_pair_jaccard_threshold"])
            ):
                semantic_match = {
                    "source_line": prior["source_line"],
                    "query_jaccard": round(query_score, 6),
                    "pair_jaccard": round(pair_score, 6),
                }
                break
        if semantic_match is not None:
            self_dedup_counts["semantic_duplicate_removed"] += 1
            rejected[candidate["source_line"]] = rejection_record(
                candidate["source_line"],
                candidate["query"],
                candidate["response"],
                ["semantic_near_duplicate"],
                "semantic_dedup",
                {"duplicate_match": semantic_match},
            )
            continue
        exact_query_lines[candidate["query_norm"]] = candidate["source_line"]
        exact_pair_lines[candidate["pair_norm"]] = candidate["source_line"]
        unique_candidates.append(candidate)

    # Remove exact and approximate intersections with V5.1.  We compare both
    # query and full query/answer pair.  Query-level blocking is intentionally
    # stronger for this safety artifact because a changed unsafe answer on an
    # already-trained prompt still leaks the prompt.
    isolated_candidates: List[Dict[str, Any]] = []
    v51_overlap_counts = Counter()
    v51_train_query_map: Dict[str, int] = {}
    v51_train_pair_map: Dict[str, int] = {}
    for ref in v51_train_refs:
        v51_train_query_map.setdefault(ref["query_norm"], ref["line"])
        v51_train_pair_map.setdefault(ref["pair_norm"], ref["line"])

    for candidate in unique_candidates:
        exact_line = v51_train_query_map.get(candidate["query_norm"])
        if exact_line is None:
            exact_line = v51_train_pair_map.get(candidate["pair_norm"])
        if exact_line is not None:
            v51_overlap_counts["v51_exact_removed"] += 1
            rejected[candidate["source_line"]] = rejection_record(
                candidate["source_line"],
                candidate["query"],
                candidate["response"],
                ["v5_1_exact_overlap"],
                "v5_1_leakage_filter",
                {"v5_1_train_line": exact_line, "v5_1_train_path": str(V51_TRAIN_PATH.resolve())},
            )
            continue
        near_match: Optional[Dict[str, Any]] = None
        for ref in v51_train_refs:
            query_score = jaccard(candidate["query_grams"], ref["query_grams"])
            pair_score = jaccard(candidate["pair_grams"], ref["pair_grams"])
            if (
                query_score >= float(config["v51_query_jaccard_threshold"])
                or pair_score >= float(config["v51_pair_jaccard_threshold"])
            ):
                near_match = {
                    "v5_1_train_line": ref["line"],
                    "query_jaccard": round(query_score, 6),
                    "pair_jaccard": round(pair_score, 6),
                }
                break
        if near_match is not None:
            v51_overlap_counts["v51_semantic_removed"] += 1
            rejected[candidate["source_line"]] = rejection_record(
                candidate["source_line"],
                candidate["query"],
                candidate["response"],
                ["v5_1_semantic_near_overlap"],
                "v5_1_leakage_filter",
                {"v5_1_train_path": str(V51_TRAIN_PATH.resolve()), "overlap_match": near_match},
            )
            continue
        isolated_candidates.append(candidate)

    # Reserve 20% as an independent blind test set, stratified by the two
    # safe response classes.  Every row belongs to exactly one output file.
    by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in isolated_candidates:
        by_class[candidate["class"]].append(candidate)
    clean_candidates: List[Dict[str, Any]] = []
    heldout_candidates: List[Dict[str, Any]] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for response_class in sorted(by_class):
        rows = list(by_class[response_class])
        rng.shuffle(rows)
        if len(rows) <= 1:
            heldout_count = 0
        else:
            heldout_count = max(1, int(round(len(rows) * float(config["heldout_fraction"]))))
            heldout_count = min(len(rows) - 1, heldout_count)
        heldout_candidates.extend(rows[:heldout_count])
        clean_candidates.extend(rows[heldout_count:])
        split_counts[response_class] = {
            "pool": len(rows),
            "clean": len(rows) - heldout_count,
            "heldout": heldout_count,
        }

    # Stable source-line order makes artifact diffs and hashes reproducible.
    clean_candidates.sort(key=lambda row: row["source_line"])
    heldout_candidates.sort(key=lambda row: row["source_line"])

    with CLEAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in clean_candidates:
            handle.write(json.dumps(output_row(candidate), ensure_ascii=False, separators=(",", ":")) + "\n")
    with HELDOUT_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in heldout_candidates:
            handle.write(json.dumps(output_row(candidate), ensure_ascii=False, separators=(",", ":")) + "\n")

    # Ensure every source line is represented exactly once in either output or
    # rejection log.  This invariant catches accidental silent drops.
    accepted_lines = {candidate["source_line"] for candidate in clean_candidates + heldout_candidates}
    all_source_lines = set(range(1, raw_counts["input_lines"] + 1))
    accounted_lines = accepted_lines | set(rejected)
    missing_lines = sorted(all_source_lines - accounted_lines)
    extra_lines = sorted(set(rejected) - all_source_lines)
    if missing_lines or extra_lines or len(accounted_lines) != raw_counts["input_lines"]:
        raise RuntimeError(
            f"Source accounting invariant failed: missing={missing_lines[:5]}, extra={extra_lines[:5]}, "
            f"accounted={len(accounted_lines)}, input={raw_counts['input_lines']}"
        )

    with REJECTION_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for line_number in sorted(rejected):
            handle.write(json.dumps(rejected[line_number], ensure_ascii=False, separators=(",", ":")) + "\n")

    # Build a stratified, reproducible audit sample.  Rejected samples are
    # sampled by primary reason; the measured rejection rate is therefore
    # explicitly 100% by construction and independently checkable.
    accepted_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in clean_candidates + heldout_candidates:
        accepted_by_class[candidate["class"]].append(candidate)
    sample_rows: List[Dict[str, Any]] = []
    sample_rng = random.Random(seed)
    per_accept = int(config["spot_samples_per_acceptance_class"])
    for response_class in sorted(accepted_by_class):
        rows = sorted(accepted_by_class[response_class], key=lambda row: row["source_line"])
        sample_rng.shuffle(rows)
        for candidate in rows[:per_accept]:
            sample_rows.append(
                {
                    "sample_type": "accepted",
                    "stratum": response_class,
                    "source_line": candidate["source_line"],
                    "query": candidate["query"],
                    "response": candidate["response"],
                    "accepted_output": "clean" if candidate in clean_candidates else "heldout",
                    "rule_verification": "safe_refusal_or_clarification_and_no_hard_block",
                }
            )
    rejection_strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in rejected.values():
        rejection_strata[record["primary_reason"]].append(record)
    per_reject = int(config["spot_samples_per_rejection_stratum"])
    for reason in sorted(rejection_strata):
        rows = sorted(rejection_strata[reason], key=lambda row: row["source_line"])
        sample_rng.shuffle(rows)
        for record in rows[:per_reject]:
            sample_rows.append(
                {
                    "sample_type": "rejected",
                    "stratum": reason,
                    "source_line": record["source_line"],
                    "query": record["query"],
                    "response": record["response"],
                    "rejection_reason_codes": record["reason_codes"],
                    "rule_verification": "rejection_reason_present",
                }
            )
    sample_rows.sort(key=lambda row: (row["sample_type"], row["stratum"], row["source_line"]))
    with SAMPLE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Re-read the files that will be consumed downstream and verify every
    # retained assistant message/question, including the 20% held-out file.
    # This full scan is stronger than the stratified spot-check below and is
    # recorded as its own auditable artifact.
    retained_verification = verify_retained_outputs(config)

    # Quantify all exact/semantic intersections after the split, both against
    # the V5.1 mixed train and between the new clean and held-out outputs.
    def candidate_overlap(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], q_threshold: float, pair_threshold: float) -> Dict[str, Any]:
        right_q = {row["query_norm"] for row in right}
        right_pair = {row["pair_norm"] for row in right}
        exact_q = 0
        exact_pair = 0
        near_q = 0
        near_pair = 0
        matches: List[Dict[str, Any]] = []
        for row in left:
            if row["query_norm"] in right_q:
                exact_q += 1
            elif row["pair_norm"] in right_pair:
                exact_pair += 1
            found_q = False
            found_pair = False
            for other in right:
                q_score = jaccard(row["query_grams"], other["query_grams"])
                pair_score = jaccard(row["pair_grams"], other["pair_grams"])
                if q_score >= q_threshold:
                    found_q = True
                if pair_score >= pair_threshold:
                    found_pair = True
                if found_q or found_pair:
                    matches.append(
                        {
                            "left_source_line": row["source_line"],
                            "right_source_line": other["source_line"],
                            "query_jaccard": round(q_score, 6),
                            "pair_jaccard": round(pair_score, 6),
                        }
                    )
                    break
            if found_q:
                near_q += 1
            if found_pair:
                near_pair += 1
        return {
            "left_count": len(left),
            "right_count": len(right),
            "exact_query_count": exact_q,
            "exact_pair_count_excluding_exact_query": exact_pair,
            "semantic_query_count": near_q,
            "semantic_pair_count": near_pair,
            "any_intersection_count": len({m["left_source_line"] for m in matches}),
            "sample_matches": matches[:20],
            "query_threshold": q_threshold,
            "pair_threshold": pair_threshold,
        }

    clean_v_held = candidate_overlap(
        clean_candidates,
        heldout_candidates,
        float(config["self_query_jaccard_threshold"]),
        float(config["self_pair_jaccard_threshold"]),
    )
    # The references need the same source_line field as candidates.  Their
    # line numbers are local to their respective V5.1 files, which is exactly
    # what the report labels.
    clean_v51_ref = [dict(row, source_line=row["line"]) for row in v51_clean_refs]
    train_v51_ref = [dict(row, source_line=row["line"]) for row in v51_train_refs]
    new_clean_v51 = candidate_overlap(
        clean_candidates,
        train_v51_ref,
        float(config["v51_query_jaccard_threshold"]),
        float(config["v51_pair_jaccard_threshold"]),
    )
    new_heldout_v51 = candidate_overlap(
        heldout_candidates,
        train_v51_ref,
        float(config["v51_query_jaccard_threshold"]),
        float(config["v51_pair_jaccard_threshold"]),
    )
    new_clean_v51_clean = candidate_overlap(
        clean_candidates,
        clean_v51_ref,
        float(config["v51_query_jaccard_threshold"]),
        float(config["v51_pair_jaccard_threshold"]),
    )
    new_heldout_v51_clean = candidate_overlap(
        heldout_candidates,
        clean_v51_ref,
        float(config["v51_query_jaccard_threshold"]),
        float(config["v51_pair_jaccard_threshold"]),
    )

    # Spot check metric is explicit and machine-verifiable.
    rejected_sample_count = sum(1 for row in sample_rows if row["sample_type"] == "rejected")
    rejected_sample_with_reason = sum(
        1 for row in sample_rows if row["sample_type"] == "rejected" and row.get("rejection_reason_codes")
    )
    spot_check = {
        "seed": seed,
        "sampling": "deterministic stratified sample; up to configured rows per primary rejection reason and acceptance class",
        "rejected_sample_count": rejected_sample_count,
        "rejected_sample_with_reason_count": rejected_sample_with_reason,
        "rejection_rate": (rejected_sample_with_reason / rejected_sample_count) if rejected_sample_count else 1.0,
        "all_sampled_rejections_have_reason": rejected_sample_count == rejected_sample_with_reason,
        "retained_full_scan_count": retained_verification["total_retained_rows"],
        "retained_full_scan_pass_count": retained_verification["full_scan_pass_rows"],
        "retained_full_scan_fail_count": retained_verification["full_scan_fail_rows"],
        "retained_full_scan_pass_rate": retained_verification["full_scan_pass_rate"],
        "retained_sample_count": retained_verification["stratified_sample"]["sample_count"],
        "retained_sample_pass_count": retained_verification["stratified_sample"]["sample_pass_count"],
        "retained_sample_pass_rate": retained_verification["stratified_sample"]["sample_pass_rate"],
    }

    reason_counts = Counter()
    reason_all_counts = Counter()
    for record in rejected.values():
        reason_counts[record["primary_reason"]] += 1
        reason_all_counts.update(record["reason_codes"])
    output_counts = {
        "clean_rows": len(clean_candidates),
        "heldout_rows": len(heldout_candidates),
        "new_pool_rows": len(isolated_candidates),
        "new_pool_by_class": dict(Counter(row["class"] for row in isolated_candidates)),
        "split_by_class": split_counts,
    }

    dedup_report = {
        "source_path": str(RAW_PATH.resolve()),
        "input_rows": raw_counts["input_lines"],
        "safe_candidates_before_dedup": len(safe_candidates),
        "text_duplicate_removed": self_dedup_counts["text_duplicate_removed"],
        "semantic_near_duplicate_removed": self_dedup_counts["semantic_duplicate_removed"],
        "after_source_dedup": len(unique_candidates),
        "v5_1_exact_overlap_removed": v51_overlap_counts["v51_exact_removed"],
        "v5_1_semantic_near_overlap_removed": v51_overlap_counts["v51_semantic_removed"],
        "isolated_pool_rows": len(isolated_candidates),
        "thresholds": {
            "shingle_type": "compact Unicode NFKC/casefold character n-grams",
            "shingle_n": int(config["shingle_n"]),
            "self_query_jaccard": float(config["self_query_jaccard_threshold"]),
            "self_pair_jaccard": float(config["self_pair_jaccard_threshold"]),
            "v5_1_query_jaccard": float(config["v51_query_jaccard_threshold"]),
            "v5_1_pair_jaccard": float(config["v51_pair_jaccard_threshold"]),
        },
        "split": {
            "seed": seed,
            "heldout_fraction": float(config["heldout_fraction"]),
            "stratified_by": "response_class",
            "clean_rows": len(clean_candidates),
            "heldout_rows": len(heldout_candidates),
        },
        "post_split_zero_intersection": {
            "clean_vs_heldout": clean_v_held,
        },
    }
    write_json(DEDUP_PATH, dedup_report)

    overlap_report = {
        "baseline_clean_path": str(V51_CLEAN_PATH.resolve()),
        "baseline_clean_rows": len(v51_clean_refs),
        "baseline_mixed_train_path": str(V51_TRAIN_PATH.resolve()),
        "baseline_mixed_train_rows": len(v51_train_refs),
        "new_clean_vs_v5_1_mixed_train": new_clean_v51,
        "new_heldout_vs_v5_1_mixed_train": new_heldout_v51,
        "new_clean_vs_v5_1_clean": new_clean_v51_clean,
        "new_heldout_vs_v5_1_clean": new_heldout_v51_clean,
        "required_zero_intersection": {
            "new_clean_exact_query": new_clean_v51["exact_query_count"],
            "new_clean_exact_pair": new_clean_v51["exact_pair_count_excluding_exact_query"],
            "new_clean_semantic_query": new_clean_v51["semantic_query_count"],
            "new_clean_semantic_pair": new_clean_v51["semantic_pair_count"],
            "new_heldout_exact_query": new_heldout_v51["exact_query_count"],
            "new_heldout_exact_pair": new_heldout_v51["exact_pair_count_excluding_exact_query"],
            "new_heldout_semantic_query": new_heldout_v51["semantic_query_count"],
            "new_heldout_semantic_pair": new_heldout_v51["semantic_pair_count"],
            "new_clean_vs_heldout_any": clean_v_held["any_intersection_count"],
        },
    }
    write_json(OVERLAP_PATH, overlap_report)

    # Copy the exact rule document into the artifact folder using the same
    # bytes used for review.  The pipeline does not overwrite any V5.1 path.
    if RULES_PATH.exists():
        ARTIFACT_RULES_PATH.write_bytes(RULES_PATH.read_bytes())

    # Stats and evidence are written after all primary outputs exist.  Their
    # own hashes are added to the manifest below.
    statistics = {
        "config": {
            key: config[key]
            for key in (
                "seed",
                "heldout_fraction",
                "min_query_chars",
                "max_query_chars",
                "min_response_chars",
                "max_response_chars",
                "shingle_n",
                "self_query_jaccard_threshold",
                "self_pair_jaccard_threshold",
                "v51_query_jaccard_threshold",
                "v51_pair_jaccard_threshold",
                "spot_samples_per_rejection_stratum",
                "spot_samples_per_acceptance_class",
                "retention_samples_per_class",
                "medical_request_requires_explicit_refusal",
                "medical_request_disallows_clarification",
                "documentary_query_context_clarification_allowed",
                "reject_response_promises_and_external_operations",
            )
            if key in config
        },
        "source": {
            "raw_path": str(RAW_PATH.resolve()),
            "raw_rows": raw_counts["input_lines"],
            "raw_sha256": file_sha256(RAW_PATH),
            "v5_1_clean_path": str(V51_CLEAN_PATH.resolve()),
            "v5_1_clean_rows": len(v51_clean_refs),
            "v5_1_clean_sha256": file_sha256(V51_CLEAN_PATH),
            "v5_1_mixed_train_path": str(V51_TRAIN_PATH.resolve()),
            "v5_1_mixed_train_rows": len(v51_train_refs),
            "v5_1_mixed_train_sha256": file_sha256(V51_TRAIN_PATH),
        },
        "filter_policy": {
            "accepted_response_classes": ["safety_refusal", "clarification"],
            "authority_policy": "source has no per-record auditable authority citation; factual medical claims are unverified and rejected",
            "direct_prescription_policy": "reject formula/herb/drug/treatment/dose/instruction content",
            "chain_policy": "responses exposing reasoning are rejected",
        },
        "raw_counts": dict(raw_counts),
        "rejection_counts_primary": dict(sorted(reason_counts.items())),
        "rejection_counts_all_reason_codes": dict(sorted(reason_all_counts.items())),
        "outputs": output_counts,
        "spot_check": spot_check,
        "retention_verification": {
            "path": str(RETAINED_VERIFY_PATH.resolve()),
            "total_retained_rows": retained_verification["total_retained_rows"],
            "full_scan_pass_rows": retained_verification["full_scan_pass_rows"],
            "full_scan_fail_rows": retained_verification["full_scan_fail_rows"],
            "full_scan_pass_rate": retained_verification["full_scan_pass_rate"],
            "stratified_sample_count": retained_verification["stratified_sample"]["sample_count"],
            "stratified_sample_pass_count": retained_verification["stratified_sample"]["sample_pass_count"],
            "stratified_sample_pass_rate": retained_verification["stratified_sample"]["sample_pass_rate"],
        },
        "overlap_zero_verified": all(
            value == 0
            for value in (
                new_clean_v51["exact_query_count"],
                new_clean_v51["exact_pair_count_excluding_exact_query"],
                new_clean_v51["semantic_query_count"],
                new_clean_v51["semantic_pair_count"],
                new_heldout_v51["exact_query_count"],
                new_heldout_v51["exact_pair_count_excluding_exact_query"],
                new_heldout_v51["semantic_query_count"],
                new_heldout_v51["semantic_pair_count"],
                clean_v_held["any_intersection_count"],
            )
        ),
        "retained_full_scan_safe": retained_verification["full_scan_fail_rows"] == 0,
    }
    write_json(STATS_PATH, statistics)

    # Manifest all material output files except the manifest itself.  Paths
    # are absolute to make the archive/review handoff unambiguous.
    manifest_targets = [
        CLEAN_PATH,
        HELDOUT_PATH,
        REJECTION_PATH,
        SAMPLE_PATH,
        ARTIFACT_RULES_PATH,
        STATS_PATH,
        DEDUP_PATH,
        OVERLAP_PATH,
        RETAINED_VERIFY_PATH,
    ]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for path in manifest_targets:
            handle.write(f"{file_sha256(path)}  {path.resolve()}\n")

    run_evidence = {
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "config_path": str(CONFIG_PATH.resolve()),
        "rules_path": str(RULES_PATH.resolve()),
        "command": f"python3 {Path(__file__).resolve()}",
        "seed": seed,
        "inputs": {
            "raw": {"path": str(RAW_PATH.resolve()), "rows": raw_counts["input_lines"], "sha256": file_sha256(RAW_PATH)},
            "v5_1_clean": {"path": str(V51_CLEAN_PATH.resolve()), "rows": len(v51_clean_refs), "sha256": file_sha256(V51_CLEAN_PATH)},
            "v5_1_mixed_train": {"path": str(V51_TRAIN_PATH.resolve()), "rows": len(v51_train_refs), "sha256": file_sha256(V51_TRAIN_PATH)},
        },
        "outputs": {
            "clean": {"path": str(CLEAN_PATH.resolve()), "rows": len(clean_candidates), "sha256": file_sha256(CLEAN_PATH)},
            "heldout": {"path": str(HELDOUT_PATH.resolve()), "rows": len(heldout_candidates), "sha256": file_sha256(HELDOUT_PATH)},
            "rejections": {"path": str(REJECTION_PATH.resolve()), "rows": len(rejected), "sha256": file_sha256(REJECTION_PATH)},
            "stratified_samples": {"path": str(SAMPLE_PATH.resolve()), "rows": len(sample_rows), "sha256": file_sha256(SAMPLE_PATH)},
            "statistics": {"path": str(STATS_PATH.resolve()), "sha256": file_sha256(STATS_PATH)},
            "dedup_report": {"path": str(DEDUP_PATH.resolve()), "sha256": file_sha256(DEDUP_PATH)},
            "overlap_report": {"path": str(OVERLAP_PATH.resolve()), "sha256": file_sha256(OVERLAP_PATH)},
            "retention_verification": {"path": str(RETAINED_VERIFY_PATH.resolve()), "sha256": file_sha256(RETAINED_VERIFY_PATH)},
            "rules": {"path": str(ARTIFACT_RULES_PATH.resolve()), "sha256": file_sha256(ARTIFACT_RULES_PATH)},
            "manifest": {"path": str(MANIFEST_PATH.resolve()), "sha256": file_sha256(MANIFEST_PATH)},
        },
        "verification": {
            "retained_total_rows": retained_verification["total_retained_rows"],
            "retained_full_scan_pass_rows": retained_verification["full_scan_pass_rows"],
            "retained_full_scan_fail_rows": retained_verification["full_scan_fail_rows"],
            "retained_full_scan_pass_rate": retained_verification["full_scan_pass_rate"],
            "direct_prescription_unapproved_rows": retained_verification["direct_prescription_unapproved_rows"],
            "unverified_claim_unapproved_rows": retained_verification["unverified_claim_unapproved_rows"],
            "promise_or_operation_unapproved_rows": retained_verification[
                "promise_or_operation_unapproved_rows"
            ],
            "medical_request_rows": retained_verification["medical_request_rows"],
            "medical_request_without_explicit_refusal_rows": retained_verification[
                "medical_request_without_explicit_refusal_rows"
            ],
            "retained_stratified_sample_count": retained_verification["stratified_sample"]["sample_count"],
            "retained_stratified_sample_pass_count": retained_verification["stratified_sample"]["sample_pass_count"],
            "retained_stratified_sample_pass_rate": retained_verification["stratified_sample"]["sample_pass_rate"],
        },
        "implementation": {
            "script": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
            "config": {"path": str(CONFIG_PATH.resolve()), "sha256": file_sha256(CONFIG_PATH)},
            "rules": {"path": str(RULES_PATH.resolve()), "sha256": file_sha256(RULES_PATH)},
        },
        "invariants": {
            "raw_rows_accounted_exactly_once": len(accounted_lines) == raw_counts["input_lines"],
            "clean_heldout_zero_exact_and_semantic_intersection": clean_v_held["any_intersection_count"] == 0,
            "v5_1_zero_exact_and_semantic_intersection": statistics["overlap_zero_verified"],
            "spot_check_rejection_rate_100_percent": spot_check["rejection_rate"] == 1.0,
            "retained_full_scan_safe": statistics["retained_full_scan_safe"],
            "retained_stratified_sample_safe": retained_verification["stratified_sample"]["sample_pass_rate"] == 1.0,
            "v5_1_paths_read_only": True,
        },
    }
    write_json(EVIDENCE_PATH, run_evidence)

    # The evidence hash itself is not in the manifest because including the
    # manifest's hash in itself is not well-defined.  Print a compact handoff
    # for the parent agent and human review.
    print(json.dumps({
        "raw_rows": raw_counts["input_lines"],
        "safe_candidates": len(safe_candidates),
        "clean_rows": len(clean_candidates),
        "heldout_rows": len(heldout_candidates),
        "rejected_rows": len(rejected),
        "rejection_primary": dict(sorted(reason_counts.items())),
        "v51_removed_exact": v51_overlap_counts["v51_exact_removed"],
        "v51_removed_semantic": v51_overlap_counts["v51_semantic_removed"],
        "overlap_zero_verified": statistics["overlap_zero_verified"],
        "clean_sha256": file_sha256(CLEAN_PATH),
        "heldout_sha256": file_sha256(HELDOUT_PATH),
        "rejections_sha256": file_sha256(REJECTION_PATH),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"v5.2 data pipeline failed: {exc}", file=sys.stderr)
        raise
