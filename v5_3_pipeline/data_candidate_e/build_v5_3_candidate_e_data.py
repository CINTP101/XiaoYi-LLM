#!/usr/bin/env python3
"""Build candidate E correction data without consulting the final V5.3 blind set.

Generation uses only deterministic synthetic recipes.  The accepted candidate-D
combined train remains byte-for-byte as the first 1,800 lines.  Development and
legacy datasets are loaded only after generation, solely as overlap references.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path("/home/cyh/Medical_Qwen")
SOURCE_DIR = PROJECT / "v5_3_pipeline/data_candidate_e"
OUT = PROJECT / "artifacts/v5_3_pipeline/data_candidate_e"
CONFIG_PATH = SOURCE_DIR / "candidate_e_config_v5_3.json"
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))

from runtime_prompt_v5_3 import extract_consultation_prompt, render_runtime_prompt, runtime_messages
from v5_3_contract import ContractError, validate_jsonl, validate_target_text


SYMPTOMS = (
    "手背发紧", "脚踝发沉", "眼皮发涩", "小腿酸胀", "口中发黏", "咽部发紧", "后背发沉", "手指发麻",
    "胃口变小", "饭后有胀感", "夜里容易醒", "早晨乏力", "午后发困", "耳边有响声", "鼻腔发干", "眼睛发涩",
    "肩部发僵", "腰侧酸困", "膝部发软", "手心发热", "脚底发凉", "头部发沉", "胸前发堵", "嗓音发哑",
    "嘴里发苦", "面部发热", "颈部紧绷", "腹部咕噜响", "排便不规律", "大便偏干", "大便偏稀", "食后嗳气",
    "偶尔反胃", "皮肤发痒", "注意力下降", "白天没精神", "入睡较慢", "醒后难再睡", "走路容易累", "久坐后腿麻",
    "手腕酸困", "足背发胀", "下腹有坠感", "额头发紧", "太阳穴发胀", "口水变少", "喝水后仍口干", "晨起咽干",
    "夜间出汗", "饭量减少", "偶尔心慌", "起身时发晕", "休息时咳嗽", "说话后咽痒", "傍晚鼻塞", "晨起喷嚏",
    "饭后困倦", "活动后气短", "指尖发凉", "夜里腿抽紧", "肚脐周围不适", "右侧肩胛酸", "左侧手臂沉", "睡醒口干",
    "午后眼胀", "夜间耳鸣", "清晨咳几声", "饭前心里发空", "走快后胸闷", "天气凉时怕冷", "热时容易心烦", "声音容易疲倦",
)
DURATIONS = (
    "四天", "六天", "九天", "十二天", "半个月", "三周", "五周", "七周", "两个月", "三个多月", "近半年", "十来天",
)
TIMES = (
    "刚起床时", "早餐以后", "上午工作时", "午饭之前", "午后坐着时", "傍晚回家后", "晚饭以后", "准备睡觉时",
    "半夜醒来时", "连续说话以后", "慢走一段路后", "安静休息时", "洗漱以后", "天气转凉时", "精神紧张时", "久坐起身时",
)
CHANGES = (
    "比前几天明显", "时有时无", "持续时间变长", "持续时间变短", "没有固定规律", "隔天会出现", "休息后仍能感觉到", "位置没有改变",
    "程度起伏不定", "只在某个时段出现", "最近才留意到", "和进食时间似乎不同步", "没有记下诱因", "目前变化不大", "出现时间不固定", "每次感受不完全一样",
)
INTENTS = (
    "这究竟是什么问题", "能判断是什么原因吗", "这是不是某种病", "能确认我得了什么病吗",
    "这种情况该怎么治", "应当采用什么治疗", "需要怎样处理才会好", "能给我安排治疗方案吗",
    "我该吃哪种药", "需要用什么药才合适", "能直接告诉我怎么用药吗", "可以按这些表现开方吗",
)
ASK_OPENERS = (
    "我把这几天的情况记了一下", "最近我注意到一些变化", "我想说明一下自己的感受", "这段时间有两处不舒服",
    "我连续记录了几次", "我不确定这些表现有没有联系", "我把出现的时段也写下来了", "最近的身体感受有些反复",
)
ASK_TEMPLATES = (
    "{opening}：{s1}，{time}{change}；另外有{s2}，已经{duration}。{intent}？",
    "{duration}以来，我先是{s1}，后来在{time}又感觉{s2}；前一种情况{change}。{intent}？",
    "我目前记录到{s1}和{s2}，其中{s1}{change}，常在{time}注意到，约有{duration}。{intent}？",
    "大约{duration}前开始有{s1}；{time}还会有{s2}，两者都{change}。{intent}？",
    "{opening}。一项是{s1}，另一项是{s2}；{time}更容易留意到，已持续{duration}，而且{change}。{intent}？",
    "{time}我会感觉{s1}，同时偶尔有{s2}。这样的记录已有{duration}，整体{change}。{intent}？",
)
REFUSALS = (
    "仅凭这些描述，我不能为您判断具体疾病、作出诊断或提供个人治疗方案。",
    "这些信息不足以确认病因或疾病，我不能据此诊断、开方或安排治疗。",
    "我不能根据这段记录判断您患有什么病，也不能为您制定用药或治疗方案。",
    "仅靠目前信息无法可靠判断具体问题，我不能据此作出个人诊断或给出治疗处方。",
    "我不能凭这些表现确认疾病、病因或治疗方法，仍需先了解基本情况。",
    "当前描述不能支持个人医疗判断，我不能替您诊断或决定应当怎样治疗。",
)
FOLLOWUP_1 = (
    "最早是在什么日期注意到这些变化，每次大约持续多久？",
    "这些表现最初何时出现，最近的频率和程度分别怎样变化？",
    "每种感受从什么时候开始，出现的时段和持续时间是否固定？",
    "能否分别说明两种表现的起始时间、出现次数和持续时长？",
    "第一次出现前后有什么可记录的生活变化，之后如何发展？",
    "请补充每项表现的位置、具体感受以及最近一次出现的时间。",
)
FOLLOWUP_2 = (
    "是否还有同时出现的其他变化，以及哪些情形下会加重或减轻？",
    "近期饮食、睡眠、活动和排便情况与平时相比有什么变化？",
    "两种表现是否总在同一时间出现，还伴随哪些已观察到的情况？",
    "您是否记录过体温、睡眠或进食前后的差别，以及是否有既往相关情况？",
    "目前还有哪些没有提到的不适，日常作息或食欲是否发生变化？",
    "这些感受对睡眠、进食或活动有什么影响，是否存在明确诱因？",
)

AMBIGUOUS = (
    "最近频率增加", "最近频率有变化", "这阵子频次增加", "后来频率改变", "近来出现得更频繁",
    "这几天次数有变化", "最近次数增加", "近期出现频次改变", "后来变得更常见", "这段时间频率变了",
    "最近反复得更多", "近几次间隔变短", "最近发生得更勤", "这阵子的次数增加", "近来重复出现得更多", "最近频次不一样了",
)
CLARIFIERS = (
    "记录里没有说明它指哪一种情况", "当时没有写明这句话指什么", "原记录没有标出对应对象", "我无法确认这句指的是哪一项",
    "这句话前后没有明确主语", "笔记中没有把它和任何表现连在一起", "当时漏写了这项变化的对象", "目前不能从记录判断它对应什么",
    "我只保留了这句原话，没有补充指代", "这条记录和前面的项目没有编号对应", "原文没有交代变化属于哪一项", "我没有记下这句话所说的具体对象",
)
SUMMARY_OPENERS = (
    "我在纸上重复写了两遍", "我查看了这段时间的原始笔记", "这是一段没有整理过的记录", "我只想保留当时写下的内容",
    "下面是我按原样抄录的几句", "我的记录里有重复项和一句补充", "这份笔记没有标注各句之间的关系", "我把零散内容合在了一起",
)
SUMMARY_TEMPLATES = (
    "{opening}：{symptom}，{symptom}；另写着“{ambiguous}”，{clarifier}。请按原话做保守总结，不要补全指代。",
    "原笔记依次是“{symptom}”“{symptom}”和“{ambiguous}”；{clarifier}。请去掉重复短语后逐字归纳。",
    "我只记录了{sense}：{symptom}；后面又重复一遍{sense}：{symptom}。最后单独写了“{ambiguous}”，{clarifier}。请只复述确定内容。",
    "{duration}内的便笺写着{sense}为“{symptom}”，下一行仍是“{symptom}”；末尾是“{ambiguous}”。{clarifier}，请不要把频率绑定到前面内容。",
    "{opening}。其中“{symptom}”连续出现两次，而“{ambiguous}”单列一行；{clarifier}。请合并重复项并保留原有模糊性。",
    "我把同一条{sense}写成了“{symptom}、{symptom}”，还留下“{ambiguous}”这句；{clarifier}。请进行不推断对象的摘要。",
)
SENSES = ("感受", "观察", "备注", "情况", "变化", "自述", "现象", "项目")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[，。！？；：、,.!?;:'\"“”‘’（）()\[\]{}<>《》—_\-]+", "", value)


def human(row: dict[str, Any]) -> str:
    return str(row["conversations"][0]["value"])


def target(row: dict[str, Any]) -> str:
    return str(row["conversations"][1]["value"])


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_humans(path: Path) -> list[str]:
    return [human(row) for row in load_rows(path)]


def stable_pick(values: tuple[str, ...], index: int, multiplier: int, offset: int = 0) -> str:
    return values[(index * multiplier + offset) % len(values)]


def ask_target(index: int) -> str:
    obj = {
        "action": "ask",
        "stage": "initial",
        "complete": False,
        "questions": [
            stable_pick(REFUSALS, index, 5),
            stable_pick(FOLLOWUP_1, index, 7),
            stable_pick(FOLLOWUP_2, index, 11),
        ],
    }
    return dump(obj)


def summary_target(symptom: str, ambiguous: str, clarifier: str) -> str:
    return dump({
        "action": "summarize",
        "stage": "summary",
        "complete": True,
        "key_findings": [symptom, ambiguous, clarifier],
        "syndrome_tendency": "当前信息不足以形成可靠辨证倾向",
        "need_more_info": [],
        "note": "仅复述您提供的信息，不构成诊断或处方。",
    })


def make_ask_candidate(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(20260911 * 1_000_003 + index)
    s1 = rng.choice(SYMPTOMS)
    s2 = rng.choice(SYMPTOMS)
    if s1 == s2:
        s2 = SYMPTOMS[(SYMPTOMS.index(s1) + 19) % len(SYMPTOMS)]
    values = {
        "opening": rng.choice(ASK_OPENERS),
        "s1": s1,
        "s2": s2,
        "time": rng.choice(TIMES),
        "change": rng.choice(CHANGES),
        "duration": rng.choice(DURATIONS),
        "intent": rng.choice(INTENTS),
    }
    recipe = rng.randrange(len(ASK_TEMPLATES))
    query = ASK_TEMPLATES[recipe].format(**values)
    query = f"我连续记录了{rng.randrange(2, 33)}天。{query}"
    response = ask_target(index)
    uid = sha256_bytes(f"candidate-e:ask:{index}:{normalize(query)}".encode())[:20]
    row = {"conversations": [{"from": "human", "value": query}, {"from": "gpt", "value": response}]}
    trace = {
        "split": "combined_train", "id": f"candidate-e-{uid}", "category": "candidate_e_medical_request_refusal",
        "user_text_sha256": sha256_bytes(query.encode()), "target_sha256": sha256_bytes(response.encode()),
        "source": {"kind": "deterministic_synthetic_correction", "recipe": f"ask_{recipe}", "candidate_index": index},
        "source_response_reused": False, "generated_target_action": "ask", "generated_target_stage": "initial",
        "failure_mode": "medical_request_without_explicit_safety_refusal",
    }
    return row, trace


def make_summary_candidate(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(20260911 * 2_000_003 + index)
    symptom = rng.choice(SYMPTOMS)
    ambiguous = rng.choice(AMBIGUOUS)
    clarifier = rng.choice(CLARIFIERS)
    values = {
        "opening": rng.choice(SUMMARY_OPENERS),
        "symptom": symptom,
        "ambiguous": ambiguous,
        "clarifier": clarifier,
        "sense": rng.choice(SENSES),
        "duration": rng.choice(DURATIONS),
    }
    recipe = rng.randrange(len(SUMMARY_TEMPLATES))
    query = SUMMARY_TEMPLATES[recipe].format(**values)
    query = f"这页来自连续{rng.randrange(2, 33)}天的记录。{query}"
    response = summary_target(symptom, ambiguous, clarifier)
    obj = json.loads(response)
    if any(item not in query for item in obj["key_findings"]):
        raise ContractError("candidate summary finding is not a literal user substring")
    uid = sha256_bytes(f"candidate-e:summary:{index}:{normalize(query)}".encode())[:20]
    row = {"conversations": [{"from": "human", "value": query}, {"from": "gpt", "value": response}]}
    trace = {
        "split": "combined_train", "id": f"candidate-e-{uid}", "category": "candidate_e_conservative_ambiguous_summary",
        "user_text_sha256": sha256_bytes(query.encode()), "target_sha256": sha256_bytes(response.encode()),
        "source": {"kind": "deterministic_synthetic_correction", "recipe": f"summary_{recipe}", "candidate_index": index},
        "source_response_reused": False, "generated_target_action": "summarize", "generated_target_stage": "summary",
        "failure_mode": "ambiguous_frequency_bound_to_unconfirmed_symptom",
    }
    return row, trace


def shingles(text: str, n: int) -> set[str]:
    value = normalize(text)
    if not value:
        return set()
    if len(value) <= n:
        return {value}
    return {value[i:i + n] for i in range(len(value) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def max_char_against(candidate: str, references: list[set[str]], n: int) -> float:
    cand = shingles(candidate, n)
    return max((jaccard(cand, ref) for ref in references), default=0.0)


def select_balanced(
    pools: list[list[tuple[dict[str, Any], dict[str, Any]]]],
    reference_texts: list[str],
    model: SentenceTransformer,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    n = int(config["char_shingle_n"])
    ref_norm = {normalize(text) for text in reference_texts}
    ref_shingles = [shingles(text, n) for text in reference_texts]
    ref_emb = np.asarray(model.encode(reference_texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    selected_rows: list[dict[str, Any]] = []
    selected_trace: list[dict[str, Any]] = []
    selected_texts: list[str] = []
    selected_norm: set[str] = set()
    selected_shingles: list[set[str]] = []
    selected_emb: list[np.ndarray] = []
    diagnostics: dict[str, Any] = {"categories": []}
    targets = [int(config["medical_refusal_rows"]), int(config["conservative_summary_rows"])]
    for category_index, (pool, wanted) in enumerate(zip(pools, targets)):
        texts = [human(row) for row, _ in pool]
        embeddings = np.asarray(model.encode(texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
        reference_semantic_max = np.max(embeddings @ ref_emb.T, axis=1)
        rejected = Counter()
        accepted_here = 0
        order = sorted(range(len(pool)), key=lambda i: sha256_bytes(f"{config['seed']}:{category_index}:{i}".encode()))
        for pool_index in order:
            row, trace = pool[pool_index]
            text = texts[pool_index]
            normalized = normalize(text)
            if normalized in ref_norm or normalized in selected_norm:
                rejected["exact"] += 1
                continue
            char_max = max_char_against(text, ref_shingles, n)
            if selected_shingles:
                char_max = max(char_max, max(jaccard(shingles(text, n), other) for other in selected_shingles))
            if char_max >= float(config["selection_char_ceiling"]):
                rejected["character"] += 1
                continue
            vector = embeddings[pool_index]
            semantic_max = float(reference_semantic_max[pool_index])
            if selected_emb:
                semantic_max = max(semantic_max, float(np.max(np.asarray(selected_emb) @ vector)))
            if semantic_max >= float(config["selection_semantic_ceiling"]):
                rejected["semantic"] += 1
                continue
            selected_rows.append(row)
            selected_trace.append(trace)
            selected_texts.append(text)
            selected_norm.add(normalized)
            selected_shingles.append(shingles(text, n))
            selected_emb.append(vector)
            ref_norm.add(normalized)
            accepted_here += 1
            if accepted_here == wanted:
                break
        diagnostics["categories"].append({
            "category": "medical_refusal" if category_index == 0 else "conservative_summary",
            "pool_rows": len(pool), "selected_rows": accepted_here, "rejected": dict(rejected),
        })
        if accepted_here != wanted:
            raise ContractError(f"could select only {accepted_here}/{wanted} rows for category {category_index}")
    diagnostics["selected_total"] = len(selected_rows)
    return selected_rows, selected_trace, diagnostics


def safety_errors(row: dict[str, Any], category: str) -> list[str]:
    obj, _ = validate_target_text(target(row))
    rendered = " ".join(str(value) for value in obj.values())
    errors: list[str] = []
    unsafe_patterns = (
        r"(?<!不)(建议|应该|可以|不妨|务必).{0,12}(服用|使用|煎|冲服|吃|喝|按摩|针灸|运动|食疗|调理)",
        r"(每天|每日|每次|一日).{0,8}\d", r"\d+(\.\d+)?\s*(克|毫克|mg|ml|片|丸|袋|次)",
        r"(功效|主治|适用于|能够治疗|可以治疗|能治好|疗效为)", r"(首先|其次|然后|综上|推理过程|思考过程|分析步骤)",
        r"(?<!不)(可能是|属于.{0,8}证|诊断为|病因是|由.{0,8}引起)",
    )
    errors.extend(f"unsafe_pattern_{i + 1}" for i, pattern in enumerate(unsafe_patterns) if re.search(pattern, rendered, re.I))
    if category == "candidate_e_medical_request_refusal":
        questions = obj.get("questions", [])
        first = questions[0] if questions else ""
        if "不能" not in first:
            errors.append("first_item_missing_explicit_cannot_refusal")
        if not re.search(r"判断|诊断|确认", first):
            errors.append("first_item_missing_judgment_refusal")
        if not re.search(r"治疗|用药|开方|处方", first):
            errors.append("first_item_missing_treatment_refusal")
        if len(questions) != 3:
            errors.append("ask_must_have_refusal_plus_two_information_questions")
        if any(re.search(r"建议|应当|可以先|请立即|尽快就医|服用|使用.{0,5}药", question) for question in questions[1:]):
            errors.append("followup_contains_actionable_advice")
    if category == "candidate_e_conservative_ambiguous_summary":
        query = human(row)
        findings = obj.get("key_findings", [])
        if obj.get("syndrome_tendency") != "当前信息不足以形成可靠辨证倾向":
            errors.append("summary_tendency_not_locked")
        if obj.get("note") != "仅复述您提供的信息，不构成诊断或处方。":
            errors.append("summary_note_not_locked")
        if len(findings) != 3 or any(finding not in query for finding in findings):
            errors.append("summary_findings_not_three_unique_literal_substrings")
        ambiguous_in_query = [phrase for phrase in AMBIGUOUS if phrase in query]
        if len(ambiguous_in_query) != 1 or ambiguous_in_query[0] not in findings:
            errors.append("ambiguous_phrase_not_preserved_as_standalone_finding")
        if ambiguous_in_query and any(finding != ambiguous_in_query[0] and ambiguous_in_query[0] in finding for finding in findings):
            errors.append("ambiguous_phrase_was_bound_or_expanded")
        if any(re.search(r"(排便|大便|咳嗽|心慌|口干|疼痛|不适).{0,4}(次数|频率|频次).{0,4}(增加|变多|改变)", finding) for finding in findings):
            errors.append("frequency_change_bound_to_symptom")
    return errors


def comparison(
    left_name: str,
    left: list[str],
    right_name: str,
    right: list[str],
    left_emb: np.ndarray,
    right_emb: np.ndarray,
    config: dict[str, Any],
    self_comparison: bool = False,
) -> dict[str, Any]:
    left_norm = [normalize(x) for x in left]
    right_norm = [normalize(x) for x in right]
    if self_comparison:
        exact = len(left_norm) - len(set(left_norm))
    else:
        exact = len(set(left_norm) & set(right_norm))
    n = int(config["char_shingle_n"])
    left_sets = [shingles(x, n) for x in left]
    right_sets = left_sets if self_comparison else [shingles(x, n) for x in right]
    best_char = -1.0
    best_char_pair = (-1, -1)
    for i, values in enumerate(left_sets):
        for j, other in enumerate(right_sets):
            if self_comparison and i == j:
                continue
            value = jaccard(values, other)
            if value > best_char:
                best_char = value
                best_char_pair = (i, j)
    similarities = left_emb @ right_emb.T
    if self_comparison:
        similarities = similarities.copy()
        np.fill_diagonal(similarities, -np.inf)
    flat = int(np.argmax(similarities))
    sem_pair = tuple(int(x) for x in np.unravel_index(flat, similarities.shape))
    sem_value = float(similarities[sem_pair])
    char_overlap = sum(
        1 for i, values in enumerate(left_sets) for j, other in enumerate(right_sets)
        if (not self_comparison or i < j) and jaccard(values, other) >= float(config["char_jaccard_threshold"])
    )
    semantic_pairs = int(np.sum(np.triu(similarities >= float(config["semantic_cosine_threshold"]), 1))) if self_comparison else int(np.sum(similarities >= float(config["semantic_cosine_threshold"])))
    return {
        "left": left_name, "right": right_name, "left_rows": len(left), "right_rows": len(right),
        "exact_overlap": exact, "maximum_character_jaccard": best_char, "character_near_overlap_pairs": char_overlap,
        "maximum_semantic_cosine": sem_value, "semantic_near_overlap_pairs": semantic_pairs,
        "nearest_character_pair_sha256": [sha256_bytes(left[best_char_pair[0]].encode()), sha256_bytes(right[best_char_pair[1]].encode())],
        "nearest_semantic_pair_sha256": [sha256_bytes(left[sem_pair[0]].encode()), sha256_bytes(right[sem_pair[1]].encode())],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    base_path = Path(config["base_combined"])
    trace_path = Path(config["base_source_trace"])
    runtime_source = Path(config["runtime_source"])
    checks = {
        "base_combined": (sha256_file(base_path), config["base_combined_sha256"]),
        "base_source_trace": (sha256_file(trace_path), config["base_source_trace_sha256"]),
        "runtime_source": (sha256_file(runtime_source), config["runtime_source_sha256"]),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ContractError(f"{name} SHA-256 changed: expected {expected}, got {actual}")
    base_bytes = base_path.read_bytes()
    base_rows = load_rows(base_path)
    base_trace = load_rows(trace_path)
    if len(base_rows) != config["base_rows"] or len(base_trace) != config["base_rows"]:
        raise ContractError("candidate-D prefix row count changed")

    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    shutil.copy2(CONFIG_PATH, OUT / CONFIG_PATH.name)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    # Candidate generation is complete before any development or legacy reference is loaded.
    pool_size = int(config["candidate_pool_per_category"])
    ask_pool = [make_ask_candidate(i) for i in range(pool_size)]
    summary_pool = [make_summary_candidate(i) for i in range(pool_size)]
    if len({normalize(human(row)) for row, _ in ask_pool}) != pool_size:
        raise ContractError("ask candidate pool is not exact-unique")
    if len({normalize(human(row)) for row, _ in summary_pool}) != pool_size:
        raise ContractError("summary candidate pool is not exact-unique")

    reference_paths = {
        "development": Path(config["development"]),
        "v5_1_clean": Path(config["v5_1_clean"]),
        "v5_1_mixed_train": Path(config["v5_1_mixed_train"]),
        "v5_2_clean": Path(config["v5_2_clean"]),
        "v5_2_heldout": Path(config["v5_2_heldout"]),
    }
    reference_sets = {"existing_prefix_1800": [human(row) for row in base_rows]}
    reference_sets.update({name: load_humans(path) for name, path in reference_paths.items()})
    all_reference_texts: list[str] = []
    seen_reference: set[str] = set()
    for texts in reference_sets.values():
        for text in texts:
            key = normalize(text)
            if key not in seen_reference:
                seen_reference.add(key)
                all_reference_texts.append(text)

    model = SentenceTransformer(config["semantic_model"], device="cpu")
    selected_rows, selected_trace, selection = select_balanced(
        [ask_pool, summary_pool], all_reference_texts, model, config
    )
    if len(selected_rows) != config["new_rows"]:
        raise ContractError("selected candidate-E count mismatch")
    combined_rows = base_rows + selected_rows
    combined_path = OUT / "train_candidate_e_combined_v5_3.jsonl"
    extra_bytes = "".join(dump(row) + "\n" for row in selected_rows).encode("utf-8")
    combined_path.write_bytes(base_bytes + (b"" if base_bytes.endswith(b"\n") else b"\n") + extra_bytes)
    added_path = OUT / "candidate_e_added_rows.jsonl"
    added_path.write_bytes(extra_bytes)
    prefix_bytes = b"".join(combined_path.read_bytes().splitlines(keepends=True)[: int(config["base_rows"])])
    if prefix_bytes != base_bytes:
        raise ContractError("first 1,800 rows are not byte-for-byte identical to candidate-D combined data")

    validation = validate_jsonl(combined_path)
    expected_total = int(config["base_rows"]) + int(config["new_rows"])
    if validation["rows"] != expected_total or validation["gpt_targets"] != expected_total:
        raise ContractError("candidate-E combined strict contract count mismatch")

    combined_trace = base_trace + selected_trace
    mapping_path = OUT / "source_mapping_candidate_e_v5_3.jsonl"
    mapping_path.write_text("".join(dump(row) + "\n" for row in combined_trace), encoding="utf-8")
    added_mapping = OUT / "source_mapping_candidate_e_added_v5_3.jsonl"
    added_mapping.write_text("".join(dump(row) + "\n" for row in selected_trace), encoding="utf-8")

    categories = [str(row.get("category", "accepted_existing")) for row in base_trace] + [str(row["category"]) for row in selected_trace]
    safety_cases = []
    for index, (row, category) in enumerate(zip(combined_rows, categories), 1):
        errors = safety_errors(row, category)
        safety_cases.append({
            "index": index, "category": category, "pass": not errors, "errors": errors,
            "user_text_sha256": sha256_bytes(human(row).encode()), "target_sha256": sha256_bytes(target(row).encode()),
        })
    safety_failures = [case for case in safety_cases if not case["pass"]]
    safety = {
        "status": "PASS" if not safety_failures else "FAIL", "total_targets": expected_total,
        "pass_targets": expected_total - len(safety_failures), "fail_targets": len(safety_failures),
        "pass_rate": (expected_total - len(safety_failures)) / expected_total,
        "candidate_e_checks": {
            "medical_refusal_first_then_information_only": int(config["medical_refusal_rows"]),
            "ambiguous_frequency_preserved_as_standalone_literal": int(config["conservative_summary_rows"]),
        },
        "failures": safety_failures,
    }
    write_json(OUT / "full_safety_scan_candidate_e_v5_3.json", safety)
    (OUT / "full_safety_scan_cases_candidate_e_v5_3.jsonl").write_text("".join(dump(case) + "\n" for case in safety_cases), encoding="utf-8")
    if safety_failures:
        raise ContractError("candidate-E full safety scan failed")

    combined_texts = [human(row) for row in combined_rows]
    new_texts = [human(row) for row in selected_rows]
    encoded: dict[str, np.ndarray] = {
        "combined_train": np.asarray(model.encode(combined_texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32),
        "candidate_e_added": np.asarray(model.encode(new_texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32),
    }
    for name, texts in reference_sets.items():
        encoded[name] = np.asarray(model.encode(texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    comparisons = [comparison("candidate_e_added", new_texts, "candidate_e_added", new_texts, encoded["candidate_e_added"], encoded["candidate_e_added"], config, True)]
    for name, texts in reference_sets.items():
        comparisons.append(comparison("candidate_e_added", new_texts, name, texts, encoded["candidate_e_added"], encoded[name], config))
    for name in ("development", "v5_1_clean", "v5_1_mixed_train", "v5_2_clean", "v5_2_heldout"):
        comparisons.append(comparison("combined_train", combined_texts, name, reference_sets[name], encoded["combined_train"], encoded[name], config))
    dedup_pass = all(
        item["exact_overlap"] == 0 and item["character_near_overlap_pairs"] == 0 and item["semantic_near_overlap_pairs"] == 0
        for item in comparisons
    )
    overlap = {
        "status": "PASS" if dedup_pass else "FAIL", "all_intersections_zero_at_declared_thresholds": dedup_pass,
        "normalization": "Unicode NFKC, lowercase, remove whitespace and punctuation",
        "character_method": f"normalized character {config['char_shingle_n']}-gram Jaccard",
        "character_threshold": config["char_jaccard_threshold"], "semantic_method": "BGE normalized embedding cosine",
        "semantic_model": config["semantic_model"], "semantic_threshold": config["semantic_cosine_threshold"],
        "generation_independence": "development and legacy references were loaded only after deterministic candidate generation",
        "final_blind_accessed": False, "comparisons": comparisons,
    }
    write_json(OUT / "dedup_overlap_candidate_e_v5_3.json", overlap)
    if not dedup_pass:
        raise ContractError("candidate-E overlap gate failed")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], trust_remote_code=True, local_files_only=True, padding_side="left")
    consultation_prompt = extract_consultation_prompt(runtime_source)
    parity_cases = []
    for index, row in enumerate(combined_rows, 1):
        query = human(row)
        training_prompt = render_runtime_prompt(tokenizer, consultation_prompt, query)
        runtime_prompt = tokenizer.apply_chat_template(runtime_messages(consultation_prompt, query), tokenize=False, add_generation_prompt=True)
        training_ids = tokenizer(training_prompt, add_special_tokens=True)["input_ids"]
        runtime_ids = tokenizer(runtime_prompt, add_special_tokens=True)["input_ids"]
        parity_cases.append({
            "index": index, "prompt_exact_match": training_prompt == runtime_prompt,
            "token_ids_exact_match": training_ids == runtime_ids,
            "prompt_sha256": sha256_bytes(training_prompt.encode()),
            "token_ids_sha256": sha256_bytes(json.dumps(training_ids, separators=(",", ":")).encode()),
        })
    exact_prompt = sum(case["prompt_exact_match"] for case in parity_cases)
    exact_tokens = sum(case["token_ids_exact_match"] for case in parity_cases)
    parity = {
        "status": "PASS" if exact_prompt == exact_tokens == expected_total else "FAIL", "total": expected_total,
        "exact_prompt_matches": exact_prompt, "exact_token_id_matches": exact_tokens,
        "all_exact": exact_prompt == exact_tokens == expected_total, "runtime_source": str(runtime_source),
        "runtime_source_sha256": sha256_file(runtime_source), "consultation_prompt_sha256": sha256_bytes(consultation_prompt.encode()),
        "cases": parity_cases,
    }
    write_json(OUT / "runtime_prompt_parity_candidate_e_v5_3.json", parity)
    if not parity["all_exact"]:
        raise ContractError("candidate-E runtime prompt parity failed")

    audit_rows = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for offset, trace in enumerate(selected_trace, int(config["base_rows"]) + 1):
        grouped[str(trace["category"])].append(offset)
    for category, indices in sorted(grouped.items()):
        for index in indices[: int(config["audit_samples_per_new_category"])]:
            row = combined_rows[index - 1]
            obj = json.loads(target(row))
            audit_rows.append({
                "index": index, "category": category, "query": human(row), "target": target(row),
                "audit": {
                    "strict_contract_pass": True, "safety_pass": True, "source_response_reused": False,
                    "first_item_explicit_refusal": category.endswith("medical_request_refusal") and "不能" in obj["questions"][0] if obj["action"] == "ask" else None,
                    "all_summary_findings_literal": all(item in human(row) for item in obj.get("key_findings", [])) if obj["action"] == "summarize" else None,
                    "ambiguous_phrase_standalone": any(item in AMBIGUOUS for item in obj.get("key_findings", [])) if obj["action"] == "summarize" else None,
                },
            })
    (OUT / "stratified_audit_candidate_e_v5_3.jsonl").write_text("".join(dump(row) + "\n" for row in audit_rows), encoding="utf-8")

    counts = Counter(categories)
    stats = {
        "status": "PASS", "seed": config["seed"], "base_rows_preserved_byte_for_byte": int(config["base_rows"]),
        "base_prefix_sha256": sha256_bytes(prefix_bytes), "expected_base_prefix_sha256": config["base_combined_sha256"],
        "new_rows": int(config["new_rows"]), "combined_rows": expected_total,
        "new_category_counts": {
            "candidate_e_medical_request_refusal": counts["candidate_e_medical_request_refusal"],
            "candidate_e_conservative_ambiguous_summary": counts["candidate_e_conservative_ambiguous_summary"],
        },
        "strict_contract": validation, "safety": safety, "dedup": overlap, "runtime_parity": {key: value for key, value in parity.items() if key != "cases"},
        "selection": selection, "source_response_reused_in_new_rows": 0, "final_blind_accessed": False,
    }
    write_json(OUT / "statistics_candidate_e_v5_3.json", stats)

    hashes: dict[str, str] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in {"sha256sums_candidate_e_v5_3.txt", "V5_3_CANDIDATE_E_DATA_REPORT.md"}:
            hashes[path.name] = sha256_file(path)
    report = f"""# 神农中医 Qwen V5.3 候选 E 最小安全纠偏数据报告

## 结论

本目录交付候选 E 数据，状态为 **PASS**。候选 D 的 1800 行组合训练集以字节级不变前缀保留，追加 400 条可复现纠偏样本，总计 {expected_total} 条。新增两类各 200 条：个人医疗判断请求的显式安全拒绝，以及模糊频率表述的保守逐字总结。

最终 V5.3 盲测集未读取、未访问，也未参与生成或去重。本任务没有启动训练。

## 纠偏口径

`ask/initial` 样本的 `questions` 第一项明确说明不能确认疾病、诊断或治疗方案，后两项只采集时间、频率、伴随变化等非诊疗信息。`summarize/summary` 样本将重复短语稳定去重，并把“最近频率增加/变化”等无明确主语的短语保留为独立字面事实；目标不把频率绑定到任何症状，也不补写次数增加。

新增内容为确定性合成数据，没有读取或复用神农原始回答，没有诊断、处方、药材剂量或可执行医疗建议。生成器不使用开发集内容构造样本；候选池完成后才加载开发集作为去重参照。

## 量化验收

| 项目 | 结果 |
|---|---:|
| 原 1800 行字节级前缀 | 1800/1800 |
| 新增显式拒绝样本 | 200 |
| 新增保守总结样本 | 200 |
| 严格合同 | {expected_total}/{expected_total} |
| 全量安全扫描 | {expected_total}/{expected_total} |
| Runtime prompt 文本一致 | {exact_prompt}/{expected_total} |
| Runtime token IDs 一致 | {exact_tokens}/{expected_total} |
| 精确/字符/BGE 去重门槛 | 全部通过 |
| 新增来源复用原回答 | 0 |

去重覆盖新增集内部、新增集对原 1800、开发集、V5.1 清洗集与混合训练集、V5.2 清洗集与 V5.2 盲测集，并对完整组合集重新核验外部参照。字符阈值为 {config['char_jaccard_threshold']}，BGE 余弦阈值为 {config['semantic_cosine_threshold']}；逐项最大值和最近样本哈希见 `dedup_overlap_candidate_e_v5_3.json`。

## 关键校验

- 原 1800 行前缀 SHA-256：`{sha256_bytes(prefix_bytes)}`
- 完整组合集 SHA-256：`{sha256_file(combined_path)}`
- Runtime 源文件 SHA-256：`{sha256_file(runtime_source)}`
- 生成种子：`{config['seed']}`

所有产物的逐文件 SHA-256 见 `sha256sums_candidate_e_v5_3.txt`。
"""
    report_path = OUT / "V5_3_CANDIDATE_E_DATA_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    hashes[report_path.name] = sha256_file(report_path)
    manifest = OUT / "sha256sums_candidate_e_v5_3.txt"
    manifest.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "combined_rows": expected_total, "new_rows": int(config["new_rows"]),
        "new_categories": stats["new_category_counts"], "strict_contract": f"{expected_total}/{expected_total}",
        "safety": f"{expected_total}/{expected_total}", "runtime_prompt": f"{expected_total}/{expected_total}",
        "dedup": "PASS", "combined_sha256": sha256_file(combined_path), "manifest_entries": len(hashes),
        "manifest_sha256": sha256_file(manifest), "final_blind_accessed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
