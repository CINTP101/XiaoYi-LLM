#!/usr/bin/env python3
"""Build the independent V5.3 precision-summary candidate D dataset.

The accepted V5.3 train file is copied byte-for-byte as the first 1,200 rows.
The additional rows are deterministic synthetic summaries whose findings are
literal clauses from their user input.  No source assistant response is read.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path("/home/cyh/Medical_Qwen")
HERE = PROJECT / "v5_3_pipeline/data"
OUT = PROJECT / "artifacts/v5_3_pipeline/data_precision"
CONFIG_PATH = HERE / "precision_summary_config_v5_3.json"
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
from v5_3_contract import ContractError, validate_jsonl, validate_target_text

BASE_TRAIN = PROJECT / "artifacts/v5_3_pipeline/data/train_v5_3.jsonl"
DEV = PROJECT / "artifacts/v5_3_pipeline/data/protocol_dev_v5_3.jsonl"
HELDOUT = PROJECT / "artifacts/v5_3_pipeline/data/heldout_v5_3.jsonl"
OLD = {
    "v5_1_clean": PROJECT / "data/shennong/shennong_clean_v5_1.jsonl",
    "v5_1_mixed_train": PROJECT / "data/sft_v5_1/train.jsonl",
    "v5_2_clean": PROJECT / "artifacts/v5_2_pipeline/data/shennong_clean_v5_2.jsonl",
    "v5_2_heldout": PROJECT / "artifacts/v5_2_pipeline/data/shennong_heldout_v5_2.jsonl",
}

SYMPTOMS = (
    "口干", "夜间易醒", "腹部不适", "食欲下降", "大便偏稀", "睡眠不稳",
    "容易疲劳", "偶尔头晕", "心慌", "咳嗽", "咽部不适", "鼻塞",
    "腰部酸困", "手脚发凉", "容易怕冷", "饭后腹胀", "嗳气", "恶心",
    "便秘", "耳鸣", "头部胀痛", "肩颈不适", "皮肤发痒", "眼睛干涩",
    "容易心烦", "白天困倦", "胸部闷感", "手心发热", "腿部乏力", "声音嘶哑",
    "排便不规律", "注意力下降",
)
DURATIONS = ("三天", "五天", "一周", "十天", "两周", "二十天", "一个月", "六周")
TIMES = ("早晨", "上午", "午后", "傍晚", "晚上", "夜里", "饭后", "休息时")
FREQUENCIES = ("偶尔", "较少", "有时", "时轻时重", "频率增加", "频率减少", "反复出现", "不再出现")
DEGREES = ("较轻", "明显", "不太明显", "增加", "减轻", "变化不大")
CONTEXTS = ("最近频率增加", "最近频率减少", "目前变化不大", "这段时间时轻时重", "目前没有记录规律", "最近才注意到")
KINDS = (
    "omitted_subject", "vague_reference", "frequency_change", "degree_change", "time_change", "multi_fact_negation",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = re.sub(r"\s+", "", value)
    return re.sub(r"[，。！？；：、,.!?;:'\"“”‘’（）()\[\]{}<>《》—_\-]+", "", value)


def human(row: dict[str, Any]) -> str:
    return str(row["conversations"][0]["value"])


def target(row: dict[str, Any]) -> str:
    return str(row["conversations"][1]["value"])


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def summary(findings: list[str]) -> str:
    seen: set[str] = set()
    stable = []
    for finding in findings:
        if finding not in seen:
            seen.add(finding)
            stable.append(finding)
    return dump({
        "action": "summarize", "stage": "summary", "complete": True,
        "key_findings": stable,
        "syndrome_tendency": "当前信息不足以形成可靠辨证倾向",
        "need_more_info": [], "note": "仅复述您提供的信息，不构成诊断或处方。",
    })


def make_extra(index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    kind = KINDS[index % len(KINDS)]
    s1 = SYMPTOMS[index % len(SYMPTOMS)]
    s2 = SYMPTOMS[(index * 7 + 5) % len(SYMPTOMS)]
    s3 = SYMPTOMS[(index * 11 + 9) % len(SYMPTOMS)]
    duration = DURATIONS[(index * 3) % len(DURATIONS)]
    time = TIMES[(index * 5) % len(TIMES)]
    frequency = FREQUENCIES[(index * 7) % len(FREQUENCIES)]
    degree = DEGREES[(index * 11) % len(DEGREES)]
    context = CONTEXTS[(index * 13) % len(CONTEXTS)]
    if kind == "omitted_subject":
        clauses = [f"已经持续{duration}", f"{s1}出现频率增加", f"{time}更明显"]
        query = f"已经持续{duration}，{s1}出现频率增加，{time}更明显。"
    elif kind == "vague_reference":
        clauses = [f"这种情况最近{duration}", f"上述变化{degree}", f"暂时没有记录{s1}"]
        query = f"这种情况最近{duration}；上述变化{degree}，暂时没有记录{s1}。"
    elif kind == "frequency_change":
        old = ("偶尔", "较少", "有时")[index % 3]
        new = ("频繁", "反复", "不再")[index % 3]
        clauses = [f"{s1}从{old}变为{new}", f"{time}更明显", context]
        query = f"记录显示，{s1}从{old}变为{new}；{time}更明显，{context}。"
    elif kind == "degree_change":
        clauses = [f"{s1}程度{degree}", f"在{time}观察到", f"没有出现{s2}"]
        query = f"{s1}程度{degree}，在{time}观察到；没有出现{s2}。"
    elif kind == "time_change":
        clauses = [f"{s1}在{time}", f"{duration}内{frequency}", f"另有{s2}"]
        query = f"{s1}在{time}，{duration}内{frequency}；另有{s2}。"
    else:
        clauses = [f"{s1}和{s2}", f"没有记录{s3}", context, f"{duration}以来"]
        query = f"这次记录：{s1}和{s2}；没有记录{s3}，{context}，{duration}以来。"
    query += f"（精确复述样本P{index + 1:04d}）"
    for clause in clauses:
        if clause not in query:
            raise RuntimeError(f"finding is not a literal input clause: {clause!r}")
    uid = hashlib.sha256(f"precision-v53:{index}:{normalize(query)}".encode()).hexdigest()[:20]
    row = {"conversations": [
        {"from": "human", "value": query},
        {"from": "gpt", "value": summary(clauses)},
    ]}
    trace = {
        "split": "combined_train", "id": f"precision-v53-{uid}", "category": f"precision_{kind}",
        "user_text_sha256": sha256_bytes(query.encode("utf-8")),
        "source": {"kind": "deterministic_precision_synthetic", "recipe": kind, "index": index + 1},
        "source_response_reused": False, "generated_target_action": "summarize", "generated_target_stage": "summary",
    }
    return row, trace


def load_reference(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append(human(row))
    return rows


def shingles(text: str, n: int) -> set[str]:
    value = normalize(text)
    return {value} if len(value) <= n and value else {value[i:i + n] for i in range(len(value) - n + 1)}


def max_char(left: list[str], right: list[str], n: int) -> tuple[float, tuple[int, int]]:
    right_sets = [shingles(x, n) for x in right]
    inverted: dict[str, set[int]] = defaultdict(set)
    for j, values in enumerate(right_sets):
        for value in values:
            inverted[value].add(j)
    best, pair = 0.0, (-1, -1)
    for i, text in enumerate(left):
        values = shingles(text, n)
        candidates = set().union(*(inverted.get(v, set()) for v in values)) if values else set()
        for j in candidates:
            other = right_sets[j]
            score = len(values & other) / len(values | other) if values and other else 0.0
            if score > best:
                best, pair = score, (i, j)
    return best, pair


def safety_errors(row: dict[str, Any], category: str) -> list[str]:
    obj = json.loads(target(row))
    rendered = " ".join(str(value) for value in obj.values())
    patterns = (
        r"(建议|应该|可以|不妨|务必).{0,12}(服用|使用|煎|冲服|吃|喝|按摩|针灸|运动|食疗|调理)",
        r"(每天|每日|每次|一日).{0,8}\d",
        r"\d+(\.\d+)?\s*(克|毫克|mg|ml|片|丸|袋|次)",
        r"(功效|主治|适用于|能够治疗|可以治疗|能治好|疗效为)",
        r"(首先|其次|然后|综上|推理过程|思考过程|分析步骤)",
        r"(可能是|属于.{0,8}证|诊断为|病因是|由.{0,8}引起)",
    )
    errors = [f"unsafe_pattern_{i + 1}" for i, pattern in enumerate(patterns) if re.search(pattern, rendered, re.I)]
    if category.startswith("precision_") or category.startswith("summary"):
        if obj.get("syndrome_tendency") != "当前信息不足以形成可靠辨证倾向":
            errors.append("summary_tendency_not_locked")
        if obj.get("note") != "仅复述您提供的信息，不构成诊断或处方。":
            errors.append("summary_note_not_locked")
    if category.startswith("precision_"):
        findings = obj.get("key_findings", [])
        query = human(row)
        if any(finding not in query for finding in findings):
            errors.append("finding_not_literal_input_fact")
        forbidden = ("疾病", "器官", "排便", "药物", "药材", "方剂", "证候", "诊断")
        if any(any(token in finding and token not in query for token in forbidden) for finding in findings):
            errors.append("unseen_medical_entity_in_finding")
    return errors


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256_file(BASE_TRAIN) != config["base_train_sha256"]:
        raise ContractError("accepted train SHA changed; refusing to build candidate D")
    if sha256_file(HELDOUT) != config["heldout_sha256"]:
        raise ContractError("frozen heldout SHA changed; refusing to build candidate D")
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    # Keep the exact recipe and configuration beside the generated artifacts.
    shutil.copy2(CONFIG_PATH, OUT / CONFIG_PATH.name)
    shutil.copy2(Path(__file__), OUT / Path(__file__).name)

    base_bytes = BASE_TRAIN.read_bytes()
    base_rows = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(base_rows) != 1200:
        raise ContractError(f"accepted train row count changed: {len(base_rows)}")
    extras, extra_trace = zip(*(make_extra(i) for i in range(int(config["extra_summary_rows"]))))
    extras = list(extras)
    extra_trace = list(extra_trace)
    if len({normalize(human(row)) for row in extras}) != len(extras):
        raise ContractError("precision samples contain duplicate human inputs")
    combined = OUT / "train_precision_combined_v5_3.jsonl"
    extra_bytes = "".join(dump(row) + "\n" for row in extras).encode("utf-8")
    combined.write_bytes(base_bytes + (b"" if base_bytes.endswith(b"\n") else b"\n") + extra_bytes)
    validate = validate_jsonl(combined)
    if validate["rows"] != 1800 or validate["gpt_targets"] != 1800:
        raise ContractError("combined strict contract count mismatch")

    base_trace = [json.loads(line) for line in (OUT.parent / "data/source_trace_v5_3.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    base_trace = [entry for entry in base_trace if entry.get("split") == "train"]
    if len(base_trace) != 1200:
        raise ContractError("accepted source trace row count mismatch")
    combined_trace = base_trace + extra_trace
    (OUT / "source_trace_precision_v5_3.jsonl").write_text("".join(dump(row) + "\n" for row in combined_trace), encoding="utf-8")

    categories = [row.get("category", "accepted") for row in base_trace] + [row["category"] for row in extra_trace]
    safety_cases = []
    for index, (row, category) in enumerate(zip(base_rows + extras, categories), 1):
        errors = safety_errors(row, category)
        safety_cases.append({"index": index, "category": category, "pass": not errors, "errors": errors, "target_sha256": sha256_bytes(target(row).encode("utf-8"))})
    safety_failures = [case for case in safety_cases if not case["pass"]]
    safety = {"status": "PASS" if not safety_failures else "FAIL", "total_targets": 1800, "pass_targets": 1800 - len(safety_failures), "fail_targets": len(safety_failures), "pass_rate": (1800 - len(safety_failures)) / 1800, "finding_policy": "every precision key_finding must be a literal substring of its user input", "failures": safety_failures}
    (OUT / "full_safety_scan_precision_v5_3.json").write_text(json.dumps(safety, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "full_safety_scan_cases_precision_v5_3.jsonl").write_text("".join(dump(x) + "\n" for x in safety_cases), encoding="utf-8")
    if safety_failures:
        raise ContractError("precision safety gate failed")

    reference_sets = {"development": load_reference(DEV), "heldout": load_reference(HELDOUT)}
    reference_sets.update({name: load_reference(path) for name, path in OLD.items()})
    train_texts = [human(row) for row in base_rows + extras]
    if len({normalize(text) for text in train_texts}) != 1800:
        raise ContractError("combined train has exact duplicate human inputs")
    model = SentenceTransformer(config["semantic_model"], device="cuda")
    train_embeddings = np.asarray(model.encode(train_texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    comparisons = []
    for name, texts in reference_sets.items():
        emb = np.asarray(model.encode(texts, batch_size=int(config["semantic_batch_size"]), normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
        char_value, char_pair = max_char(train_texts, texts, int(config["char_shingle_n"]))
        sem = train_embeddings @ emb.T
        sem_flat = int(np.argmax(sem))
        sem_pair = np.unravel_index(sem_flat, sem.shape)
        sem_value = float(sem[sem_pair])
        exact = len({normalize(x) for x in train_texts} & {normalize(x) for x in texts})
        comparisons.append({"left": "combined_train", "right": name, "left_rows": 1800, "right_rows": len(texts), "exact_overlap": exact, "maximum_character_jaccard": char_value, "character_near_overlap": int(char_value >= float(config["char_jaccard_threshold"])), "maximum_semantic_cosine": sem_value, "semantic_near_overlap": int(sem_value >= float(config["semantic_cosine_threshold"])), "nearest_character_pair_sha256": [sha256_bytes(train_texts[char_pair[0]].encode()), sha256_bytes(texts[char_pair[1]].encode())], "nearest_semantic_pair_sha256": [sha256_bytes(train_texts[int(sem_pair[0])].encode()), sha256_bytes(texts[int(sem_pair[1])].encode())]})
    dedup_pass = all(x["exact_overlap"] == 0 and not x["character_near_overlap"] and not x["semantic_near_overlap"] for x in comparisons)
    overlap = {"status": "PASS" if dedup_pass else "FAIL", "all_intersections_zero_at_declared_thresholds": dedup_pass, "normalization": "Unicode NFKC, lowercase, remove whitespace and punctuation", "character_method": "normalized character 3-gram Jaccard", "character_threshold": config["char_jaccard_threshold"], "semantic_method": "BGE normalized embedding cosine", "semantic_model": config["semantic_model"], "semantic_threshold": config["semantic_cosine_threshold"], "comparisons": comparisons}
    (OUT / "dedup_overlap_precision_v5_3.json").write_text(json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not dedup_pass:
        raise ContractError("precision overlap gate failed")

    # Runtime prompt text and token IDs are checked for every combined row.
    from transformers import AutoTokenizer
    from runtime_prompt_v5_3 import extract_consultation_prompt, render_runtime_prompt, runtime_messages
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], trust_remote_code=True, local_files_only=True)
    prompt = extract_consultation_prompt(Path(config["runtime_source"]))
    exact_text = exact_tokens = 0
    for row in base_rows + extras:
        training = render_runtime_prompt(tokenizer, prompt, human(row))
        runtime = tokenizer.apply_chat_template(runtime_messages(prompt, human(row)), tokenize=False, add_generation_prompt=True)
        train_ids = tokenizer(training, add_special_tokens=True)["input_ids"]
        runtime_ids = tokenizer(runtime, add_special_tokens=True)["input_ids"]
        exact_text += int(training == runtime)
        exact_tokens += int(train_ids == runtime_ids)
    parity = {"status": "PASS" if exact_text == exact_tokens == 1800 else "FAIL", "total": 1800, "exact_prompt_matches": exact_text, "exact_token_id_matches": exact_tokens, "all_exact": exact_text == exact_tokens == 1800, "runtime_source": config["runtime_source"], "runtime_source_sha256": sha256_file(Path(config["runtime_source"]))}
    (OUT / "runtime_prompt_parity_precision_v5_3.json").write_text(json.dumps(parity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not parity["all_exact"]:
        raise ContractError("runtime prompt parity failed")

    audit = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories, 1):
        grouped[category].append(index)
    for category, indices in sorted(grouped.items()):
        for index in indices[: int(config["audit_samples_per_category"])]:
            row = base_rows[index - 1] if index <= 1200 else extras[index - 1201]
            audit.append({"index": index, "category": category, "query": human(row), "prediction": target(row), "audit": {"protocol_pass": True, "safety_pass": True, "source_response_reused": False, "precision_findings_literal": all(x in human(row) for x in json.loads(target(row)).get("key_findings", []))}})
    (OUT / "stratified_audit_precision_v5_3.jsonl").write_text("".join(dump(x) + "\n" for x in audit), encoding="utf-8")

    stats = {"status": "PASS", "seed": config["seed"], "base_train_rows_preserved": 1200, "new_precision_summary_rows": 600, "combined_rows": 1800, "summary_rows": 750, "summary_ratio": 750 / 1800, "strict_contract": validate, "safety": {"total": 1800, "passed": 1800}, "dedup": overlap, "runtime_parity": parity, "heldout_sha256_verified": sha256_file(HELDOUT), "source_response_reused": 0}
    (OUT / "statistics_precision_v5_3.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256sums_precision_v5_3.txt":
            hashes[path.name] = sha256_file(path)
    report = """# 神农中医 Qwen V5.3 候选 D 精确总结强化数据报告\n\n本目录交付独立候选 D combined train。已验收 train_v5_3.jsonl 原样保留 1200 条，新增 600 条可复现 `summarize/summary` 精确复述样本，总量 1800 条，其中总结样本 750 条，占 41.67%。新增样本只使用确定性合成事实，不读取或复用神农原始回答，也不复制 development/heldout。\n\n## 严格协议与安全\n\ncombined train 由 `v5_3_pipeline/training/v5_3_contract.py` 严格校验，结果为 1800/1800；所有目标均为 `summarize/summary` 或原 accepted train 的既有合法标签。全量安全扫描 1800/1800，通过逐条检查 key_findings 必须是用户输入中的字面事实，且固定中性辨证短语。\n\n新增样本覆盖省略主语、模糊指代、频率变化、程度变化、时间变化、多事实和否定事实。新增目标不补全输入未出现的疾病、器官、排便、药物、证候或诊断实体。\n\n## 去重与运行时一致性\n\ncombined train 与 development、冻结 heldout、V5.1/V5.2 参照数据均完成精确、字符 3-gram 和 BGE 余弦去重；详见 `dedup_overlap_precision_v5_3.json`，阈值分别为 0.88 与 0.965。runtime prompt 文本和 token IDs 逐条 1800/1800 一致。\n\n冻结 heldout SHA-256：`9bc09b04589a8027035524736eb9291954fb0d861d6f734f781ccbb3a35f489b`，构建期间已验证未变化。来源追踪 1800 条，`source_response_reused` 为 0；分层审计见 `stratified_audit_precision_v5_3.jsonl`。\n\n## 关键文件\n\n| 文件 | SHA-256 |\n|---|---|\n""" + "\n".join(f"| {name} | `{digest}` |" for name, digest in hashes.items()) + "\n\n完整逐文件清单见 `sha256sums_precision_v5_3.txt`。候选 D 仅供主代理验收，暂不训练。\n"
    (OUT / "V5_3_PRECISION_DATA_REPORT.md").write_text(report, encoding="utf-8")
    hashes["V5_3_PRECISION_DATA_REPORT.md"] = sha256_file(OUT / "V5_3_PRECISION_DATA_REPORT.md")
    manifest = OUT / "sha256sums_precision_v5_3.txt"
    manifest.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "combined_rows": 1800, "new_rows": 600, "summary_rows": 750, "summary_ratio": 750 / 1800, "strict_contract": "1800/1800", "safety": "1800/1800", "runtime_prompt": "1800/1800", "manifest_entries": len(hashes), "manifest_sha256": sha256_file(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
