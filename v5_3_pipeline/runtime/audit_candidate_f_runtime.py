#!/usr/bin/env python3
"""Audit Candidate F prompt safety, Candidate E target compatibility, and token parity."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT = Path("/home/cyh/Medical_Qwen")
SOURCE = PROJECT / "v5_3_pipeline/runtime"
OUT = PROJECT / "artifacts/v5_3_pipeline/runtime_candidate_f"
TRAIN = PROJECT / "artifacts/v5_3_pipeline/data_candidate_e/train_candidate_e_combined_v5_3.jsonl"
EXPECTED_TRAIN_SHA = "011b5feaa926e9b354903d90e701cf3d9f40bbeaeb7b03c6342cb0191c23f8db"
CURRENT_RUNTIME = PROJECT / "tcm_chat_v5.py"
EXPECTED_RUNTIME_SHA = "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
PROMPT_SOURCE = SOURCE / "consultation_prompt_candidate_f.py"
TOKENIZER_PATH = PROJECT / "models/Qwen2.5-1.5B-Instruct"

sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
sys.path.insert(0, str(SOURCE))

from candidate_f_policy import policy_errors
from runtime_prompt_v5_3 import render_runtime_prompt, runtime_messages
from v5_3_contract import strict_json_loads, validate_jsonl, validate_target_text


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_literal() -> str:
    tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))
    forbidden = (ast.Import, ast.ImportFrom, ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        raise RuntimeError("prompt-only module contains executable imports, calls, or definitions")
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "CONSULTATION_PROMPT" for target in node.targets
    )]
    if len(assignments) != 1:
        raise RuntimeError("prompt-only module must define CONSULTATION_PROMPT exactly once")
    prompt = ast.literal_eval(assignments[0].value)
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("CONSULTATION_PROMPT must be a non-empty literal string")
    return prompt


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Candidate E combined training SHA changed")
    if sha256_file(CURRENT_RUNTIME) != EXPECTED_RUNTIME_SHA:
        raise RuntimeError("tcm_chat_v5.py changed; refusing Candidate F audit")
    prompt = load_prompt_literal()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    for name in ("consultation_prompt_candidate_f.py", "candidate_f_policy.py", "test_candidate_f_prompt.py", "audit_candidate_f_runtime.py"):
        shutil.copy2(SOURCE / name, OUT / name)

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern="test_candidate_f_prompt.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    test_result = {
        "status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "output": stream.getvalue(), "prompt_module_side_effect_free": True,
    }
    write_json(OUT / "candidate_f_static_test_results.json", test_result)
    if not result.wasSuccessful():
        raise RuntimeError("Candidate F static unit tests failed")

    contract = validate_jsonl(TRAIN)
    cases = []
    exact_pass = equivalent_pass = 0
    exact_reasons: Counter[str] = Counter()
    equivalent_reasons: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    with TRAIN.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            row = strict_json_loads(raw_line, context=f"line {index}")
            human = row["conversations"][0]["value"]
            target_text = row["conversations"][1]["value"]
            target, label = validate_target_text(target_text, context=f"line {index}.target")
            label_name = f"{label[0]}/{label[1]}"
            labels[label_name] += 1
            exact_errors = policy_errors(human, target, allow_equivalent_boundary=False)
            equivalent_errors = policy_errors(human, target, allow_equivalent_boundary=True)
            exact_pass += int(not exact_errors)
            equivalent_pass += int(not equivalent_errors)
            exact_reasons.update(exact_errors)
            equivalent_reasons.update(equivalent_errors)
            cases.append({
                "index": index, "label": label_name, "user_text_sha256": sha256_bytes(human.encode()),
                "target_sha256": sha256_bytes(target_text.encode()), "shared_contract_pass": True,
                "candidate_f_fixed_policy_pass": not exact_errors, "candidate_f_fixed_policy_errors": exact_errors,
                "candidate_f_equivalent_policy_pass": not equivalent_errors, "candidate_f_equivalent_policy_errors": equivalent_errors,
            })
    (OUT / "candidate_e_target_compatibility_cases.jsonl").write_text("".join(dump(case) + "\n" for case in cases), encoding="utf-8")
    compatibility = {
        "status": "PASS_WITH_DOCUMENTED_TARGET_MISMATCH", "input": str(TRAIN), "input_sha256": sha256_file(TRAIN),
        "total_targets": len(cases), "shared_json_contract_compatible": len(cases), "shared_json_contract_rate": 1.0,
        "label_counts": dict(labels),
        "candidate_f_fixed_policy_compatible": exact_pass, "candidate_f_fixed_policy_rate": exact_pass / len(cases),
        "candidate_f_equivalent_policy_compatible": equivalent_pass, "candidate_f_equivalent_policy_rate": equivalent_pass / len(cases),
        "fixed_policy_failure_reasons": dict(exact_reasons), "equivalent_policy_failure_reasons": dict(equivalent_reasons),
        "interpretation": "The output schema is fully compatible; existing targets are not all aligned with Candidate F's stricter semantic policy.",
        "training_data_modified": False, "final_blind_accessed": False,
    }
    write_json(OUT / "candidate_e_target_compatibility.json", compatibility)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True, padding_side="left")
    prompt_cases = []
    with TRAIN.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            row = json.loads(raw_line)
            human = row["conversations"][0]["value"]
            training_prompt = render_runtime_prompt(tokenizer, prompt, human)
            candidate_runtime_prompt = tokenizer.apply_chat_template(
                runtime_messages(prompt, human), tokenize=False, add_generation_prompt=True
            )
            training_ids = tokenizer(training_prompt, add_special_tokens=True)["input_ids"]
            runtime_ids = tokenizer(candidate_runtime_prompt, add_special_tokens=True)["input_ids"]
            prompt_cases.append({
                "index": index, "text_exact_match": training_prompt == candidate_runtime_prompt,
                "token_ids_exact_match": training_ids == runtime_ids,
                "prompt_sha256": sha256_bytes(training_prompt.encode()),
                "token_ids_sha256": sha256_bytes(json.dumps(training_ids, separators=(",", ":")).encode()),
            })
    text_matches = sum(case["text_exact_match"] for case in prompt_cases)
    token_matches = sum(case["token_ids_exact_match"] for case in prompt_cases)
    parity = {
        "status": "PASS" if text_matches == token_matches == len(cases) else "FAIL", "total": len(cases),
        "text_exact_matches": text_matches, "token_id_exact_matches": token_matches,
        "candidate_f_prompt_source": str(PROMPT_SOURCE), "candidate_f_prompt_source_sha256": sha256_file(PROMPT_SOURCE),
        "candidate_f_prompt_sha256": sha256_bytes(prompt.encode()), "tokenizer": str(TOKENIZER_PATH),
        "wrapper": "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "current_runtime_source": str(CURRENT_RUNTIME), "current_runtime_source_sha256": sha256_file(CURRENT_RUNTIME),
        "current_runtime_modified": False, "training_data_modified": False, "final_blind_accessed": False,
        "cases": prompt_cases,
    }
    write_json(OUT / "candidate_f_prompt_text_token_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError("Candidate F prompt text/token parity failed")

    report = f"""# V5.3 Candidate F 安全运行提示词协议

## 结论

Candidate F 独立 prompt-only 源文件已通过静态安全检查和现有 JSON contract 兼容性检查。该模块只含一个字面量 `CONSULTATION_PROMPT`，没有 import、函数、类或调用，不会加载模型，也没有修改当前 `tcm_chat_v5.py`。本任务没有修改训练数据、没有启动训练，并且没有读取或访问最终盲测集。

## 安全协议

- `ask/initial` 的 `questions` 为 1–3 个非空唯一字符串；首项固定为“仅凭您提供的信息，我不能为您判断疾病或证型、作出诊断、开方或制定治疗方案。”，其后最多两个问题且只收集非诊疗信息。
- `summarize/summary` 的 `key_findings` 稳定去重，每项必须是用户明确提供的字面事实。主语不清的频率变化必须作为独立原短语保留，不得绑定症状或补成次数增加。
- `syndrome_tendency`、`need_more_info` 和 `note` 使用指定固定值。
- 两种动作都禁止药物、药材、方剂、剂量、可执行医疗建议、诊断和治疗规划。

## 静态测试

静态单元测试 {result.testsRun}/{result.testsRun} 通过，覆盖 prompt-only 模块无副作用、闭集 JSON 字段、固定拒绝、最多三个问题、非诊疗追问、字面总结、模糊频率独立保留、稳定去重、固定总结字段、剂量和可执行建议禁令。

## Candidate E 目标兼容性

Candidate E 组合训练目标共 {len(cases)} 条，现有严格 JSON contract 兼容 {len(cases)}/{len(cases)}。其中 `ask/initial` {labels['ask/initial']} 条，`summarize/summary` {labels['summarize/summary']} 条。

Candidate F 固定安全句与完整保守总结策略严格兼容 {exact_pass}/{len(cases)}；允许首项使用覆盖全部安全边界的等价表达时兼容 {equivalent_pass}/{len(cases)}。因此，新提示词与现有解析器和 JSON schema 完全兼容，但与既有训练目标的安全语义并非全部一致。使用该提示词做候选推理评估是可行的；若未来以它继续训练，应另行构建完全对齐的目标数据，不能把 2200/2200 的结构兼容当作语义兼容。

具体失败原因计数见 `candidate_e_target_compatibility.json`，逐条证据以用户文本和目标哈希绑定，见 `candidate_e_target_compatibility_cases.jsonl`。

## Prompt 文本与 Token 一致性

使用本地 Qwen2.5-1.5B-Instruct tokenizer，对 Candidate E 的 {len(cases)} 条输入逐条复现 Candidate F wrapper。提示词文本一致 {text_matches}/{len(cases)}，Token IDs 一致 {token_matches}/{len(cases)}。这份证据只验证候选提示词两条构造路径等价，不表示当前运行文件已经切换。

## 保护性校验

- 当前 `tcm_chat_v5.py` SHA-256：`{sha256_file(CURRENT_RUNTIME)}`，与基线一致。
- Candidate E 组合训练集 SHA-256：`{sha256_file(TRAIN)}`，未修改。
- Candidate F prompt 字面量 SHA-256：`{sha256_bytes(prompt.encode())}`。
- 最终盲测集访问：否。
"""
    (OUT / "CANDIDATE_F_RUNTIME_PROTOCOL.md").write_text(report, encoding="utf-8")
    hashes: dict[str, str] = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256s_runtime_candidate_f.txt":
            hashes[path.name] = sha256_file(path)
    manifest = OUT / "sha256sums_runtime_candidate_f.txt"
    manifest.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "tests": f"{result.testsRun}/{result.testsRun}",
        "shared_contract": f"{len(cases)}/{len(cases)}", "fixed_policy": f"{exact_pass}/{len(cases)}",
        "equivalent_policy": f"{equivalent_pass}/{len(cases)}", "prompt_text_parity": f"{text_matches}/{len(cases)}",
        "prompt_token_parity": f"{token_matches}/{len(cases)}", "manifest_entries": len(hashes),
        "manifest_sha256": sha256_file(manifest), "runtime_modified": False, "training_data_modified": False,
        "final_blind_accessed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
