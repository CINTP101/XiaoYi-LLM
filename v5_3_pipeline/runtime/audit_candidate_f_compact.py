#!/usr/bin/env python3
"""Build immutable evidence for the compact Candidate F prompt."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

PROJECT = Path("/home/cyh/Medical_Qwen")
SOURCE = PROJECT / "v5_3_pipeline/runtime"
OUT = PROJECT / "artifacts/v5_3_pipeline/runtime_candidate_f_compact"
PROMPT_SOURCE = SOURCE / "consultation_prompt_candidate_f_compact.py"
POLICY_SOURCE = SOURCE / "candidate_f_policy.py"
TEST_SOURCE = SOURCE / "test_candidate_f_compact_prompt.py"
AUDIT_SOURCE = SOURCE / "audit_candidate_f_compact.py"
OLD_PROMPT_SOURCE = SOURCE / "consultation_prompt_candidate_f.py"
TRAIN = PROJECT / "artifacts/v5_3_pipeline/data_candidate_e/train_candidate_e_combined_v5_3.jsonl"
EXPECTED_TRAIN_SHA = "011b5feaa926e9b354903d90e701cf3d9f40bbeaeb7b03c6342cb0191c23f8db"
CURRENT_RUNTIME = PROJECT / "tcm_chat_v5.py"
EXPECTED_RUNTIME_SHA = "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
TOKENIZER_PATH = PROJECT / "models/Qwen2.5-1.5B-Instruct"
MAX_FIXED_TOKENS = 600
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
sys.path.insert(0, str(SOURCE))

from runtime_prompt_v5_3 import render_runtime_prompt, runtime_messages


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_literal(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = (ast.Import, ast.ImportFrom, ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    if any(isinstance(node, forbidden) for node in ast.walk(tree)):
        raise RuntimeError(f"prompt module has side-effect-capable node: {path}")
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "CONSULTATION_PROMPT" for target in node.targets
    )]
    if len(assignments) != 1 or [type(node).__name__ for node in tree.body] != ["Expr", "Assign"]:
        raise RuntimeError(f"prompt module shape is not docstring plus one literal assignment: {path}")
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("prompt is not a non-empty literal string")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if sha256_file(TRAIN) != EXPECTED_TRAIN_SHA:
        raise RuntimeError("Candidate E training data changed")
    if sha256_file(CURRENT_RUNTIME) != EXPECTED_RUNTIME_SHA:
        raise RuntimeError("tcm_chat_v5.py changed")
    prompt = prompt_literal(PROMPT_SOURCE)
    old_prompt = prompt_literal(OLD_PROMPT_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
    raw_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    fixed_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True,
    )
    old_fixed_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": old_prompt}, {"role": "user", "content": ""}],
        tokenize=True, add_generation_prompt=True,
    )
    if len(fixed_ids) > MAX_FIXED_TOKENS:
        raise RuntimeError(f"compact fixed prompt is {len(fixed_ids)} tokens, exceeds {MAX_FIXED_TOKENS}")

    OUT.mkdir(parents=True, exist_ok=True)
    for path in OUT.iterdir():
        if path.is_file():
            path.unlink()
    for path in (PROMPT_SOURCE, POLICY_SOURCE, TEST_SOURCE, AUDIT_SOURCE):
        shutil.copy2(path, OUT / path.name)

    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern="test_candidate_f_compact_prompt.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    tests = {
        "status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "output": stream.getvalue(), "compact_module_side_effect_free": True,
    }
    write_json(OUT / "candidate_f_compact_static_test_results.json", tests)
    if not result.wasSuccessful():
        raise RuntimeError("compact static unit tests failed")

    token_report = {
        "status": "PASS", "maximum_fixed_prompt_tokens": MAX_FIXED_TOKENS,
        "tokenizer": str(TOKENIZER_PATH), "measurement": "apply_chat_template(system compact prompt + empty user slot, tokenize=True, add_generation_prompt=True)",
        "compact_prompt_raw_literal_tokens": len(raw_ids), "compact_prompt_fixed_template_tokens": len(fixed_ids),
        "available_tokens_with_max_input_length_768": 768 - len(fixed_ids),
        "previous_candidate_f_fixed_template_tokens_same_measurement": len(old_fixed_ids),
        "compact_prompt_source_sha256": sha256_file(PROMPT_SOURCE), "compact_prompt_literal_sha256": sha256_bytes(prompt.encode()),
        "previous_prompt_source_sha256": sha256_file(OLD_PROMPT_SOURCE), "previous_prompt_modified": False,
    }
    write_json(OUT / "candidate_f_compact_token_budget.json", token_report)

    text_matches = token_matches = rows = 0
    parity_cases = []
    with TRAIN.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, 1):
            row = json.loads(raw_line)
            human = row["conversations"][0]["value"]
            left = render_runtime_prompt(tokenizer, prompt, human)
            right = tokenizer.apply_chat_template(runtime_messages(prompt, human), tokenize=False, add_generation_prompt=True)
            left_ids = tokenizer(left, add_special_tokens=True)["input_ids"]
            right_ids = tokenizer(right, add_special_tokens=True)["input_ids"]
            text_equal = left == right
            token_equal = left_ids == right_ids
            text_matches += int(text_equal)
            token_matches += int(token_equal)
            rows += 1
            parity_cases.append({
                "index": index, "text_exact_match": text_equal, "token_ids_exact_match": token_equal,
                "rendered_prompt_sha256": sha256_bytes(left.encode()),
                "token_ids_sha256": sha256_bytes(json.dumps(left_ids, separators=(",", ":")).encode()),
            })
    parity = {
        "status": "PASS" if text_matches == token_matches == rows else "FAIL", "total": rows,
        "text_exact_matches": text_matches, "token_id_exact_matches": token_matches,
        "input_sha256": sha256_file(TRAIN), "training_data_modified": False,
        "current_runtime_sha256": sha256_file(CURRENT_RUNTIME), "current_runtime_modified": False,
        "final_blind_accessed": False, "cases": parity_cases,
    }
    write_json(OUT / "candidate_f_compact_prompt_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError("compact prompt parity failed")

    report = f"""# V5.3 Candidate F Compact 安全运行协议

## 结论

压缩版 prompt-only 模块独立交付，状态 **PASS**。Qwen tokenizer 的提示词字面量为 {len(raw_ids)} Token；按实际 chat template 加入 system 提示词、空 user 槽和 generation prompt 后，固定开销为 {len(fixed_ids)} Token，不超过 600，给 `max-input-length=768` 保留 {768 - len(fixed_ids)} Token。旧 Candidate F 提示词与旧证据未覆盖、未修改。

## 保留的安全门

- 输出限定为单个 JSON 对象，action/stage 仅允许 `ask/initial` 与 `summarize/summary`，字段闭集与现有 contract 一致。
- 每个 `ask/initial` 的 `questions[0]` 使用固定完整安全边界；总数 1–3，其后最多两个非诊疗信息收集问题。
- `summarize/summary` 的 `key_findings` 只含用户明确提供的非空字面事实，唯一并按首次出现顺序稳定去重。
- 主语不清的频率变化原短语独立逐字保留，不绑定任何症状或器官，不补成次数增加。
- `syndrome_tendency`、`need_more_info`、`note` 使用指定固定值。
- 禁止诊断、证型或病因结论，禁止药物、药材、方剂、剂量、可执行医疗建议、治疗规划和编造事实。

## 验证

- 静态与 tokenizer 单元测试：{result.testsRun}/{result.testsRun}。
- Candidate E 2200 条输入的 compact prompt 文本 parity：{text_matches}/{rows}。
- Candidate E 2200 条输入的 compact prompt Token IDs parity：{token_matches}/{rows}。
- 当前 `tcm_chat_v5.py` SHA-256：`{sha256_file(CURRENT_RUNTIME)}`，未修改。
- Candidate E 训练集 SHA-256：`{sha256_file(TRAIN)}`，未修改。
- 最终盲测集未读取、未访问；没有启动模型生成或训练。
"""
    (OUT / "CANDIDATE_F_COMPACT_RUNTIME_PROTOCOL.md").write_text(report, encoding="utf-8")
    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256sums_runtime_candidate_f_compact.txt":
            hashes[path.name] = sha256_file(path)
    manifest = OUT / "sha256sums_runtime_candidate_f_compact.txt"
    manifest.write_text("\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "raw_tokens": len(raw_ids), "fixed_template_tokens": len(fixed_ids),
        "remaining_at_768": 768 - len(fixed_ids), "tests": f"{result.testsRun}/{result.testsRun}",
        "text_parity": f"{text_matches}/{rows}", "token_parity": f"{token_matches}/{rows}",
        "manifest_entries": len(hashes), "manifest_sha256": sha256_file(manifest),
        "old_prompt_modified": False, "runtime_modified": False, "training_data_modified": False,
        "final_blind_accessed": False, "training_started": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
