#!/usr/bin/env python3
"""Static and tokenizer-bound unit tests for the compact Candidate F prompt."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

from transformers import AutoTokenizer

PROJECT = Path("/home/cyh/Medical_Qwen")
RUNTIME_DIR = PROJECT / "v5_3_pipeline/runtime"
PROMPT_SOURCE = RUNTIME_DIR / "consultation_prompt_candidate_f_compact.py"
TOKENIZER_PATH = PROJECT / "models/Qwen2.5-1.5B-Instruct"
MAX_FIXED_TOKENS = 600
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
sys.path.insert(0, str(RUNTIME_DIR))

from candidate_f_policy import FIXED_NOTE, FIXED_SAFETY_BOUNDARY, FIXED_TENDENCY, require_policy
from v5_3_contract import ContractError, validate_target_object


def load_literal_prompt() -> str:
    tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "CONSULTATION_PROMPT" for target in node.targets
    )]
    if len(assignments) != 1:
        raise AssertionError("compact module must define CONSULTATION_PROMPT exactly once")
    prompt = ast.literal_eval(assignments[0].value)
    if not isinstance(prompt, str) or not prompt.strip():
        raise AssertionError("compact prompt must be a non-empty literal string")
    return prompt


class CandidateFCompactPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompt = load_literal_prompt()

    def test_module_has_no_side_effect_nodes(self) -> None:
        tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"))
        forbidden = (ast.Import, ast.ImportFrom, ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        self.assertFalse(any(isinstance(node, forbidden) for node in ast.walk(tree)))
        self.assertEqual([type(node).__name__ for node in tree.body], ["Expr", "Assign"])

    def test_qwen_fixed_template_is_at_most_600_tokens(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True, local_files_only=True)
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": self.prompt}, {"role": "user", "content": ""}],
            tokenize=True, add_generation_prompt=True,
        )
        self.assertLessEqual(len(ids), MAX_FIXED_TOKENS)

    def test_closed_action_stage_and_json_fields_present(self) -> None:
        for value in (
            '"action":"ask"', '"stage":"initial"', '"complete":false', '"questions":[]',
            '"action":"summarize"', '"stage":"summary"', '"complete":true', '"key_findings":[]',
        ):
            self.assertIn(value, self.prompt)

    def test_all_fixed_phrases_present(self) -> None:
        for value in (FIXED_SAFETY_BOUNDARY, FIXED_TENDENCY, FIXED_NOTE, '"need_more_info":[]'):
            self.assertIn(value, self.prompt)

    def test_ask_limits_and_information_only_rule_present(self) -> None:
        for value in ("1至3个", "其后最多2项", "非诊疗信息", "不重复已回答"):
            self.assertIn(value, self.prompt)

    def test_literal_unique_stable_summary_rule_present(self) -> None:
        for value in ("字面事实", "非空、唯一", "首次出现顺序稳定去重", "不推断或补全"):
            self.assertIn(value, self.prompt)

    def test_ambiguous_frequency_rule_present(self) -> None:
        for value in ("最近频率增加", "独立逐字保留", "禁止绑定", "大便次数增加"):
            self.assertIn(value, self.prompt)

    def test_all_prohibitions_present(self) -> None:
        for value in ("禁止诊断", "证型或病因结论", "药物、药材、方剂、剂量", "可执行医疗建议", "治疗规划", "禁止编造"):
            self.assertIn(value, self.prompt)

    def test_safe_ask_passes_shared_policy(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False,
                  "questions": [FIXED_SAFETY_BOUNDARY, "这种情况从何时开始，每次持续多久？"]}
        validate_target_object(target)
        require_policy("我最近口干，这是什么问题？", target)

    def test_ask_without_boundary_fails(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False, "questions": ["这种情况多久了？"]}
        with self.assertRaises(ContractError):
            require_policy("我最近口干", target)

    def test_more_than_three_questions_fails(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False,
                  "questions": [FIXED_SAFETY_BOUNDARY, "一？", "二？", "三？"]}
        with self.assertRaises(ContractError):
            validate_target_object(target)

    def test_actionable_followup_fails(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False,
                  "questions": [FIXED_SAFETY_BOUNDARY, "建议立即服用某药。"]}
        with self.assertRaises(ContractError):
            require_policy("我最近口干", target)

    def test_safe_ambiguous_summary_passes(self) -> None:
        human = "手背发紧，手背发紧；最近频率增加，但没有写明指什么。"
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["手背发紧", "最近频率增加", "没有写明指什么"],
                  "syndrome_tendency": FIXED_TENDENCY, "need_more_info": [], "note": FIXED_NOTE}
        require_policy(human, target)

    def test_nonliteral_or_bound_summary_fails(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["大便次数增加"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("大便偏稀。最近频率增加。", target)

    def test_duplicate_summary_fails(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["口干", "口干"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            validate_target_object(target)

    def test_fixed_summary_fields_fail_when_changed(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["口干"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": ["脉象"], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("口干", target)

    def test_dosage_summary_fails(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["服用某物10克"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("服用某物10克", target)


if __name__ == "__main__":
    unittest.main()
