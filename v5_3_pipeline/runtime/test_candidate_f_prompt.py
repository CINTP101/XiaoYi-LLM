#!/usr/bin/env python3
"""Static unit tests for the Candidate F prompt and its closed JSON policy."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

PROJECT = Path("/home/cyh/Medical_Qwen")
RUNTIME_DIR = PROJECT / "v5_3_pipeline/runtime"
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))
sys.path.insert(0, str(RUNTIME_DIR))

from candidate_f_policy import FIXED_NOTE, FIXED_SAFETY_BOUNDARY, FIXED_TENDENCY, require_policy
from v5_3_contract import ContractError, validate_target_object


PROMPT_SOURCE = RUNTIME_DIR / "consultation_prompt_candidate_f.py"


def load_literal_prompt() -> str:
    tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"), filename=str(PROMPT_SOURCE))
    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "CONSULTATION_PROMPT" for target in node.targets):
                assignments.append(node)
    if len(assignments) != 1:
        raise AssertionError("prompt module must define CONSULTATION_PROMPT exactly once")
    value = ast.literal_eval(assignments[0].value)
    if not isinstance(value, str) or not value.strip():
        raise AssertionError("prompt must be a non-empty literal string")
    return value


class CandidateFPromptTests(unittest.TestCase):
    def test_prompt_module_has_no_imports_calls_or_definitions(self) -> None:
        tree = ast.parse(PROMPT_SOURCE.read_text(encoding="utf-8"))
        forbidden = (ast.Import, ast.ImportFrom, ast.Call, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        self.assertFalse(any(isinstance(node, forbidden) for node in ast.walk(tree)))
        load_literal_prompt()

    def test_prompt_contains_closed_contract_and_fixed_phrases(self) -> None:
        prompt = load_literal_prompt()
        for required in (
            '"action": "ask"', '"stage": "initial"', '"complete": false',
            '"action": "summarize"', '"stage": "summary"', '"complete": true',
            FIXED_SAFETY_BOUNDARY, FIXED_TENDENCY, FIXED_NOTE,
            "最近频率增加", "不得把它连接到任何症状", "最多再问 2 个问题",
            "禁止输出具体或泛化的药物、药材、方剂、剂量",
        ):
            self.assertIn(required, prompt)

    def test_safe_ask_passes_contract_and_policy(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False, "questions": [
            FIXED_SAFETY_BOUNDARY, "这种变化最早何时出现，每次持续多久？", "是否还有同时出现的其他变化？",
        ]}
        validate_target_object(target)
        require_policy("最近口中发黏，这是什么问题？", target)

    def test_safe_summary_passes_contract_and_policy(self) -> None:
        human = "手背发紧，手背发紧；另记最近频率增加，但没有说明指什么。"
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["手背发紧", "最近频率增加", "没有说明指什么"],
                  "syndrome_tendency": FIXED_TENDENCY, "need_more_info": [], "note": FIXED_NOTE}
        validate_target_object(target)
        require_policy(human, target)

    def test_ask_without_fixed_boundary_fails(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False, "questions": ["这种情况多久了？"]}
        with self.assertRaises(ContractError):
            require_policy("我不舒服", target)

    def test_more_than_three_questions_fails_shared_contract(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False,
                  "questions": [FIXED_SAFETY_BOUNDARY, "问题一？", "问题二？", "问题三？"]}
        with self.assertRaises(ContractError):
            validate_target_object(target)

    def test_actionable_followup_fails(self) -> None:
        target = {"action": "ask", "stage": "initial", "complete": False,
                  "questions": [FIXED_SAFETY_BOUNDARY, "建议立即服用某药。"]}
        with self.assertRaises(ContractError):
            require_policy("我不舒服", target)

    def test_nonliteral_summary_fact_fails(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["大便次数增加"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("大便偏稀。最近频率增加。", target)

    def test_ambiguous_phrase_must_be_standalone(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["口干最近频率增加"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("口干最近频率增加，但没有写明频率指什么。", target)

    def test_duplicate_summary_finding_fails_shared_contract(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["口干", "口干"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            validate_target_object(target)

    def test_fixed_summary_fields_are_enforced(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["口干"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": ["舌象"], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("口干", target)

    def test_dosage_in_summary_fails(self) -> None:
        target = {"action": "summarize", "stage": "summary", "complete": True,
                  "key_findings": ["服用某物10克"], "syndrome_tendency": FIXED_TENDENCY,
                  "need_more_info": [], "note": FIXED_NOTE}
        with self.assertRaises(ContractError):
            require_policy("服用某物10克", target)


if __name__ == "__main__":
    unittest.main()
