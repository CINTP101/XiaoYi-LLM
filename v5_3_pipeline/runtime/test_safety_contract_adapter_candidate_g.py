#!/usr/bin/env python3
"""Unit tests for the deterministic Candidate G safety adapter."""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

PROJECT = Path("/home/cyh/Medical_Qwen")
RUNTIME = PROJECT / "v5_3_pipeline/runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(PROJECT / "v5_3_pipeline/training"))

from safety_contract_adapter_candidate_g import (
    FIXED_NOTE,
    FIXED_SAFETY_BOUNDARY,
    FIXED_TENDENCY,
    adapt_prediction,
    adapt_prediction_object,
    contains_prescription_or_treatment_risk,
    detect_closed_action,
)
from v5_3_contract import validate_target_text


class CandidateGAdapterTests(unittest.TestCase):
    def assert_strict(self, output: str, action: str) -> dict:
        obj, label = validate_target_text(output)
        self.assertEqual(label[0], action)
        return obj

    def test_module_has_no_file_network_model_or_gpu_imports(self) -> None:
        source = RUNTIME / "safety_contract_adapter_candidate_g.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imports, {"__future__", "json", "re", "typing"})
        forbidden_calls = {"open", "print", "exec", "eval", "compile", "input"}
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls for node in ast.walk(tree)))

    def test_ask_discards_all_raw_questions(self) -> None:
        raw = '{"action":"ask","questions":["建议服用某药10克","异常措辞"]}'
        obj = self.assert_strict(adapt_prediction("我口干", raw), "ask")
        self.assertEqual(obj["questions"], [FIXED_SAFETY_BOUNDARY])

    def test_unknown_action_defaults_to_fixed_ask(self) -> None:
        obj = self.assert_strict(adapt_prediction("我口干", '{"action":"diagnose"}'), "ask")
        self.assertEqual(obj["questions"], [FIXED_SAFETY_BOUNDARY])

    def test_unparseable_output_defaults_to_fixed_ask(self) -> None:
        obj = self.assert_strict(adapt_prediction("我口干", "这不是JSON"), "ask")
        self.assertEqual(obj["questions"], [FIXED_SAFETY_BOUNDARY])

    def test_empty_user_defaults_to_fixed_ask(self) -> None:
        obj = self.assert_strict(adapt_prediction("   ", '{"action":"summarize"}'), "ask")
        self.assertEqual(obj["questions"], [FIXED_SAFETY_BOUNDARY])

    def test_summary_keeps_only_literal_findings(self) -> None:
        user = "口干，夜里容易醒。"
        raw = '{"action":"summarize","key_findings":["口干","夜里容易醒","阴虚"]}'
        obj = self.assert_strict(adapt_prediction(user, raw), "summarize")
        self.assertEqual(obj["key_findings"], ["口干", "夜里容易醒"])

    def test_summary_stable_deduplicates(self) -> None:
        user = "大便偏稀，大便偏稀，最近频率增加。"
        raw = '{"action":"summarize","key_findings":["大便偏稀","大便偏稀","最近频率增加"]}'
        obj = self.assert_strict(adapt_prediction(user, raw), "summarize")
        self.assertEqual(obj["key_findings"], ["大便偏稀", "最近频率增加"])

    def test_ambiguous_frequency_is_not_bound_or_completed(self) -> None:
        user = "大便偏稀。最近频率增加。"
        raw = '{"action":"summarize","key_findings":["大便次数增加","最近频率增加"]}'
        obj = self.assert_strict(adapt_prediction(user, raw), "summarize")
        self.assertEqual(obj["key_findings"], ["最近频率增加"])

    def test_no_literal_finding_falls_back_to_complete_user_text(self) -> None:
        user = "我只写了这一段原文。"
        raw = '{"action":"summarize","key_findings":["不存在的改写"]}'
        obj = self.assert_strict(adapt_prediction(user, raw), "summarize")
        self.assertEqual(obj["key_findings"], [user])

    def test_broken_tail_quote_still_recognizes_summary(self) -> None:
        user = "口干，夜里容易醒。"
        raw = '{"action":"summarize","key_findings":["口干","夜里容易醒"],"note":"坏引号”}'
        self.assertEqual(detect_closed_action(raw), "summarize")
        obj = self.assert_strict(adapt_prediction(user, raw), "summarize")
        self.assertEqual(obj["key_findings"], ["口干", "夜里容易醒"])

    def test_summary_fields_are_always_fixed(self) -> None:
        raw = '{"action":"summarize","key_findings":["口干"],"syndrome_tendency":"阴虚","need_more_info":["脉象"],"note":"改写"}'
        obj = self.assert_strict(adapt_prediction("口干", raw), "summarize")
        self.assertEqual(obj["syndrome_tendency"], FIXED_TENDENCY)
        self.assertEqual(obj["need_more_info"], [])
        self.assertEqual(obj["note"], FIXED_NOTE)

    def test_drug_request_summary_falls_back_to_ask(self) -> None:
        user = "我口干，应该吃什么药？"
        raw = '{"action":"summarize","key_findings":["口干"]}'
        self.assertTrue(contains_prescription_or_treatment_risk(user))
        self.assert_strict(adapt_prediction(user, raw), "ask")

    def test_dosage_summary_falls_back_to_ask(self) -> None:
        user = "记录里写了某物10克。"
        raw = '{"action":"summarize","key_findings":["记录里写了某物10克"]}'
        self.assert_strict(adapt_prediction(user, raw), "ask")

    def test_formula_summary_falls_back_to_ask(self) -> None:
        user = "有人让我试试小柴胡汤。"
        raw = '{"action":"summarize","key_findings":["小柴胡汤"]}'
        self.assert_strict(adapt_prediction(user, raw), "ask")

    def test_non_string_raw_prediction_defaults_to_ask(self) -> None:
        self.assert_strict(adapt_prediction("口干", {"action": "summarize"}), "ask")

    def test_output_is_deterministic(self) -> None:
        user = "口干，最近频率有变化。"
        raw = '{"action":"summarize","key_findings":["口干","最近频率有变化"]}<|im_end|>'
        outputs = {adapt_prediction(user, raw) for _ in range(20)}
        self.assertEqual(len(outputs), 1)

    def test_object_and_string_interfaces_match(self) -> None:
        user = "口干。"
        raw = '{"action":"summarize","key_findings":["口干"]}'
        self.assertEqual(json.loads(adapt_prediction(user, raw)), adapt_prediction_object(user, raw))

    def test_action_regex_is_anchored_and_closed(self) -> None:
        self.assertIsNone(detect_closed_action('说明文字 {"action":"summarize"}'))
        self.assertIsNone(detect_closed_action('{"action":"other"}'))
        self.assertEqual(detect_closed_action('{"action":"ask","stage":"initial"}'), "ask")


if __name__ == "__main__":
    unittest.main()
