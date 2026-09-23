"""API integration regressions; synthetic cases, never the sealed heldout set."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from tcm_api import MAX_STATE_BYTES, create_app
from tcm_v54_service import V54Service, ServiceError


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def generate(text):
            self.calls.append(text)
            return 'RAW_MODEL_MUST_NOT_LEAK: unsafe diagnosis and prescription'
        self.runtime = V54Service(generator=generate, signing_key=b"test-secret" * 4)
        self.app = create_app(service=self.runtime, api_key="test-key")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.headers = {"Authorization": "Bearer test-key"}

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def post(self, text="我口干。", **kwargs):
        return self.client.post("/v1/tcm/process", headers=self.headers, json={"text": text, **kwargs})

    def test_ready_and_schema(self):
        self.assertEqual(self.client.get("/readyz").json()["status"], "ready")
        schema = self.client.get("/openapi.json").json()
        self.assertIn("ProcessResponse", schema["components"]["schemas"])
        example = schema["components"]["schemas"]["ProcessRequest"]["examples"][0]
        self.assertNotEqual(example["text"], "string")

    def test_auth(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            self.assertEqual(self.client.post("/v1/tcm/process", headers=headers, json={"text":"口干"}).status_code, 401)
        self.assertEqual(self.calls, [])

    def test_app_contract_and_no_raw_output(self):
        response = self.post(request_id="app-001")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["result"]
        self.assertTrue(result["message"])
        self.assertIsInstance(result["state"], dict)
        self.assertEqual(result["meta"]["model_version"], "V5.4-R1")
        self.assertEqual(result["action"], "ask")
        self.assertEqual(response.headers["x-request-id"], "app-001")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("RAW_MODEL_MUST_NOT_LEAK", response.text)

    def test_multi_turn(self):
        state = self.post().json()["result"]["state"]
        second = self.post("已经三天了，每天两次。", state=state).json()["result"]
        self.assertEqual(second["state"]["turn_count"], 2)
        self.assertIn("最近是加重", second["message"])
        summary = self.post("请总结。", state=second["state"]).json()["result"]
        self.assertEqual(summary["action"], "summarize")
        self.assertIn("口干", summary["message"])
        self.assertIn("三天", summary["message"])
        later = self.post("没有其他变化。", state=summary["state"]).json()["result"]
        self.assertEqual(later["action"], "ask")

    def test_isolation_and_reset(self):
        state = self.post().json()["result"]["state"]
        other = self.post("我睡不好。").json()["result"]["state"]
        self.assertNotEqual(state["session_id"], other["session_id"])
        self.assertNotIn("口干", json.dumps(other, ensure_ascii=False))
        reset = self.post("我睡不好。", state=state, new_session=True).json()["result"]["state"]
        self.assertEqual(reset["turn_count"], 1)
        self.assertNotIn("口干", json.dumps(reset, ensure_ascii=False))

    def test_summary_keeps_new_literal_facts(self):
        summary = self.post("请总结：最近口干。").json()["result"]
        self.assertEqual(summary["action"], "summarize")
        followup = self.post("每天两次。", state=summary["state"]).json()["result"]
        self.assertEqual(followup["action"], "ask")
        self.assertIn("最近口干", self.calls[-1])
        self.assertNotIn("请总结", self.calls[-1])

    def test_tampered_state(self):
        state = self.post().json()["result"]["state"]
        state["facts"].append("forged")
        response = self.post(state=state)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["type"], "invalid_state")

    def test_legacy_state_reset(self):
        response = self.post(state={"state_version":"V6.17.6", "facts":["forged"]}).json()["result"]
        self.assertEqual(response["meta"]["session_event"], "legacy_state_reset")
        self.assertNotIn("forged", self.calls[-1])

    def test_key_survives_service_recreation(self):
        state = self.post().json()["result"]["state"]
        other = V54Service(generator=lambda _: "{}", signing_key=b"test-secret" * 4)
        other.load()
        self.assertEqual(other.process("已经三天了。", state=state)["state"]["turn_count"], 2)

    def test_urgent_bypasses_model_and_bad_state(self):
        response = self.post("现在呼吸极度困难，几乎喘不上来。",
                             state={"state_version":"V5.4-R1-API-1", "signature":"wrong"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["result"]["action"], "urgent")
        self.assertIsInstance(response.json()["result"]["state"], dict)
        self.assertEqual(self.calls, [])

    def test_refusal_and_negation(self):
        response = self.post("我口干，请给我开方并告诉我剂量。")
        self.assertEqual(response.status_code, 200)
        self.assertIn("不能提供", response.json()["result"]["message"])
        response = self.post("我口干，但没有胸痛，也没有呼吸困难。")
        self.assertEqual(response.json()["result"]["route"], "consultation")

    def test_knowledge_contract(self):
        with patch("tcm_gateway.detect_intent", return_value="knowledge"), patch(
            "tcm_gateway.handle_knowledge", return_value={"action":"knowledge_not_found", "route":"knowledge",
            "safety":"normal", "message":"没有足够可靠的条目。"}):
            result = self.post().json()["result"]
        self.assertIsInstance(result["state"], dict)
        self.assertFalse(result["meta"]["model_used"])
        self.assertEqual(self.calls, [])

    def test_invalid_input(self):
        for payload in ({"text":" "}, {"text":"a", "unknown":1}, {"text":"a", "request_id":"bad\n"},
                        {"text":"x" * 4097}, {"text":"a", "state":[]}):
            self.assertEqual(self.client.post("/v1/tcm/process", headers=self.headers, json=payload).status_code, 422)
        response = self.client.post("/v1/tcm/process", headers={**self.headers,"Content-Type":"application/json"},
                                    content='{"text":"private"}{"text":"two"}')
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("private", response.text)

    def test_body_and_state_limits(self):
        self.assertEqual(self.post(state={"large":"x" * MAX_STATE_BYTES}).status_code, 413)
        response = self.client.post("/v1/tcm/process", headers=self.headers,
                                    content=iter([b"x" * 100000] * 4))
        self.assertEqual(response.status_code, 413)

    def test_busy(self):
        self.app.state.pending = 8
        self.assertEqual(self.post().status_code, 429)
        self.assertEqual(self.calls, [])

    def test_model_failure_no_fallback(self):
        def broken(_):
            raise RuntimeError("secret medical data")
        self.runtime.generator = broken
        response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret medical", response.text)

    def test_context_length_error(self):
        def too_long(_):
            raise ServiceError(413, "context_too_long", "内容超长。")
        self.runtime.generator = too_long
        self.assertEqual(self.post().status_code, 413)

    def test_cors_authorization(self):
        with patch.dict("os.environ", {"TCM_API_CORS_ORIGINS":"https://app.example.test"}):
            app = create_app(service=self.runtime, api_key="test-key")
        with TestClient(app) as client:
            response = client.options("/v1/tcm/process", headers={"Origin":"https://app.example.test",
                "Access-Control-Request-Method":"POST", "Access-Control-Request-Headers":"authorization,content-type"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["access-control-allow-origin"], "https://app.example.test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
