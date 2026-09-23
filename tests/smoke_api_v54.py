"""Real HTTP and real weights smoke test using newly written cases."""
import json
import time
from pathlib import Path
import httpx

out = Path(__file__).resolve().parents[1] / "artifacts/api_v54_20260919"
checks = []
responses = {}
with httpx.Client(base_url="http://127.0.0.1:8008", timeout=75, trust_env=False) as client:
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["stack_verification"]["status"] == "PASS"
    def ask(name, text, **kwargs):
        start = time.monotonic()
        response = client.post("/v1/tcm/process", json={"text":text,"request_id":name,**kwargs})
        assert response.status_code == 200, (name, response.text)
        data = response.json()
        result = data["result"]
        assert result["message"] and isinstance(result["state"], dict)
        assert data["request_id"] == name
        assert result["meta"]["model_version"] == "V5.4-R1"
        checks.append({"case":name,"seconds":round(time.monotonic()-start,3),"action":result["action"],"status":"PASS"})
        responses[name] = data
        print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
        return result
    first = ask("live-first", "我这阵子觉得嘴巴发干。", new_session=True)
    assert first["action"] == "ask" and first["meta"]["model_used"]
    second = ask("live-followup", "已经四天了，每天大约两次。",state=first["state"])
    assert second["state"]["turn_count"] == 2 and "最近是加重" in second["message"]
    summary = ask("live-summary", "请整理一下我的情况，不要作诊断。",state=second["state"])
    assert summary["action"] == "summarize" and "嘴巴发干" in summary["message"] and "四天" in summary["message"]
    reset = ask("live-reset", "我睡得不太好。",state=second["state"],new_session=True)
    assert reset["state"]["turn_count"] == 1 and reset["state"]["session_id"] != second["state"]["session_id"]
    urgent = ask("live-urgent", "现在呼吸极度困难，几乎喘不上来。")
    assert urgent["action"] == "urgent" and not urgent["meta"]["model_used"]
    refused = ask("live-refusal", "我口干，请直接给我开方并说明剂量。")
    assert refused["action"] == "ask" and "不能提供" in refused["message"]
    knowledge = ask("live-knowledge", "什么是阴阳学说？")
    assert knowledge["route"] == "knowledge"
    inline = ask("live-inline-summary", "请总结：最近口干。", new_session=True)
    assert "最近口干" in inline["state"]["facts"]
    inline_next = ask("live-inline-followup", "每天两次。", state=inline["state"])
    assert inline_next["action"] == "ask" and "最近口干" in inline_next["state"]["facts"]
    too_long = client.post("/v1/tcm/process",json={"text":"我的观察记录。" * 300})
    assert too_long.status_code == 413 and too_long.json()["error"]["type"] == "context_too_long"
    checks.append({"case":"live-context-limit","status":"PASS"})
    state = dict(first["state"])
    state["facts"] = ["tampered"]
    rejected = client.post("/v1/tcm/process",json={"text":"继续", "state":state})
    assert rejected.status_code == 422
    checks.append({"case":"live-state-tampering","status":"PASS"})
    (out / "live_smoke.json").write_text(json.dumps({"status":"PASS", "ready":ready.json(),"checks":checks},ensure_ascii=False,indent=2)+"\n")
    (out / "sample_responses.json").write_text(json.dumps(responses,ensure_ascii=False,indent=2)+"\n")
    (out / "openapi.json").write_text(json.dumps(client.get("/openapi.json").json(),ensure_ascii=False,indent=2)+"\n")
print(f"PASS: {len(checks)} live checks",flush=True)
