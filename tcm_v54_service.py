"""Frozen V5.4 R1 inference plus the Candidate H v2 public protocol.

Only user facts are passed between turns. Raw generation never reaches clients.
No training or heldout data is read by this module.
"""
from __future__ import annotations

import ast
import hashlib
import hmac
import json
import os
import secrets
import threading
import types
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_VERSION = "V5.4-R1"
STATE_VERSION = "V5.4-R1-API-1"
STACK = ROOT / "artifacts/v5_4_pipeline/final_stack_r1"
WEIGHTS = ROOT / "output/tcm-qwen-1.5b-v5-4-r1"
MANIFEST_SHA = "56e667bc1623f1361c2152ea91aed083e179ffd9ee6b6f675aba48b9d55356d1"
ADAPTER_SHA = "7a1772d3594dba6ddb30dab9c8a6db9ca1e82b44701756ca45b140097884568a"
RUNTIME_SHA = "ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
BASE_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(code)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_adapter():
    source = (STACK / "candidate_h_v2_adapter.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != ADAPTER_SHA:
        raise RuntimeError("Frozen Candidate H v2 hash mismatch")
    module = types.ModuleType("frozen_candidate_h_v2")
    exec(compile(source, str(STACK / "candidate_h_v2_adapter.py"), "exec"), module.__dict__)
    return module


def verify_stack() -> dict:
    manifest = STACK / "v5-4-r1_weights.sha256"
    if sha256(manifest) != MANIFEST_SHA:
        raise RuntimeError("V5.4 R1 manifest hash mismatch")
    expected = set()
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = WEIGHTS / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(WEIGHTS.resolve()):
            raise RuntimeError("Invalid frozen weight path")
        if sha256(path) != digest:
            raise RuntimeError("V5.4 R1 weight hash mismatch")
        expected.add(name)
    actual = {str(p.relative_to(WEIGHTS)) for p in WEIGHTS.rglob("*") if p.is_file()}
    if expected != actual or len(expected) != 51:
        raise RuntimeError("V5.4 R1 weight inventory mismatch")
    if sha256(ROOT / "tcm_chat_v5.py") != RUNTIME_SHA:
        raise RuntimeError("Frozen prompt source hash mismatch")
    if sha256(ROOT / "models/Qwen2.5-1.5B-Instruct/model.safetensors") != BASE_SHA:
        raise RuntimeError("Base model hash mismatch")
    frozen_adapter()
    return {"weights_files": len(expected), "weights_manifest_sha256": MANIFEST_SHA,
            "adapter_sha256": ADAPTER_SHA, "status": "PASS"}


def state_key() -> bytes:
    configured = os.getenv("TCM_API_STATE_SECRET")
    if configured:
        if len(configured.encode()) < 32:
            raise RuntimeError("TCM_API_STATE_SECRET must contain at least 32 bytes")
        return configured.encode()
    directory = ROOT / ".runtime"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / "state.key"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
    if len(key) < 32:
        raise RuntimeError("Invalid state signing key")
    return key


class V54Service:
    def __init__(self, *, generator=None, signing_key: bytes | None = None):
        self.adapter = frozen_adapter()
        self.generator = generator
        self.key = signing_key
        self.ready = False
        self.verification: dict = {}
        self.load_lock = threading.Lock()

    def load(self):
        with self.load_lock:
            if self.ready:
                return
            self.key = self.key or state_key()
            if self.generator is None:
                self.verification = verify_stack()
                import torch
                from peft import PeftModel
                from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

                base = ROOT / "models/Qwen2.5-1.5B-Instruct"
                self.tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, padding_side="left")
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                tree = ast.parse((ROOT / "tcm_chat_v5.py").read_text())
                self.prompt = next(ast.literal_eval(node.value) for node in tree.body
                                   if isinstance(node, ast.Assign) and any(
                                       isinstance(t, ast.Name) and t.id == "CONSULTATION_PROMPT" for t in node.targets))
                model = AutoModelForCausalLM.from_pretrained(
                    str(base), torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto", local_files_only=True, low_cpu_mem_usage=True)
                self.model = PeftModel.from_pretrained(model, str(WEIGHTS), is_trainable=False)
                self.model.eval()
                self.config = GenerationConfig(max_new_tokens=256, do_sample=False, repetition_penalty=1.05,
                                               eos_token_id=self.tokenizer.eos_token_id,
                                               pad_token_id=self.tokenizer.pad_token_id)
                self.generator = self._generate
            self.ready = True

    def _generate(self, text: str) -> str:
        import torch
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.prompt}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=True)
        length = encoded["input_ids"].shape[1]
        if length > 768:
            raise ServiceError(413, "context_too_long", "本次问诊内容超过长度上限，请缩短描述或新建问诊。")
        encoded = {k: v.to(self.model.get_input_embeddings().weight.device) for k, v in encoded.items()}
        with torch.inference_mode():
            result = self.model.generate(**encoded, generation_config=self.config)
        return self.tokenizer.decode(result[0][length:], skip_special_tokens=True).strip()

    def _sign(self, data: dict) -> dict:
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return {**data, "signature": hmac.new(self.key, encoded, hashlib.sha256).hexdigest()}

    def _state(self, state: dict | None, reset: bool) -> tuple[dict, str]:
        if reset or not state or state.get("state_version") != STATE_VERSION:
            event = "new_session" if reset or not state else "legacy_state_reset"
            return {"state_version": STATE_VERSION, "session_id": uuid.uuid4().hex,
                    "turn_count": 0, "facts": []}, event
        data = {k: v for k, v in state.items() if k != "signature"}
        signature = state.get("signature")
        if not isinstance(signature, str) or not hmac.compare_digest(self._sign(data)["signature"], signature):
            raise ServiceError(422, "invalid_state", "会话状态无效，请新建问诊。")
        return data, "continued"

    def process(self, text: str, *, state=None, new_session=False) -> dict[str, Any]:
        from tcm_gateway import (normalize_safety_result, safety_route, urgent_response, refer_response,
                                 finalize_gateway_result, detect_intent, handle_knowledge)
        if not self.ready:
            raise ServiceError(503, "model_not_ready", "模型尚未就绪，请稍后重试。")
        # Safety routing always gets the current message, even if the supplied state is invalid.
        safety = normalize_safety_result(safety_route(text))
        if safety in {"urgent", "refer"}:
            try:
                current, event = self._state(state, new_session)
            except ServiceError:
                current, event = self._state(None, True)
            result = finalize_gateway_result(urgent_response() if safety == "urgent" else refer_response())
        else:
            current, event = self._state(state, new_session)
            if detect_intent(text) == "knowledge":
                result = finalize_gateway_result(handle_knowledge(text))
            else:
                context = "。".join([*current["facts"], text])
                if current["turn_count"] >= 24 or len(context) > 8192:
                    raise ServiceError(413, "context_too_long", "本次问诊内容超过长度上限，请新建问诊。")
                raw = self.generator(context)
                protocol = self.adapter.adapt_interaction(context, raw)
                if protocol["action"] == "ask":
                    message = "\n".join(protocol["questions"])
                else:
                    findings = protocol["key_findings"]
                    message = ("已记录的情况：\n" + "\n".join("• " + f for f in findings)
                               if findings else "目前没有可整理的症状描述，请先补充你的情况。")
                    message += "\n" + protocol["note"]
                result = {"gateway_version": "V6.21.0", "action": protocol["action"],
                          "route": "consultation", "safety": "normal", "message": message,
                          "complete": protocol["complete"], "questions": protocol.get("questions", []),
                          "knowledge": None, "retrieval": None, "error": None, "meta": {"protocol": protocol}}
                facts = list(current["facts"])
                if protocol["action"] == "summarize":
                    # A request such as "请总结：口干三天" also supplies new facts.
                    candidates = protocol["key_findings"]
                else:
                    candidates = [self.adapter._information_scope(text)]
                # Commands from prior turns must not select the action of later turns.
                for fact in candidates:
                    if fact and not self.adapter._has_explicit_summary_command(fact) and not self.adapter._is_dangerous_request(fact):
                        if fact not in facts:
                            facts.append(fact)
                current = {**current, "turn_count": current["turn_count"] + 1, "facts": facts}
        result["state"] = self._sign(current)
        result.setdefault("meta", {}).update({"model_version": MODEL_VERSION, "adapter_version": "Candidate-H-v2",
                                               "model_used": result["route"] == "consultation",
                                               "raw_output_returned": False, "session_event": event})
        return result
