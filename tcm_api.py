"""App-compatible HTTP API for the frozen V5.4 R1 stack."""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from tcm_v54_service import MODEL_VERSION, ServiceError, V54Service

API_VERSION = "V6.21.0"
DEFAULT_HOST, DEFAULT_PORT = "127.0.0.1", 8008
MAX_STATE_BYTES = 256 * 1024
MAX_BODY_BYTES = 320 * 1024
logger = logging.getLogger("tcm_api")


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "text": "我最近口干，而且晚上会出汗。", "state": None, "new_session": True,
        "request_id": "app-001"}]})
    text: str = Field(min_length=1, max_length=4096)
    state: Optional[dict[str, Any]] = None
    new_session: bool = False
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128,
                                      pattern=r"^[A-Za-z0-9._:-]+$")

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class GatewayResult(BaseModel):
    gateway_version: str
    action: str
    route: str
    safety: str
    message: str
    complete: bool
    questions: list[str]
    knowledge: Optional[dict[str, Any]] = None
    retrieval: Optional[dict[str, Any]] = None
    state: dict[str, Any]
    error: Optional[dict[str, Any]] = None
    meta: dict[str, Any]


class ProcessResponse(BaseModel):
    api_version: str
    request_id: str
    result: GatewayResult


def error_response(status: int, code: str, message: str, request_id: str | None = None):
    request_id = request_id or uuid.uuid4().hex
    return JSONResponse(status_code=status, content={"api_version": API_VERSION,
                        "request_id": request_id, "error": {"type": code, "message": message}},
                        headers={"X-Request-ID": request_id, "Cache-Control": "no-store",
                                 **({"WWW-Authenticate": "Bearer"} if status == 401 else {}),
                                 **({"Retry-After": "5"} if status == 429 else {})})


class BodyLimitMiddleware:
    """Bound bytes before JSON parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                response = error_response(413, "request_too_large", "请求内容过大。")
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(processor: Optional[Callable] = None, *, service=None, api_key=None) -> FastAPI:
    runtime = service or (V54Service() if processor is None else None)
    expected_key = os.getenv("TCM_API_KEY", "") if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(app):
        if runtime is not None:
            await run_in_threadpool(runtime.load)
        yield

    app = FastAPI(title="XiaoMedInsight TCM API", version=API_VERSION, lifespan=lifespan,
                  description="V5.4 R1 + Candidate H v2。接收用户描述，返回追问或事实摘要。")
    app.add_middleware(BodyLimitMiddleware)
    origins = [s.strip() for s in os.getenv("TCM_API_CORS_ORIGINS", "").split(",") if s.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
                           expose_headers=["X-Request-ID"])
    app.state.runtime = runtime
    app.state.processor = processor or runtime.process
    app.state.process_lock = asyncio.Lock()
    app.state.pending = 0
    bearer = HTTPBearer(auto_error=False)

    async def authorize(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        if expected_key and (credentials is None or not hmac.compare_digest(
                credentials.credentials.encode(), expected_key.encode())):
            raise ServiceError(401, "unauthorized", "访问凭据无效或未提供。")

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error_response(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(422, "invalid_request", "请求格式错误：请提交一个 JSON 对象，text 必须为非空文本。")

    @app.get("/", include_in_schema=False)
    async def index():
        return {"service": "tcm-gateway-api", "api_version": API_VERSION,
                "model_version": MODEL_VERSION, "docs": "/docs", "endpoint": "/v1/tcm/process"}

    @app.get("/healthz", tags=["system"])
    async def health():
        return {"status": "ok", "api_version": API_VERSION, "model_version": MODEL_VERSION}

    @app.get("/readyz", tags=["system"])
    async def ready():
        is_ready = runtime is None or runtime.ready
        return JSONResponse(status_code=200 if is_ready else 503, content={
            "status": "ready" if is_ready else "not_ready", "api_version": API_VERSION,
            "model_version": MODEL_VERSION, "adapter_version": "Candidate-H-v2",
            "authentication": "bearer" if expected_key else "disabled",
            "stack_verification": runtime.verification if runtime else {"status": "test_processor"}})

    @app.post("/v1/tcm/process", tags=["gateway"], response_model=ProcessResponse,
              dependencies=[Depends(authorize)], responses={401: {"description": "访问凭据无效"},
                  413: {"description": "内容或上下文超长"}, 429: {"description": "队列已满，稍后重试"},
                  503: {"description": "模型未就绪"}})
    async def process_turn(request: ProcessRequest):
        request_id = request.request_id or uuid.uuid4().hex
        if len(json.dumps(request.state, ensure_ascii=False).encode()) > MAX_STATE_BYTES:
            return error_response(413, "state_too_large", "state 超过 256 KiB。", request_id)
        if app.state.pending >= 8:
            return error_response(429, "server_busy", "服务繁忙，请稍后重试。", request_id)
        app.state.pending += 1
        locked = False
        try:
            try:
                await asyncio.wait_for(app.state.process_lock.acquire(), timeout=10)
                locked = True
            except TimeoutError:
                return error_response(429, "server_busy", "服务繁忙，请稍后重试。", request_id)
            result = await run_in_threadpool(app.state.processor, request.text,
                                            state=request.state, new_session=request.new_session)
            if not isinstance(result, dict):
                return error_response(500, "invalid_gateway_result", "服务返回无效结果。", request_id)
            response = ProcessResponse(api_version=API_VERSION, request_id=request_id, result=result)
            return JSONResponse(content=response.model_dump(), headers={"X-Request-ID": request_id,
                                                                        "Cache-Control": "no-store"})
        except ServiceError as exc:
            return error_response(exc.status, exc.code, exc.message, request_id)
        except Exception as exc:
            # Do not log patient text, state, raw generation or exception messages.
            logger.error("request failed request_id=%s exception=%s", request_id, type(exc).__name__)
            return error_response(500, "internal_error", "请求处理失败，请稍后重试。", request_id)
        finally:
            if locked:
                app.state.process_lock.release()
            app.state.pending -= 1

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Run V5.4 R1 App API")
    parser.add_argument("--host", default=os.getenv("TCM_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("TCM_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--allow-unauthenticated", action="store_true", help="仅用于受控局域网调试")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("TCM_API_KEY") and not args.allow_unauthenticated:
        parser.error("非本机监听需设置 TCM_API_KEY；受控局域网调试可使用 --allow-unauthenticated")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1, proxy_headers=False)


if __name__ == "__main__":
    main()
