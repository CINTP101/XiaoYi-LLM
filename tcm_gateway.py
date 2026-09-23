"""
TCM Gateway V6.17.3

Unified pipeline:

User/App
 ↓
Safety Router
 ├ urgent -> END
 ├ refer  -> END
 └ normal
      ↓
 Intent Router
 ├ knowledge
 │    ↓
 │ Retriever
 │    ↓
 │ Verified KB
 │
 └ consultation
      ↓
 Stateful Consultation Adapter
      ↓
 V5.1
      ↓
 ask / summarize

V6.17.3:
- Gateway accepts consultation state
- Gateway returns consultation state
- State can be round-tripped by App
- Existing one-argument consultation handlers remain compatible
"""

import inspect

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

from safety_router import (
    route as safety_route,
)

from intent_router import (
    detect_intent,
)

GATEWAY_VERSION = "V6.17.6"


# =========================================================
# Safety normalization
# =========================================================

def normalize_safety_result(
    raw: Any,
) -> str:

    if raw is None:
        return "normal"


    if isinstance(
        raw,
        str,
    ):

        value = raw.strip().lower()

        if value == "urgent":
            return "urgent"

        if value == "refer":
            return "refer"

        return "normal"


    if isinstance(
        raw,
        dict,
    ):

        for key in (
            "route",
            "action",
            "safety",
            "level",
            "result",
        ):

            value = raw.get(
                key
            )

            if not isinstance(
                value,
                str,
            ):
                continue

            value = (
                value
                .strip()
                .lower()
            )

            if value == "urgent":
                return "urgent"

            if value == "refer":
                return "refer"


    return "normal"


# =========================================================
# Raw internal result
# =========================================================

def gateway_result(
    *,
    action: str,
    route: str,
    safety: str,
    message: Optional[str] = None,
    **extra,
) -> Dict[str, Any]:

    result = {
        "gateway_version":
            GATEWAY_VERSION,

        "action":
            action,

        "route":
            route,

        "safety":
            safety,

        "message":
            message,
    }

    result.update(
        extra
    )

    return result


# =========================================================
# Safety responses
# =========================================================

def urgent_response():

    return gateway_result(
        action="urgent",
        route="urgent",
        safety="urgent",
        message=(
            "当前描述可能涉及需要及时评估的情况。"
            "请尽快寻求急诊或当地紧急医疗服务，"
            "不要继续依赖在线问诊判断。"
        ),
    )


def refer_response():

    return gateway_result(
        action="refer",
        route="refer",
        safety="refer",
        message=(
            "目前情况建议进行线下面诊或进一步检查，"
            "以便获得更完整的医学评估。"
        ),
    )


# =========================================================
# Knowledge path
# =========================================================

def handle_knowledge(
    text: str,
):

    # Loading the embedding model is relatively expensive.
    # Keep it out of urgent/refer/consultation cold paths so
    # the safety router can respond without retrieval startup.
    from rag.retriever import retrieve

    result = retrieve(
        text,
        top_k=3,
    )


    if not result.get(
        "found"
    ):

        return gateway_result(
            action=
                "knowledge_not_found",

            route=
                "knowledge",

            safety=
                "normal",

            message=(
                "当前已验证知识库中没有找到"
                "足够可靠的对应条目。"
            ),

            retrieval={
                "found":
                    False,

                "method":
                    result.get(
                        "method"
                    ),

                "confidence":
                    result.get(
                        "confidence"
                    ),

                "query_type":
                    result.get(
                        "query_type"
                    ),

                "category_hint":
                    result.get(
                        "category_hint"
                    ),
            },
        )


    results = result.get(
        "results",
        [],
    )


    if not results:

        return gateway_result(
            action=
                "knowledge_not_found",

            route=
                "knowledge",

            safety=
                "normal",

            message=(
                "检索结果异常："
                "未获得可用知识条目。"
            ),
        )


    top = results[0]

    content = top.get(
        "content"
    )


    if not content:

        return gateway_result(
            action=
                "knowledge_not_found",

            route=
                "knowledge",

            safety=
                "normal",

            message=(
                "知识条目存在，"
                "但缺少可验证正文。"
            ),
        )


    return gateway_result(
        action="answer",
        route="knowledge",
        safety="normal",

        # Verified KB content is returned directly.
        message=content,

        knowledge={
            "id":
                top.get("id"),

            "title":
                top.get("title"),

            "category":
                top.get("category"),

            "source":
                top.get("source"),

            "verified":
                top.get("verified"),
        },

        retrieval={
            "found":
                True,

            "method":
                result.get(
                    "method"
                ),

            "confidence":
                result.get(
                    "confidence"
                ),

            "query_type":
                result.get(
                    "query_type"
                ),

            "category_hint":
                result.get(
                    "category_hint"
                ),

            "field_rescue":
                result.get(
                    "field_rescue"
                ),
        },
    )


# =========================================================
# Consultation handler compatibility
# =========================================================

def handler_accepts_state(
    handler,
) -> bool:

    """
    V6.17.3 state-aware handlers:
        handler(text, state=state)

    Legacy regression mocks:
        handler(text)

    Both remain supported.
    """

    try:

        signature = inspect.signature(
            handler
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


    parameters = signature.parameters


    if "state" in parameters:

        return True


    return any(
        p.kind
        == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )


def call_consultation_handler(
    handler,
    text,
    state,
):

    if handler_accepts_state(
        handler
    ):

        return handler(
            text,
            state=state,
        )


    # Legacy handler compatibility.
    return handler(
        text
    )


# =========================================================
# Consultation path
# =========================================================

def handle_consultation(
    text: str,
    consultation_handler: Optional[
        Callable
    ] = None,
    state: Optional[
        Dict[str, Any]
    ] = None,
):

    # -----------------------------------------------------
    # Default V6.17 state-aware adapter
    # -----------------------------------------------------

    if consultation_handler is None:

        from tcm_consultation_adapter import (
            consult as consultation_handler,
        )


    response = call_consultation_handler(
        consultation_handler,
        text,
        state,
    )


    if not isinstance(
        response,
        dict,
    ):

        return gateway_result(
            action="error",
            route="consultation",
            safety="normal",
            message=(
                "问诊模块返回了非结构化结果。"
            ),
        )


    result = dict(
        response
    )


    action = str(
        result.get(
            "action",
            "",
        )
    ).strip().lower()


    # =====================================================
    # Defense in depth
    #
    # Consultation model may only:
    # ask / summarize / error
    #
    # It cannot authoritatively route:
    # urgent / refer / knowledge / answer
    # =====================================================

    if action not in (
        "ask",
        "summarize",
        "error",
    ):

        return gateway_result(
            action="error",
            route="consultation",
            safety="normal",
            message=(
                "问诊模块返回了不允许的动作。"
            ),
        )


    # Gateway fields are authoritative.
    result[
        "gateway_version"
    ] = GATEWAY_VERSION

    result[
        "route"
    ] = "consultation"

    result[
        "safety"
    ] = "normal"


    return result


# =========================================================
# Internal raw process
# =========================================================

def _process_raw(
    text: str,
    consultation_handler=None,
    state=None,
):

    if text is None:

        return gateway_result(
            action="error",
            route="invalid",
            safety="normal",
            message="输入不能为空。",
        )


    text = str(
        text
    ).strip()


    if not text:

        return gateway_result(
            action="error",
            route="invalid",
            safety="normal",
            message="输入不能为空。",
        )


    # =====================================================
    # 1. SAFETY FIRST
    # =====================================================

    raw_safety = safety_route(
        text
    )

    safety = normalize_safety_result(
        raw_safety
    )


    if safety == "urgent":

        return urgent_response()


    if safety == "refer":

        return refer_response()


    # =====================================================
    # 2. INTENT
    # =====================================================

    intent = detect_intent(
        text
    )


    # =====================================================
    # 3. KNOWLEDGE
    # =====================================================

    if intent == "knowledge":

        return handle_knowledge(
            text
        )


    # =====================================================
    # 4. CONSULTATION
    # =====================================================

    return handle_consultation(
        text,
        consultation_handler=
            consultation_handler,
        state=state,
    )


# =========================================================
# Unified public output contract
# =========================================================

PUBLIC_KEYS = (
    "gateway_version",
    "action",
    "route",
    "safety",
    "message",
    "complete",
    "questions",
    "knowledge",
    "retrieval",
    "state",
    "error",
    "meta",
)


def _normalize_questions(
    value,
):

    if value is None:
        return []


    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        return (
            [value]
            if value
            else []
        )


    if isinstance(
        value,
        list,
    ):

        result = []

        for item in value:

            if not isinstance(
                item,
                str,
            ):
                continue

            item = item.strip()

            if item:
                result.append(
                    item
                )

        return result


    return []


def normalize_public_state(
    raw_state,
    request_state=None,
):

    """
    Consultation response state has priority.

    For urgent / refer / knowledge:
    preserve an existing incoming consultation state
    without modifying it.

    No state supplied:
    return None for non-consultation requests.
    """

    from tcm_consultation_state import (
        normalize_state,
    )


    if isinstance(
        raw_state,
        dict,
    ):

        return normalize_state(
            raw_state
        )


    if isinstance(
        request_state,
        dict,
    ):

        return normalize_state(
            request_state
        )


    return None


def finalize_gateway_result(
    raw,
    request_state=None,
):

    if not isinstance(
        raw,
        dict,
    ):

        raw = {
            "action":
                "error",

            "route":
                "invalid",

            "safety":
                "normal",

            "message":
                "Gateway返回了无效结果。",

            "error_type":
                "invalid_gateway_result",
        }


    action = str(
        raw.get(
            "action",
            "error",
        )
    ).strip().lower()


    route = str(
        raw.get(
            "route",
            "invalid",
        )
    ).strip().lower()


    safety = str(
        raw.get(
            "safety",
            "normal",
        )
    ).strip().lower()


    message = raw.get(
        "message"
    )


    if (
        message is not None
        and not isinstance(
            message,
            str,
        )
    ):

        message = str(
            message
        )


    # =====================================================
    # Complete
    # =====================================================

    if "complete" in raw:

        complete = bool(
            raw.get(
                "complete"
            )
        )

    elif action in (
        "urgent",
        "refer",
        "answer",
        "knowledge_not_found",
        "summarize",
    ):

        complete = True

    else:

        complete = False


    if action == "error":

        complete = False


    # =====================================================
    # Questions
    # =====================================================

    questions = _normalize_questions(
        raw.get(
            "questions"
        )
    )


    # =====================================================
    # Knowledge / Retrieval
    # =====================================================

    knowledge = raw.get(
        "knowledge"
    )

    if not isinstance(
        knowledge,
        dict,
    ):

        knowledge = None


    retrieval = raw.get(
        "retrieval"
    )

    if not isinstance(
        retrieval,
        dict,
    ):

        retrieval = None


    # =====================================================
    # V6.17 State
    # =====================================================

    public_state = normalize_public_state(
        raw.get(
            "state"
        ),
        request_state=
            request_state,
    )


    # =====================================================
    # Error
    # =====================================================

    error = None


    if action == "error":

        error = {
            "type": (
                raw.get(
                    "error_type"
                )
                or "gateway_error"
            )
        }


    # =====================================================
    # Metadata
    # =====================================================

    meta = {}


    adapter_version = raw.get(
        "adapter_version"
    )

    if adapter_version:

        meta[
            "adapter_version"
        ] = adapter_version


    binding = raw.get(
        "binding"
    )

    if isinstance(
        binding,
        dict,
    ):

        meta[
            "binding"
        ] = binding


    question_policy = raw.get(
        "question_policy"
    )

    if isinstance(
        question_policy,
        dict,
    ):

        meta[
            "question_policy"
        ] = question_policy


    session_event = raw.get(
        "session_event"
    )

    if isinstance(
        session_event,
        str,
    ):

        meta[
            "session_event"
        ] = session_event


    return {
        "gateway_version":
            GATEWAY_VERSION,

        "action":
            action,

        "route":
            route,

        "safety":
            safety,

        "message":
            message,

        "complete":
            complete,

        "questions":
            questions,

        "knowledge":
            knowledge,

        "retrieval":
            retrieval,

        "state":
            public_state,

        "error":
            error,

        "meta":
            meta,
    }


# =========================================================
# Official public API
# =========================================================

def process(
    text,
    consultation_handler=None,
    state=None,
    new_session=False,
):

    # =====================================================
    # V6.17.6 Explicit New Consultation
    #
    # App usage:
    #
    # process(
    #     text,
    #     state=old_state,
    #     new_session=True,
    # )
    #
    # This must discard old consultation facts/history.
    # =====================================================

    if new_session:

        from tcm_consultation_state import (
            new_state,
        )

        state = new_state()


    raw = _process_raw(
        text,
        consultation_handler=
            consultation_handler,
        state=state,
    )


    result = finalize_gateway_result(
        raw,
        request_state=state,
    )


    # Explicit reset is authoritative even when the
    # current request routes to knowledge/safety.
    if new_session:

        result[
            "meta"
        ][
            "session_event"
        ] = "new_session"


    return result


route_request = process
