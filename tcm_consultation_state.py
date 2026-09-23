import re
"""
TCM Consultation State Machine V6.17.1

Deterministic multi-turn state layer.

Principles:
- Missing information remains None.
- Never invent tongue / pulse / age / gender / name.
- A short answer can be bound to the field asked in the
  previous turn.
- "I don't know my pulse" marks pulse as unknown so the
  system does not repeatedly ask for it.
- State is JSON serializable.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional
import uuid


STATE_VERSION = "V6.17.6"


# =========================================================
# Canonical consultation fields
# =========================================================

FACT_FIELDS = (
    "name",
    "age",
    "gender",

    "chief_complaint",
    "duration",
    "associated_symptoms",

    "sleep",
    "appetite",
    "stool",
    "urination",

    "temperature",
    "sweating",

    "tongue",
    "pulse",
)


UNKNOWN_PHRASES = (
    "不知道",
    "不清楚",
    "不晓得",
    "不会看",
    "没看",
    "没有看",
    "没注意",
    "没有注意",
    "说不清",
    "不确定",
)


# =========================================================
# State creation
# =========================================================

def new_state(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:

    if not session_id:

        session_id = uuid.uuid4().hex


    return {
        "state_version":
            STATE_VERSION,

        "session_id":
            session_id,

        "turn_count":
            0,

        "facts": {
            field: None
            for field in FACT_FIELDS
        },

        # User explicitly said they do not know this field.
        "unknown_fields": [],

        # V6.17.5
        # Fields that should no longer be automatically
        # re-asked because the retry limit was reached.
        #
        # This is NOT equivalent to "unknown".
        "deferred_fields": [],

        # Number of times each field has been requested.
        "asked_fields": {},

        # Which field(s) the assistant's last question
        # was asking for.
        "pending_fields": [],

        # Lightweight conversation history.
        "history": [],

        "last_action": None,

        "complete": False,
    }


# =========================================================
# Validation / normalization
# =========================================================

def normalize_state(
    state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    if not isinstance(
        state,
        dict,
    ):

        return new_state()


    result = deepcopy(
        state
    )


    result[
        "state_version"
    ] = STATE_VERSION


    result.setdefault(
        "session_id",
        uuid.uuid4().hex,
    )


    result.setdefault(
        "turn_count",
        0,
    )


    facts = result.get(
        "facts"
    )

    if not isinstance(
        facts,
        dict,
    ):

        facts = {}


    result["facts"] = {
        field: facts.get(field)
        for field in FACT_FIELDS
    }


    unknown = result.get(
        "unknown_fields"
    )

    if not isinstance(
        unknown,
        list,
    ):

        unknown = []


    result[
        "unknown_fields"
    ] = [
        field
        for field in unknown
        if field in FACT_FIELDS
    ]


    deferred = result.get(
        "deferred_fields"
    )

    if not isinstance(
        deferred,
        list,
    ):
        deferred = []


    result[
        "deferred_fields"
    ] = [
        field
        for field in deferred
        if (
            field in FACT_FIELDS
            and field not in result[
                "unknown_fields"
            ]
        )
    ]


    asked = result.get(
        "asked_fields"
    )

    if not isinstance(
        asked,
        dict,
    ):

        asked = {}


    result[
        "asked_fields"
    ] = {
        field: int(count)
        for field, count in asked.items()
        if (
            field in FACT_FIELDS
            and isinstance(
                count,
                (int, float),
            )
        )
    }


    pending = result.get(
        "pending_fields"
    )

    if not isinstance(
        pending,
        list,
    ):

        pending = []


    result[
        "pending_fields"
    ] = [
        field
        for field in pending
        if field in FACT_FIELDS
    ]


    history = result.get(
        "history"
    )

    if not isinstance(
        history,
        list,
    ):

        history = []


    result[
        "history"
    ] = history


    result.setdefault(
        "last_action",
        None,
    )

    result.setdefault(
        "complete",
        False,
    )


    return result



# =========================================================
# V6.17.6 Session Lifecycle
# =========================================================

def reset_state(
    state: Optional[Dict[str, Any]] = None,
    keep_session_id: bool = False,
) -> Dict[str, Any]:

    """
    Start a completely clean consultation state.

    keep_session_id=False:
        true new consultation

    keep_session_id=True:
        clear medical state while keeping the same external
        session identifier if an application needs it.
    """

    session_id = None

    if (
        keep_session_id
        and isinstance(state, dict)
    ):

        old = normalize_state(
            state
        )

        session_id = old.get(
            "session_id"
        )


    return new_state(
        session_id=session_id
    )


def prepare_for_consultation_turn(
    state: Optional[Dict[str, Any]],
):

    """
    Prepare state before a new consultation turn.

    Returns:
        (state, session_event)

    session_event:
        new_session
        continued
        reopened_after_summary
    """

    if not isinstance(
        state,
        dict,
    ):

        return (
            new_state(),
            "new_session",
        )


    state = normalize_state(
        state
    )


    if state.get(
        "complete"
    ):

        # A phase summary does not permanently lock the
        # session. The user may continue adding information.
        #
        # Existing verified facts/history are preserved.
        state[
            "complete"
        ] = False

        state[
            "pending_fields"
        ] = []

        state[
            "last_action"
        ] = None


        return (
            state,
            "reopened_after_summary",
        )


    return (
        state,
        "continued",
    )



# =========================================================
# History
# =========================================================

def append_history(
    state: Dict[str, Any],
    role: str,
    content: str,
):

    state = normalize_state(
        state
    )


    content = str(
        content
    ).strip()


    if not content:

        return state


    state[
        "history"
    ].append(
        {
            "role": role,
            "content": content,
        }
    )


    return state


# =========================================================
# Facts
# =========================================================

def set_fact(
    state: Dict[str, Any],
    field: str,
    value: Any,
):

    state = normalize_state(
        state
    )


    if field not in FACT_FIELDS:

        raise ValueError(
            f"Unknown consultation field: {field}"
        )


    if value is None:

        state[
            "facts"
        ][field] = None

        return state


    if isinstance(
        value,
        str,
    ):

        value = value.strip()

        if not value:

            value = None


    state[
        "facts"
    ][field] = value


    if (
        value is not None
        and field
        in state["unknown_fields"]
    ):

        state[
            "unknown_fields"
        ].remove(
            field
        )


    if (
        value is not None
        and field
        in state["deferred_fields"]
    ):

        state[
            "deferred_fields"
        ].remove(
            field
        )


    return state


def mark_unknown(
    state: Dict[str, Any],
    field: str,
):

    state = normalize_state(
        state
    )


    if field not in FACT_FIELDS:

        raise ValueError(
            f"Unknown consultation field: {field}"
        )


    # Important:
    # Unknown does NOT mean fabricate a value.
    state[
        "facts"
    ][field] = None


    if field not in state[
        "unknown_fields"
    ]:

        state[
            "unknown_fields"
        ].append(
            field
        )


    if field in state[
        "deferred_fields"
    ]:

        state[
            "deferred_fields"
        ].remove(
            field
        )


    return state



# =========================================================
# V6.17.5 Deferred Field
# =========================================================

def defer_field(
    state: Dict[str, Any],
    field: str,
):

    state = normalize_state(
        state
    )

    if field not in FACT_FIELDS:

        raise ValueError(
            f"Unknown consultation field: {field}"
        )


    # Known/explicitly unknown fields never need deferral.
    if (
        state["facts"].get(field)
        is not None
    ):

        return state


    if field in state[
        "unknown_fields"
    ]:

        return state


    if field not in state[
        "deferred_fields"
    ]:

        state[
            "deferred_fields"
        ].append(
            field
        )


    # It must no longer remain an active pending question.
    state[
        "pending_fields"
    ] = [
        x
        for x in state[
            "pending_fields"
        ]
        if x != field
    ]


    return state



# =========================================================
# Question tracking
# =========================================================

def mark_asked(
    state: Dict[str, Any],
    fields: Iterable[str],
):

    state = normalize_state(
        state
    )


    valid = []

    for field in fields:

        if field not in FACT_FIELDS:
            continue

        valid.append(
            field
        )

        state[
            "asked_fields"
        ][field] = (
            state[
                "asked_fields"
            ].get(
                field,
                0,
            )
            + 1
        )


    state[
        "pending_fields"
    ] = valid


    return state


def should_ask_field(
    state: Dict[str, Any],
    field: str,
) -> bool:

    state = normalize_state(
        state
    )


    if field not in FACT_FIELDS:

        return False


    # Already known.
    if state[
        "facts"
    ].get(field) is not None:

        return False


    # User explicitly does not know.
    # Especially important for pulse.
    if field in state[
        "unknown_fields"
    ]:

        return False


    # V6.17.5 retry-limit deferral.
    if field in state[
        "deferred_fields"
    ]:

        return False


    return True


# =========================================================
# Expected-answer binding
# =========================================================

def looks_unknown(
    text: str,
) -> bool:

    q = str(
        text
    ).strip()


    return any(
        phrase in q
        for phrase in UNKNOWN_PHRASES
    )


def bind_pending_answer(
    state: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:

    """
    Bind a user's answer to the previous question.

    Conservative rule:
    - exactly ONE pending field:
        bind the raw user response to that field
    - multiple pending fields:
        do NOT guess which fragment maps to which field
    """

    state = normalize_state(
        state
    )


    text = str(
        text
    ).strip()


    pending = list(
        state[
            "pending_fields"
        ]
    )


    result = {
        "state":
            state,

        "bound":
            False,

        "field":
            None,

        "value":
            None,

        "unknown":
            False,

        "needs_structured_parse":
            False,
    }


    if not text:

        return result


    if len(pending) == 0:

        return result


    if len(pending) > 1:

        # Never guess between multiple requested fields.
        result[
            "needs_structured_parse"
        ] = True

        return result


    field = pending[0]


    if looks_unknown(
        text
    ):

        state = mark_unknown(
            state,
            field,
        )

        state[
            "pending_fields"
        ] = []


        result.update(
            {
                "state": state,
                "bound": True,
                "field": field,
                "value": None,
                "unknown": True,
            }
        )

        return result


    state = set_fact(
        state,
        field,
        text,
    )


    state[
        "pending_fields"
    ] = []


    result.update(
        {
            "state": state,
            "bound": True,
            "field": field,
            "value": text,
            "unknown": False,
        }
    )


    return result



# =========================================================
# V6.17.4 Multi-field Structured Answer Parsing
# =========================================================

MULTI_FIELD_PATTERNS = {

    "sleep": (
        "睡眠",
        "睡得",
        "睡觉",
        "入睡",
        "失眠",
        "早醒",
        "多梦",
    ),

    "appetite": (
        "食欲",
        "胃口",
        "饮食",
        "吃饭",
    ),

    "stool": (
        "大便",
        "排便",
        "便秘",
        "便溏",
        "腹泻",
    ),

    "urination": (
        "小便",
        "排尿",
        "尿频",
        "夜尿",
        "尿量",
    ),

    "temperature": (
        "怕冷",
        "怕热",
        "寒热",
        "发热",
        "手脚冷",
        "手足冷",
    ),

    "sweating": (
        "出汗",
        "汗多",
        "盗汗",
        "自汗",
        "容易出汗",
    ),

    "tongue": (
        "舌象",
        "舌苔",
        "舌质",
        "舌头",
    ),

    "pulse": (
        "脉象",
        "脉搏",
        "把脉",
    ),

    "associated_symptoms": (
        "伴随",
        "其他不适",
        "还会",
        "还有",
        "同时",
    ),
}


DURATION_PATTERN = re.compile(
    r"(?:"
    r"\d+"
    r"|一|两|二|三|四|五|六|七|八|九|十"
    r"|半"
    r")"
    r"(?:天|周|星期|个月|月|年)"
)


def split_answer_clauses(
    text: str,
):

    text = str(text).strip()

    if not text:
        return []

    parts = re.split(
        r"[，,；;。！？!?\n]+",
        text,
    )

    return [
        x.strip()
        for x in parts
        if x.strip()
    ]


def detect_clause_fields(
    clause: str,
    pending_fields,
):

    fields = []

    for field in pending_fields:

        patterns = MULTI_FIELD_PATTERNS.get(
            field,
            (),
        )

        if any(
            pattern in clause
            for pattern in patterns
        ):

            fields.append(field)


    # Duration is special because replies such as
    # "两个月左右" do not normally contain the word "病程".
    if (
        "duration" in pending_fields
        and "duration" not in fields
        and DURATION_PATTERN.search(clause)
    ):

        fields.append(
            "duration"
        )


    return fields


def parse_multi_pending_answer(
    state: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:

    """
    Conservative deterministic parser.

    Only a clause with exactly ONE identifiable pending
    field is bound.

    Ambiguous clauses are ignored.

    Example:
        pending:
            sleep/appetite/stool/urination

        answer:
            睡得一般，胃口还行，大便偏干，小便正常

        -> four explicit facts.

    If a field cannot be identified:
        it remains pending and its fact remains None.
    """

    state = normalize_state(
        state
    )

    text = str(text).strip()

    pending = list(
        state[
            "pending_fields"
        ]
    )


    result = {
        "state":
            state,

        "bound":
            False,

        "field":
            None,

        "value":
            None,

        "unknown":
            False,

        "needs_structured_parse":
            False,

        "parsed_fields":
            {},

        "unknown_parsed_fields":
            [],

        "unresolved_fields":
            list(pending),
    }


    if (
        not text
        or len(pending) <= 1
    ):

        return result


    clauses = split_answer_clauses(
        text
    )


    parsed_fields = {}
    unknown_fields = []


    for clause in clauses:

        detected = detect_clause_fields(
            clause,
            pending,
        )


        # Never guess an ambiguous clause.
        if len(detected) != 1:
            continue


        field = detected[0]


        # Already parsed from an earlier explicit clause.
        if field in parsed_fields:
            continue


        if field in unknown_fields:
            continue


        if looks_unknown(
            clause
        ):

            state = mark_unknown(
                state,
                field,
            )

            unknown_fields.append(
                field
            )

            continue


        state = set_fact(
            state,
            field,
            clause,
        )

        parsed_fields[
            field
        ] = clause


    resolved = set(
        parsed_fields
    ) | set(
        unknown_fields
    )


    unresolved = [
        field
        for field in pending
        if field not in resolved
    ]


    state[
        "pending_fields"
    ] = unresolved


    result.update(
        {
            "state":
                state,

            "bound":
                bool(resolved),

            "needs_structured_parse":
                bool(unresolved),

            "parsed_fields":
                parsed_fields,

            "unknown_parsed_fields":
                unknown_fields,

            "unresolved_fields":
                unresolved,
        }
    )


    return result



# =========================================================
# Turn management
# =========================================================

def begin_user_turn(
    state: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:

    state = normalize_state(
        state
    )


    state[
        "turn_count"
    ] += 1


    state = append_history(
        state,
        "user",
        text,
    )


    return state


def record_assistant_turn(
    state: Dict[str, Any],
    message: str,
    action: Optional[str] = None,
) -> Dict[str, Any]:

    state = normalize_state(
        state
    )


    state = append_history(
        state,
        "assistant",
        message,
    )


    state[
        "last_action"
    ] = action


    if action == "summarize":

        state[
            "complete"
        ] = True


    return state


# =========================================================
# Public summary for model / API
# =========================================================

def state_snapshot(
    state: Dict[str, Any],
) -> Dict[str, Any]:

    state = normalize_state(
        state
    )


    return {
        "state_version":
            state[
                "state_version"
            ],

        "session_id":
            state[
                "session_id"
            ],

        "turn_count":
            state[
                "turn_count"
            ],

        "facts":
            deepcopy(
                state[
                    "facts"
                ]
            ),

        "unknown_fields":
            list(
                state[
                    "unknown_fields"
                ]
            ),

        "deferred_fields":
            list(
                state[
                    "deferred_fields"
                ]
            ),

        "asked_fields":
            dict(
                state[
                    "asked_fields"
                ]
            ),

        "pending_fields":
            list(
                state[
                    "pending_fields"
                ]
            ),

        "last_action":
            state[
                "last_action"
            ],

        "complete":
            bool(
                state[
                    "complete"
                ]
            ),
    }
