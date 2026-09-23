import re

import rag.retriever as rr


# =========================================================
# Basic text helpers
# =========================================================

def flat(value):

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set)):
        return " ".join(
            flat(x)
            for x in value
        )

    if isinstance(value, dict):
        return " ".join(
            flat(x)
            for x in value.values()
        )

    return str(value)


def norm(text):

    text = flat(text).lower()

    return re.sub(
        r"""[\s，。；、：？！,.!?;:"'（）()【】\[\]<>《》\-_/]+""",
        "",
        text,
    )


def doc_text(doc):

    return " ".join(
        [
            flat(doc.get("title")),
            flat(doc.get("aliases")),
            flat(doc.get("content")),
        ]
    )


def ngrams(text, n=2):

    t = norm(text)

    if not t:
        return set()

    if len(t) < n:
        return {t}

    return {
        t[i:i+n]
        for i in range(
            len(t) - n + 1
        )
    }


def jaccard(a, b, n=2):

    aa = ngrams(a, n)
    bb = ngrams(b, n)

    if not aa or not bb:
        return 0.0

    return (
        len(aa & bb)
        / len(aa | bb)
    )


def coverage(query, content, n=2):

    q = ngrams(
        query,
        n
    )

    if not q:
        return 0.0

    d = ngrams(
        content,
        n
    )

    return (
        len(q & d)
        / len(q)
    )


# =========================================================
# Herb name inventory
# =========================================================

HERB_NAMES = []

for doc in rr.documents:

    if doc.get("category") != "herb":
        continue

    names = [
        doc.get("title")
    ]

    aliases = doc.get(
        "aliases",
        []
    )

    if isinstance(aliases, str):
        names.append(aliases)

    elif isinstance(aliases, list):
        names.extend(aliases)

    for name in names:

        if not name:
            continue

        name = norm(name)

        if len(name) >= 2:
            HERB_NAMES.append(
                name
            )


HERB_NAMES = sorted(
    set(HERB_NAMES),
    key=len,
    reverse=True,
)


def query_herbs(query):

    q = norm(query)

    found = [
        name
        for name in HERB_NAMES
        if name in q
    ]

    output = []

    for name in found:

        # 保留最长匹配
        if any(
            name != other
            and name in other
            for other in found
        ):
            continue

        output.append(name)

    return output


# =========================================================
# Herb structured fields
# =========================================================

TASTES = [
    "酸",
    "苦",
    "甘",
    "辛",
    "咸",
    "淡",
    "涩",
]

TEMPERATURES = [
    "大寒",
    "大热",
    "微寒",
    "微温",
    "寒",
    "热",
    "温",
    "凉",
    "平",
]

MERIDIANS = [
    "心包",
    "三焦",
    "大肠",
    "小肠",
    "膀胱",
    "肝",
    "心",
    "脾",
    "肺",
    "肾",
    "胃",
    "胆",
]


def extract_tastes(query):

    q = norm(query)

    return {
        x
        for x in TASTES
        if x in q
    }


def extract_temperature(query):

    q = norm(query)

    result = set()

    for x in TEMPERATURES:

        if x not in q:
            continue

        if x in (
            "寒",
            "温",
        ):

            if (
                ("微" + x) in q
                or ("大" + x) in q
            ):
                continue

        result.add(x)

    return result


def extract_meridians(query):

    text = flat(query)

    matches = re.findall(
        r"归([^。；，]+?)经",
        text
    )

    result = set()

    for segment in matches:

        for meridian in MERIDIANS:

            if meridian in segment:
                result.add(
                    meridian
                )

    return result


def set_match(values, text):

    if not values:
        return None

    d = norm(text)

    hit = sum(
        1
        for value in values
        if norm(value) in d
    )

    return (
        hit
        / len(values)
    )


# =========================================================
# Field Evidence
# =========================================================

def evidence_score(query, doc):

    category = doc.get(
        "category"
    )

    title = flat(
        doc.get("title")
    )

    text = doc_text(doc)

    nq = norm(query)
    nt = norm(title)


    cov2 = coverage(
        query,
        text,
        2
    )

    jac2 = jaccard(
        query,
        text,
        2
    )

    generic = (
        0.70 * cov2
        + 0.30 * jac2
    )


    title_bonus = 0.0

    if (
        len(nt) >= 2
        and nt in nq
    ):
        title_bonus = 1.0


    # =====================================================
    # Herb
    # =====================================================

    if category == "herb":

        tastes = extract_tastes(
            query
        )

        temps = extract_temperature(
            query
        )

        meridians = extract_meridians(
            query
        )


        taste_score = set_match(
            tastes,
            text
        )

        temp_score = set_match(
            temps,
            text
        )

        meridian_score = set_match(
            meridians,
            text
        )


        components = [
            (
                0.35,
                generic,
            ),
            (
                0.20,
                title_bonus,
            ),
        ]


        if taste_score is not None:

            components.append(
                (
                    0.10,
                    taste_score,
                )
            )


        if temp_score is not None:

            components.append(
                (
                    0.10,
                    temp_score,
                )
            )


        if meridian_score is not None:

            components.append(
                (
                    0.25,
                    meridian_score,
                )
            )


        total_weight = sum(
            weight
            for weight, _ in components
        )

        score = (
            sum(
                weight * value
                for weight, value
                in components
            )
            / total_weight
        )


        return score, {
            "generic": generic,
            "title": title_bonus,
            "taste": taste_score,
            "temp": temp_score,
            "meridian": meridian_score,
        }


    # =====================================================
    # Formula
    # =====================================================

    if category == "formula":

        herbs = query_herbs(
            query
        )

        ingredient = None

        if herbs:

            dt = norm(text)

            hit = sum(
                1
                for herb in herbs
                if herb in dt
            )

            ingredient = (
                hit
                / len(herbs)
            )


        if ingredient is None:

            score = (
                0.75 * generic
                + 0.25 * title_bonus
            )

        else:

            score = (
                0.40 * generic
                + 0.45 * ingredient
                + 0.15 * title_bonus
            )


        return score, {
            "generic": generic,
            "title": title_bonus,
            "ingredient": ingredient,
            "query_herbs": herbs,
        }


    # =====================================================
    # Theory
    # =====================================================

    if category == "theory":

        score = (
            0.55 * generic
            + 0.45 * title_bonus
        )

        return score, {
            "generic": generic,
            "title": title_bonus,
        }


    # =====================================================
    # Syndrome
    # =====================================================

    score = (
        0.80 * generic
        + 0.20 * title_bonus
    )

    return score, {
        "generic": generic,
        "title": title_bonus,
    }


# =========================================================
# Global Field Ranking
# =========================================================

def rank_by_field_evidence(
    query,
    top_k=5,
):

    scored = []

    for doc in rr.documents:

        score, detail = evidence_score(
            query,
            doc
        )

        scored.append(
            (
                score,
                doc,
                detail,
            )
        )


    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return scored[:top_k]



# =========================================================
# V6.14.3 Explicit Named Entity Rescue Guard
# =========================================================

ENTITY_SUFFIXES = (
    "汤",
    "丸",
    "散",
    "饮",
    "膏",
    "丹",
    "方",
    "剂",
    "证",
)


def explicit_named_entity_candidate(query):

    """
    仅用于判断：
    用户是否已经明确点名了一个完整知识实体。

    例如：
        请说明百合玄元汤的组成和传统功用
        -> 百合玄元汤

    注意：
    这是Field Rescue的保护层，
    不负责一般语义检索。
    """

    text = flat(query).strip()

    # 去句末标点
    text = re.sub(
        r"[？?。！!\s]+$",
        "",
        text,
    )


    patterns = [

        # 请说明XX的组成和传统功用
        r"^(?:请)?(?:说明|介绍|解释|讲解)"
        r"(.{2,20}?)"
        r"(?:的)?"
        r"(?:组成|配伍)"
        r"(?:和|及|与)?"
        r"(?:传统)?"
        r"(?:功用|功效|作用)",

        # XX的组成和传统功用是什么
        r"^(.{2,20}?)"
        r"(?:的)?"
        r"(?:组成|配伍)"
        r"(?:和|及|与)?"
        r"(?:传统)?"
        r"(?:功用|功效|作用)",

        # XX有什么功用/功效
        r"^(?:请)?(?:说明|介绍)?"
        r"(.{2,20}?)"
        r"(?:有何|有什么)"
        r"(?:传统)?"
        r"(?:功用|功效|作用)",
    ]


    for pattern in patterns:

        m = re.search(
            pattern,
            text,
        )

        if not m:
            continue

        candidate = (
            m.group(1)
            .strip(" ：:，,。；;")
        )

        # 防止把整段描述当实体
        if not (
            2 <= len(candidate) <= 20
        ):
            continue


        # 此Guard重点保护明确命名实体。
        # 方剂/证候等通常具有明显后缀。
        if candidate.endswith(
            ENTITY_SUFFIXES
        ):
            return candidate


    return None


def entity_exists_in_kb(candidate):

    if not candidate:
        return False

    nc = rr.normalize(
        candidate
    )

    return bool(
        rr.entity_map.get(
            nc,
            []
        )
    )


def should_block_field_rescue(query):

    candidate = explicit_named_entity_candidate(
        query
    )

    if not candidate:

        return False, None


    exists = entity_exists_in_kb(
        candidate
    )


    if exists:

        return False, {
            "candidate": candidate,
            "exists": True,
            "reason": "explicit_entity_exists",
        }


    return True, {
        "candidate": candidate,
        "exists": False,
        "reason": "explicit_entity_not_in_kb",
    }



# =========================================================
# Conservative Rescue Decision
# =========================================================

def should_rescue(
    query,
    ranked,
):

    if not ranked:
        return False, None


    score1, doc1, detail1 = ranked[0]

    score2 = (
        ranked[1][0]
        if len(ranked) > 1
        else 0.0
    )

    margin = (
        score1
        - score2
    )

    category = doc1.get(
        "category"
    )


    # -----------------------------------------------------
    # A. Explicit title evidence
    #
    # 五行相生这个概念...
    # -----------------------------------------------------

    if (
        detail1.get("title") == 1.0
        and score1 >= 0.50
        and margin >= 0.04
    ):

        return True, {
            "reason": "exact_title_evidence",
            "score": score1,
            "margin": margin,
        }


    # -----------------------------------------------------
    # B. Herb structured evidence
    #
    # 性味 + 温度 + 归经
    # -----------------------------------------------------

    if category == "herb":

        structured = [
            detail1.get("taste"),
            detail1.get("temp"),
            detail1.get("meridian"),
        ]

        available = [
            x
            for x in structured
            if x is not None
        ]


        if (
            len(available) >= 2
            and min(available) >= 0.90
            and score1 >= 0.52
            and margin >= 0.08
        ):

            return True, {
                "reason": "herb_structured_evidence",
                "score": score1,
                "margin": margin,
            }


        # -------------------------------------------------
        # Herb efficacy profile rescue
        #
        # 黄芪传统功效：
        # 没有性味归经，但整段功效文本高度吻合。
        # 必须要求很强generic overlap和很大margin。
        # -------------------------------------------------

        if (
            not available
            and detail1.get(
                "generic",
                0.0
            ) >= 0.50
            and score1 >= 0.30
            and margin >= 0.15
        ):

            return True, {
                "reason": "herb_profile_evidence",
                "score": score1,
                "margin": margin,
            }


    # -----------------------------------------------------
    # C. Formula ingredient coverage
    # -----------------------------------------------------

    if category == "formula":

        ingredient = detail1.get(
            "ingredient"
        )

        if (
            ingredient is not None
            and ingredient >= 0.99
            and score1 >= 0.54
            and margin >= 0.02
        ):

            return True, {
                "reason": "formula_ingredient_evidence",
                "score": score1,
                "margin": margin,
            }


    return False, {
        "reason": "insufficient_field_evidence",
        "score": score1,
        "margin": margin,
    }


# =========================================================
# V6.14 Retrieve Wrapper
# =========================================================

def retrieve_v614(
    query,
    top_k=3,
):

    base = rr.retrieve_base(
        query,
        top_k=max(
            top_k,
            5,
        )
    )


    # -----------------------------------------------------
    # Existing successful result:
    # NEVER replace it in V6.14.2.
    # -----------------------------------------------------

    if base.get("found"):

        return base


    # -----------------------------------------------------
    # Guards remain authoritative.
    # Unknown entities / structural negatives must never
    # be rescued by semantic evidence.
    # -----------------------------------------------------

    if base.get("method") in (
        "exact_guard",
        "structure_guard",
    ):

        return base


    # -----------------------------------------------------
    # Rescue only description_to_entity.
    # -----------------------------------------------------

    if (
        base.get("query_type")
        != "description_to_entity"
    ):

        return base


    # =====================================================
    # V6.14.3
    # Explicit named entity guard
    #
    # 用户已经明确点名一个完整方名/证候，
    # 且该完整实体不在KB时，
    # 禁止Field Rescue把内部真实药名或相似方剂
    # 强行“救活”。
    # =====================================================

    block_rescue, guard_info = should_block_field_rescue(
        query
    )

    if block_rescue:

        result = dict(base)

        result[
            "field_rescue"
        ] = False

        result[
            "field_guard"
        ] = guard_info

        return result


    ranked = rank_by_field_evidence(
        query,
        top_k=max(
            top_k,
            5,
        )
    )


    rescue, evidence = should_rescue(
        query,
        ranked,
    )


    if not rescue:

        result = dict(base)

        result[
            "field_rescue"
        ] = False

        result[
            "field_evidence"
        ] = evidence

        return result


    # -----------------------------------------------------
    # Convert ranked docs to standard result shape
    # -----------------------------------------------------

    results = []

    for score, doc, detail in ranked[:top_k]:

        item = dict(doc)

        item[
            "score"
        ] = float(score)

        item[
            "field_score"
        ] = float(score)

        item[
            "field_detail"
        ] = detail

        results.append(
            item
        )


    top_doc = ranked[0][1]


    result = dict(base)

    result[
        "found"
    ] = True

    result[
        "results"
    ] = results

    result[
        "method"
    ] = "field_rescue"

    # 保持现有confidence枚举兼容
    result[
        "confidence"
    ] = "medium"

    result[
        "category_hint"
    ] = top_doc.get(
        "category"
    )

    result[
        "field_rescue"
    ] = True

    result[
        "field_evidence"
    ] = evidence

    return result
