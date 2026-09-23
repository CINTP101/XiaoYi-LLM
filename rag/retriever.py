import json
import re
import sys
from pathlib import Path

import faiss


# =========================================================
# Project
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from rag.embedder import encode_queries


INDEX_PATH = (
    PROJECT_ROOT
    / "rag"
    / "index"
    / "tcm.index"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "rag"
    / "index"
    / "metadata.json"
)


# =========================================================
# Thresholds
# =========================================================

VECTOR_MIN_SCORE = 0.65
VECTOR_MIN_MARGIN = 0.08

CATEGORY_MIN_SCORE = {
    "herb": 0.60,
    "formula": 0.60,
    "syndrome": 0.60,
    "theory": 0.50,
}

CATEGORY_MIN_MARGIN = 0.03


RERANK_VECTOR_WEIGHT = 0.75
RERANK_LEXICAL_WEIGHT = 0.25


# =========================================================
# Load Index
# =========================================================

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS索引不存在: {INDEX_PATH}"
    )

print("[Retriever] 加载FAISS...")

index = faiss.read_index(
    str(INDEX_PATH)
)


with METADATA_PATH.open(
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)


documents = metadata.get(
    "documents",
    []
)


if len(documents) != index.ntotal:
    raise RuntimeError(
        "metadata与FAISS数量不一致"
    )


print(
    f"[Retriever] 知识条目: {index.ntotal}"
)

print(
    "[Retriever] Ready ✅"
)


# =========================================================
# Normalization
# =========================================================

def normalize(text):

    text = str(
        text or ""
    ).lower()

    text = re.sub(
        r"[，。？！、；：,.!?;:\s]",
        "",
        text
    )

    return text


# =========================================================
# Entity Index
# =========================================================

entity_map = {}


for doc in documents:

    names = [
        doc.get(
            "title",
            ""
        )
    ]

    aliases = doc.get(
        "aliases",
        []
    )

    if isinstance(
        aliases,
        list
    ):
        names.extend(
            aliases
        )


    # 同一个document内部：
    # title和alias可能完全相同，
    # 必须先对规范化后的实体名去重。
    seen_names = set()


    for name in names:

        nn = normalize(
            name
        )

        if not nn:
            continue


        if nn in seen_names:
            continue


        seen_names.add(
            nn
        )


        bucket = entity_map.setdefault(
            nn,
            []
        )


        # 再按document id做一次保险去重
        doc_id = doc.get(
            "id"
        )


        if any(
            x.get("id") == doc_id
            for x in bucket
        ):
            continue


        bucket.append(
            doc
        )



# =========================================================
# V6.13 Entity Mention Resolver
# =========================================================

def find_entity_mentions(query):

    q = normalize(query)

    matches = []

    for name, docs in entity_map.items():

        # 单字实体不做宽松substring扫描，
        # 防止“气/血”等普通字造成误命中。
        if len(name) < 2:
            continue

        if name not in q:
            continue

        for doc in docs:

            matches.append(
                (
                    len(name),
                    name,
                    doc,
                )
            )


    matches.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return matches


def resolve_mentioned_entity(query):

    matches = find_entity_mentions(
        query
    )

    if not matches:
        return []


    # 最长实体优先：
    # 麻黄汤 > 麻黄
    max_len = matches[0][0]

    selected = [
        x
        for x in matches
        if x[0] == max_len
    ]


    results = []
    seen = set()

    for _, name, doc in selected:

        doc_id = doc.get("id")

        if doc_id in seen:
            continue

        seen.add(doc_id)

        item = dict(doc)

        item["score"] = 1.0
        item["retrieval_method"] = "exact"

        results.append(item)


    return results



# =========================================================
# Description Query Patterns
# =========================================================

DESCRIPTION_PATTERNS = (

    # -----------------------------------------------------
    # Herb
    # -----------------------------------------------------

    "有一味中药",
    "某味药",
    "药性资料",
    "药性特点",
    "哪味药",
    "哪味中药",
    "哪一味药",
    "哪一味中药",
    "性味归经和传统功能",
    "性味归经及传统功能",
    "根据以下中药资料",
    "判断药名",

    # -----------------------------------------------------
    # Formula
    # -----------------------------------------------------

    "某经典方",
    "一个传统方",
    "哪首方剂",
    "是哪首方剂",
    "这个方剂叫什么",
    "方剂是什么",
    "经典方是什么",
    "配伍而成",
    "组成并具有",
    "组成且具有",
    "组成和传统功用",
    "配伍和功用",
    "以下配伍和功用",
    "这些药物组成和功用",

    # -----------------------------------------------------
    # Syndrome
    # -----------------------------------------------------

    "辨证资料提示",
    "辨证信息表现为",
    "根据以下辨证资料",
    "证候名称判断",
    "最符合哪一证",
    "最接近哪一个中医证候",
    "属于哪个证候",
    "属于什么证候",
    "属于什么证",
    "属于哪种证",
    "判断证候名称",

    # -----------------------------------------------------
    # Theory
    # -----------------------------------------------------

    "中医基础理论中",
    "中医基础概念",
    "基础理论术语",
    "概念描述为",
    "对应的理论术语",
    "这个术语是什么",
)


def looks_like_description_query(query):

    q = normalize(query)


    # =====================================================
    # 1. 已有明确 description patterns
    # =====================================================

    if any(
        normalize(x) in q
        for x in DESCRIPTION_PATTERNS
    ):
        return True


    # =====================================================
    # 2. 理论“关系”反推
    #
    # 例如：
    #
    # 五行之间相互资生和促进的关系叫什么？
    # 五行之间出现反向克制属于什么关系？
    #
    # 虽然包含“五行”，但不能直接返回“五行”。
    # =====================================================

    relation_patterns = [

        r"关系叫什么",

        r"关系是什么",

        r"属于什么关系",

        r"属于哪种关系",

        r"是什么关系",

        r"称为什么关系",
    ]


    if any(
        re.search(pattern, q)
        for pattern in relation_patterns
    ):
        return True


    # =====================================================
    # 3. 理论 / 辨证方法反推
    #
    # 例如：
    #
    # 阴阳表里寒热虚实八个纲领组成的辨证方法是什么？
    #
    # 不能因为出现“阴阳”而直接返回阴阳。
    # =====================================================

    method_patterns = [

        r"辨证方法是什么",

        r"辨证方法叫什么",

        r"属于什么辨证方法",

        r"是哪种辨证方法",

        r"这种辨证方法叫什么",
    ]


    if any(
        re.search(pattern, q)
        for pattern in method_patterns
    ):
        return True


    # =====================================================
    # 4. 方剂描述反推
    #
    # 例如：
    #
    # 以补气生血为功用并由黄芪和当归组成的经典方是什么？
    #
    # 黄芪/当归只是组成成分，不是目标实体。
    # =====================================================

    asks_formula = any(
        phrase in q
        for phrase in (
            "经典方是什么",
            "方剂是什么",
            "是哪首方",
            "哪首方剂",
            "方名是什么",
            "对应什么方",
            "对应哪首方",
        )
    )


    has_formula_description = (
        "组成" in q
        or "配伍" in q
        or (
            "功用" in q
            and "由" in q
        )
    )


    if (
        asks_formula
        and has_formula_description
    ):
        return True


    # =====================================================
    # 5. 通用反推句式
    # =====================================================

    patterns = [

        r"对应.{0,12}(哪|什么|名称|条目|术语|方名|证候|中药)",

        r"判断.{0,15}(哪|什么|名称|条目|术语|方名|证候|中药)",

        r"最接近.{0,12}(哪|什么|名称|条目|证候)",

        r"最匹配.{0,12}(哪|什么|名称|条目)",

        r"根据.{0,20}资料.{0,20}(判断|对应)",

        r"依据描述.{0,20}(判断|匹配)",

        r"某证候定义为.{0,80}(?:临床可见|常见表现)",

        r"药性为.{0,80}功能为.{0,80}(?:药叫什么|药名)",

        r"配伍包括.{0,160}功用是.{0,80}(?:辨认方名|方名)",
    ]


    return any(
        re.search(pattern, q)
        for pattern in patterns
    )



# =========================================================
# Comparison Query Patterns
# =========================================================

COMPARISON_PATTERNS = (

    "有什么区别",
    "有何区别",
    "有什么不同",
    "有何不同",

    "怎么区别",
    "如何区别",
    "怎么区分",
    "如何区分",

    "比较一下",
    "比较",
    "区别",
    "不同之处",

    "和什么不同",
    "与什么不同",
)


def detect_query_type(query):

    q = normalize(query)

    # A complete direct pattern naming a known entity is
    # stronger than generic wording such as "基础概念".
    direct_candidate = extract_direct_candidate(query)

    if (
        direct_candidate
        and normalize(direct_candidate) in entity_map
    ):
        return "direct_entity"


    # -----------------------------------------------------
    # 描述反推必须先于实体substring扫描
    # -----------------------------------------------------

    if looks_like_description_query(
        query
    ):

        return "description_to_entity"


    if any(
        normalize(x) in q
        for x in COMPARISON_PATTERNS
    ):

        return "comparison"


    # -----------------------------------------------------
    # 传统Direct Pattern
    # -----------------------------------------------------

    candidate = direct_candidate


    if candidate:

        if normalize(
            candidate
        ) in entity_map:

            return "direct_entity"

        return "unknown_entity"


    # -----------------------------------------------------
    # V6.13：
    # 自然问法中出现明确知识库实体
    #
    # 例如：
    # “能介绍一下黄芪在传统中药中的主要作用吗”
    # -----------------------------------------------------

    mentioned = resolve_mentioned_entity(
        query
    )


    if mentioned:

        return "direct_entity"


    return "free_semantic"


# =========================================================
# Direct Entity Candidate Extraction
# =========================================================

DIRECT_PATTERNS = [

    r"中医基础概念(.+?)的规范含义是什么",

    # 请介绍 / 请说明
    r"请介绍一下(.+?)的传统功能",
    r"请介绍一下(.+?)的传统功效",
    r"请介绍一下(.+?)的传统功用",

    r"请说明(.+?)的传统功能",
    r"请说明(.+?)的传统功效",
    r"请说明(.+?)的传统功用",

    # 中医所说的
    r"中医所说的(.+?)是什么意思",
    r"中医里什么是(.+)",
    r"中医中什么是(.+)",

    # 组成
    r"(.+?)的(?:完整)?组成是什么",
    r"(.+?)的组成是什么",
    r"(.+?)有哪些组成药物",
    r"(.+?)由哪些药组成",
    r"(.+?)由哪些组成",
    r"(.+?)由什么组成",

    # 表现
    r"(.+?)通常有哪些表现",
    r"(.+?)有哪些常见表现",
    r"(.+?)有什么常见表现",
    r"(.+?)有哪些表现",
    r"(.+?)有什么表现",

    # 功效
    r"(.+?)有什么传统功效",
    r"(.+?)有什么传统功能",
    r"(.+?)有什么传统功用",
    r"(.+?)有什么功用",
    r"(.+?)有什么功效",
    r"(.+?)有什么功能",
    r"(.+?)有什么作用",

    # 其他知识
    r"(.+?)归什么经",
    r"(.+?)归哪些经",
    r"(.+?)主治什么",
    r"(.+?)是什么意思",

    # 什么是 X 放最后
    r"什么叫(.+)",
    r"什么是(.+)",
]


def extract_direct_candidate(query):

    q = normalize(
        query
    )

    q = re.sub(
        r"^(?:方剂|证候|中药|理论)(?:资料)?检索",
        "",
        q,
    )



    # =====================================================
    # V6.13.3.3 ACTIVE Compound Syndrome Guard
    #
    # 完整复合证候优先：
    #
    # 肾气不固兼精血逆乱证通常有哪些辨证表现
    # ->
    # 肾气不固兼精血逆乱证
    # =====================================================

    if (
        "兼" in q
        and (
            "辨证表现" in q
            or "证候表现" in q
        )
    ):
        compat_pos = q.find("兼")

        syndrome_end = q.find(
            "证",
            compat_pos + 1
        )

        if syndrome_end != -1:
            candidate = q[:syndrome_end + 1].strip()

            if (
                "兼" in candidate
                and 3 <= len(candidate) <= 30
            ):
                return candidate

    for pattern in DIRECT_PATTERNS:

        m = re.fullmatch(
            pattern,
            q
        )

        if not m:
            continue

        candidate = m.group(
            1
        ).strip()

        if not candidate:
            continue

        return candidate


    return None


# =========================================================
# Direct Entity Resolution
# =========================================================

def resolve_direct_entity(query):

    candidate = extract_direct_candidate(
        query
    )


    if candidate:

        nc = normalize(
            candidate
        )

        docs = entity_map.get(
            nc,
            []
        )

        results = []

        seen = set()

        for doc in docs:

            doc_id = doc.get("id")

            if doc_id in seen:
                continue

            seen.add(doc_id)

            item = dict(doc)

            item["score"] = 1.0
            item["retrieval_method"] = "exact"

            results.append(item)


        if results:
            return results


    # -----------------------------------------------------
    # 自然语言实体扫描fallback
    # -----------------------------------------------------

    return resolve_mentioned_entity(
        query
    )


# =========================================================
# Category Router V6.10
# =========================================================

def infer_category(
    query,
    query_type=None,
    resolved_docs=None
):

    q = normalize(query)


    # -----------------------------------------------------
    # Direct resolved entity
    # -----------------------------------------------------

    if resolved_docs:

        categories = {
            x.get("category")
            for x in resolved_docs
        }

        categories.discard(None)

        if len(categories) == 1:

            return next(
                iter(categories)
            )


    if query_type == "comparison":

        return None


    # =====================================================
    # Explicit Herb
    # =====================================================

    herb_patterns = (

        "有一味中药",
        "某味药",
        "哪味药",
        "哪味中药",
        "哪一味药",
        "哪一味中药",
        "药性资料",
        "药性特点",
        "性味归经",
        "传统中药资料",
        "判断药名",
        "中药资料",
        "对应哪味中药",
    )


    if any(
        normalize(x) in q
        for x in herb_patterns
    ):

        return "herb"


    # =====================================================
    # Explicit Formula
    # =====================================================

    formula_patterns = (

        "某经典方",
        "经典方是什么",
        "一个传统方",
        "传统方",
        "哪首方",
        "哪首方剂",
        "是什么方剂",
        "方剂是什么",
        "方剂资料",
        "哪一个方名",
        "方名",
        "这个方剂叫什么",
        "方剂叫什么",
        "药物配伍",
        "配伍而成",
        "配伍包括",
        "组成并具有",
        "组成且具有",
        "组成和传统功用",
        "配伍和功用",
        "药物组成和功用",
        "以下配伍",
    )


    if any(
        normalize(x) in q
        for x in formula_patterns
    ):

        return "formula"


    # =====================================================
    # Explicit Syndrome
    # =====================================================

    syndrome_patterns = (

        "辨证资料",
        "辨证信息",
        "哪个证候",
        "什么证候",
        "中医证候名称",
        "证候名称",
        "属于哪个证候",
        "属于什么证候",
        "属于什么证",
        "属于哪种证",
        "最符合哪一证",
        "最接近哪一个中医证候",
        "属于哪一证",
        "判断证候",
    )


    if any(
        normalize(x) in q
        for x in syndrome_patterns
    ):

        return "syndrome"


    # =====================================================
    # Explicit Theory
    # =====================================================

    theory_patterns = (

        "中医基础理论",
        "中医基础概念",
        "基础理论术语",
        "理论术语",
        "理论概念",
        "基础概念",
        "概念描述",
        "理论中的",
        "对应的理论术语",
        "术语是什么",
        "关系叫什么",
        "关系是什么",
        "什么关系",
        "属于什么关系",
        "属于哪种关系",
        "功能叫什么",
        "合称什么",
        "总称是什么",
        "辨证方法",
        "病理产物",
    )


    if any(
        normalize(x) in q
        for x in theory_patterns
    ):

        return "theory"


    # =====================================================
    # Structural Feature: Herb
    #
    # 例如：
    # 甘、微苦，凉，归心、肺、肾经……
    # =====================================================

    if re.search(
        r"归.{1,25}经",
        q
    ):

        return "herb"


    # =====================================================
    # Structural Feature: Formula
    #
    # 一个描述里出现至少3个已知中药名，
    # 通常是方剂组成描述。
    # =====================================================

    herb_names = set()

    for name, docs in entity_map.items():

        if len(name) < 2:
            continue

        if not any(
            d.get("category") == "herb"
            for d in docs
        ):
            continue

        if name in q:
            herb_names.add(name)


    # 去除互相包含造成的重复计数
    herb_names = {
        name
        for name in herb_names
        if not any(
            name != other
            and name in other
            for other in herb_names
        )
    }


    if len(herb_names) >= 3:

        return "formula"


    # =====================================================
    # Structural Feature: Syndrome
    # =====================================================

    if "证候" in q:

        return "syndrome"


    # =====================================================
    # V6.13 Vector Category Probe
    #
    # 当语言规则仍然无法判断时，
    # 用每个category的最高BGE候选决定搜索空间。
    #
    # 注意：
    # 这只是category选择，不直接判found。
    # 最终仍经过原confidence规则。
    # =====================================================

    if query_type in (
        "description_to_entity",
        "free_semantic",
    ):

        return probe_category(
            query
        )


    return None



# =========================================================
# V6.13 Category Probe
# =========================================================

def probe_category(query):

    embed_query = clean_query_for_embedding(
        query,
        None
    )


    vector = encode_queries(
        [
            embed_query
        ]
    )


    scores, ids = index.search(
        vector,
        int(index.ntotal)
    )


    best = {}


    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        idx = int(idx)

        if idx < 0:
            continue


        doc = documents[idx]

        category = doc.get(
            "category"
        )


        if category not in (
            "herb",
            "formula",
            "syndrome",
            "theory",
        ):
            continue


        score = float(score)


        if (
            category not in best
            or score > best[category]
        ):

            best[category] = score


    if not best:

        return None


    ranked = sorted(
        best.items(),
        key=lambda x: x[1],
        reverse=True
    )


    category, score = ranked[0]


    # 太弱的语义证据仍然不强制分类
    if score < 0.45:

        return None


    return category


# =========================================================
# Closed-set Structural Guard
# =========================================================

CLOSED_SET_COUNTS = {
    "五行": 5,
    "五脏": 5,
    "六腑": 6,
    "八纲辨证": 8,
    "八纲": 8,
    "七情": 7,
    "六淫": 6,
    "四诊": 4,
    "三焦": 3,
}


CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def parse_small_number(text):

    text = str(
        text
    )


    if text.isdigit():
        return int(
            text
        )


    if text == "十":
        return 10


    if "十" in text:

        left, right = text.split(
            "十",
            1
        )

        tens = (
            CN_DIGITS.get(
                left,
                1
            )
            if left
            else 1
        )

        ones = (
            CN_DIGITS.get(
                right,
                0
            )
            if right
            else 0
        )

        return (
            tens * 10
            + ones
        )


    return CN_DIGITS.get(
        text
    )


def closed_set_guard(query):

    q = normalize(
        query
    )


    # =====================================================
    # V6.13.1 Generic structural extension guard
    #
    # “X之外固定的第N……”
    # 表示用户要求人为扩展一个固定结构，
    # 不允许substring实体命中后返回已有条目。
    # =====================================================

    generic = re.search(
        r"之外(?:还|再)?固定的?"
        r"第([零一二两三四五六七八九十\\d]+)",
        q
    )


    if generic:

        number = parse_small_number(
            generic.group(1)
        )


        return {
            "concept": "generic_closed_set_extension",
            "requested_index": number,
            "max_count": None,
        }


    m = re.search(
        r"第([零一二两三四五六七八九十\d]+)",
        q
    )


    if not m:
        return None


    number = parse_small_number(
        m.group(1)
    )


    if number is None:
        return None


    for concept, count in CLOSED_SET_COUNTS.items():

        if normalize(
            concept
        ) not in q:
            continue


        if number > count:

            return {
                "concept": concept,
                "requested_index": number,
                "max_count": count,
            }


    return None


# =========================================================
# Semantic Query Cleaning
# =========================================================

QUERY_BOILERPLATE = (

    "有一味中药",
    "药性资料为",
    "传统应用特点包括",
    "根据这些信息",
    "最可能是哪味药",
    "最可能是哪味中药",
    "根据以下中药资料判断药名",

    "某经典方",
    "配伍而成",
    "这个方剂叫什么",
    "以下配伍和功用对应哪一个方剂",
    "这些药物组成和功用对应哪首方剂",

    "辨证资料提示",
    "按照中医证候名称判断",
    "最符合哪一证",
    "根据以下辨证资料判断证候名称",

    "中医基础理论中",
    "有一个概念描述为",
    "这个术语是什么",

    "请根据",
    "根据下面",
    "根据以下",
    "判断",
)



V613_QUERY_BOILERPLATE = (

    "某味药的药性特点为",
    "传统上用于",
    "按这些资料应当对应哪味中药",

    "一个传统方的药物配伍包括",
    "其功用概括为",
    "这对应哪一个方名",

    "辨证信息表现为",
    "临床常见",
    "最接近哪一个中医证候名称",

    "有一个中医基础概念",
    "其含义可概括为",
    "请判断对应的理论术语",

    "根据下面这段中医药资料",
    "判断知识库中最对应的条目名称",

    "下面描述对应一个容易与其他条目混淆的中医药概念",
    "请只依据描述判断最匹配的条目",

)


def clean_query_for_embedding(
    query,
    category=None
):

    text = str(
        query
    )


    for phrase in QUERY_BOILERPLATE + V613_QUERY_BOILERPLATE:

        text = text.replace(
            phrase,
            ""
        )


    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    if len(
        normalize(
            text
        )
    ) < 4:

        return str(
            query
        )


    return text


# =========================================================
# Lexical Rerank
# =========================================================

LEXICAL_STOP_PHRASES = (

    "是什么",
    "是什么意思",
    "有什么",
    "叫什么",
    "属于什么",
    "属于哪个",
    "最符合",
    "哪味药",
    "哪味中药",
    "哪首方剂",
    "哪些",
    "什么",
    "属于",
    "组成",
    "具有",
    "作用",
    "功能",
    "证候",
    "方剂",
    "经典方",
    "同时",
    "以及",
    "怎么",
    "如何",

    "有一味中药",
    "药性资料",
    "传统应用特点",
    "根据这些信息",
    "某经典方",
    "配伍而成",
    "辨证资料提示",
    "中医基础理论",
    "概念描述",
)


def clean_for_lexical(text):

    text = normalize(
        text
    )


    for phrase in LEXICAL_STOP_PHRASES:

        text = text.replace(
            normalize(
                phrase
            ),
            ""
        )


    return text


def make_bigrams(text):

    text = clean_for_lexical(
        text
    )


    if len(text) < 2:
        return set()


    return {
        text[i:i + 2]
        for i in range(
            len(text) - 1
        )
    }


def lexical_overlap_score(
    query,
    doc
):

    aliases = doc.get(
        "aliases",
        []
    )


    if not isinstance(
        aliases,
        list
    ):
        aliases = []


    doc_text = " ".join(
        [
            str(
                doc.get(
                    "title",
                    ""
                )
            ),

            " ".join(
                str(x)
                for x in aliases
            ),

            str(
                doc.get(
                    "content",
                    ""
                )
            ),
        ]
    )


    qgrams = make_bigrams(
        query
    )

    dgrams = make_bigrams(
        doc_text
    )


    if not qgrams:
        return 0.0


    return (
        len(
            qgrams
            & dgrams
        )
        /
        len(
            qgrams
        )
    )


# =========================================================
# Vector Search
# =========================================================

def vector_search(
    query,
    top_k=3,
    category=None
):

    embed_query = clean_query_for_embedding(
        query,
        category
    )


    vector = encode_queries(
        [
            embed_query
        ]
    )


    # 300条，直接搜全库
    scores, ids = index.search(
        vector,
        int(
            index.ntotal
        )
    )


    candidates = []


    for vector_score, idx in zip(
        scores[0],
        ids[0]
    ):

        idx = int(
            idx
        )


        if idx < 0:
            continue


        doc = documents[
            idx
        ]


        if (
            category is not None
            and doc.get(
                "category"
            ) != category
        ):
            continue


        vector_score = float(
            vector_score
        )


        if category is None:

            lexical_score = 0.0

            final_score = (
                vector_score
            )


        else:

            lexical_score = (
                lexical_overlap_score(
                    query,
                    doc
                )
            )


            final_score = (
                RERANK_VECTOR_WEIGHT
                * vector_score
                +
                RERANK_LEXICAL_WEIGHT
                * lexical_score
            )


        item = dict(
            doc
        )


        item[
            "vector_score"
        ] = vector_score

        item[
            "lexical_score"
        ] = float(
            lexical_score
        )

        item[
            "score"
        ] = float(
            final_score
        )

        item[
            "retrieval_method"
        ] = (
            "hybrid"
            if category
            else "vector"
        )


        candidates.append(
            item
        )


    candidates.sort(
        key=lambda x: x[
            "score"
        ],
        reverse=True
    )


    return candidates[
        :top_k
    ]


# =========================================================
# Unified Retrieve V6.10
# =========================================================

def retrieve_base(
    query,
    top_k=3
):

    query = str(
        query or ""
    ).strip()


    if not query:

        return {
            "found": False,
            "confidence": "none",
            "method": "none",
            "query_type": "empty",
            "category_hint": None,
            "results": [],
        }


    # =====================================================
    # 1. Query type
    # =====================================================

    query_type = detect_query_type(
        query
    )


    # =====================================================
    # 2. Structural guard
    # =====================================================

    structural = closed_set_guard(
        query
    )


    if structural:

        return {
            "found": False,
            "confidence": "structural_not_found",
            "method": "structure_guard",
            "query_type": query_type,
            "category_hint": "theory",
            "results": [],
            "guard_detail": structural,
        }


    # =====================================================
    # 3. Direct known entity
    # =====================================================

    if query_type == "direct_entity":

        exact = resolve_direct_entity(
            query
        )


        if exact:

            category = infer_category(
                query,
                query_type=query_type,
                resolved_docs=exact
            )


            return {
                "found": True,
                "confidence": "high",
                "method": "exact",
                "query_type": query_type,
                "category_hint": category,
                "results": exact[
                    :top_k
                ],
            }


    # =====================================================
    # 4. Explicit unknown entity
    #
    # 用户明确点名一个KB不存在的实体，
    # 禁止向量“猜一个相似答案”。
    # =====================================================

    if query_type == "unknown_entity":

        candidate = extract_direct_candidate(
            query
        )


        return {
            "found": False,
            "confidence": "entity_not_in_kb",
            "method": "exact_guard",
            "query_type": query_type,
            "category_hint": None,
            "results": [],
            "requested_entity": candidate,
        }


    # =====================================================
    # 5. Category
    # =====================================================

    category = infer_category(
        query,
        query_type=query_type
    )


    # =====================================================
    # 6. Vector / Hybrid
    # =====================================================

    results = vector_search(
        query,
        top_k=top_k,
        category=category
    )


    if not results:

        return {
            "found": False,
            "confidence": "none",
            "method": "vector",
            "query_type": query_type,
            "category_hint": category,
            "results": [],
        }


    top1 = float(
        results[0][
            "score"
        ]
    )


    top2 = (
        float(
            results[1][
                "score"
            ]
        )
        if len(results) > 1
        else 0.0
    )


    margin = (
        top1
        - top2
    )


    # =====================================================
    # 7. Threshold
    # =====================================================

    if category:

        min_score = (
            CATEGORY_MIN_SCORE.get(
                category,
                0.60
            )
        )

        min_margin = (
            CATEGORY_MIN_MARGIN
        )


    else:

        min_score = (
            VECTOR_MIN_SCORE
        )

        min_margin = (
            VECTOR_MIN_MARGIN
        )


    # =====================================================
    # 8. Low
    # =====================================================

    if top1 < min_score:

        return {
            "found": False,
            "confidence": "low",
            "method": "vector",
            "query_type": query_type,
            "category_hint": category,
            "results": results,
        }


    # =====================================================
    # 9. Strong lexical evidence override
    # =====================================================

    lexical_override = False


    if (
        category is not None
        and len(results) >= 2
    ):

        lexical1 = float(
            results[0].get(
                "lexical_score",
                0.0
            )
        )

        lexical2 = float(
            results[1].get(
                "lexical_score",
                0.0
            )
        )

        lexical_gap = (
            lexical1
            - lexical2
        )


        # Herb单独规则
        if category == "herb":

            if (
                lexical1 >= 0.54
                and lexical_gap >= 0.08
            ):
                lexical_override = True


        else:

            if (
                lexical1 >= 0.55
                and lexical_gap >= 0.05
            ):
                lexical_override = True


    if lexical_override:

        return {
            "found": True,
            "confidence": "medium",
            "method": "vector",
            "query_type": query_type,
            "category_hint": category,
            "results": results,
        }


    # =====================================================
    # 10. Margin
    # =====================================================

    if margin < min_margin:

        return {
            "found": False,
            "confidence": "ambiguous",
            "method": "vector",
            "query_type": query_type,
            "category_hint": category,
            "results": results,
        }


    return {
        "found": True,
        "confidence": "medium",
        "method": "vector",
        "query_type": query_type,
        "category_hint": category,
        "results": results,
    }


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 64)
    print("TCM Retriever V6.13.2")
    print("Structured Query Parser")
    print("+ Direct Entity Resolver")
    print("+ Unknown Entity Guard")
    print("+ Category Router")
    print("+ BGE")
    print("+ Field-aware Hybrid Rerank")
    print("+ FAISS")
    print("=" * 64)


    while True:

        try:

            q = input(
                "\nQuery: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            break


        if q.lower() == "exit":
            break


        if not q:
            continue


        result = retrieve(
            q
        )


        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )


# =========================================================
# TCM Retriever V6.14.4 Integrated Public Interface
# =========================================================

def retrieve(
    query,
    top_k=3,
):
    """
    Official V6.14 public Retriever interface.

    Pipeline:
        V6.13.3.3 base retriever
        -> conservative field-aware rescue
        -> final result

    Safety principles:
        - existing successful result is never replaced
        - exact_guard remains authoritative
        - structure_guard remains authoritative
        - field rescue only applies to rejected
          description_to_entity queries
    """

    # Late import intentionally avoids circular import:
    #
    # field_reranker
    #     imports rag.retriever
    #
    # therefore we only import it after retriever.py
    # has completed module initialization.
    from rag.field_reranker import retrieve_v614

    return retrieve_v614(
        query,
        top_k=top_k,
    )
