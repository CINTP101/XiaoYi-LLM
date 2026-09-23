import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from pathlib import Path

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from safety_router import route as safety_route
from intent_router import detect_intent
from rag.retriever import retrieve


# =========================================================
# 配置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASE_MODEL = (
    PROJECT_ROOT
    / "models"
    / "Qwen2.5-1.5B-Instruct"
)

LORA_MODEL = (
    PROJECT_ROOT
    / "output"
    / "tcm-qwen-1.5b-v5-1"
)

MAX_HISTORY_MESSAGES = 12


# =========================================================
# 问诊 Prompt
# =========================================================

CONSULTATION_PROMPT = """
你是中医问诊辅助模型。

当前任务是患者个人问诊。

只能输出合法JSON。

action只能为：
ask
summarize

规则：

1. 信息不足时使用ask。
2. 每轮最多询问3个最重要的问题。
3. 优先询问病程、伴随症状、寒热、汗、饮食、
   二便、睡眠。
4. 不重复患者已经回答的问题。
5. 用户明确不知道某项信息时，不得反复追问。
6. 不得编造姓名、年龄、性别、舌象、脉象、
   既往病史和用药史。
7. 未提供的信息不等于正常。
8. 信息不足时不得推荐具体中药或方剂。
9. 信息基本充分时可以进行阶段性辨证总结。
10. 辨证只能使用“倾向”“可能”等表达。

ask格式：

{
  "action": "ask",
  "stage": "initial",
  "complete": false,
  "questions": [
    "问题1",
    "问题2"
  ]
}

summarize格式：

{
  "action": "summarize",
  "stage": "summary",
  "complete": true,
  "key_findings": [],
  "syndrome_tendency": "",
  "need_more_info": [],
  "note": ""
}

禁止输出JSON之外的内容。
"""


# =========================================================
# 加载 V5.1
# =========================================================

print("=" * 60)
print("正在启动 TCM V6.2")
print("=" * 60)

print("正在加载基础模型...")

tokenizer = AutoTokenizer.from_pretrained(
    str(BASE_MODEL),
    trust_remote_code=True,
    local_files_only=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    str(BASE_MODEL),
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)

print("正在加载 V5.1 LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    str(LORA_MODEL)
)

model.eval()

try:
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
except Exception:
    pass

print("V5.1 加载完成 ✅")


# =========================================================
# 问诊历史
# =========================================================

consultation_history = []


# =========================================================
# JSON解析
# =========================================================

def extract_json(text):

    if not text:
        return None

    text = str(text).strip()

    text = text.replace("```json", "")
    text = text.replace("```JSON", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            pass

    return None


# =========================================================
# 结构标准化
# =========================================================

def normalize_questions(value):

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, dict):

        q = (
            value.get("question")
            or value.get("text")
            or value.get("value")
        )

        return [str(q).strip()] if q else []

    if isinstance(value, list):

        result = []

        for item in value:

            if isinstance(item, str):

                item = item.strip()

                if item:
                    result.append(item)

            elif isinstance(item, dict):

                q = (
                    item.get("question")
                    or item.get("text")
                    or item.get("value")
                )

                if q:
                    result.append(
                        str(q).strip()
                    )

        return result

    return []


def normalize_list(value):

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, list):

        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    return []


# =========================================================
# 问诊输出
# =========================================================

def normalize_consultation(result):

    if not isinstance(result, dict):

        return {
            "action": "error",
            "stage": "output_validation",
            "complete": False,
            "message": "问诊模型没有返回合法JSON。"
        }

    action = str(
        result.get("action", "")
    ).lower().strip()


    # -----------------------------------------------------
    # ASK
    # -----------------------------------------------------

    if action == "ask":

        questions = (
            result.get("questions")
            or result.get("question")
            or result.get("need_more_info")
        )

        questions = normalize_questions(
            questions
        )

        if not questions:

            questions = [
                "这种情况大约持续多久了？",
                "还有没有其他伴随不适？",
                "最近饮食、睡眠和大小便情况怎么样？"
            ]

        return {
            "action": "ask",
            "stage": result.get(
                "stage",
                "initial"
            ),
            "complete": False,
            "questions": questions[:3]
        }


    # -----------------------------------------------------
    # SUMMARIZE
    # -----------------------------------------------------

    if action == "summarize":

        return {
            "action": "summarize",
            "stage": "summary",
            "complete": True,

            "key_findings": normalize_list(
                result.get(
                    "key_findings",
                    []
                )
            ),

            "syndrome_tendency": str(
                result.get(
                    "syndrome_tendency",
                    ""
                )
            ).strip(),

            "need_more_info": normalize_list(
                result.get(
                    "need_more_info",
                    []
                )
            ),

            "note": str(
                result.get(
                    "note",
                    "当前仅为阶段性辨证总结。"
                )
            ).strip()
        }


    # -----------------------------------------------------
    # 防止个人症状直接answer
    # -----------------------------------------------------

    if action == "answer":

        return {
            "action": "ask",
            "stage": "initial",
            "complete": False,
            "questions": [
                "这种情况大约持续多久了？",
                "还有没有其他伴随不适？",
                "最近饮食、睡眠和大小便情况怎么样？"
            ]
        }


    return {
        "action": "error",
        "stage": "output_validation",
        "complete": False,
        "message": "未知问诊状态。"
    }


# =========================================================
# 调用模型
# =========================================================

def generate(messages):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    device = next(
        model.parameters()
    ).device

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()


# =========================================================
# Consultation
# =========================================================

def run_consultation(text):

    messages = [
        {
            "role": "system",
            "content": CONSULTATION_PROMPT
        }
    ]

    messages.extend(
        consultation_history[
            -MAX_HISTORY_MESSAGES:
        ]
    )

    messages.append(
        {
            "role": "user",
            "content": text
        }
    )

    raw = generate(
        messages
    )

    result = normalize_consultation(
        extract_json(raw)
    )

    consultation_history.append(
        {
            "role": "user",
            "content": text
        }
    )

    consultation_history.append(
        {
            "role": "assistant",
            "content": json.dumps(
                result,
                ensure_ascii=False
            )
        }
    )

    if len(
        consultation_history
    ) > MAX_HISTORY_MESSAGES:

        del consultation_history[
            :-MAX_HISTORY_MESSAGES
        ]

    return result


# =========================================================
# Source标准化
# =========================================================

def build_source(item):

    return {
        "id": item.get(
            "id",
            ""
        ),

        "title": item.get(
            "title",
            ""
        ),

        "category": item.get(
            "category",
            ""
        ),

        "source": item.get(
            "source",
            ""
        ),

        "source_ref": item.get(
            "source_ref",
            ""
        ),

        "source_type": item.get(
            "source_type",
            ""
        ),

        "retrieval_method": item.get(
            "retrieval_method",
            ""
        ),

        "score": round(
            float(
                item.get(
                    "score",
                    0
                )
            ),
            4
        )
    }


# =========================================================
# Knowledge
# =========================================================

def run_knowledge(text):

    retrieval = retrieve(
        text,
        top_k=3
    )

    print(
        "[Retriever]",
        f"found={retrieval.get('found')}",
        f"confidence={retrieval.get('confidence')}",
        f"method={retrieval.get('method')}"
    )

    results = retrieval.get(
        "results",
        []
    )


    # -----------------------------------------------------
    # 没有可靠资料
    # -----------------------------------------------------

    if not retrieval.get(
        "found"
    ):

        sources = [
            build_source(x)
            for x in results[:3]
        ]

        return {
            "action": "knowledge_not_found",
            "stage": "knowledge",
            "complete": True,
            "answer": (
                "当前知识库没有检索到足够可靠的相关资料，"
                "暂不基于模型记忆自行回答。"
            ),
            "confidence": retrieval.get(
                "confidence",
                "none"
            ),
            "sources": sources
        }


    # -----------------------------------------------------
    # 找到知识
    # -----------------------------------------------------

    top = results[0]

    content = str(
        top.get(
            "content",
            ""
        )
    ).strip()


    if not content:

        return {
            "action": "knowledge_not_found",
            "stage": "knowledge",
            "complete": True,
            "answer": "检索到了知识条目，但该条目没有有效正文。",
            "sources": [
                build_source(
                    top
                )
            ]
        }


    # -----------------------------------------------------
    # 当前阶段直接使用知识库正文
    #
    # 不让1.5B重新编写医学事实
    # -----------------------------------------------------

    return {
        "action": "answer",
        "stage": "knowledge_verified",
        "complete": True,
        "answer": content,

        "confidence": retrieval.get(
            "confidence"
        ),

        "retrieval_method": retrieval.get(
            "method"
        ),

        "sources": [
            build_source(
                top
            )
        ]
    }


# =========================================================
# CLI
# =========================================================


def main():
    print()
    print("=" * 60)

    print("TCM V6.2 已启动")

    print()
    print("Safety Router")
    print("+ Intent Router")
    print("+ V5.1 Consultation")
    print("+ Hybrid RAG Retriever")
    print("  ├─ Exact title / alias")
    print("  └─ BGE + FAISS")
    print("+ Verified Knowledge Answer")

    print()

    print(
        "clear = 清除问诊历史"
    )

    print(
        "exit  = 退出"
    )

    print("=" * 60)
    print()


    while True:

        try:

            text = input(
                "User: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                "\n程序退出。"
            )

            break


        if not text:
            continue


        if text.lower() == "exit":

            print(
                "程序退出。"
            )

            break


        if text.lower() == "clear":

            consultation_history.clear()

            print(
                "History cleared."
            )

            continue


        # =====================================================
        # 1. Safety
        # =====================================================

        safety = safety_route(
            text
        )


        if safety is not None:

            print(
                "Assistant:",
                json.dumps(
                    safety,
                    ensure_ascii=False,
                    indent=2
                )
            )

            continue


        # =====================================================
        # 2. Intent
        # =====================================================

        intent = detect_intent(
            text
        )


        print(
            f"[Router] intent = {intent}"
        )


        # =====================================================
        # 3. 分流
        # =====================================================

        if intent == "knowledge":

            result = run_knowledge(
                text
            )

        else:

            result = run_consultation(
                text
            )


        print(
            "Assistant:",
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )



if __name__ == "__main__":
    main()
