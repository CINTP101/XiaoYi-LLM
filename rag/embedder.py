import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "bge-small-zh-v1.5"
)

QUERY_INSTRUCTION = (
    "为这个句子生成表示以用于检索相关文章："
)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"BGE模型不存在: {MODEL_PATH}"
    )


print("[Embedder] 加载 BGE tokenizer...")

# 强制 slow tokenizer，避开当前 fast tokenizer 报错
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    use_fast=False,
    local_files_only=True
)


print("[Embedder] 加载 BGE model...")

model = AutoModel.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.to("cpu")
model.eval()

print("[Embedder] BGE 加载完成 ✅")


def _encode(
    texts,
    batch_size=32
):
    """
    使用BGE官方推荐的CLS pooling：
    last_hidden_state[:, 0]
    然后进行L2归一化。
    """

    texts = [
        str(x)
        for x in texts
    ]

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:
            start + batch_size
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(
                **encoded
            )

            embeddings = (
                outputs.last_hidden_state[
                    :, 0
                ]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.append(
            embeddings.cpu()
        )

    if not all_embeddings:

        return np.empty(
            (0, 0),
            dtype=np.float32
        )

    embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    return embeddings.numpy().astype(
        np.float32
    )


def encode_documents(
    texts,
    batch_size=32
):
    """
    文档不加Query instruction。
    """

    return _encode(
        texts,
        batch_size=batch_size
    )


def encode_queries(
    queries,
    batch_size=32
):
    """
    短查询添加BGE中文检索instruction。
    """

    queries = [
        QUERY_INSTRUCTION
        + str(q)
        for q in queries
    ]

    return _encode(
        queries,
        batch_size=batch_size
    )


if __name__ == "__main__":

    test = encode_queries(
        [
            "麻黄汤的组成是什么？"
        ]
    )

    print(
        "Embedding shape:",
        test.shape
    )

    print(
        "Embedding dtype:",
        test.dtype
    )

    print(
        "BGE直接编码测试成功 ✅"
    )
