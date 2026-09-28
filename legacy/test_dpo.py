# -*- coding: utf-8 -*-
"""
测试DPO微调后模型的文本生成指标（BLEU1-4、ROUGE-L、METEOR）
适配MedicalGPT的DPO训练逻辑，兼容中文医疗数据集
"""
import os
import json
from glob import glob
from typing import List, Dict, Iterable
import math

import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel  # 加载LoRA微调后的模型

# 下载nltk所需资源（首次运行需要）
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# ========== 配置参数（根据你的实际情况修改） ==========
class TestConfig:
    # 模型路径：DPO微调后的输出目录
    model_name_or_path = "/home/gyl/project/Medical_Image_Analysis/R2GenCSR/Qwen/Qwen1.5-1.8B-Chat"
    # LoRA权重路径（如果用PEFT微调，填LoRA权重目录；否则设为None）
    peft_model_path = "outputs-pt-qwen-v1/checkpoint-79"
    # 测试数据集路径（和训练时的validation_file_dir一致）
    test_data_dir = "/home/gyl/DataSets/medical/pretrain/test_encyclopedia.json"
    # Prompt模板名称（和训练时的template_name一致）
    template_name = "vicuna"
    # 模型加载配置
    load_in_8bit = False
    load_in_4bit = True
    torch_dtype = torch.bfloat16
    device_map = "auto"
    trust_remote_code = True
    # 生成参数
    max_new_tokens = 512  # 最大生成长度（和训练时的max_target_length一致）
    temperature = 0.1  # 生成温度（越低越稳定）
    top_p = 0.9
    do_sample = False  # 关闭采样，保证生成结果可复现
    # 【新增】批量生成配置
    batch_size = 25  # 测试批量大小（4bit量化建议2-4，8bit建议1-2，根据显存调整）
    max_prompt_length = 2048  # 单个prompt的最大长度（和训练时的max_source_length一致）
    # 指标计算配置
    use_chinese_char_split = True  # 中文用分字计算指标（更适配医疗文本）


# ========== 复用训练时的Prompt模板逻辑（必须和训练一致） ==========
def get_conv_template(template_name: str = "vicuna"):
    """复用训练代码中的prompt模板，确保输入格式一致"""

    class ConvTemplate:
        def __init__(self):
            self.stop_str = "</s>"  # 对应训练时的eos_token

        def get_prompt(self, messages: List[List[str]], system_prompt: str = "") -> str:
            """构建vicuna格式的prompt"""
            prompt = system_prompt + "\n" if system_prompt else ""
            for q, a in messages:
                if a:
                    prompt += f"USER: {q}\nASSISTANT: {a}{self.stop_str}\n"
                else:
                    prompt += f"USER: {q}\nASSISTANT: "
            return prompt.strip()

    return ConvTemplate()


# ========== 数据集加载（复用训练逻辑） ==========
def load_test_dataset(config: TestConfig) -> Dict[str, List[str]]:
    """加载测试数据集，返回{prompt: 输入提示, reference: 参考回复（chosen）}"""
    data_files = glob(f'{config.test_data_dir}/**/*.json', recursive=True) + glob(
        f'{config.test_data_dir}/**/*.jsonl', recursive=True)
    logger.info(f"加载测试文件：{', '.join(data_files)}")

    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir=None,
    )["train"]  # 测试集统一用train split

    # 复用训练时的prompt构建逻辑
    prompt_template = get_conv_template(config.template_name)
    prompts = []
    references = []

    for example in raw_datasets:
        # 构建prompt（和训练时一致）
        system = example.get("system", "")
        history = example.get("history", [])
        question = example.get("question", "")
        history_with_question = history + [[question, '']] if history else [[question, '']]
        prompt = prompt_template.get_prompt(messages=history_with_question, system_prompt=system)
        # 参考回复（chosen是人工偏好的正确回复）
        reference = example.get("response_chosen", "")

        if prompt and reference:  # 过滤空样本
            prompts.append(prompt)
            references.append(reference)

    logger.info(f"测试数据集加载完成，有效样本数：{len(prompts)}")
    return {
        "prompts": prompts,
        "references": references
    }

# 【新增】批量数据拆分工具函数
def batchify(data: List, batch_size: int) -> Iterable[List]:
    """将列表按batch_size拆分，最后一个批次不足则保留"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# ========== 中文适配的指标计算函数 ==========
def chinese_char_split(text: str) -> List[str]:
    """中文文本分字（适配BLEU/ROUGE计算）"""
    return [char for char in text if char.strip()]


def calculate_bleu(reference: str, candidate: str, n: int, use_char_split: bool = True) -> float:
    """
    计算BLEU-n分数（n=1/2/3/4）
    :param reference: 参考文本
    :param candidate: 生成文本
    :param n: BLEU阶数
    :param use_char_split: 是否分字（中文必选）
    :return: BLEU分数（0-1）
    """
    if use_char_split:
        ref_tokens = [chinese_char_split(reference)]  # sentence_bleu要求参考是二维列表
        cand_tokens = chinese_char_split(candidate)
    else:
        ref_tokens = [nltk.word_tokenize(reference.lower())]
        cand_tokens = nltk.word_tokenize(candidate.lower())

    # 权重配置（BLEU-1: (1,0,0,0), BLEU-2: (0.5,0.5,0,0)...）
    weights = [1.0 / n] * n + [0.0] * (4 - n)

    # 平滑函数（避免0分）
    smooth_fn = SmoothingFunction().method4

    try:
        bleu_score = sentence_bleu(
            ref_tokens, cand_tokens,
            weights=weights,
            smoothing_function=smooth_fn
        )
    except:
        bleu_score = 0.0
    return bleu_score


def calculate_rouge_l(reference: str, candidate: str, use_char_split: bool = True) -> float:
    """计算ROUGE-L分数（F1值）"""
    if use_char_split:
        # 中文分字后拼接成空格分隔的字符串（适配rouge-score库）
        ref_text = " ".join(chinese_char_split(reference))
        cand_text = " ".join(chinese_char_split(candidate))
    else:
        ref_text = reference
        cand_text = candidate

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = scorer.score(ref_text, cand_text)
    return scores['rougeL'].fmeasure


def calculate_meteor(reference: str, candidate: str, use_char_split: bool = True) -> float:
    """计算METEOR分数"""
    if use_char_split:
        ref_tokens = chinese_char_split(reference)
        cand_tokens = chinese_char_split(candidate)
    else:
        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())

    try:
        meteor = meteor_score([ref_tokens], cand_tokens)
    except:
        meteor = 0.0
    return meteor


# ========== 模型生成函数（【修改】批量生成） ==========
def load_model_and_tokenizer(config: TestConfig):
    """加载微调后的模型和分词器"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=False,
        padding_side='left'  # 核心修复：decoder-only模型必须左填充
    )
    # 复用训练时的特殊令牌配置
    prompt_template = get_conv_template(config.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=config.torch_dtype,
    ) if (config.load_in_4bit or config.load_in_8bit) else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True
    )

    # 加载LoRA权重（如果用PEFT微调）
    if config.peft_model_path is not None:
        model = PeftModel.from_pretrained(model, config.peft_model_path)
        logger.info(f"成功加载LoRA权重：{config.peft_model_path}")

    # 推理模式（禁用梯度计算）
    model.eval()
    return model, tokenizer


# 【修改】批量生成回复函数
def generate_batch_responses(
        model,
        tokenizer,
        prompts: List[str],
        config: TestConfig
) -> List[str]:
    """批量生成模型回复"""
    # 1. 批量编码prompt（添加padding和truncation）
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,  # 批量padding到最长prompt长度
        truncation=True,  # 截断超长prompt
        max_length=config.max_prompt_length,
        add_special_tokens=True
    ).to(model.device)  # 移到模型所在设备

    # 2. 构建批量生成参数
    generate_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "do_sample": config.do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": encodings["attention_mask"],  # 避免padding部分参与注意力计算
        "return_dict_in_generate": True,
        "output_scores": False,
    }

    # 3. 批量生成回复
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encodings["input_ids"],
            **generate_kwargs
        )

    # 4. 解码生成结果（只保留新增部分，去掉原prompt）
    generated_responses = []
    for i, prompt in enumerate(prompts):
        # 提取当前样本的生成token（去掉原prompt的token）
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=True))
        generated_tokens = outputs.sequences[i][prompt_len:]
        # 解码为文本
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        # 移除停止符
        if tokenizer.eos_token in response:
            response = response.replace(tokenizer.eos_token, "").strip()
        generated_responses.append(response)

    return generated_responses


# ========== 主测试逻辑（【修改】批量处理） ==========
def main():
    # 1. 初始化配置
    config = TestConfig()
    logger.info(f"测试配置：{config.__dict__}")

    # 2. 加载模型和分词器
    logger.info("开始加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("模型和分词器加载完成！")

    # 3. 加载测试数据集
    logger.info("开始加载测试数据集...")
    test_data = load_test_dataset(config)
    prompts = test_data["prompts"]
    references = test_data["references"]
    total_samples = len(prompts)
    total_batches = math.ceil(total_samples / config.batch_size)

    # 4. 生成回复并计算指标
    logger.info(f"开始批量生成回复（总样本数：{total_samples}，总批次：{total_batches}，批次大小：{config.batch_size}）...")
    metrics = {
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [],
        "rougeL": [], "meteor": []
    }
    examples = []  # 保存前5个样本示例

    # 按批次处理数据
    batch_idx = 0
    for prompt_batch, ref_batch in zip(batchify(prompts, config.batch_size), batchify(references, config.batch_size)):
        batch_idx += 1
        if batch_idx % 5 == 0:
            logger.info(f"处理进度：批次 {batch_idx}/{total_batches}")

        # 批量生成回复
        pred_batch = generate_batch_responses(model, tokenizer, prompt_batch, config)

        # 逐样本计算指标
        for idx_in_batch, (prompt, ref, pred) in enumerate(zip(prompt_batch, ref_batch, pred_batch)):
            # 计算指标
            metrics["bleu1"].append(calculate_bleu(ref, pred, n=1, use_char_split=config.use_chinese_char_split))
            metrics["bleu2"].append(calculate_bleu(ref, pred, n=2, use_char_split=config.use_chinese_char_split))
            metrics["bleu3"].append(calculate_bleu(ref, pred, n=3, use_char_split=config.use_chinese_char_split))
            metrics["bleu4"].append(calculate_bleu(ref, pred, n=4, use_char_split=config.use_chinese_char_split))
            metrics["rougeL"].append(calculate_rouge_l(ref, pred, use_char_split=config.use_chinese_char_split))
            metrics["meteor"].append(calculate_meteor(ref, pred, use_char_split=config.use_chinese_char_split))

            # 保存前5个示例
            global_idx = (batch_idx - 1) * config.batch_size + idx_in_batch
            if global_idx < 5:
                examples.append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "reference": ref,
                    "prediction": pred
                })

    # 5. 计算指标均值
    avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    # 6. 输出结果
    logger.info("=" * 50 + " 测试结果 " + "=" * 50)
    # 输出整体指标
    logger.info("【整体平均指标】")
    for k, v in avg_metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}")
    # 输出示例
    logger.info("\n【前5个样本示例】")
    for i, ex in enumerate(examples):
        logger.info(f"\n示例{i + 1}：")
        logger.info(f"Prompt: {ex['prompt']}")
        logger.info(f"参考回复: {ex['reference']}")
        logger.info(f"模型回复: {ex['prediction']}")

    # 7. 保存结果到文件
    result_file = os.path.join(config.model_name_or_path, "test_metrics.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "avg_metrics": avg_metrics,
            "examples": examples,
            "test_config": config.__dict__  # 保存测试配置（含batch_size）
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"测试结果已保存到：{result_file}")


if __name__ == "__main__":
    # 安装依赖（首次运行执行）
    # pip install nltk rouge-score
    main()