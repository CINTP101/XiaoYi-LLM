# -*- coding: utf-8 -*-
import os
import json
import torch
import nltk
import argparse
from loguru import logger
from typing import List, Dict

# 下载nltk所需资源（首次运行需要）
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 导入指标计算和模型相关库
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel


# ======================== 核心指标计算函数 ========================
def chinese_char_split(text: str) -> List[str]:
    """中文文本分字（适配BLEU/ROUGE/METEOR计算）"""
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


def compute_all_metrics(
        references: List[str],
        candidates: List[str],
        use_chinese_char_split: bool = True
) -> Dict[str, float]:
    """
    批量计算所有生成指标
    :param references: 参考文本列表
    :param candidates: 生成文本列表
    :param use_chinese_char_split: 中文分字开关
    :return: 指标均值字典
    """
    logger.info(f"开始计算 {len(references)} 条样本的生成指标...")
    metrics = {
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [],
        "rougeL": [], "meteor": []
    }

    # 逐样本计算
    for idx, (ref, cand) in enumerate(zip(references, candidates)):
        if idx % 10 == 0:
            logger.info(f"计算进度: {idx}/{len(references)}")

        # 过滤空文本
        if not ref or not cand:
            logger.warning(
                f"跳过空文本（第{idx}条）: ref={ref[:20] if ref else '空'}, cand={cand[:20] if cand else '空'}")
            continue

        metrics["bleu1"].append(calculate_bleu(ref, cand, n=1, use_char_split=use_chinese_char_split))
        metrics["bleu2"].append(calculate_bleu(ref, cand, n=2, use_char_split=use_chinese_char_split))
        metrics["bleu3"].append(calculate_bleu(ref, cand, n=3, use_char_split=use_chinese_char_split))
        metrics["bleu4"].append(calculate_bleu(ref, cand, n=4, use_char_split=use_chinese_char_split))
        metrics["rougeL"].append(calculate_rouge_l(ref, cand, use_char_split=use_chinese_char_split))
        metrics["meteor"].append(calculate_meteor(ref, cand, use_char_split=use_chinese_char_split))

    # 计算均值
    avg_metrics = {k: round(sum(v) / len(v), 4) if v else 0.0 for k, v in metrics.items()}

    # 打印结果
    logger.info("\n===== 生成指标结果 =====")
    for k, v in avg_metrics.items():
        logger.info(f"{k.upper()}: {v}")

    return avg_metrics


# ======================== 数据加载函数 ========================
def load_test_data(data_path: str, max_samples: int = None) -> tuple[List[str], List[str]]:
    """
    加载测试数据（json/jsonl格式，包含conversations字段）
    :param data_path: 测试数据文件/文件夹路径
    :param max_samples: 最大加载样本数（None表示全部）
    :return: (输入问题列表, 参考回答列表)
    """
    logger.info(f"加载测试数据: {data_path}")

    # 支持文件或文件夹
    if os.path.isdir(data_path):
        data_files = []
        for ext in ["json", "jsonl"]:
            data_files.extend([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(f".{ext}")])
        if not data_files:
            raise ValueError(f"文件夹 {data_path} 中未找到json/jsonl文件")
    else:
        data_files = [data_path]

    # 加载数据集
    raw_datasets = load_dataset(
        'json',
        data_files=data_files,
        cache_dir="./cache"
    )["train"]

    # 截取样本数
    if max_samples and max_samples > 0:
        raw_datasets = raw_datasets.select(range(min(max_samples, len(raw_datasets))))

    # 提取对话对（最后一轮human-gpt）
    input_texts = []
    reference_texts = []
    roles = ["human", "gpt"]

    for idx, sample in enumerate(raw_datasets):
        conversations = sample.get("conversations", [])
        if not conversations:
            logger.warning(f"第{idx}条样本无conversations字段，跳过")
            continue

        # 过滤system开头的消息
        system_prompt = ""
        if len(conversations) > 0 and conversations[0].get("from") == "system":
            system_prompt = conversations[0].get("value", "")
            conversations = conversations[1:]

        # 提取human和gpt的消息
        human_msgs = []
        gpt_msgs = []
        for msg in conversations:
            msg_from = msg.get("from", "")
            msg_value = msg.get("value", "").strip()
            if msg_from == "human" and msg_value:
                human_msgs.append(msg_value)
            elif msg_from == "gpt" and msg_value:
                gpt_msgs.append(msg_value)

        # 只保留最后一轮有效对话
        if len(human_msgs) > 0 and len(gpt_msgs) > 0:
            last_human = human_msgs[-1]
            last_gpt = gpt_msgs[-1]

            # 拼接system prompt（如果有）
            if system_prompt:
                input_text = f"{system_prompt}\n{last_human}"
            else:
                input_text = last_human

            input_texts.append(input_text)
            reference_texts.append(last_gpt)

    logger.info(f"成功加载 {len(input_texts)} 条有效对话对")
    return input_texts, reference_texts


# ======================== 模型加载与生成函数 ========================
def load_model_and_tokenizer(
        base_model_path: str,
        peft_model_path: str = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        trust_remote_code: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和tokenizer（支持4bit/8bit量化、LoRA）
    """
    logger.info(f"加载基础模型: {base_model_path}")

    # 量化配置
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        logger.info(f"启用量化: load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}")
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit和load_in_8bit不能同时启用")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4" if load_in_4bit else None
        )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left"  # 生成时左填充更合理
    )
    # 补充特殊token（适配Qwen等模型）
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True
    )

    # 加载LoRA权重
    if peft_model_path:
        logger.info(f"加载LoRA模型: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
        model = model.merge_and_unload()  # 合并LoRA权重（可选）

    # 模型设为评估模式
    model.eval()
    logger.info("模型加载完成")
    return model, tokenizer


@torch.no_grad()
def batch_generate(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        input_texts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 4,
        device: str = "cuda"
) -> List[str]:
    """
    批量生成模型回答
    """
    logger.info(f"开始批量生成（共{len(input_texts)}条，批次大小{batch_size}）")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.05  # 轻微抑制重复
    )

    predictions = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        logger.info(f"生成进度: {min(i + batch_size, len(input_texts))}/{len(input_texts)}")

        # 编码输入
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length - max_new_tokens,
            padding=True
        ).to(device)

        # 生成回答
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

        # 解码并提取生成部分（去掉输入）
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for input_text, gen_text in zip(batch_texts, generated_texts):
            # 提取输入后的生成内容
            if gen_text.startswith(input_text):
                pred_text = gen_text[len(input_text):].strip()
            else:
                pred_text = gen_text.strip()
            predictions.append(pred_text)

    logger.info("批量生成完成")
    return predictions


# ======================== 主函数 ========================
def main():
    # 参数解析
    parser = argparse.ArgumentParser(description="测试微调模型的生成指标（BLEU/ROUGE/METEOR）")

    # 模型相关参数
    parser.add_argument("--base_model_path", required=True, help="基础模型路径（如Qwen1.5-1.8B-Chat）")
    parser.add_argument("--peft_model_path", default=None, help="LoRA模型路径（微调后的checkpoint）")
    parser.add_argument("--load_in_4bit", action="store_true", help="是否4bit量化加载模型")
    parser.add_argument("--load_in_8bit", action="store_true", help="是否8bit量化加载模型")
    parser.add_argument("--device_map", default="auto", help="设备映射（auto/cuda:0/cpu）")

    # 数据相关参数
    parser.add_argument("--test_data_path", required=True, help="测试数据文件/文件夹路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大测试样本数（调试用）")
    parser.add_argument("--use_chinese_char_split", action="store_true", default=True, help="中文是否分字计算指标")

    # 生成相关参数
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成文本最大长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成top_p")
    parser.add_argument("--batch_size", type=int, default=4, help="生成批次大小")

    # 输出相关参数
    parser.add_argument("--output_dir", default="./test_results", help="结果保存目录")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载测试数据
    input_texts, reference_texts = load_test_data(args.test_data_path, args.max_samples)
    if len(input_texts) == 0:
        logger.error("未加载到有效测试数据，退出")
        return

    # 2. 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(
        base_model_path=args.base_model_path,
        peft_model_path=args.peft_model_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device_map=args.device_map
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. 批量生成回答
    pred_texts = batch_generate(
        model=model,
        tokenizer=tokenizer,
        input_texts=input_texts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        device=device
    )

    # 4. 计算指标
    metrics = compute_all_metrics(
        references=reference_texts,
        candidates=pred_texts,
        use_chinese_char_split=args.use_chinese_char_split
    )

    # 5. 保存结果
    # 保存指标
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标结果已保存到: {metrics_path}")

    # 保存生成示例（前20条）
    examples_path = os.path.join(args.output_dir, "generation_examples.json")
    examples = []
    for i in range(min(20, len(input_texts))):
        examples.append({
            "input": input_texts[i],
            "reference": reference_texts[i],
            "prediction": pred_texts[i]
        })
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    logger.info(f"生成示例已保存到: {examples_path}")

    # 保存所有生成结果
    all_results_path = os.path.join(args.output_dir, "all_results.json")
    all_results = []
    for i in range(len(input_texts)):
        all_results.append({
            "input": input_texts[i],
            "reference": reference_texts[i],
            "prediction": pred_texts[i]
        })
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"所有生成结果已保存到: {all_results_path}")

    logger.info("测试完成！")


if __name__ == "__main__":
    main()