# -*- coding: utf-8 -*-
"""
测试CLM微调后模型的核心指标（Perplexity + BLEU1-4 + ROUGE-L + METEOR）
适配中文医疗文本，支持LoRA/QLoRA/4bit/8bit量化
"""
import math
import os
import json
from dataclasses import dataclass, field
from glob import glob
from typing import Optional, List, Dict, Any, Iterable
import torch
import nltk  # 新增：nltk依赖
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # 新增
from nltk.translate.meteor_score import meteor_score  # 新增
from rouge_score import rouge_scorer  # 新增
from datasets import load_dataset
from loguru import logger
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.integrations import is_deepspeed_zero3_enabled

# 新增：下载nltk所需资源（首次运行需要）
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# ========== 复用原代码的参数结构（新增中文分字配置） ==========
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "微调后的模型权重路径/基础模型路径"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "分词器路径（默认和模型路径一致）"}
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "是否8bit加载模型"})
    load_in_4bit: bool = field(default=False, metadata={"help": "是否4bit加载模型"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "模型缓存目录"})
    model_revision: Optional[str] = field(default="main", metadata={"help": "模型版本"})
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "HF Hub令牌"})
    use_fast_tokenizer: bool = field(default=False, metadata={"help": "是否使用快速分词器"})
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"], "help": "模型加载精度"}
    )
    device_map: Optional[str] = field(default="auto", metadata={"help": "设备映射"})
    trust_remote_code: bool = field(default=True, metadata={"help": "是否信任远程代码"})

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("必须指定model_name_or_path")


@dataclass
class DataArguments:
    test_file_dir: Optional[str] = field(default=None, metadata={"help": "测试数据文件/目录路径"})
    max_test_samples: Optional[int] = field(default=None, metadata={"help": "最大测试样本数（调试用）"})
    block_size: Optional[int] = field(default=1024, metadata={"help": "输入序列长度"})
    overwrite_cache: bool = field(default=False, metadata={"help": "是否覆盖缓存"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "预处理进程数"})
    keep_linebreaks: bool = field(default=True, metadata={"help": "是否保留换行符"})
    streaming: bool = field(default=False, metadata={"help": "是否流式加载数据"})


@dataclass
class TestArguments:
    peft_model_path: Optional[str] = field(default=None, metadata={"help": "LoRA权重路径（微调后输出目录）"})
    batch_size: int = field(default=4, metadata={"help": "测试批量大小（4bit建议4，8bit建议2）"})
    max_new_tokens: int = field(default=256, metadata={"help": "生成文本最大长度"})
    temperature: float = field(default=0.7, metadata={"help": "生成温度"})
    top_p: float = field(default=0.9, metadata={"help": "核采样概率"})
    do_sample: bool = field(default=True, metadata={"help": "是否采样生成"})
    seed: int = field(default=42, metadata={"help": "随机种子"})
    save_results_path: Optional[str] = field(default="test_results.json", metadata={"help": "测试结果保存路径"})
    num_generate_examples: int = field(default=5, metadata={"help": "生成示例数量"})
    use_chinese_char_split: bool = field(default=True, metadata={"help": "中文是否分字计算指标（必选True）"})  # 新增


# ========== 新增：中文适配的基础函数 + 指标计算函数 ==========
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


def calculate_generation_metrics(
        references: List[str],
        candidates: List[str],
        use_chinese_char_split: bool = True
) -> Dict[str, float]:
    """
    批量计算生成文本的所有指标（BLEU1-4 + ROUGE-L + METEOR）
    :param references: 参考文本列表
    :param candidates: 生成文本列表
    :param use_chinese_char_split: 中文分字开关
    :return: 指标均值字典
    """
    logger.info("开始计算生成文本指标（BLEU1-4/ROUGE-L/METEOR）...")
    metrics = {
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": [],
        "rougeL": [], "meteor": []
    }

    # 逐样本计算
    for ref, cand in zip(references, candidates):
        # 过滤空文本
        if not ref or not cand:
            logger.warning(f"空文本跳过：ref={ref[:20]}..., cand={cand[:20]}...")
            continue

        metrics["bleu1"].append(calculate_bleu(ref, cand, n=1, use_char_split=use_chinese_char_split))
        metrics["bleu2"].append(calculate_bleu(ref, cand, n=2, use_char_split=use_chinese_char_split))
        metrics["bleu3"].append(calculate_bleu(ref, cand, n=3, use_char_split=use_chinese_char_split))
        metrics["bleu4"].append(calculate_bleu(ref, cand, n=4, use_char_split=use_chinese_char_split))
        metrics["rougeL"].append(calculate_rouge_l(ref, cand, use_char_split=use_chinese_char_split))
        metrics["meteor"].append(calculate_meteor(ref, cand, use_char_split=use_chinese_char_split))

    # 计算均值
    avg_metrics = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}
    logger.info("生成文本指标计算完成：")
    for k, v in avg_metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}")
    return avg_metrics


# ========== 复用原代码的核心工具函数 ==========
def find_all_linear_names(peft_model, int4=False, int8=False):
    """查找模型中所有线性层名称（适配QLoRA）"""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            if 'lm_head' in name or 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def batchify(data: List, batch_size: int) -> Iterable[List]:
    """批量拆分数据"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# ========== 数据加载（完全复用原代码逻辑） ==========
def load_test_dataset(model_args: ModelArguments, data_args: DataArguments) -> Any:
    """加载测试数据集（支持txt/json/jsonl，兼容文件/目录路径）"""
    if data_args.streaming:
        require_version("datasets>=2.0.0", "流式加载需要datasets>=2.0.0")

    # 处理数据路径（兼容文件/目录）
    if os.path.isfile(data_args.test_file_dir):
        data_files = {"test": [data_args.test_file_dir]}
    elif os.path.isdir(data_args.test_file_dir):
        data_files = {"test": glob(f'{data_args.test_file_dir}/**/*.txt', recursive=True) +
                              glob(f'{data_args.test_file_dir}/**/*.json', recursive=True) +
                              glob(f'{data_args.test_file_dir}/**/*.jsonl', recursive=True)}
    else:
        raise ValueError(f"测试数据路径不存在：{data_args.test_file_dir}")

    if not data_files["test"]:
        raise ValueError(f"未找到测试文件，请检查路径：{data_args.test_file_dir}")
    logger.info(f"加载测试文件：{data_files['test']}")

    # 确定数据格式（txt/json/jsonl）
    types = [f.split('.')[-1] for f in data_files["test"]]
    if len(set(types)) > 1:
        raise ValueError(f"测试文件必须统一格式（全txt/全json/全jsonl），当前：{types}")
    extension = "text" if types[0] == "txt" else "json"

    # 加载数据集
    dataset_args = {}
    if extension == "text":
        dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        streaming=data_args.streaming,
        **dataset_args,
    )

    # 截断测试样本（调试用）
    test_dataset = raw_datasets["test"]
    if data_args.max_test_samples is not None and data_args.max_test_samples > 0:
        max_test_samples = min(len(test_dataset), data_args.max_test_samples)
        test_dataset = test_dataset.select(range(max_test_samples))
    logger.info(f"测试数据集加载完成，样本数：{len(test_dataset)}")

    return test_dataset


def preprocess_test_dataset(test_dataset, tokenizer, data_args: DataArguments):
    """预处理测试数据集（适配CLM任务）"""
    column_names = list(test_dataset.features)

    def tokenize_function(examples):
        """分词函数（复用原代码逻辑）"""
        tokenized_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=data_args.block_size,
            return_overflowing_tokens=False,
        )
        # CLM任务：labels = input_ids（自回归预测）
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    # 分词处理
    tokenized_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="分词测试数据集",
    )

    # 转换为PyTorch格式
    tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized_dataset


# ========== 模型加载（修复参数重复问题） ==========
def load_model_and_tokenizer(model_args: ModelArguments, test_args: TestArguments):
    """加载模型和分词器（支持4bit/8bit/LoRA）"""
    # 设置随机种子
    set_seed(test_args.seed)

    # 加载分词器
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

    # 修复decoder-only模型的padding问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 左填充（适配自回归模型）

    # 加载模型配置
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    config_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # 量化配置（复用原代码逻辑）
    load_in_4bit = model_args.load_in_4bit
    load_in_8bit = model_args.load_in_8bit
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit和load_in_8bit不能同时开启")
    elif load_in_8bit or load_in_4bit:
        logger.info(f"量化加载模型：4bit={load_in_4bit}, 8bit={load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3不兼容量化")

        if load_in_8bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            config_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )

    # 加载基础模型（修复trust_remote_code重复问题）
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        device_map=model_args.device_map,
        **config_kwargs,
    )

    # 加载LoRA权重（如果指定）
    if test_args.peft_model_path is not None:
        logger.info(f"加载LoRA权重：{test_args.peft_model_path}")
        model = PeftModel.from_pretrained(model, test_args.peft_model_path)
    model.eval()  # 推理模式
    logger.info("模型加载完成！")
    return model, tokenizer


# ========== 核心评估函数 ==========
@torch.no_grad()
def calculate_perplexity(model, tokenizer, test_dataset, data_args: DataArguments, test_args: TestArguments):
    """计算CLM核心指标：困惑度（Perplexity）"""
    logger.info("开始计算Perplexity...")
    total_loss = 0.0
    total_tokens = 0

    # 批量处理测试数据
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_args.batch_size,
        shuffle=False,
        collate_fn=lambda x: {k: torch.stack([item[k] for item in x]) for k in x[0].keys()}
    )

    for batch_idx, batch in enumerate(dataloader):
        # 移到模型设备
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # 前向计算损失
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # 累计损失和token数
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += attention_mask.sum().item()

        # 打印进度
        if batch_idx % 10 == 0:
            logger.info(f"处理批次 {batch_idx}/{len(dataloader)} | 累计损失: {total_loss:.4f}")

    # 计算平均损失和困惑度
    avg_loss = total_loss / len(test_dataset)
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float("inf")
    logger.info(f"Perplexity计算完成 | 平均损失: {avg_loss:.4f} | Perplexity: {perplexity:.4f}")
    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens
    }


@torch.no_grad()
def generate_text_for_metrics(
        model,
        tokenizer,
        test_dataset,
        test_args: TestArguments
) -> Dict[str, List[str]]:
    """
    批量生成测试集文本（用于计算指标）
    :return: {"references": 参考文本列表, "candidates": 生成文本列表, "examples": 示例详情}
    """
    logger.info(f"批量生成测试集文本（共{len(test_dataset)}样本）...")
    references = []  # 参考文本（完整文本）
    candidates = []  # 生成文本（续写部分）
    examples = []  # 前N个示例详情

    # 批量处理
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_args.batch_size,
        shuffle=False
    )

    for batch_idx, batch in enumerate(dataloader):
        batch_texts = batch["text"] if isinstance(batch["text"], list) else batch["text"].tolist()

        # 批量编码prompt（取前半部分作为输入）
        prompts = [text[:min(len(text) // 2, tokenizer.model_max_length - test_args.max_new_tokens)] for text in
                   batch_texts]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length - test_args.max_new_tokens,
            padding=True
        ).to(model.device)

        # 批量生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=test_args.max_new_tokens,
            temperature=test_args.temperature,
            top_p=test_args.top_p,
            do_sample=test_args.do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # 解码并收集结果
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for idx, (prompt, full_text, gen_text) in enumerate(zip(prompts, batch_texts, generated_texts)):
            # 提取生成的续写部分（去掉prompt）
            gen_text = gen_text[len(prompt):].strip()
            # 参考文本取prompt后的对应长度（对齐生成长度）
            ref_text = full_text[len(prompt):len(prompt) + len(gen_text)].strip()

            references.append(ref_text)
            candidates.append(gen_text)

            # 保存前N个示例
            global_idx = batch_idx * test_args.batch_size + idx
            if global_idx < test_args.num_generate_examples:
                examples.append({
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "reference": ref_text,
                    "generated": gen_text,
                    "full_text": full_text[:200] + "..." if len(full_text) > 200 else full_text
                })

        if batch_idx % 5 == 0:
            logger.info(f"生成进度：批次 {batch_idx}/{len(dataloader)}")

    return {
        "references": references,
        "candidates": candidates,
        "examples": examples
    }


# ========== 主测试逻辑 ==========
def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TestArguments))
    model_args, data_args, test_args = parser.parse_args_into_dataclasses()

    # 打印配置
    logger.info(f"模型配置: {model_args}")
    logger.info(f"数据配置: {data_args}")
    logger.info(f"测试配置: {test_args}")

    # 1. 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args, test_args)

    # 2. 加载测试数据集
    raw_test_dataset = load_test_dataset(model_args, data_args)
    processed_test_dataset = preprocess_test_dataset(raw_test_dataset, tokenizer, data_args)

    # 3. 计算Perplexity（CLM核心指标）
    perplexity_metrics = calculate_perplexity(model, tokenizer, processed_test_dataset, data_args, test_args)

    # 4. 批量生成文本并收集参考/生成文本
    generation_results = generate_text_for_metrics(model, tokenizer, raw_test_dataset, test_args)

    # 5. 计算生成文本指标（BLEU1-4/ROUGE-L/METEOR）
    generation_metrics = calculate_generation_metrics(
        references=generation_results["references"],
        candidates=generation_results["candidates"],
        use_chinese_char_split=test_args.use_chinese_char_split
    )

    # 6. 整合所有结果
    test_results = {
        "model_config": model_args.__dict__,
        "test_config": test_args.__dict__,
        "perplexity_metrics": perplexity_metrics,
        "generation_metrics": generation_metrics,  # 新增：BLEU/ROUGE/METEOR
        "generation_examples": generation_results["examples"],
        "total_test_samples": len(raw_test_dataset)
    }

    # 7. 保存结果
    with open(test_args.save_results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    logger.info(f"所有测试结果已保存到: {test_args.save_results_path}")

    # 8. 打印最终汇总
    logger.info("=" * 80)
    logger.info("测试结果汇总")
    logger.info("=" * 80)
    logger.info("【CLM核心指标】")
    logger.info(f"Perplexity: {perplexity_metrics['perplexity']:.4f} | 平均损失: {perplexity_metrics['avg_loss']:.4f}")
    logger.info("\n【生成文本指标】")
    for k, v in generation_metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}")
    logger.info(f"\n生成示例数量: {len(generation_results['examples'])}")
    logger.info(f"测试样本总数: {len(raw_test_dataset)}")


if __name__ == "__main__":
    """
    运行示例（命令行）：
    python test_pt.py \
        --model_name_or_path /home/gyl/project/Medical_Image_Analysis/R2GenCSR/Qwen/Qwen1.5-1.8B-Chat \
        --peft_model_path outputs-pt-qwen-v1/checkpoint-79 \
        --test_file_dir /home/gyl/DataSets/medical/pretrain/test_encyclopedia.json \
        --load_in_4bit True \
        --batch_size 4 \
        --max_new_tokens 256 \
        --num_generate_examples 5 \
        --use_chinese_char_split True
    """
    main()