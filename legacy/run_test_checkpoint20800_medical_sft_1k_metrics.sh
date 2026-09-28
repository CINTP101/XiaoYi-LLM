#!/usr/bin/env bash
# 使用指定全参 checkpoint-20800，在 medical_sft_1K_format.jsonl 上评测：
# BLEU-1~4 / ROUGE-L / BERTScore(P/R/F1)
#
# 依赖：
#   pip install nltk rouge-score bert-score
# （本项目 evaluate_sft_qwen.py 默认按中文“逐字”计算 BLEU/ROUGE）
#
# 若环境无法直连 Hugging Face 下载 BERTScore 模型，可用镜像：
#   export HF_ENDPOINT="https://hf-mirror.com"
#
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export HF_HOME="${HF_HOME:-/tmp/medical_qwen_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/medical_qwen_hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/medical_qwen_transformers_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

CHECKPOINT="/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/Medical_Qwen/outputs-sft-fullft-qwen3-5-4b-generate-eval-ds-continue-training-2/checkpoint-20800"
# checkpoint 通常不含 tokenizer 文件，必须从基座模型目录加载 tokenizer
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/models/models/Qwen/Qwen3___5-4B}"
TEST_JSONL="/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/Medical_Qwen/data/finetune/medical_sft_1K_format.jsonl"
OUT_DIR="$(pwd)/test/eval_medical_sft_1k_ckpt_20800_metrics"

# BERTScore 模型：可用本地目录替换（离线环境）
BERTSCORE_MODEL_TYPE="${BERTSCORE_MODEL_TYPE:-bert-base-chinese}"

# 只评测前 N 条（默认 128；设为 -1/空表示全量）
MAX_SAMPLES="${MAX_SAMPLES:-128}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "错误：checkpoint 不存在: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TEST_JSONL}" ]]; then
  echo "错误：测试集不存在: ${TEST_JSONL}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[Eval] disable_thinking=True  max_samples=${MAX_SAMPLES}  out=${OUT_DIR}" >&2
python -u evaluate_sft_qwen.py \
  --base_model_path "${CHECKPOINT}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --test_data_path "${TEST_JSONL}" \
  --template_name qwen \
  --disable_thinking \
  --max_samples "${MAX_SAMPLES}" \
  --max_input_length 4096 \
  --max_new_tokens 512 \
  --batch_size "${BATCH_SIZE:-128}" \
  --torch_dtype bfloat16 \
  --device_map auto \
  --output_dir "${OUT_DIR}" \
  --enable_bertscore \
  --bertscore_model_type "${BERTSCORE_MODEL_TYPE}" \
  --bertscore_device "${BERTSCORE_DEVICE:-}" \
  --print_every_batch \
  --show_examples_per_batch 1

OUT_DIR_THINK="${OUT_DIR}_think"
mkdir -p "${OUT_DIR_THINK}"
echo "[Eval] disable_thinking=False (thinking enabled)  max_samples=${MAX_SAMPLES}  out=${OUT_DIR_THINK}" >&2
python -u evaluate_sft_qwen.py \
  --base_model_path "${CHECKPOINT}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --test_data_path "${TEST_JSONL}" \
  --template_name qwen \
  --no-disable_thinking \
  --save_think_traces \
  --max_samples "${MAX_SAMPLES}" \
  --max_input_length 4096 \
  --max_new_tokens 512 \
  --batch_size "${BATCH_SIZE:-128}" \
  --torch_dtype bfloat16 \
  --device_map auto \
  --output_dir "${OUT_DIR_THINK}" \
  --enable_bertscore \
  --bertscore_model_type "${BERTSCORE_MODEL_TYPE}" \
  --bertscore_device "${BERTSCORE_DEVICE:-}" \
  --print_every_batch \
  --show_examples_per_batch 1

echo "评测完成，输出目录: ${OUT_DIR} 与 ${OUT_DIR_THINK}"

