#!/usr/bin/env bash
set -euo pipefail

# 医学问答向增量训练（默认 SFT：仅对 Response 段算 loss；需整条 CLM 时设 TRAIN_MODE=clm）
# - 数据：去重 + 清洗（redacted_thinking、过长答案截断、简单复读压缩）；生成评测固定种子子集 + 停止串截断
# - 环境与 run_sft_fullft_auto2gpu_deepspeed.sh 对齐：HF_HOME / datasets / transformers 缓存默认走 /tmp
# - DeepSpeed ZeRO-2（bf16），torchrun 多卡
# - 跑满 num_train_epochs；WARMUP_STEPS 优先于长 warmup_ratio，避免只训几百 step 时有效 lr 过小
# - LoRA 试验示例：USE_PEFT=True LEARNING_RATE=5e-6 LORA_RANK=16 bash run_pt_incremental_deepspeed.sh（并确认 pretraining 支持对应参数）
# - 环境变量：MODEL_NAME_OR_PATH、TRAIN_FILE_DIR、OUTPUT_DIR、WARMUP_STEPS、USE_PEFT、LEARNING_RATE 等

export MASTER_PORT="${MASTER_PORT:-29518}"
export HF_HOME="${HF_HOME:-/tmp/medical_qwen_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/medical_qwen_hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/medical_qwen_transformers_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

cd "$(dirname "$0")/.."
MEDICAL_QWEN_ROOT="$(pwd)"

detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    python - <<'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
print(len([x for x in v.split(",") if x.strip()!=""]) if v else 0)
PY
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -L | wc -l
    else
      python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
    fi
  fi
}

NUM_GPUS="$(detect_gpus)"
if [[ "${NUM_GPUS}" -le 0 ]]; then
  echo "No GPU detected. Please set CUDA_VISIBLE_DEVICES or ensure CUDA is available." 1>&2
  exit 1
fi
if [[ "${NUM_GPUS}" -gt 2 ]]; then
  NUM_GPUS=2
fi
echo "Using NUM_GPUS=${NUM_GPUS}"

DS_CONFIG="${DS_CONFIG:-${MEDICAL_QWEN_ROOT}/legacy/configs/ds_zero2_bf16.json}"
CACHE_DIR="${CACHE_DIR:-/tmp/medical_qwen_pt_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${MEDICAL_QWEN_ROOT}/outputs-pt-incremental-qwen35-4b-medical-medicalds}"
# 含 train_*.json / valid_*.json 的目录（JSONL 行、Alpaca 字段）
_DEFAULT_DS="/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/DataSets/medical/pretrain"
if [[ -d "${_DEFAULT_DS}" ]]; then
  _TRAIN_DEFAULT="${_DEFAULT_DS}"
else
  _TRAIN_DEFAULT="${MEDICAL_QWEN_ROOT}/data/finetune"
fi
TRAIN_FILE_DIR="${TRAIN_FILE_DIR:-${_TRAIN_DEFAULT}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/models/models/Qwen/Qwen3___5-4B}"
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-${MODEL_NAME_OR_PATH}}"

BLOCK_SIZE="${BLOCK_SIZE:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"
TRAIN_MODE="${TRAIN_MODE:-sft}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
USE_PEFT="${USE_PEFT:-False}"
LORA_RANK="${LORA_RANK:-16}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-500}"
EVAL_GEN_MAX_SAMPLES="${EVAL_GEN_MAX_SAMPLES:-64}"

echo "Using DS_CONFIG=${DS_CONFIG}"
echo "Using CACHE_DIR=${CACHE_DIR}"
echo "Using OUTPUT_DIR=${OUTPUT_DIR}"
echo "Using TRAIN_FILE_DIR=${TRAIN_FILE_DIR}"
echo "Using MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "Using TRAIN_MODE=${TRAIN_MODE} WARMUP_STEPS=${WARMUP_STEPS} USE_PEFT=${USE_PEFT}"

torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NUM_GPUS}" legacy/pretraining.py \
  --deepspeed "${DS_CONFIG}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
  --train_file_dir "${TRAIN_FILE_DIR}" \
  --train_mode "${TRAIN_MODE}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size 2 \
  --do_train \
  --do_eval \
  --use_peft "${USE_PEFT}" \
  --lora_rank "${LORA_RANK}" \
  --seed 42 \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --validation_split_percentage 5 \
  --eval_generation_metrics True \
  --eval_gen_max_samples "${EVAL_GEN_MAX_SAMPLES}" \
  --eval_gen_max_new_tokens 192 \
  --eval_gen_stop_strings "### Instruction:,### Input:" \
  --eval_gen_repetition_penalty 1.12 \
  --eval_gen_subset_seed 42 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --warmup_ratio 0.0 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 10 \
  --eval_steps 50 \
  --eval_strategy steps \
  --save_steps 50 \
  --save_strategy steps \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --preprocessing_num_workers 4 \
  --block_size "${BLOCK_SIZE}" \
  --clm_group_by_length false \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --optim adamw_torch \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --cache_dir "${CACHE_DIR}"
