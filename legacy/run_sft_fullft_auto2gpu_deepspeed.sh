#!/usr/bin/env bash
set -euo pipefail

# 正式训练（全参微调 + DeepSpeed ZeRO-2）
# - 每次验证后会在 output_dir 更新 train_curve.csv、eval_curve.csv、training_curves.png（见 supervised_finetuning.py 中 SaveTrainingCurvesOnEvaluateCallback）
# - 自动检测 GPU 数量（当前默认最多用 2 张卡）
# - 降低 model_max_length 以缓解 OOM（默认 384）
# - 默认 deepspeed：ds_zero2_bf16.json（不把 optimizer offload 到 CPU，避免触发 CPUAdam 编译）
#   若系统 CUDA 工具链与 torch 的 CUDA 版本不一致，CPU offload 会报 CUDAMismatchException。
#   仍需要 CPU offload 时：安装与 torch 匹配的 CUDA toolkit，或设置
#   DS_CONFIG=$(pwd)/ds_zero2_cpu_offload_bf16.json

export MASTER_PORT="${MASTER_PORT:-29517}"
export HF_HOME="${HF_HOME:-/tmp/medical_qwen_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/medical_qwen_hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/medical_qwen_transformers_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

cd "$(dirname "$0")/.."

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

TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-200000}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-384}"
DS_CONFIG="${DS_CONFIG:-$(pwd)/legacy/configs/ds_zero2_bf16.json}"
# 全参 + ZeRO-2 每次 save 体积大；save_total_limit=999 极易把盘写满（OSError 28）。load_best_model_at_end 仍会保留最优 checkpoint。
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-4}"

echo "Using TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"
echo "Using MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH}"
echo "Using DS_CONFIG=${DS_CONFIG}"
echo "Using SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT}"

# 从仅含权重的 checkpoint 继续训时，checkpoint 目录往往没有 tokenizer；与评测脚本一致指向基座
TOKENIZER_NAME_OR_PATH="${TOKENIZER_NAME_OR_PATH:-/home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/models/models/Qwen/Qwen3___5-4B}"
echo "Using TOKENIZER_NAME_OR_PATH=${TOKENIZER_NAME_OR_PATH}"

torchrun --master_port "${MASTER_PORT}" --nproc_per_node "${NUM_GPUS}" supervised_finetuning.py \
  --deepspeed "${DS_CONFIG}" \
  --model_name_or_path /home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/Medical_Qwen/outputs-sft-fullft-qwen3-5-4b-generate-eval-ds-continue-training-2/checkpoint-3400 \
  --tokenizer_name_or_path "${TOKENIZER_NAME_OR_PATH}" \
  --train_file_dir /home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/DataSets/huatuo_medical_qa_sharegpt_jsonl \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --do_train \
  --do_eval \
  --template_name qwen \
  --disable_thinking True \
  --use_peft False \
  --max_train_samples 200000 \
  --max_eval_samples 200 \
  --max_test_samples 200 \
  --validation_split_percentage 1 \
  --test_split_percentage 1 \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --num_train_epochs 50 \
  --max_steps "${TRAIN_MAX_STEPS}" \
  --learning_rate 2e-5 \
  --warmup_ratio 0.05 \
  --weight_decay 0.05 \
  --logging_strategy steps \
  --logging_steps 50 \
  --eval_steps 200 \
  --eval_strategy steps \
  --save_steps 200 \
  --save_strategy steps \
  --load_best_model_at_end True \
  --metric_for_best_model eval_bleu1 \
  --greater_is_better True \
  --early_stopping_patience 100 \
  --early_stopping_threshold 0.0 \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 4 \
  --output_dir outputs-sft-fullft-qwen3-5-4b-generate-eval-ds-continue-training-2 \
  --overwrite_output_dir \
  --ddp_timeout 30000 \
  --logging_first_step True \
  --torch_dtype bfloat16 \
  --bf16 \
  --report_to tensorboard \
  --optim adamw_torch \
  --ddp_find_unused_parameters False \
  --gradient_checkpointing True \
  --cache_dir /tmp/medical_qwen_cache \
  --flash_attn False \
  --predict_with_generate True \
  --generation_max_length 512 \
  --generation_num_beams 1

