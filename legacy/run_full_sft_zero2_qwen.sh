#!/usr/bin/env bash
set -euo pipefail

export MASTER_PORT="${MASTER_PORT:-29518}"
export HF_HOME="${HF_HOME:-/tmp/medical_qwen_hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/medical_qwen_hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/medical_qwen_transformers_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port "${MASTER_PORT}" --nproc_per_node 8 supervised_finetuning.py \
    --model_name_or_path /home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/models/models/Qwen/Qwen3___5-4B \
    --train_file_dir /home/notebook/data/group/guoyulong/code/image_enhance/vlm-prx/SuperResolution_train_prx/andes_vl/DataSets/huatuo_medical_qa_sharegpt_jsonl \
    --template_name qwen \
    --disable_thinking True \
    --use_peft False \
    --do_train \
    --do_eval \
    --max_train_samples -1 \
    --max_eval_samples 200 \
    --max_test_samples 200 \
    --validation_split_percentage 1 \
    --test_split_percentage 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --model_max_length 4096 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 100 \
    --save_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model eval_bleu4 \
    --greater_is_better True \
    --early_stopping_patience 6 \
    --early_stopping_threshold 0.0 \
    --save_total_limit 5 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-full-sft-qwen4b-zero2 \
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
    --deepspeed ./legacy/configs/zero2.json
