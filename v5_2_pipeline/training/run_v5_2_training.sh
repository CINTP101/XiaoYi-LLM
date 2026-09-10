#!/usr/bin/env bash
# Runs only after the parent agent has accepted Luna's clean JSONL and hash.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
run_dir="${project_dir}/artifacts/v5_2_pipeline/training"
v5_1_dir="${project_dir}/output/tcm-qwen-1.5b-v5-1"
v5_2_dir="${project_dir}/output/tcm-qwen-1.5b-v5-2"
isolated_input_dir="${run_dir}/input_clean_only"
input_manifest="${run_dir}/training_input_manifest.json"
plan_path="${run_dir}/training_plan.json"
execution_report="${run_dir}/training_execution_parameters.md"

if [[ "${LUNA_ACCEPTED:-}" != "YES" ]]; then
  echo "Refusing to train: set LUNA_ACCEPTED=YES only after the parent agent accepts Luna's data." >&2
  exit 2
fi

: "${CLEAN_SOURCE_JSONL:?CLEAN_SOURCE_JSONL is required}"
: "${CLEAN_SOURCE_SHA256:?CLEAN_SOURCE_SHA256 is required}"

if [[ "${CLEAN_SOURCE_JSONL}" != /* ]]; then
  echo "Refusing to train: CLEAN_SOURCE_JSONL must be an absolute file path." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" ]]; then
  echo "Refusing to train: tcm_llm Python is unavailable at ${runtime_python}." >&2
  exit 2
fi
if [[ -e "${v5_2_dir}" || -e "${isolated_input_dir}" || -e "${input_manifest}" || -e "${plan_path}" || -e "${execution_report}" ]]; then
  echo "Refusing to train: a V5.2 output or input-isolation artifact already exists." >&2
  exit 2
fi

"${runtime_python}" - <<'PY'
import matplotlib
import rouge_score
print(f"matplotlib={matplotlib.__version__}")
print("rouge-score=available")
PY

"${runtime_python}" "${run_dir}/prepare_v5_2_training_input.py" \
  --source-clean-jsonl "${CLEAN_SOURCE_JSONL}" \
  --expected-sha256 "${CLEAN_SOURCE_SHA256}" \
  --isolated-dir "${isolated_input_dir}" \
  --manifest "${input_manifest}" \
  --plan "${plan_path}" \
  | tee "${run_dir}/training_input_preparation.log"

find "${isolated_input_dir}" -maxdepth 1 -mindepth 1 -printf '%f\t%s bytes\n' \
  | sort | tee "${run_dir}/training_input_files.txt"
input_file_count="$(find "${isolated_input_dir}" -maxdepth 1 -type f -printf '.' | wc -c)"
if [[ "${input_file_count}" != "1" || ! -f "${isolated_input_dir}/clean_train.jsonl" || -L "${isolated_input_dir}/clean_train.jsonl" ]]; then
  echo "Refusing to train: isolated input directory must contain exactly one regular clean_train.jsonl." >&2
  exit 2
fi
sha256sum "${isolated_input_dir}/clean_train.jsonl" | tee "${run_dir}/training_input.sha256"

read -r gradient_accumulation_steps num_train_epochs logging_steps eval_steps save_steps < <(
  "${runtime_python}" -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); print(p["gradient_accumulation_steps"], p["num_train_epochs"], p["logging_steps"], p["eval_steps"], p["save_steps"])' "${plan_path}"
)

"${runtime_python}" "${run_dir}/write_v5_2_execution_report.py" \
  --input-manifest "${input_manifest}" \
  --output "${execution_report}"

(
  cd "${v5_1_dir}"
  sha256sum -c ../../artifacts/v5_2_pipeline/v5-1_baseline.sha256
) | tee "${run_dir}/v5-1_pretrain_hash_check.log"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader \
  | tee "${run_dir}/gpu_pretrain_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < 6000 )); then
  echo "Refusing to train: GPU free memory must be at least 6000 MiB; observed ${free_memory_mib:-unknown} MiB." >&2
  exit 2
fi

"${runtime_python}" - <<'PY' | tee "${run_dir}/tcm_llm_runtime.json"
import importlib.metadata as metadata
import json
import sys
import torch
packages = ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "tensorboard", "matplotlib", "rouge-score"]
result = {"python": sys.version.replace("\n", " "), "executable": sys.executable}
for package in packages:
    try:
        result[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        result[package] = "NOT_INSTALLED"
result["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    result["gpu"] = torch.cuda.get_device_name(0)
    result["cuda_runtime"] = torch.version.cuda
print(json.dumps(result, ensure_ascii=False, indent=2))
PY

cd "${project_dir}"
set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" supervised_finetuning.py \
  --model_name_or_path ./models/Qwen2.5-1.5B-Instruct \
  --peft_path ./output/tcm-qwen-1.5b-v5-1 \
  --train_file_dir "${isolated_input_dir}" \
  --output_dir "${v5_2_dir}" \
  --do_train \
  --do_eval \
  --template_name qwen \
  --disable_thinking True \
  --use_peft True \
  --model_max_length 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${gradient_accumulation_steps}" \
  --num_train_epochs "${num_train_epochs}" \
  --learning_rate 1e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --optim adamw_torch_fused \
  --torch_dtype bfloat16 \
  --bf16 True \
  --gradient_checkpointing True \
  --logging_strategy steps \
  --logging_steps "${logging_steps}" \
  --logging_first_step True \
  --eval_strategy steps \
  --eval_steps "${eval_steps}" \
  --save_strategy steps \
  --save_steps "${save_steps}" \
  --save_total_limit 2 \
  --validation_split_percentage 5 \
  --test_split_percentage 0 \
  --seed 42 \
  --data_seed 42 \
  --device_map auto \
  --flash_attn False \
  --report_to tensorboard \
  --cache_dir /tmp/medical_qwen_v5_2_cache \
  2>&1 | tee "${run_dir}/train_v5_2.log"
training_exit_code="${PIPESTATUS[0]}"
set -e
if (( training_exit_code != 0 )); then
  echo "V5.2 training failed with exit code ${training_exit_code}; no resume command has been issued." >&2
  exit "${training_exit_code}"
fi

for filename in train_curve.csv eval_curve.csv training_curves.png train_results.json eval_results.json test_results.json all_results.json trainer_state.json; do
  if [[ -f "${v5_2_dir}/${filename}" ]]; then
    cp "${v5_2_dir}/${filename}" "${run_dir}/${filename}"
  fi
done
find "${v5_2_dir}" -maxdepth 1 -type f -name 'eval_samples_*.json' -exec cp {} "${run_dir}/" \;

(
  cd "${v5_2_dir}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${run_dir}/v5-2_weights.sha256"

(
  cd "${v5_1_dir}"
  sha256sum -c ../../artifacts/v5_2_pipeline/v5-1_baseline.sha256
) | tee "${run_dir}/v5-1_posttrain_hash_check.log"

echo "V5.2 training completed. Blind-test evaluation is intentionally a separate command."
