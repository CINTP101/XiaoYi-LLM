#!/usr/bin/env bash
# Train the isolated, stronger candidate C only from the read-only V5.2 adapter.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
v5_1_dir="${project_dir}/output/tcm-qwen-1.5b-v5-1"
v5_2_dir="${project_dir}/output/tcm-qwen-1.5b-v5-2"
v5_1_baseline="${project_dir}/artifacts/v5_2_pipeline/v5-1_baseline.sha256"
v5_2_baseline="${project_dir}/artifacts/v5_3_pipeline/stage0/v5-2_baseline.sha256"
isolated_dir="${run_dir}/input_clean_only"
input_manifest="${run_dir}/training_input_manifest.json"
candidate_dir="${run_dir}/candidates/hardened_from_v5_2"
candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2"
training_min_free_memory_mib=6000
model_max_length=768
learning_rate=2e-5
epochs=3
gradient_accumulation_steps=8
logging_steps=1
eval_save_steps=35
seed=42

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_HARDENED_CANDIDATE_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing hardened candidate: accepted data and explicit authorization are required." >&2
  exit 2
fi
: "${CLEAN_SOURCE_JSONL:?CLEAN_SOURCE_JSONL is required}"
: "${CLEAN_SOURCE_SHA256:?CLEAN_SOURCE_SHA256 is required}"
if [[ "${CLEAN_SOURCE_JSONL}" != /* || ! -f "${CLEAN_SOURCE_JSONL}" || -L "${CLEAN_SOURCE_JSONL}" || "${CLEAN_SOURCE_JSONL}" != *.jsonl ]]; then
  echo "Refusing hardened candidate: source must be one absolute regular JSONL file." >&2; exit 2
fi
if [[ -e "${candidate_dir}" || -e "${candidate_output}" ]]; then
  echo "Refusing hardened candidate: candidate C artifact or weight directory already exists." >&2; exit 2
fi
if [[ ! -d "${isolated_dir}" || ! -f "${isolated_dir}/clean_train.jsonl" || -L "${isolated_dir}/clean_train.jsonl" || ! -f "${input_manifest}" ]]; then
  echo "Refusing hardened candidate: accepted isolated training input is missing." >&2; exit 2
fi
source_hash="$(sha256sum "${CLEAN_SOURCE_JSONL}" | awk '{print $1}')"
isolated_hash="$(sha256sum "${isolated_dir}/clean_train.jsonl" | awk '{print $1}')"
if [[ "${source_hash}" != "${CLEAN_SOURCE_SHA256,,}" || "${isolated_hash}" != "${CLEAN_SOURCE_SHA256,,}" ]]; then
  echo "Refusing hardened candidate: source or isolated training hash mismatch." >&2; exit 2
fi
if [[ "$(find "${isolated_dir}" -maxdepth 1 -type f -printf . | wc -c)" != "1" ]]; then
  echo "Refusing hardened candidate: isolation directory must contain exactly one file." >&2; exit 2
fi

mkdir -p "${candidate_dir}"
printf '%s\n' "{\"candidate\":\"hardened_from_v5_2\",\"base_adapter\":\"${v5_2_dir}\",\"train_sha256\":\"${isolated_hash}\",\"epochs\":${epochs},\"learning_rate\":${learning_rate},\"batch_size\":1,\"gradient_accumulation_steps\":${gradient_accumulation_steps},\"model_max_length\":${model_max_length},\"seed\":${seed},\"data_seed\":${seed},\"logging_steps\":${logging_steps},\"eval_steps\":${eval_save_steps},\"save_steps\":${eval_save_steps},\"expected_optimizer_steps\":429}" > "${candidate_dir}/hardened_training_plan.json"
printf '%s  %s\n' "${isolated_hash}" "${isolated_dir}/clean_train.jsonl" | tee "${candidate_dir}/isolated_input.sha256"
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
printf '%s  %s\n' "${runtime_hash}" "${runtime_source}" | tee "${candidate_dir}/runtime_wrapper.sha256"
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" --jsonl "${isolated_dir}/clean_train.jsonl" --expected-sha256 "${isolated_hash}" --base-model-path "${base_model_dir}" --runtime-source "${runtime_source}" --expected-runtime-source-sha256 "${runtime_hash}" --output "${candidate_dir}/training_runtime_prompt_parity.json"
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_pretrain_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_pretrain_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_pretrain_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < training_min_free_memory_mib )); then
  echo "Refusing hardened candidate: GPU free memory must be at least ${training_min_free_memory_mib} MiB." >&2; exit 2
fi
set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_sft_v5_3.py" \
  --base-model-path "${base_model_dir}" --peft-path "${v5_2_dir}" \
  --train-jsonl "${isolated_dir}/clean_train.jsonl" --expected-train-sha256 "${isolated_hash}" \
  --runtime-source "${runtime_source}" --expected-runtime-source-sha256 "${runtime_hash}" \
  --output-dir "${candidate_output}" --model-max-length "${model_max_length}" \
  --per-device-train-batch-size 1 --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${gradient_accumulation_steps}" --num-train-epochs "${epochs}" \
  --learning-rate "${learning_rate}" --logging-steps "${logging_steps}" --eval-steps "${eval_save_steps}" --save-steps "${eval_save_steps}" \
  --seed "${seed}" --data-seed "${seed}" 2>&1 | tee "${candidate_dir}/train.log"
training_exit_code="${PIPESTATUS[0]}"
set -e
if (( training_exit_code != 0 )); then echo "Hardened candidate training failed; no checkpoint continuation was issued." >&2; exit "${training_exit_code}"; fi
(
  cd "${candidate_output}"; find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${candidate_dir}/candidate_weights.sha256"
(
  cd "${candidate_output}"; sha256sum -c "${candidate_dir}/candidate_weights.sha256"
) | tee "${candidate_dir}/candidate_weights_hash_check.log"
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_posttrain_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_posttrain_hash_check.log"
echo "Hardened candidate C completed. It never read development or heldout data."
