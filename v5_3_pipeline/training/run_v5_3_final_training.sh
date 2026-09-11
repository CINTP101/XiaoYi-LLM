#!/usr/bin/env bash
# Train the selected V5.3 candidate start into the only final V5.3 output directory.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
v5_1_dir="${project_dir}/output/tcm-qwen-1.5b-v5-1"
v5_2_dir="${project_dir}/output/tcm-qwen-1.5b-v5-2"
v5_3_dir="${project_dir}/output/tcm-qwen-1.5b-v5-3"
v5_1_baseline="${project_dir}/artifacts/v5_2_pipeline/v5-1_baseline.sha256"
v5_2_baseline="${project_dir}/artifacts/v5_3_pipeline/stage0/v5-2_baseline.sha256"
isolated_dir="${run_dir}/input_clean_only"
input_manifest="${run_dir}/training_input_manifest.json"
plan_path="${run_dir}/training_plan.json"
final_dir="${run_dir}/final_training"
training_min_free_memory_mib=6000
model_max_length=768

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_FINAL_TRAINING_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing final V5.3 training: accepted data and explicit final-training authorization are required." >&2
  exit 2
fi
: "${SELECTED_START:?SELECTED_START must be v5-1 or v5-2}"
: "${CANDIDATE_SELECTION_REPORT:?CANDIDATE_SELECTION_REPORT is required}"
if [[ ! -f "${CANDIDATE_SELECTION_REPORT}" || -L "${CANDIDATE_SELECTION_REPORT}" ]]; then
  echo "Refusing final V5.3 training: selection report must be a regular file." >&2
  exit 2
fi
if [[ -e "${v5_3_dir}" || -e "${final_dir}" ]]; then
  echo "Refusing final V5.3 training: final output or final artifact directory already exists." >&2
  exit 2
fi
if [[ ! -d "${isolated_dir}" || ! -f "${isolated_dir}/clean_train.jsonl" || -L "${isolated_dir}/clean_train.jsonl" || ! -f "${input_manifest}" || ! -f "${plan_path}" ]]; then
  echo "Refusing final V5.3 training: accepted one-file isolation and training plan must already exist from the A/B preparation stage." >&2
  exit 2
fi
input_file_count="$(find "${isolated_dir}" -maxdepth 1 -type f -printf '.' | wc -c)"
if [[ "${input_file_count}" != "1" ]]; then
  echo "Refusing final V5.3 training: isolation directory contains more than one file." >&2
  exit 2
fi

case "${SELECTED_START}" in
  v5-1) peft_dir="${v5_1_dir}" ;;
  v5-2) peft_dir="${v5_2_dir}" ;;
  *) echo "Refusing final V5.3 training: SELECTED_START must be v5-1 or v5-2." >&2; exit 2 ;;
esac

mkdir -p "${final_dir}"
sha256sum "${isolated_dir}/clean_train.jsonl" | tee "${final_dir}/isolated_input.sha256"
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
printf '%s  %s\n' "${runtime_hash}" "${runtime_source}" | tee "${final_dir}/runtime_wrapper.sha256"
input_hash="$(awk '{print $1}' "${final_dir}/isolated_input.sha256")"
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${isolated_dir}/clean_train.jsonl" \
  --expected-sha256 "${input_hash}" \
  --base-model-path "${base_model_dir}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output "${final_dir}/training_runtime_prompt_parity.json"

(
  cd "${v5_1_dir}"
  sha256sum -c "${v5_1_baseline}"
) | tee "${final_dir}/v5-1_pretrain_hash_check.log"
(
  cd "${v5_2_dir}"
  sha256sum -c "${v5_2_baseline}"
) | tee "${final_dir}/v5-2_pretrain_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${final_dir}/gpu_pretrain_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < training_min_free_memory_mib )); then
  echo "Refusing final V5.3 training: GPU free memory must be at least ${training_min_free_memory_mib} MiB." >&2
  exit 2
fi
read -r gradient_accumulation_steps num_train_epochs logging_steps eval_steps save_steps < <(
  "${runtime_python}" - "${plan_path}" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
print(plan["gradient_accumulation_steps"], plan["num_train_epochs"], plan["logging_steps"], plan["eval_steps"], plan["save_steps"])
PY
)
set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_sft_v5_3.py" \
  --base-model-path "${base_model_dir}" \
  --peft-path "${peft_dir}" \
  --train-jsonl "${isolated_dir}/clean_train.jsonl" \
  --expected-train-sha256 "${input_hash}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output-dir "${v5_3_dir}" \
  --model-max-length "${model_max_length}" \
  --gradient-accumulation-steps "${gradient_accumulation_steps}" \
  --num-train-epochs "${num_train_epochs}" \
  --logging-steps "${logging_steps}" \
  --eval-steps "${eval_steps}" \
  --save-steps "${save_steps}" \
  2>&1 | tee "${final_dir}/train.log"
training_exit_code="${PIPESTATUS[0]}"
set -e
if (( training_exit_code != 0 )); then
  echo "Final V5.3 training failed with exit code ${training_exit_code}; no interrupted-run continuation was issued." >&2
  exit "${training_exit_code}"
fi
(
  cd "${v5_3_dir}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${final_dir}/v5-3_weights.sha256"
(
  cd "${v5_3_dir}"
  sha256sum -c "${final_dir}/v5-3_weights.sha256"
) | tee "${final_dir}/v5-3_weights_hash_check.log"
(
  cd "${v5_1_dir}"
  sha256sum -c "${v5_1_baseline}"
) | tee "${final_dir}/v5-1_posttrain_hash_check.log"
(
  cd "${v5_2_dir}"
  sha256sum -c "${v5_2_baseline}"
) | tee "${final_dir}/v5-2_posttrain_hash_check.log"
echo "Final V5.3 training completed. The final blind-test script is a separate one-time operation."
