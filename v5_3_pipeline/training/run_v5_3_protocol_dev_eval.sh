#!/usr/bin/env bash
# Evaluate one internal V5.3 A/B candidate on Luna's non-blind protocol development set.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
evaluation_min_free_memory_mib=5800
model_max_length=768
minimum_protocol_dev_samples=50

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_DEV_EVAL_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing protocol-development evaluation: accepted data and explicit authorization are required." >&2
  exit 2
fi
: "${PROTOCOL_DEV_JSONL:?PROTOCOL_DEV_JSONL is required}"
: "${PROTOCOL_DEV_SHA256:?PROTOCOL_DEV_SHA256 is required}"
: "${CANDIDATE_START:?CANDIDATE_START must be v5-1 or v5-2}"
if [[ "${PROTOCOL_DEV_JSONL}" != /* || ! -f "${PROTOCOL_DEV_JSONL}" || -L "${PROTOCOL_DEV_JSONL}" || "${PROTOCOL_DEV_JSONL}" != *.jsonl ]]; then
  echo "Refusing protocol-development evaluation: PROTOCOL_DEV_JSONL must be one absolute regular JSONL file." >&2
  exit 2
fi

case "${CANDIDATE_START}" in
  v5-1)
    candidate_name="from_v5_1"
    candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-from-v5-1"
    ;;
  v5-2)
    candidate_name="from_v5_2"
    candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-from-v5-2"
    ;;
  *)
    echo "Refusing protocol-development evaluation: CANDIDATE_START must be v5-1 or v5-2." >&2
    exit 2
    ;;
esac
candidate_dir="${run_dir}/candidates/${candidate_name}"
evaluation_dir="${candidate_dir}/protocol_dev_eval"
review_template="${candidate_dir}/protocol_dev_manual_review_template.json"
if [[ ! -d "${candidate_output}" || ! -d "${candidate_dir}" || -e "${evaluation_dir}" || -e "${review_template}" ]]; then
  echo "Refusing protocol-development evaluation: candidate or fresh-artifact checks failed." >&2
  exit 2
fi
actual_hash="$(sha256sum "${PROTOCOL_DEV_JSONL}" | awk '{print $1}')"
if [[ "${actual_hash}" != "${PROTOCOL_DEV_SHA256,,}" ]]; then
  echo "Refusing protocol-development evaluation: SHA-256 mismatch." >&2
  exit 2
fi
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_predev_eval_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < evaluation_min_free_memory_mib )); then
  echo "Refusing protocol-development evaluation: GPU free memory must be at least ${evaluation_min_free_memory_mib} MiB." >&2
  exit 2
fi

"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${PROTOCOL_DEV_JSONL}" \
  --expected-sha256 "${actual_hash}" \
  --base-model-path "${base_model_dir}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output "${candidate_dir}/protocol_dev_runtime_prompt_parity.json"

CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_evaluate_v5_3.py" \
  --base-model-path "${base_model_dir}" \
  --peft-path "${candidate_output}" \
  --test-jsonl "${PROTOCOL_DEV_JSONL}" \
  --expected-test-sha256 "${actual_hash}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output-dir "${evaluation_dir}" \
  --max-input-length "${model_max_length}" \
  --max-new-tokens 256 \
  --batch-size 1 \
  --require-v5-3-contract 2>&1 | tee "${candidate_dir}/protocol_dev_eval.log"

"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" \
  --blind-test-jsonl "${PROTOCOL_DEV_JSONL}" \
  --expected-blind-sha256 "${actual_hash}" \
  --generation-results "${evaluation_dir}/all_results.json" \
  --output-dir "${evaluation_dir}" \
  --minimum-samples "${minimum_protocol_dev_samples}" \
  --review-template-output "${review_template}" \
  --template-only

echo "Protocol-development generation completed. A human must complete ${review_template} before candidate selection. This script never reads the final blind-test set."
