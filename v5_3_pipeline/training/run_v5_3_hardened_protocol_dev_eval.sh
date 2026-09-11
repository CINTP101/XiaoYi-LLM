#!/usr/bin/env bash
# Evaluate hardened candidate C once on the accepted non-blind development set.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
candidate_dir="${run_dir}/candidates/hardened_from_v5_2"
candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2"
evaluation_dir="${candidate_dir}/protocol_dev_eval"
review_template="${candidate_dir}/protocol_dev_manual_review_template.json"
evaluation_min_free_memory_mib=5800
model_max_length=768

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_HARDENED_DEV_EVAL_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing hardened development evaluation: accepted data and explicit authorization are required." >&2; exit 2
fi
: "${PROTOCOL_DEV_JSONL:?PROTOCOL_DEV_JSONL is required}"
: "${PROTOCOL_DEV_SHA256:?PROTOCOL_DEV_SHA256 is required}"
if [[ ! -d "${candidate_dir}" || ! -d "${candidate_output}" || -e "${evaluation_dir}" || -e "${review_template}" || ! -f "${PROTOCOL_DEV_JSONL}" || -L "${PROTOCOL_DEV_JSONL}" ]]; then
  echo "Refusing hardened development evaluation: candidate, fresh output, or regular development input check failed." >&2; exit 2
fi
actual_hash="$(sha256sum "${PROTOCOL_DEV_JSONL}" | awk '{print $1}')"
if [[ "${actual_hash}" != "${PROTOCOL_DEV_SHA256,,}" ]]; then echo "Refusing hardened development evaluation: SHA-256 mismatch." >&2; exit 2; fi
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_prehardened_dev_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < evaluation_min_free_memory_mib )); then echo "Refusing hardened development evaluation: GPU free memory must be at least ${evaluation_min_free_memory_mib} MiB." >&2; exit 2; fi
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" --jsonl "${PROTOCOL_DEV_JSONL}" --expected-sha256 "${actual_hash}" --base-model-path "${base_model_dir}" --runtime-source "${runtime_source}" --expected-runtime-source-sha256 "${runtime_hash}" --output "${candidate_dir}/protocol_dev_runtime_prompt_parity.json"
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_evaluate_v5_3.py" \
  --base-model-path "${base_model_dir}" --peft-path "${candidate_output}" --test-jsonl "${PROTOCOL_DEV_JSONL}" --expected-test-sha256 "${actual_hash}" \
  --runtime-source "${runtime_source}" --expected-runtime-source-sha256 "${runtime_hash}" --output-dir "${evaluation_dir}" \
  --max-input-length "${model_max_length}" --max-new-tokens 256 --batch-size 1 --require-v5-3-contract 2>&1 | tee "${candidate_dir}/protocol_dev_eval.log"
"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" --blind-test-jsonl "${PROTOCOL_DEV_JSONL}" --expected-blind-sha256 "${actual_hash}" --generation-results "${evaluation_dir}/all_results.json" --output-dir "${evaluation_dir}" --minimum-samples 50 --review-template-output "${review_template}" --template-only
echo "Hardened development generation completed. Complete its bound 180-row safety review before any comparison."
