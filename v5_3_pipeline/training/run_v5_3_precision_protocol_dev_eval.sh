#!/usr/bin/env bash
# Generate candidate D outputs only on the existing non-final development set.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
expected_runtime_sha256="ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
development_jsonl="${project_dir}/artifacts/v5_3_pipeline/data/protocol_dev_v5_3.jsonl"
development_sha256="e382d7e641e0bbb626d4303cfeefd1259c9a6a7ef9883d81ec5298fb4f4e2edf"
candidate_dir="${run_dir}/candidates/precision_from_c"
candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-precision-from-c"
evaluation_dir="${candidate_dir}/protocol_dev_eval"
review_template="${candidate_dir}/protocol_dev_manual_review_template.json"
evaluation_min_free_memory_mib=5800

if [[ "${GO_D:-}" != "YES" || "${GO_D_DEV_EVAL:-}" != "YES" ]]; then
  echo "Refusing candidate D development evaluation: GO_D=YES and GO_D_DEV_EVAL=YES are required." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" || ! -d "${base_model_dir}" || ! -f "${runtime_source}" || ! -f "${development_jsonl}" || -L "${development_jsonl}" || ! -d "${candidate_output}" || -L "${candidate_output}" || ! -f "${candidate_dir}/candidate-d_weights.sha256" || -e "${evaluation_dir}" || -e "${review_template}" ]]; then
  echo "Refusing candidate D development evaluation: protected input or fresh output check failed." >&2
  exit 2
fi
actual_dev_hash="$(sha256sum "${development_jsonl}" | awk '{print $1}')"
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
if [[ "${actual_dev_hash}" != "${development_sha256}" || "${runtime_hash}" != "${expected_runtime_sha256}" ]]; then
  echo "Refusing candidate D development evaluation: development or runtime SHA-256 mismatch." >&2
  exit 2
fi
(
  cd "${candidate_output}"
  sha256sum -c "${candidate_dir}/candidate-d_weights.sha256"
) | tee "${candidate_dir}/candidate-d_predev_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_predev_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < evaluation_min_free_memory_mib )); then
  echo "Refusing candidate D development evaluation: GPU free memory must be at least ${evaluation_min_free_memory_mib} MiB." >&2
  exit 2
fi
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${development_jsonl}" --expected-sha256 "${development_sha256}" \
  --base-model-path "${base_model_dir}" --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${expected_runtime_sha256}" \
  --output "${candidate_dir}/protocol_dev_runtime_prompt_parity.json"
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_evaluate_v5_3.py" \
  --base-model-path "${base_model_dir}" --peft-path "${candidate_output}" \
  --test-jsonl "${development_jsonl}" --expected-test-sha256 "${development_sha256}" \
  --runtime-source "${runtime_source}" --expected-runtime-source-sha256 "${expected_runtime_sha256}" \
  --output-dir "${evaluation_dir}" --max-input-length 768 --max-new-tokens 256 --batch-size 1 \
  --require-v5-3-contract 2>&1 | tee "${candidate_dir}/protocol_dev_eval.log"
"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" \
  --blind-test-jsonl "${development_jsonl}" --expected-blind-sha256 "${development_sha256}" \
  --generation-results "${evaluation_dir}/all_results.json" --output-dir "${evaluation_dir}" \
  --minimum-samples 180 --review-template-output "${review_template}" --template-only
echo "Candidate D development generation completed. A bound 180-row manual safety review is required before a gate decision."
