#!/usr/bin/env bash
# Generate Candidate E outputs only on the frozen 180-row development set.
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
v5_1_dir="${project_dir}/output/tcm-qwen-1.5b-v5-1"
v5_2_dir="${project_dir}/output/tcm-qwen-1.5b-v5-2"
candidate_c_dir="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-hardened-from-v5-2"
candidate_d_dir="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-precision-from-c"
v5_1_baseline="${project_dir}/artifacts/v5_2_pipeline/v5-1_baseline.sha256"
v5_2_baseline="${project_dir}/artifacts/v5_3_pipeline/stage0/v5-2_baseline.sha256"
candidate_c_baseline="${run_dir}/candidates/hardened_from_v5_2/candidate_weights.sha256"
candidate_d_baseline="${run_dir}/candidates/precision_from_c/candidate-d_weights.sha256"
candidate_dir="${run_dir}/candidates/minimal_correction_from_d"
candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-e-minimal-from-d"
candidate_e_baseline="${candidate_dir}/candidate-e_weights.sha256"
evaluation_dir="${candidate_dir}/protocol_dev_eval"
review_template="${candidate_dir}/protocol_dev_manual_review_template.json"
evaluation_min_free_memory_mib=5800

if [[ "${GO_E:-}" != "YES" || "${GO_E_DEV_EVAL:-}" != "YES" ]]; then
  echo "Refusing Candidate E development evaluation: GO_E=YES and GO_E_DEV_EVAL=YES are required." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" ]]; then
  echo "Refusing Candidate E development evaluation: the tcm_llm Python runtime is unavailable." >&2
  exit 2
fi
for required in "${base_model_dir}" "${runtime_source}" "${development_jsonl}" "${v5_1_dir}" "${v5_2_dir}" "${candidate_c_dir}" "${candidate_d_dir}" "${candidate_output}" "${v5_1_baseline}" "${v5_2_baseline}" "${candidate_c_baseline}" "${candidate_d_baseline}" "${candidate_e_baseline}"; do
  if [[ ! -e "${required}" || -L "${required}" ]]; then
    echo "Refusing Candidate E development evaluation: required input is missing or symlinked: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${evaluation_dir}" || -e "${review_template}" || -e "${candidate_dir}/protocol_dev_runtime_prompt_parity.json" ]]; then
  echo "Refusing Candidate E development evaluation: fresh output checks failed." >&2
  exit 2
fi
if [[ "$(sha256sum "${development_jsonl}" | awk '{print $1}')" != "${development_sha256}" || "$(sha256sum "${runtime_source}" | awk '{print $1}')" != "${expected_runtime_sha256}" ]]; then
  echo "Refusing Candidate E development evaluation: development or runtime SHA-256 mismatch." >&2
  exit 2
fi
if [[ "$(wc -l < "${v5_1_baseline}")" != "23" || "$(wc -l < "${v5_2_baseline}")" != "41" || "$(wc -l < "${candidate_c_baseline}")" != "51" || "$(wc -l < "${candidate_d_baseline}")" != "51" ]]; then
  echo "Refusing Candidate E development evaluation: a protected manifest has an unexpected entry count." >&2
  exit 2
fi
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_predev_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_predev_hash_check.log"
(
  cd "${candidate_c_dir}"; sha256sum -c "${candidate_c_baseline}"
) | tee "${candidate_dir}/candidate-c_predev_hash_check.log"
(
  cd "${candidate_d_dir}"; sha256sum -c "${candidate_d_baseline}"
) | tee "${candidate_dir}/candidate-d_predev_hash_check.log"
(
  cd "${candidate_output}"; sha256sum -c "${candidate_e_baseline}"
) | tee "${candidate_dir}/candidate-e_predev_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_predev_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < evaluation_min_free_memory_mib )); then
  echo "Refusing Candidate E development evaluation: GPU free memory must be at least ${evaluation_min_free_memory_mib} MiB." >&2
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
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) > "${candidate_dir}/v5-1_postdev_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) > "${candidate_dir}/v5-2_postdev_hash_check.log"
(
  cd "${candidate_c_dir}"; sha256sum -c "${candidate_c_baseline}"
) > "${candidate_dir}/candidate-c_postdev_hash_check.log"
(
  cd "${candidate_d_dir}"; sha256sum -c "${candidate_d_baseline}"
) > "${candidate_dir}/candidate-d_postdev_hash_check.log"
(
  cd "${candidate_output}"; sha256sum -c "${candidate_e_baseline}"
) > "${candidate_dir}/candidate-e_postdev_hash_check.log"
echo "Candidate E development generation completed. A bound 180-row manual audit is required before any next decision."
