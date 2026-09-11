#!/usr/bin/env bash
# One-time generation on the frozen final blind set after the final V5.3 model is frozen.
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
final_training_dir="${run_dir}/final_training"
evaluation_dir="${run_dir}/final_blind_eval"
evaluation_min_free_memory_mib=5800
model_max_length=768

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_FINAL_BLIND_EVAL_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing final blind evaluation: accepted replacement data and explicit one-time authorization are required." >&2
  exit 2
fi
: "${FINAL_BLIND_JSONL:?FINAL_BLIND_JSONL is required}"
: "${FINAL_BLIND_SHA256:?FINAL_BLIND_SHA256 is required}"
: "${CANDIDATE_SELECTION_REPORT:?CANDIDATE_SELECTION_REPORT is required}"
if [[ ! -f "${FINAL_BLIND_JSONL}" || -L "${FINAL_BLIND_JSONL}" || "${FINAL_BLIND_JSONL}" != *.jsonl || ! -f "${CANDIDATE_SELECTION_REPORT}" || -L "${CANDIDATE_SELECTION_REPORT}" ]]; then
  echo "Refusing final blind evaluation: blind input and selection report must be regular files." >&2
  exit 2
fi
if [[ ! -d "${v5_3_dir}" || ! -f "${final_training_dir}/v5-3_weights.sha256" || -e "${evaluation_dir}" ]]; then
  echo "Refusing final blind evaluation: frozen final weights, their manifest, and a fresh evaluation directory are required." >&2
  exit 2
fi
"${runtime_python}" - "${CANDIDATE_SELECTION_REPORT}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("selection_status") != "PASS" or report.get("selected_start") not in {"v5-1", "v5-2"}:
    raise SystemExit("selection report must declare selection_status=PASS and selected_start=v5-1|v5-2")
PY
actual_hash="$(sha256sum "${FINAL_BLIND_JSONL}" | awk '{print $1}')"
if [[ "${actual_hash}" != "${FINAL_BLIND_SHA256,,}" ]]; then
  echo "Refusing final blind evaluation: frozen blind-set SHA-256 mismatch." >&2
  exit 2
fi
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
mkdir -p "${evaluation_dir}"
printf '%s  %s\n' "${actual_hash}" "${FINAL_BLIND_JSONL}" | tee "${evaluation_dir}/final_blind_input.sha256"
printf '%s  %s\n' "${runtime_hash}" "${runtime_source}" | tee "${evaluation_dir}/runtime_wrapper.sha256"
(
  cd "${v5_3_dir}"
  sha256sum -c "${final_training_dir}/v5-3_weights.sha256"
) | tee "${evaluation_dir}/v5-3_preblind_weights_hash_check.log"
(
  cd "${v5_1_dir}"
  sha256sum -c "${v5_1_baseline}"
) | tee "${evaluation_dir}/v5-1_preblind_hash_check.log"
(
  cd "${v5_2_dir}"
  sha256sum -c "${v5_2_baseline}"
) | tee "${evaluation_dir}/v5-2_preblind_hash_check.log"

"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${FINAL_BLIND_JSONL}" \
  --expected-sha256 "${actual_hash}" \
  --base-model-path "${base_model_dir}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output "${evaluation_dir}/final_blind_runtime_prompt_parity.json"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${evaluation_dir}/gpu_preblind_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < evaluation_min_free_memory_mib )); then
  echo "Refusing final blind evaluation: GPU free memory must be at least ${evaluation_min_free_memory_mib} MiB." >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_evaluate_v5_3.py" \
  --base-model-path "${base_model_dir}" \
  --peft-path "${v5_3_dir}" \
  --test-jsonl "${FINAL_BLIND_JSONL}" \
  --expected-test-sha256 "${actual_hash}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output-dir "${evaluation_dir}/generation" \
  --max-input-length "${model_max_length}" \
  --max-new-tokens 256 \
  --batch-size 1 \
  --require-v5-3-contract 2>&1 | tee "${evaluation_dir}/generation.log"
"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" \
  --blind-test-jsonl "${FINAL_BLIND_JSONL}" \
  --expected-blind-sha256 "${actual_hash}" \
  --generation-results "${evaluation_dir}/generation/all_results.json" \
  --output-dir "${evaluation_dir}/generation" \
  --minimum-samples 100 \
  --review-template-output "${evaluation_dir}/manual_safety_review_template.json" \
  --template-only
echo "Final blind generation completed once. A human must complete ${evaluation_dir}/manual_safety_review_template.json before the release gate can run."
