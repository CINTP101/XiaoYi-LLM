#!/usr/bin/env bash
# Complete the human safety audit for one non-blind A/B protocol-development run.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_DEV_AUDIT_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing protocol-development audit: accepted replacement data and explicit authorization are required." >&2
  exit 2
fi
: "${PROTOCOL_DEV_JSONL:?PROTOCOL_DEV_JSONL is required}"
: "${PROTOCOL_DEV_SHA256:?PROTOCOL_DEV_SHA256 is required}"
: "${CANDIDATE_START:?CANDIDATE_START must be v5-1 or v5-2}"
: "${MANUAL_REVIEW:?MANUAL_REVIEW must be the completed review worksheet}"

case "${CANDIDATE_START}" in
  v5-1) candidate_name="from_v5_1" ;;
  v5-2) candidate_name="from_v5_2" ;;
  *) echo "Refusing protocol-development audit: CANDIDATE_START must be v5-1 or v5-2." >&2; exit 2 ;;
esac
candidate_dir="${run_dir}/candidates/${candidate_name}"
evaluation_dir="${candidate_dir}/protocol_dev_eval"
if [[ ! -f "${PROTOCOL_DEV_JSONL}" || -L "${PROTOCOL_DEV_JSONL}" || ! -f "${MANUAL_REVIEW}" || -L "${MANUAL_REVIEW}" || ! -f "${evaluation_dir}/all_results.json" ]]; then
  echo "Refusing protocol-development audit: required regular input, review, or generation artifact is missing." >&2
  exit 2
fi
if [[ -e "${evaluation_dir}/contract_metrics.json" || -e "${evaluation_dir}/contract_cases.json" || -e "${evaluation_dir}/manual_safety_cases.json" ]]; then
  echo "Refusing protocol-development audit: contract output already exists; never overwrite a completed audit." >&2
  exit 2
fi
actual_hash="$(sha256sum "${PROTOCOL_DEV_JSONL}" | awk '{print $1}')"
if [[ "${actual_hash}" != "${PROTOCOL_DEV_SHA256,,}" ]]; then
  echo "Refusing protocol-development audit: development-set SHA-256 mismatch." >&2
  exit 2
fi

"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" \
  --blind-test-jsonl "${PROTOCOL_DEV_JSONL}" \
  --expected-blind-sha256 "${actual_hash}" \
  --generation-results "${evaluation_dir}/all_results.json" \
  --output-dir "${evaluation_dir}" \
  --minimum-samples 50 \
  --manual-review "${MANUAL_REVIEW}"

echo "Protocol-development audit completed. This result may inform A/B selection only; it does not consume the final blind test."
