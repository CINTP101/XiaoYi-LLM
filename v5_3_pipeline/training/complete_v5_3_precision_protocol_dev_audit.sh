#!/usr/bin/env bash
# Close the candidate D development gate only with a completed bound manual review.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
development_jsonl="${project_dir}/artifacts/v5_3_pipeline/data/protocol_dev_v5_3.jsonl"
development_sha256="e382d7e641e0bbb626d4303cfeefd1259c9a6a7ef9883d81ec5298fb4f4e2edf"
candidate_dir="${run_dir}/candidates/precision_from_c"
evaluation_dir="${candidate_dir}/protocol_dev_eval"

if [[ "${GO_D:-}" != "YES" || "${GO_D_DEV_AUDIT:-}" != "YES" ]]; then
  echo "Refusing candidate D development audit: GO_D=YES and GO_D_DEV_AUDIT=YES are required." >&2
  exit 2
fi
: "${PRECISION_COMPLETED_MANUAL_REVIEW:?PRECISION_COMPLETED_MANUAL_REVIEW is required}"
if [[ ! -f "${PRECISION_COMPLETED_MANUAL_REVIEW}" || -L "${PRECISION_COMPLETED_MANUAL_REVIEW}" || ! -f "${development_jsonl}" || -L "${development_jsonl}" || ! -f "${evaluation_dir}/all_results.json" || -e "${evaluation_dir}/contract_metrics.json" || -e "${evaluation_dir}/contract_cases.json" || -e "${evaluation_dir}/manual_safety_cases.json" ]]; then
  echo "Refusing candidate D development audit: completed review or fresh audit output check failed." >&2
  exit 2
fi
review_path="$(realpath -e -- "${PRECISION_COMPLETED_MANUAL_REVIEW}")"
candidate_root="$(realpath -e -- "${candidate_dir}")"
case "${review_path}" in
  "${candidate_root}"/*) ;;
  *) echo "Refusing candidate D development audit: manual review must be stored under the candidate artifact directory." >&2; exit 2 ;;
esac
actual_dev_hash="$(sha256sum "${development_jsonl}" | awk '{print $1}')"
if [[ "${actual_dev_hash}" != "${development_sha256}" ]]; then
  echo "Refusing candidate D development audit: frozen development SHA-256 mismatch." >&2
  exit 2
fi
"${runtime_python}" "${code_dir}/evaluate_v5_3_contract.py" \
  --blind-test-jsonl "${development_jsonl}" --expected-blind-sha256 "${development_sha256}" \
  --generation-results "${evaluation_dir}/all_results.json" --output-dir "${evaluation_dir}" \
  --minimum-samples 180 --manual-review "${review_path}" | tee "${candidate_dir}/protocol_dev_contract_audit.log"
"${runtime_python}" - "${evaluation_dir}/contract_metrics.json" <<'PY' | tee "${candidate_dir}/protocol_dev_audit_status.json"
import json
import sys
metrics = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "development_gate_status": metrics["contract_gate_status"],
    "next_action": "Only a PASS permits a parent decision about formal V5.3; this script never creates formal weights or reads final evaluation data.",
}, ensure_ascii=False, indent=2))
PY
