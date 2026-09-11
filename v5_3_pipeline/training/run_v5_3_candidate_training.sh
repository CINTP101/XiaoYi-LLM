#!/usr/bin/env bash
# Train one internal V5.3 A/B candidate. It never reads a dev or blind-test file.
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
plan_path="${run_dir}/training_plan.json"
training_min_free_memory_mib=6000
model_max_length=768

if [[ "${LUNA_V5_3_ACCEPTED:-}" != "YES" || "${V5_3_CANDIDATE_AUTHORIZED:-}" != "YES" ]]; then
  echo "Refusing candidate training: accepted data and explicit candidate authorization are both required." >&2
  exit 2
fi
: "${CLEAN_SOURCE_JSONL:?CLEAN_SOURCE_JSONL is required}"
: "${CLEAN_SOURCE_SHA256:?CLEAN_SOURCE_SHA256 is required}"
: "${CANDIDATE_START:?CANDIDATE_START must be v5-1 or v5-2}"
if [[ "${CLEAN_SOURCE_JSONL}" != /* || ! -f "${CLEAN_SOURCE_JSONL}" || -L "${CLEAN_SOURCE_JSONL}" || "${CLEAN_SOURCE_JSONL}" != *.jsonl ]]; then
  echo "Refusing candidate training: CLEAN_SOURCE_JSONL must be one absolute regular JSONL file." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" || ! -f "${runtime_source}" ]]; then
  echo "Refusing candidate training: tcm_llm runtime or runtime wrapper is unavailable." >&2
  exit 2
fi

case "${CANDIDATE_START}" in
  v5-1)
    peft_dir="${v5_1_dir}"
    candidate_name="from_v5_1"
    candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-from-v5-1"
    ;;
  v5-2)
    peft_dir="${v5_2_dir}"
    candidate_name="from_v5_2"
    candidate_output="${project_dir}/output/tcm-qwen-1.5b-v5-3-candidate-from-v5-2"
    ;;
  *)
    echo "Refusing candidate training: CANDIDATE_START must be v5-1 or v5-2." >&2
    exit 2
    ;;
esac
candidate_dir="${run_dir}/candidates/${candidate_name}"
if [[ -e "${candidate_output}" || -e "${candidate_dir}" ]]; then
  echo "Refusing candidate training: candidate output or artifact directory already exists." >&2
  exit 2
fi

mkdir -p "${candidate_dir}"
if [[ ! -e "${isolated_dir}" && ! -e "${input_manifest}" && ! -e "${plan_path}" ]]; then
  "${runtime_python}" "${code_dir}/prepare_v5_3_training_input.py" \
    --source-clean-jsonl "${CLEAN_SOURCE_JSONL}" \
    --expected-sha256 "${CLEAN_SOURCE_SHA256}" \
    --isolated-dir "${isolated_dir}" \
    --manifest "${input_manifest}" \
    --plan "${plan_path}" | tee "${candidate_dir}/training_input_preparation.log"
elif [[ -d "${isolated_dir}" && -f "${input_manifest}" && -f "${plan_path}" ]]; then
  "${runtime_python}" - "${input_manifest}" "${CLEAN_SOURCE_SHA256}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("source_sha256") != sys.argv[2].lower():
    raise SystemExit("existing isolated input source hash differs from this authorized clean source")
PY
else
  echo "Refusing candidate training: shared input-isolation artifacts are partial or inconsistent." >&2
  exit 2
fi

input_file_count="$(find "${isolated_dir}" -maxdepth 1 -type f -printf '.' | wc -c)"
if [[ "${input_file_count}" != "1" || ! -f "${isolated_dir}/clean_train.jsonl" || -L "${isolated_dir}/clean_train.jsonl" ]]; then
  echo "Refusing candidate training: isolated input directory must contain exactly clean_train.jsonl." >&2
  exit 2
fi
isolated_hash="$(sha256sum "${isolated_dir}/clean_train.jsonl" | awk '{print $1}')"
if [[ "${isolated_hash}" != "${CLEAN_SOURCE_SHA256,,}" ]]; then
  echo "Refusing candidate training: isolated clean input hash does not match accepted hash." >&2
  exit 2
fi
printf '%s  %s\n' "${isolated_hash}" "${isolated_dir}/clean_train.jsonl" | tee "${candidate_dir}/isolated_input.sha256"

runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
printf '%s  %s\n' "${runtime_hash}" "${runtime_source}" | tee "${candidate_dir}/runtime_wrapper.sha256"
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${isolated_dir}/clean_train.jsonl" \
  --expected-sha256 "${isolated_hash}" \
  --base-model-path "${base_model_dir}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output "${candidate_dir}/training_runtime_prompt_parity.json"

(
  cd "${v5_1_dir}"
  sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_pretrain_hash_check.log"
(
  cd "${v5_2_dir}"
  sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_pretrain_hash_check.log"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_pretrain_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < training_min_free_memory_mib )); then
  echo "Refusing candidate training: GPU free memory must be at least ${training_min_free_memory_mib} MiB; observed ${free_memory_mib:-unknown} MiB." >&2
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
"${runtime_python}" - <<'PY' | tee "${candidate_dir}/tcm_llm_runtime.json"
import importlib.metadata as metadata
import json
import sys
import torch
result = {"python": sys.version.replace("\\n", " "), "executable": sys.executable}
for package in ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "tensorboard"):
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

set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_sft_v5_3.py" \
  --base-model-path "${base_model_dir}" \
  --peft-path "${peft_dir}" \
  --train-jsonl "${isolated_dir}/clean_train.jsonl" \
  --expected-train-sha256 "${isolated_hash}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${runtime_hash}" \
  --output-dir "${candidate_output}" \
  --model-max-length "${model_max_length}" \
  --gradient-accumulation-steps "${gradient_accumulation_steps}" \
  --num-train-epochs "${num_train_epochs}" \
  --logging-steps "${logging_steps}" \
  --eval-steps "${eval_steps}" \
  --save-steps "${save_steps}" \
  2>&1 | tee "${candidate_dir}/train.log"
training_exit_code="${PIPESTATUS[0]}"
set -e
if (( training_exit_code != 0 )); then
  echo "Candidate training failed with exit code ${training_exit_code}; no interrupted-run continuation was issued." >&2
  exit "${training_exit_code}"
fi

(
  cd "${candidate_output}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${candidate_dir}/candidate_weights.sha256"
(
  cd "${candidate_output}"
  sha256sum -c "${candidate_dir}/candidate_weights.sha256"
) | tee "${candidate_dir}/candidate_weights_hash_check.log"
(
  cd "${v5_1_dir}"
  sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_posttrain_hash_check.log"
(
  cd "${v5_2_dir}"
  sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_posttrain_hash_check.log"

echo "Internal candidate ${candidate_name} completed. It has not read the protocol dev or final blind-test set."
