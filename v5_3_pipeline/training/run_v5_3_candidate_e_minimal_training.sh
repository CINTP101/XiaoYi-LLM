#!/usr/bin/env bash
# Train Candidate E only after parent acceptance of one exact correction dataset.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
code_dir="${project_dir}/v5_3_pipeline/training"
run_dir="${project_dir}/artifacts/v5_3_pipeline/training"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
base_model_dir="${project_dir}/models/Qwen2.5-1.5B-Instruct"
runtime_source="${project_dir}/tcm_chat_v5.py"
expected_runtime_sha256="ff6e7337ed1c54217cbb153473aa417af6680e8d185db3e5bb8364f125c087fe"
accepted_data_root="${project_dir}/artifacts/v5_3_pipeline/data_candidate_e"
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
input_manifest="${candidate_dir}/candidate_e_training_input_manifest.json"
plan_path="${candidate_dir}/candidate_e_training_plan.json"
training_min_free_memory_mib=6000

if [[ "${CANDIDATE_E_DATA_ACCEPTED:-}" != "YES" || "${GO_E:-}" != "YES" ]]; then
  echo "Refusing Candidate E training: CANDIDATE_E_DATA_ACCEPTED=YES and GO_E=YES are required." >&2
  exit 2
fi
: "${CANDIDATE_E_TRAIN_JSONL:?CANDIDATE_E_TRAIN_JSONL is required}"
: "${CANDIDATE_E_TRAIN_SHA256:?CANDIDATE_E_TRAIN_SHA256 is required}"
if [[ "${CANDIDATE_E_TRAIN_JSONL}" != /* || ! -f "${CANDIDATE_E_TRAIN_JSONL}" || -L "${CANDIDATE_E_TRAIN_JSONL}" || "${CANDIDATE_E_TRAIN_JSONL}" != *.jsonl ]]; then
  echo "Refusing Candidate E training: accepted input must be one absolute regular JSONL file." >&2
  exit 2
fi
if [[ ! "${CANDIDATE_E_TRAIN_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Refusing Candidate E training: accepted SHA-256 must be 64 lowercase hexadecimal characters." >&2
  exit 2
fi
source_path="$(realpath -e -- "${CANDIDATE_E_TRAIN_JSONL}")"
data_root_path="$(realpath -e -- "${accepted_data_root}")"
case "${source_path}" in
  "${data_root_path}"/*) ;;
  *) echo "Refusing Candidate E training: accepted input must remain below ${data_root_path}." >&2; exit 2 ;;
esac
if [[ "$(sha256sum "${source_path}" | awk '{print $1}')" != "${CANDIDATE_E_TRAIN_SHA256}" ]]; then
  echo "Refusing Candidate E training: accepted input SHA-256 differs from GO_E." >&2
  exit 2
fi
if [[ -e "${candidate_dir}" || -e "${candidate_output}" ]]; then
  echo "Refusing Candidate E training: artifact or independent output directory already exists." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" ]]; then
  echo "Refusing Candidate E training: the tcm_llm Python runtime is unavailable." >&2
  exit 2
fi
for required in "${base_model_dir}" "${runtime_source}" "${v5_1_dir}" "${v5_2_dir}" "${candidate_c_dir}" "${candidate_d_dir}" "${v5_1_baseline}" "${v5_2_baseline}" "${candidate_c_baseline}" "${candidate_d_baseline}"; do
  if [[ ! -e "${required}" || -L "${required}" ]]; then
    echo "Refusing Candidate E training: required protected input is missing or symlinked: ${required}" >&2
    exit 2
  fi
done
if [[ "$(wc -l < "${v5_1_baseline}")" != "23" || "$(wc -l < "${v5_2_baseline}")" != "41" || "$(wc -l < "${candidate_c_baseline}")" != "51" || "$(wc -l < "${candidate_d_baseline}")" != "51" ]]; then
  echo "Refusing Candidate E training: a protected manifest has an unexpected entry count." >&2
  exit 2
fi
runtime_hash="$(sha256sum "${runtime_source}" | awk '{print $1}')"
if [[ "${runtime_hash}" != "${expected_runtime_sha256}" ]]; then
  echo "Refusing Candidate E training: frozen runtime wrapper SHA-256 mismatch." >&2
  exit 2
fi
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < training_min_free_memory_mib )); then
  echo "Refusing Candidate E training: GPU free memory must be at least ${training_min_free_memory_mib} MiB; observed ${free_memory_mib:-unknown} MiB." >&2
  exit 2
fi

mkdir -p "${candidate_dir}"
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_pretrain_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_pretrain_hash_check.log"
(
  cd "${candidate_c_dir}"; sha256sum -c "${candidate_c_baseline}"
) | tee "${candidate_dir}/candidate-c_pretrain_hash_check.log"
(
  cd "${candidate_d_dir}"; sha256sum -c "${candidate_d_baseline}"
) | tee "${candidate_dir}/candidate-d_pretrain_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_pretrain_snapshot.csv"

"${runtime_python}" "${code_dir}/prepare_v5_3_candidate_e_training_input.py" \
  --source-train-jsonl "${source_path}" \
  --expected-sha256 "${CANDIDATE_E_TRAIN_SHA256}" \
  --manifest "${input_manifest}" \
  --plan "${plan_path}" | tee "${candidate_dir}/candidate_e_input_preparation.log"
printf '%s  %s\n' "${CANDIDATE_E_TRAIN_SHA256}" "${source_path}" | tee "${candidate_dir}/accepted_training_input.sha256"
printf '%s  %s\n' "${runtime_hash}" "${runtime_source}" | tee "${candidate_dir}/runtime_wrapper.sha256"
"${runtime_python}" "${code_dir}/verify_v5_3_prompt_parity.py" \
  --jsonl "${source_path}" --expected-sha256 "${CANDIDATE_E_TRAIN_SHA256}" \
  --base-model-path "${base_model_dir}" --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${expected_runtime_sha256}" \
  --output "${candidate_dir}/training_runtime_prompt_parity.json"

plan_values="$("${runtime_python}" - "${plan_path}" <<'PY'
import json
import sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1,
    "learning_rate": 3e-6,
    "warmup_ratio": 0.03,
    "model_max_length": 768,
    "seed": 42,
    "data_seed": 42,
    "logging_steps": 1,
    "resume_from_checkpoint": False,
}
for key, value in required.items():
    if plan.get(key) != value:
        raise SystemExit(f"Candidate E plan has unexpected {key}: {plan.get(key)!r}")
if plan.get("training_eval_events_expected", 0) < 4:
    raise SystemExit("Candidate E plan has fewer than four internal evaluations")
print(plan["gradient_accumulation_steps"], plan["num_train_epochs"], plan["learning_rate"], plan["warmup_ratio"], plan["model_max_length"], plan["logging_steps"], plan["eval_steps"], plan["save_steps"])
PY
)"
read -r gradient_accumulation_steps num_train_epochs learning_rate warmup_ratio model_max_length logging_steps eval_steps save_steps <<< "${plan_values}"

set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" "${code_dir}/runtime_sft_v5_3.py" \
  --base-model-path "${base_model_dir}" \
  --peft-path "${candidate_d_dir}" \
  --train-jsonl "${source_path}" \
  --expected-train-sha256 "${CANDIDATE_E_TRAIN_SHA256}" \
  --runtime-source "${runtime_source}" \
  --expected-runtime-source-sha256 "${expected_runtime_sha256}" \
  --output-dir "${candidate_output}" \
  --model-max-length "${model_max_length}" \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps "${gradient_accumulation_steps}" \
  --num-train-epochs "${num_train_epochs}" \
  --learning-rate "${learning_rate}" \
  --warmup-ratio "${warmup_ratio}" \
  --logging-steps "${logging_steps}" \
  --eval-steps "${eval_steps}" \
  --save-steps "${save_steps}" \
  --seed 42 --data-seed 42 --report-to tensorboard 2>&1 | tee "${candidate_dir}/train.log"
training_exit_code="${PIPESTATUS[0]}"
set -e
if (( training_exit_code != 0 )); then
  echo "Candidate E training failed with exit code ${training_exit_code}; no resume or alternate training was issued." >&2
  exit "${training_exit_code}"
fi

"${runtime_python}" "${code_dir}/verify_v5_3_precision_training_evidence.py" \
  --training-output "${candidate_output}" --plan "${plan_path}" \
  --output "${candidate_dir}/training_evidence.json"
(
  cd "${candidate_output}"; find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${candidate_dir}/candidate-e_weights.sha256"
(
  cd "${candidate_output}"; sha256sum -c "${candidate_dir}/candidate-e_weights.sha256"
) | tee "${candidate_dir}/candidate-e_weights_hash_check.log"
(
  cd "${v5_1_dir}"; sha256sum -c "${v5_1_baseline}"
) | tee "${candidate_dir}/v5-1_posttrain_hash_check.log"
(
  cd "${v5_2_dir}"; sha256sum -c "${v5_2_baseline}"
) | tee "${candidate_dir}/v5-2_posttrain_hash_check.log"
(
  cd "${candidate_c_dir}"; sha256sum -c "${candidate_c_baseline}"
) | tee "${candidate_dir}/candidate-c_posttrain_hash_check.log"
(
  cd "${candidate_d_dir}"; sha256sum -c "${candidate_d_baseline}"
) | tee "${candidate_dir}/candidate-d_posttrain_hash_check.log"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader | tee "${candidate_dir}/gpu_posttrain_snapshot.csv"
echo "Candidate E training completed. Development evaluation remains separately gated."
