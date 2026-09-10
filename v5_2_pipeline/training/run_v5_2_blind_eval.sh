#!/usr/bin/env bash
# Runs V5.2 blind evaluation only after training has completed successfully.
set -euo pipefail

project_dir="/home/cyh/Medical_Qwen"
runtime_python="/home/cyh/miniconda3/envs/tcm_llm/bin/python"
run_dir="${project_dir}/artifacts/v5_2_pipeline/training"
v5_1_dir="${project_dir}/output/tcm-qwen-1.5b-v5-1"
v5_2_dir="${project_dir}/output/tcm-qwen-1.5b-v5-2"
evaluation_dir="${run_dir}/blind_eval"

if [[ "${LUNA_ACCEPTED:-}" != "YES" ]]; then
  echo "Refusing to evaluate: set LUNA_ACCEPTED=YES only after the parent agent accepts Luna's data." >&2
  exit 2
fi

: "${BLIND_TEST_JSONL:?BLIND_TEST_JSONL is required}"
: "${BLIND_TEST_SHA256:?BLIND_TEST_SHA256 is required}"
if [[ "${BLIND_TEST_JSONL}" != /* || ! -f "${BLIND_TEST_JSONL}" || -L "${BLIND_TEST_JSONL}" || "${BLIND_TEST_JSONL}" != *.jsonl ]]; then
  echo "Refusing to evaluate: BLIND_TEST_JSONL must be one absolute regular .jsonl file." >&2
  exit 2
fi
if [[ ! -x "${runtime_python}" || ! -d "${v5_2_dir}" || -e "${evaluation_dir}" ]]; then
  echo "Refusing to evaluate: runtime, V5.2 output, or fresh blind-evaluation directory check failed." >&2
  exit 2
fi

actual_blind_hash="$(sha256sum "${BLIND_TEST_JSONL}" | awk '{print $1}')"
expected_blind_hash="$(printf '%s' "${BLIND_TEST_SHA256}" | tr '[:upper:]' '[:lower:]')"
if [[ "${actual_blind_hash}" != "${expected_blind_hash}" ]]; then
  echo "Refusing to evaluate: blind-test SHA-256 mismatch." >&2
  exit 2
fi

"${runtime_python}" - <<'PY'
import rouge_score
print("rouge-score=available")
PY

(
  cd "${v5_1_dir}"
  sha256sum -c ../../artifacts/v5_2_pipeline/v5-1_baseline.sha256
) | tee "${run_dir}/v5-1_preeval_hash_check.log"

nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader \
  | tee "${run_dir}/gpu_preeval_snapshot.csv"
free_memory_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk 'NR==1 {gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print $0}')"
if ! [[ "${free_memory_mib}" =~ ^[0-9]+$ ]] || (( free_memory_mib < 6000 )); then
  echo "Refusing to evaluate: GPU free memory must be at least 6000 MiB; observed ${free_memory_mib:-unknown} MiB." >&2
  exit 2
fi

cd "${project_dir}"
set +e
CUDA_VISIBLE_DEVICES=0 "${runtime_python}" evaluate_sft_qwen.py \
  --base_model_path ./models/Qwen2.5-1.5B-Instruct \
  --peft_model_path ./output/tcm-qwen-1.5b-v5-2 \
  --test_data_path "${BLIND_TEST_JSONL}" \
  --template_name qwen \
  --disable_thinking \
  --torch_dtype bfloat16 \
  --device_map auto \
  --max_input_length 512 \
  --max_new_tokens 256 \
  --batch_size 1 \
  --repetition_penalty 1.0 \
  --output_dir "${evaluation_dir}" \
  2>&1 | tee "${run_dir}/blind_eval.log"
evaluation_exit_code="${PIPESTATUS[0]}"
set -e
if (( evaluation_exit_code != 0 )); then
  echo "Blind evaluation failed with exit code ${evaluation_exit_code}." >&2
  exit "${evaluation_exit_code}"
fi

"${runtime_python}" "${run_dir}/evaluate_v5_2_safety_structure.py" \
  --blind-test-jsonl "${BLIND_TEST_JSONL}" \
  --generation-results "${evaluation_dir}/all_results.json" \
  --output-dir "${evaluation_dir}" \
  | tee "${run_dir}/safety_structure_eval.log"

sha256sum "${BLIND_TEST_JSONL}" | tee "${evaluation_dir}/blind_test_input.sha256"
(
  cd "${evaluation_dir}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "${run_dir}/blind_eval_artifacts.sha256"

(
  cd "${v5_1_dir}"
  sha256sum -c ../../artifacts/v5_2_pipeline/v5-1_baseline.sha256
) | tee "${run_dir}/v5-1_posteval_hash_check.log"

echo "Blind evaluation completed with deterministic greedy decoding and dynamic action/stage taxonomy."
