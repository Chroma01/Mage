#!/usr/bin/env bash
# Overlay Mage-VL-Cowork agent, guest bundle, and runner into an OSWorld-Verified checkout and launch evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-microsoft/Mage-VL-Cowork}"
OSWORLD_ROOT="${OSWORLD_ROOT:-}"
SGLANG_BASE_URL="${SGLANG_BASE_URL:-http://127.0.0.1:30001}"
NUM_ENVS="${NUM_ENVS:-4}"
RESULT_DIR="${RESULT_DIR:-./results}"

if [[ -z "${OSWORLD_ROOT}" || ! -d "${OSWORLD_ROOT}/desktop_env" ]]; then
  echo "Usage: OSWORLD_ROOT=/path/to/OSWorld [MODEL_DIR=/path/to/Mage-VL-Cowork] bash ${BASH_SOURCE[0]} [extra args...]" >&2
  exit 1
fi

cp -f "${EVAL_BUNDLE_DIR}/desktop_env/pydeps_bundle.py" "${OSWORLD_ROOT}/desktop_env/pydeps_bundle.py"
cp -f "${EVAL_BUNDLE_DIR}/mm_agents/mage_gui_agent.py" "${OSWORLD_ROOT}/mm_agents/mage_gui_agent.py"
cp -f "${EVAL_BUNDLE_DIR}/mm_agents/mage_hybrid_agent.py" "${OSWORLD_ROOT}/mm_agents/mage_hybrid_agent.py"
mkdir -p "${OSWORLD_ROOT}/run" "${OSWORLD_ROOT}/evaluation_examples"
cp -f "${EVAL_BUNDLE_DIR}/run/run_multienv_mage_cowork.py" "${OSWORLD_ROOT}/run/run_multienv_mage_cowork.py"
cp -f "${EVAL_BUNDLE_DIR}/evaluation_examples/test_nogdrive.json" "${OSWORLD_ROOT}/evaluation_examples/test_nogdrive.json"

cd "${OSWORLD_ROOT}"
exec python3 run/run_multienv_mage_cowork.py \
  --headless \
  --provider_name docker \
  --model Mage-VL-Cowork \
  --processor_model_name "${MODEL_DIR}" \
  --api_backend sglang_generate \
  --base_url "${SGLANG_BASE_URL}" \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --max_steps 50 \
  --history_n 3 \
  --images_to_keep 3 \
  --temperature 0.7 \
  --top_p 0.8 \
  --max_tokens 4096 \
  --sleep_after_execution 3.0 \
  --num_envs "${NUM_ENVS}" \
  --result_dir "${RESULT_DIR}" \
  "$@"
