#!/usr/bin/env bash
set -euo pipefail

readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_CONFIG="configs/sft_text_simplification_203m.yaml"
readonly RESUME_CONFIG="configs/sft_text_simplification_203m_resume.yaml"
readonly BASE_CHECKPOINT="model/murmur_203m_base_weights_only.pt"
readonly LATEST_CHECKPOINT="outputs/text_simplification_203m/latest.pt"
readonly TRAIN_DATA="data/text_simplification/processed/train.jsonl"
readonly VALIDATION_DATA="data/text_simplification/processed/validation.jsonl"

install_dependencies() {
  if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    python -m pip install -r requirements.txt
  fi
}

prepare_data_if_needed() {
  if [[ -f "${TRAIN_DATA}" && -f "${VALIDATION_DATA}" ]]; then
    return
  fi
  if [[ -z "${TEXT_SIMPLIFICATION_DATASET:-}" ]]; then
    echo "Prepared data is missing. Set TEXT_SIMPLIFICATION_DATASET to the source JSON file." >&2
    exit 2
  fi
  python scripts/prepare_text_simplification_data.py \
    --input "${TEXT_SIMPLIFICATION_DATASET}" \
    --output-dir "$(dirname "${TRAIN_DATA}")"
}

fetch_checkpoint_if_needed() {
  if [[ -f "${BASE_CHECKPOINT}" ]]; then
    return
  fi
  if [[ -z "${MODELSCOPE_MODEL_ID:-}" ]]; then
    echo "Base checkpoint is missing. Set MODELSCOPE_MODEL_ID to its ModelScope model ID." >&2
    exit 2
  fi
  python -m pip install "modelscope>=1.18,<2"
  local revision_args=()
  if [[ -n "${MODELSCOPE_MODEL_REVISION:-}" ]]; then
    revision_args=(--revision "${MODELSCOPE_MODEL_REVISION}")
  fi
  python scripts/fetch_modelscope_checkpoint.py \
    --model-id "${MODELSCOPE_MODEL_ID}" \
    "${revision_args[@]}" \
    --output "${BASE_CHECKPOINT}"
}

select_config() {
  if [[ -f "${LATEST_CHECKPOINT}" ]]; then
    python scripts/prepare_resume_config.py \
      --base "${BASE_CONFIG}" \
      --output "${RESUME_CONFIG}" \
      --checkpoint "../${LATEST_CHECKPOINT}"
    printf '%s\n' "${RESUME_CONFIG}"
  else
    printf '%s\n' "${BASE_CONFIG}"
  fi
}

main() {
  cd "${ROOT_DIR}"
  install_dependencies
  prepare_data_if_needed
  fetch_checkpoint_if_needed
  python scripts/validate_text_simplification_setup.py --config "${BASE_CONFIG}"

  local config
  config="$(select_config)"
  echo "Starting text-simplification SFT with ${config}"
  exec python scripts/pretrain.py --config "${config}"
}

main "$@"
