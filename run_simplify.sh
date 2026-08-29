#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "用法: ./run_simplify.sh '需要简化的中文文本'" >&2
  exit 2
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT_DIR/scripts/simplify_text.py" \
  --config "$ROOT_DIR/configs/inference_text_simplification_203m.yaml" \
  --text "$1" \
  "${@:2}"
