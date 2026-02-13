#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 ]]; then
  echo "style_ablation.sh always runs full pipeline (mode=all)." >&2
  echo "Use scripts/style_ablation.py for custom options." >&2
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/style_ablation.py" --mode all
