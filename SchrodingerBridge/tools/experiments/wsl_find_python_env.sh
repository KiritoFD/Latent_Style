#!/usr/bin/env bash
set -euo pipefail

candidate_pythons=(
  "${BASELINE_PYTHON:-}"
  "/root/venvs/samam/bin/python"
  "/home/xy/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python"
  "/usr/bin/python3"
)

check_python() {
  local py="$1"
  [[ -n "$py" ]] || return 1
  [[ -x "$py" ]] || return 1
  "$py" - <<'PY' >/dev/null 2>&1
import sys
try:
    import torch
except Exception:
    raise SystemExit(11)
if not hasattr(torch, "cuda"):
    raise SystemExit(12)
if not torch.cuda.is_available():
    raise SystemExit(13)
raise SystemExit(0)
PY
}

for py in "${candidate_pythons[@]}"; do
  if check_python "$py"; then
    printf '%s\n' "$py"
    exit 0
  fi
done

echo "No usable WSL Python environment with torch.cuda was found." >&2
exit 1
