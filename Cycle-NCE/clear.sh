#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="/tmp/latent_style_clear.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another clear.sh is running (lock: ${LOCK_FILE})."
  exit 1
fi

MODE="move"
DO_KILL=0
DRY_RUN=0
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${REPO_ROOT}/_archive/${STAMP}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--move|--delete] [--kill] [--dry-run]

Default behavior:
  --move    Move experiment artifacts to _archive/<timestamp>

Options:
  --delete  Permanently delete matched artifacts
  --kill    Kill likely leftover training/eval python processes first
  --dry-run Print actions only, no filesystem changes
EOF
}

for arg in "$@"; do
  case "$arg" in
    --move) MODE="move" ;;
    --delete) MODE="delete" ;;
    --kill) DO_KILL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

kill_leftovers() {
  local patterns=(
    "python .*run.py --config"
    "python .*run_small_experiment.py"
    "python .*run_evaluation.py"
    "python3.12 .*run.py --config"
    "python3.12 .*run_small_experiment.py"
    "python3.12 .*run_evaluation.py"
  )
  for p in "${patterns[@]}"; do
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "[dry-run] pkill -f \"$p\""
    else
      pkill -f "$p" >/dev/null 2>&1 || true
    fi
  done
}

TARGETS=(
  "small-exp-*"
  "overfit50*"
  "adacut_overfit*"
  "50-probability*"
  "cycle-classify"
  "logs"
  "eval_cache"
  "torch_compile_cache"
  "_tmp_util_probe*"
  "__pycache__"
)

move_target() {
  local path="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] mv \"$path\" \"$ARCHIVE_DIR/\""
    return
  fi
  mkdir -p "${ARCHIVE_DIR}"
  if ! mv "$path" "${ARCHIVE_DIR}/"; then
    echo "[warn] failed to move: $path" >&2
    return 1
  fi
}

delete_target() {
  local path="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] rm -rf \"$path\""
    return
  fi
  if ! rm -rf "$path"; then
    echo "[warn] failed to delete: $path" >&2
    return 1
  fi
}

if [[ "${DO_KILL}" -eq 1 ]]; then
  echo "Killing possible leftover training/eval processes..."
  kill_leftovers
fi

echo "clear.sh mode=${MODE} dry_run=${DRY_RUN}"
shopt -s nullglob
cd "${REPO_ROOT}"

for pat in "${TARGETS[@]}"; do
  matches=( $pat )
  for item in "${matches[@]}"; do
    [[ -e "$item" ]] || continue
    if [[ "${MODE}" == "delete" ]]; then
      echo "[delete] $item"
      delete_target "$item" || true
    else
      echo "[move] $item"
      move_target "$item" || true
    fi
  done
done

if [[ "${MODE}" == "move" && "${DRY_RUN}" -eq 0 ]]; then
  echo "Archived artifacts to: ${ARCHIVE_DIR}"
fi
echo "Done."
