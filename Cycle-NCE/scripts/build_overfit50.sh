#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC_ROOT="${REPO_ROOT}/../latent-256"
DST_ROOT="${REPO_ROOT}/../latents_overfit50"
PER_STYLE=50
SEED=42
STYLES_CSV=""
CLEAN=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Build overfit-50 subset by sampling latents per style.

Usage:
  bash scripts/build_overfit50.sh [options]

Options:
  --src <dir>         Source root containing style subdirs (default: ../latent-256)
  --dst <dir>         Output root (default: ../latents_overfit50)
  --per-style <int>   Number of samples per style (default: 50)
  --styles <csv>      Comma-separated styles. Empty => auto detect from --src
  --seed <int>        RNG seed for deterministic sampling (default: 42)
  --clean             Remove destination root before writing
  --dry-run           Only print planned operations, do not copy
  -h, --help          Show this help

Examples:
  bash scripts/build_overfit50.sh --styles photo,Hayao --per-style 50 --seed 42 --clean
  bash scripts/build_overfit50.sh --src ../latent-256 --dst ../latents_overfit50 --per-style 50
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_ROOT="$2"
      shift 2
      ;;
    --dst)
      DST_ROOT="$2"
      shift 2
      ;;
    --per-style)
      PER_STYLE="$2"
      shift 2
      ;;
    --styles)
      STYLES_CSV="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "$PER_STYLE" =~ ^[0-9]+$ ]] || [[ "$PER_STYLE" -le 0 ]]; then
  echo "--per-style must be a positive integer, got: ${PER_STYLE}" >&2
  exit 2
fi

if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
  echo "--seed must be an integer, got: ${SEED}" >&2
  exit 2
fi

SRC_ROOT="$(cd "$SRC_ROOT" && pwd)"
mkdir -p "$DST_ROOT"
DST_ROOT="$(cd "$DST_ROOT" && pwd)"

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "Source root not found: $SRC_ROOT" >&2
  exit 1
fi

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[clean] rm -rf $DST_ROOT"
  rm -rf "$DST_ROOT"
  mkdir -p "$DST_ROOT"
fi

if [[ -z "$STYLES_CSV" ]]; then
  mapfile -t STYLES < <(find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
else
  IFS=',' read -r -a STYLES <<< "$STYLES_CSV"
fi

if [[ "${#STYLES[@]}" -eq 0 ]]; then
  echo "No style subdirs found under: $SRC_ROOT" >&2
  exit 1
fi

export OVERFIT_SRC_ROOT="$SRC_ROOT"
export OVERFIT_DST_ROOT="$DST_ROOT"
export OVERFIT_PER_STYLE="$PER_STYLE"
export OVERFIT_SEED="$SEED"
export OVERFIT_DRY_RUN="$DRY_RUN"
export OVERFIT_STYLES_CSV="$(IFS=,; echo "${STYLES[*]}")"

python - <<'PY'
import os
import random
import shutil
import sys
from pathlib import Path

src_root = Path(os.environ["OVERFIT_SRC_ROOT"])
dst_root = Path(os.environ["OVERFIT_DST_ROOT"])
per_style = int(os.environ["OVERFIT_PER_STYLE"])
seed = int(os.environ["OVERFIT_SEED"])
dry_run = int(os.environ["OVERFIT_DRY_RUN"]) == 1
styles_csv = os.environ["OVERFIT_STYLES_CSV"].strip()
styles = [s.strip() for s in styles_csv.split(",") if s.strip()]

rng = random.Random(seed)
all_ext = {".pt", ".npy"}

def list_latents(style_dir: Path):
    files = [p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in all_ext]
    files.sort(key=lambda p: p.name)
    return files

print(f"[info] src={src_root}")
print(f"[info] dst={dst_root}")
print(f"[info] per_style={per_style} seed={seed} styles={styles}")

for style in styles:
    src_style = src_root / style
    dst_style = dst_root / style
    if not src_style.is_dir():
        print(f"[error] style dir missing: {src_style}", file=sys.stderr)
        sys.exit(1)

    files = list_latents(src_style)
    n = len(files)
    if n < per_style:
        print(f"[error] style={style} has only {n} files (< {per_style})", file=sys.stderr)
        sys.exit(1)

    chosen = rng.sample(files, per_style)
    chosen.sort(key=lambda p: p.name)

    print(f"[style] {style}: choose {per_style}/{n}")
    if dry_run:
        continue

    dst_style.mkdir(parents=True, exist_ok=True)
    for src_file in chosen:
        shutil.copy2(src_file, dst_style / src_file.name)

print("[done] overfit subset prepared")
PY
