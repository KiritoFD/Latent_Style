#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/mnt/c/Users/Administrator/phase616_schedule_smoke"
mkdir -p "${OUT_DIR}"

{
  echo "smoke_start=$(date '+%F %T %Z')"
  echo "whoami=$(whoami)"
  echo "pwd=$(pwd)"
  echo "python=$(command -v python || true)"
} >> "${OUT_DIR}/smoke.log"
