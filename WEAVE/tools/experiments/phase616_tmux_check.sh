#!/usr/bin/env bash
set -euo pipefail

if command -v tmux >/dev/null 2>&1; then
  command -v tmux
  tmux -V
else
  echo "NO_TMUX"
fi
