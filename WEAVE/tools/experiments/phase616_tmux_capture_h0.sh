#!/usr/bin/env bash
set -euo pipefail

export TERM=xterm

tmux capture-pane -p -t phase616_h0_resume | tail -80
