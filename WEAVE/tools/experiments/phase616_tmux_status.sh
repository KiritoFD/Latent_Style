#!/usr/bin/env bash
set -euo pipefail

export TERM=xterm

tmux list-sessions
tmux list-panes -t phase616_h0_resume -F '#{pane_id} #{pane_current_command} #{pane_dead} #{pane_pid}'
