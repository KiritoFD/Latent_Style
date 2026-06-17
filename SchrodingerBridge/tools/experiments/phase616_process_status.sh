#!/usr/bin/env bash
set -euo pipefail

ps -ef | grep -E 'src/run.py|launch_all.sh|phase616_auto.py|h0_vertical_fm' | grep -v grep || true
