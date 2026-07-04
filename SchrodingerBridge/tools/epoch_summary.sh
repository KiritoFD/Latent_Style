#!/bin/bash
# Extract epoch summaries
set -euo pipefail
LOG="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge/620_nswd_gate03_smoke/train.log"
grep "Epoch.*loss=" "$LOG"