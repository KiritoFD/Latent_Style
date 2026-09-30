#!/usr/bin/env bash
set -euo pipefail
mkdir -p ~/Latent_Style/SchrodingerBridge_phase616
tar -xf /mnt/c/Users/Administrator/phase616_patch1.tar -C ~/Latent_Style/SchrodingerBridge_phase616
chmod +x ~/Latent_Style/SchrodingerBridge_phase616/tools/experiments/run_phase616_ot_vertical_round1.sh
cd ~/Latent_Style/SchrodingerBridge_phase616
python3 -m py_compile src/config_schema.py src/losses.py src/trainer.py
echo PATCH_SYNC_OK