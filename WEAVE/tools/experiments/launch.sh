#!/usr/bin/env bash
# 用法: bash launch.sh <exp_dir>
DIR="${1:?Usage: bash launch.sh <exp_dir>}"
CFG="${DIR}/config.json"

[ -f "$CFG" ] || { echo "No config.json in $DIR"; exit 1; }

cd /mnt/i/Github/Latent_Style/SchrodingerBridge

python src/run.py --config "$CFG" > "${DIR}/run.log" 2>&1 &
disown
echo "$(date) $(basename $DIR) PID=$!" | tee -a /tmp/launch_616.log
