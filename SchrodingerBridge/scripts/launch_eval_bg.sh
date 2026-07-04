#!/usr/bin/env bash
# Launches evaluation in background and exits
nohup bash /mnt/c/Users/Administrator/run_eval_adain_wct_256.sh > /mnt/c/Users/Administrator/eval_adain_wct_256.log 2>&1 &
echo "PID=$!"
echo "Log: /mnt/c/Users/Administrator/eval_adain_wct_256.log"
