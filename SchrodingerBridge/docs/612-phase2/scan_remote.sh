#!/bin/bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge/exp
for d in */; do
    dd=${d%/}
    summary="${dd}/full_eval/epoch_0001/summary.json"
    if [ -f "$summary" ]; then
        echo "$dd"
    fi
done
