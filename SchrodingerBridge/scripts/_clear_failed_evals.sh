#!/usr/bin/env bash
set -uo pipefail
cp /mnt/c/Users/Administrator/ablation_batch_eval.sh /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
cp /mnt/c/Users/Administrator/ablation_eval_override.json /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/
chmod +x /mnt/i/Github/Latent_Style/SchrodingerBridge/scripts/ablation_batch_eval.sh
echo COPIED

# Clear all failed eval outputs (no summary.json means failed)
for d in /mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620/*/; do
    name=$(basename "$d")
    fe="$d/full_eval"
    if [ -d "$fe" ]; then
        # Check if any summary.json exists in full_eval
        if ! find "$fe" -name "summary.json" 2>/dev/null | grep -q .; then
            echo "CLEARING $name (no summary.json)"
            rm -rf "$fe"
        fi
    fi
done
echo "===CLEARED==="
