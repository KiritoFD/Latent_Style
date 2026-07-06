#!/usr/bin/env bash
# Clear DA02 partial output and relaunch batch eval in foreground (kept alive by SSH session).
set -uo pipefail
EXP_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620

# Clear any partial/empty full_eval dirs (no summary.json)
for d in "$EXP_DIR"/*/; do
    name=$(basename "$d")
    fe="$d/full_eval"
    if [ -d "$fe" ]; then
        if ! find "$fe" -name "summary.json" 2>/dev/null | grep -q .; then
            echo "CLEARING $name (no summary.json)"
            rm -rf "$fe"
        fi
    fi
done
echo "===CLEARED==="
