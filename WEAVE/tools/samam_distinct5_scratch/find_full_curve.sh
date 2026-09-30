#!/usr/bin/env bash
echo "=== Searching for SaMam old curve data with more checkpoints ==="
# Search for any curve_metrics.csv in samam related dirs
for dir in /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/*; do
    if [ -d "$dir" ]; then
        csv="$dir/curve_metrics.csv"
        if [ -f "$csv" ]; then
            lines=$(wc -l < "$csv")
            echo "FOUND: $csv ($lines lines)"
        fi
    fi
done
echo "---"
# Search in other samam dirs
for dir in /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba*; do
    if [ -d "$dir" ]; then
        for sub in "$dir"/*; do
            if [ -d "$sub" ]; then
                csv="$sub/curve_metrics.csv"
                if [ -f "$csv" ]; then
                    lines=$(wc -l < "$csv")
                    echo "FOUND: $csv ($lines lines)"
                fi
            fi
        done
    fi
done
echo "---"
# Search for any json with curve data
find /mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag -name '*.json' 2>/dev/null | head -10
echo "=== DONE ==="
