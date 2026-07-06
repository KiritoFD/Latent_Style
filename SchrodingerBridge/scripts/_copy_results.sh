#!/bin/bash
# Copy aggregation results to Windows-accessible location
cp /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/ablation_results.md /mnt/c/Users/Administrator/ablation_results.md
cp /mnt/i/Github/Latent_Style/SchrodingerBridge/docs/ablation_results.csv /mnt/c/Users/Administrator/ablation_results.csv
echo "Files copied to /mnt/c/Users/Administrator/"
ls -la /mnt/c/Users/Administrator/ablation_results.*
