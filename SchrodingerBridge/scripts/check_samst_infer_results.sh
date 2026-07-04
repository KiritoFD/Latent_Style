#!/bin/bash
echo "=== Recent infer log ==="
tail -80 /mnt/i/exp_samst_latent_infer.log 2>/dev/null
echo ""
echo "=== Output dir contents ==="
ls -la /mnt/i/exp_samst_latent_eval/ 2>/dev/null
echo ""
echo "=== curve_metrics.csv ==="
cat /mnt/i/exp_samst_latent_eval/curve_metrics.csv 2>/dev/null
echo ""
echo "=== curve_metrics.json ==="
cat /mnt/i/exp_samst_latent_eval/curve_metrics.json 2>/dev/null
