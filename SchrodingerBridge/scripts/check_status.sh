#!/usr/bin/env bash
echo "=== latent512_e7 log tail ==="
grep -i -E "art_fid|inception|fid|musiq|error|fail" /mnt/i/exp_our_models_eval/logs/latent512_e7_eval.log 2>/dev/null | tail -40
echo ""
echo "=== image count ==="
ls /mnt/i/exp_our_models_eval/latent512_e7/images/ 2>/dev/null | wc -l
echo ""
echo "=== last 10 lines of log ==="
tail -10 /mnt/i/exp_our_models_eval/logs/latent512_e7_eval.log 2>/dev/null
