#!/bin/bash
# Collect all eval results from ablation experiments
set -euo pipefail

EXP_BASE="/mnt/i/Github/Latent_Style/exp/620_spatial_bridge"
OUTFILE="/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/ablation256/results.csv"

echo "name,epoch,clip_style,content_lpips,clip_dir,loss_fm,loss_swd,loss_edge,style_gate_value,cross_attn_entropy,velocity_abs,endpoint_abs" > "$OUTFILE"

for DIR in "$EXP_BASE"/abl_*; do
  NAME=$(basename "$DIR")
  # Find latest epoch eval
  EVAL_DIR=$(ls -d "$DIR"/full_eval/epoch_* 2>/dev/null | sort -V | tail -1)
  if [ -z "$EVAL_DIR" ]; then
    # Try numeric_debug
    continue
  fi
  CSV="$EVAL_DIR/metrics.csv"
  if [ ! -f "$CSV" ]; then continue; fi
  EPOCH=$(basename "$EVAL_DIR" | sed 's/epoch_//')
  # Extract mean values
  python3 -c "
import csv, sys
with open('$CSV') as f:
    r = list(csv.DictReader(f))
if not r: sys.exit(0)
clip = sum(float(x.get('clip_style',0)) for x in r)/len(r)
lpips = sum(float(x.get('content_lpips',0)) for x in r)/len(r)
clipd = sum(float(x.get('clip_dir',0)) for x in r)/len(r)
print(f'$NAME,$EPOCH,{clip:.4f},{lpips:.4f},{clipd:.4f}')
  " >> "$OUTFILE" 2>/dev/null || true
done

echo "Results written to $OUTFILE"
echo "Total experiments with results: $(wc -l < $OUTFILE)"
