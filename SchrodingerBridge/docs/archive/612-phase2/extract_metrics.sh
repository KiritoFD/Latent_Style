#!/bin/bash
# Extract key metrics from all remote summary.json files
BASE="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp"

for d in "$BASE"/*/; do
    dd=$(basename "$d")
    # latest epoch
    latest=""
    for e in "$d"/full_eval/epoch_*/; do
        latest=$(basename "$e")
    done
    if [ -z "$latest" ]; then continue; fi
    
    summary="$d/full_eval/$latest/summary.json"
    if [ ! -f "$summary" ]; then continue; fi
    
    # Extract metrics using python3 (one-shot to avoid jq dependency)
    python3 -c "
import json, sys
with open('$summary') as f:
    data = json.load(f)
allpairs = data.get('all_pairs_overview', {})
transfer = data.get('style_transfer_ability', {})
identity = data.get('identity_reconstruction', {})
timing = data.get('timings_sec', {})

ap_style = allpairs.get('clip_style', 0)
ap_lpips = allpairs.get('content_lpips', 0)
tr_style = transfer.get('clip_style', 0)
tr_lpips = transfer.get('content_lpips', 0)
id_style = identity.get('clip_style', 0)
id_lpips = identity.get('content_lpips', 0)
wall = timing.get('wall_total', 0)
gen = timing.get('generation', 0)

print(f'EXPERIMENT|$dd|$latest|{tr_style:.6f}|{tr_lpips:.6f}|{ap_style:.6f}|{ap_lpips:.6f}|{id_style:.6f}|{id_lpips:.6f}|{wall:.1f}|{gen:.1f}')
" 2>/dev/null
done
