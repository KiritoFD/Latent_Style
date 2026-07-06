#!/usr/bin/env bash
set -u
EXP_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/exp_ablation_620
echo "=== Ablation experiments ==="
total=0
has_ckpt=0
has_eval=0
empty_eval=0
for d in "$EXP_DIR"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    total=$((total+1))
    ckpt=""
    if [ -f "$d/epoch_0003.pt" ]; then
        ckpt="epoch_0003.pt"
        has_ckpt=$((has_ckpt+1))
    elif [ -f "$d/epoch_0002.pt" ]; then
        ckpt="epoch_0002.pt"
        has_ckpt=$((has_ckpt+1))
    elif [ -f "$d/epoch_0001.pt" ]; then
        ckpt="epoch_0001.pt"
        has_ckpt=$((has_ckpt+1))
    fi
    fe="$d/full_eval"
    eval_state="no_eval"
    if [ -d "$fe" ]; then
        # check for any summary.json
        if find "$fe" -name "summary.json" 2>/dev/null | grep -q .; then
            has_eval=$((has_eval+1))
            eval_state="has_summary"
        else
            empty_eval=$((empty_eval+1))
            eval_state="empty"
        fi
    fi
    echo "$name | ckpt=$ckpt | eval=$eval_state"
done
echo ""
echo "=== Summary ==="
echo "Total: $total"
echo "Has checkpoint: $has_ckpt"
echo "Has summary.json (done): $has_eval"
echo "Empty full_eval (failed): $empty_eval"
echo "Remaining to evaluate: $((has_ckpt - has_eval))"
