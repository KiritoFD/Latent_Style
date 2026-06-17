#!/usr/bin/env bash
# ============================================================
# VRAM Probe: 找出每种 tokenizer 在不超 11.2GB 下的最大 batch
# 结果用于设置正式实验的 batch_size
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WARM_CKPT="exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt"
BASE_CFG="$(dirname "$WARM_CKPT")/config.json"

echo "=== VRAM Batch Probe ==="
echo "Target: 9-11.2 GB, prefer multiples of 16"
echo "Test: pure_latent_spatial (default), smoe_translator, affine_connection"
echo ""

probe_batch() {
    local tag="$1" tok_family="$2" batch="$3"
    local dir="exp/phase616_vramprobe_${tag}_b${batch}"
    mkdir -p "$dir"

    python3 -c "
import json
c = json.load(open('$BASE_CFG'))
c['model']['tokenizer_family'] = '$tok_family'
c['bridge']['bridge_path_mode'] = 'vertical'
c['training']['num_epochs'] = 1
c['training']['batch_size'] = $batch
c['training']['virtual_length_multiplier'] = 0.05
c['training']['full_eval_each_epoch'] = False
c['checkpoint']['save_dir'] = './$dir'
json.dump(c, open('$dir/config.json', 'w'), indent=2)
"
    cp "$WARM_CKPT" "$dir/epoch_0001.pt"

    echo -n "  $tag b=$batch: "
    python src/run.py --config "$dir/config.json" --resume "$dir/epoch_0001.pt" 2>&1 | \
        grep "cuda_peak_allocated" | tail -1 | awk '{print $NF}' || echo "OOM?"
}

# pure_latent_spatial (当前默认)
for b in 8 12 16 20 24 28 32; do
    probe_batch "purelatent" "pure_latent_spatial" $b
done

# smoe_translator (更多参数)
for b in 4 6 8 10 12 14 16; do
    probe_batch "smoe" "smoe_translator" $b
done

# affine_connection (最轻量?)
for b in 8 12 16 20 24 28 32; do
    probe_batch "affine" "affine_connection_tokenizer" $b
done

echo ""
echo "=== DONE ==="
echo "Use the largest batch that stays under 11.2 GB for each tokenizer"
