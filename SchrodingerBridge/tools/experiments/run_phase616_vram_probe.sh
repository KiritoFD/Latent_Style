#!/usr/bin/env bash
# 快速VRAM探针
set -euo pipefail

ROOT_DIR="/mnt/i/Github/Latent_Style/SchrodingerBridge"
cd "$ROOT_DIR"

BASE_CFG="/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/config.json"
CKPT="/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt"

# 目标 VRAM: 9-11.2 GB, 尽量靠近 11GB 以提高效率
# 记录 b 及对应的峰值 VRAM

echo "tokenizer,batch,cuda_peak_gb"

for tok in pure_latent_spatial smoe_translator affine_connection_tokenizer; do
    # 每种 tokenizer 的 batch 范围
    case "$tok" in
        pure_latent_spatial) batches="8 12 16 20 24 28 32 36 40";;
        smoe_translator)     batches="4 6 8 10 12 14 16 18 20";;
        affine_connection_tokenizer) batches="8 12 16 20 24 28 32";;
    esac

    for b in $batches; do
        d="exp/_vram_${tok}_b${b}"
        mkdir -p "$d"

        python3 -c "
import json
c = json.load(open('$BASE_CFG'))
c['model']['tokenizer_family'] = '$tok'
c['training']['num_epochs'] = 1
c['training']['batch_size'] = $b
c['training']['virtual_length_multiplier'] = 0.01
c['training']['full_eval_each_epoch'] = False
c['checkpoint']['save_dir'] = './$d'
# warmstart from topogate (non-strict, ignores tokenizer mismatch)
c['training']['resume_model_strict'] = False
json.dump(c, open('$d/config.json', 'w'), indent=2)
"

        python src/run.py --config "$d/config.json" \
            --resume "exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b12a1/epoch_0001.pt" \
            2>/dev/null || true

        # 读取峰值 VRAM
        peak=$(grep "cuda_peak_allocated_gb" "$d/config.json" 2>/dev/null || true)
        if [ -f "$d/logs/training_"*.csv ]; then
            logfile=$(ls "$d/logs/training_"*.csv | tail -1)
            # 从训练日志取最后的 cuda_peak_allocated_gb
            peak=$(tail -1 "$logfile" 2>/dev/null | python3 -c "import sys; print(sys.stdin.readline().split(',')[-9].strip())" 2>/dev/null || echo "OOM")
        else
            peak="OOM_or_no_log"
        fi
        echo "$tok,$b,$peak"
    done
done

echo ""
echo "DONE. Pick largest b with peak < 11.2 for each tokenizer"
