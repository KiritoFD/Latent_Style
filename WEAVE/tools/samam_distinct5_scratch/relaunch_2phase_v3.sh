#!/usr/bin/env bash
set -uo pipefail

# Kill any existing
pkill -f "eval_samam_metrics_phase2" 2>/dev/null || true
pkill -f "gen_samam_images_phase1" 2>/dev/null || true
tmux kill-server 2>/dev/null || true
sleep 3

# Copy fixed phase1 script
cp /mnt/c/Users/Administrator/gen_samam_images_phase1.py /mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch/gen_samam_images_phase1.py

# WSL keepalive
nohup bash -c 'while true; do sleep 3600; done' >/dev/null 2>&1 &

source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export TRANSFORMERS_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_hf_750_batched
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch

# Run Phase 1 then Phase 2 with nohup+setsid (no tmux dependency)
cat > $TRAIN_OUT/run_2phase.sh << 'INNEREOF'
#!/usr/bin/env bash
set -uo pipefail
source /home/xy/venvs/samam312/bin/activate
cd /mnt/i/Github/Latent_Style
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/mnt/i/Github/Latent_Style/eval_cache/hf
export TRANSFORMERS_CACHE=/mnt/i/Github/Latent_Style/eval_cache/hf
export PYTHONPATH=/mnt/i/Github/Latent_Style/Related_Works/repos/SaMam:/mnt/i/Github/Latent_Style

TRAIN_OUT=/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_scratch_7k_250eval_remote
CKPT_DIR=$TRAIN_OUT/step_checkpoints
OUTPUT_ROOT=$TRAIN_OUT/curve_eval_hf_750_batched
IMAGE_ROOT=/mnt/i/wikiart_distinct5_samam_512_classview/test
STYLE_NAMES="Early_Renaissance,Impressionism,Minimalism,Rococo,Ukiyo_e"
SCRIPT_DIR=/mnt/i/Github/Latent_Style/SchrodingerBridge/tools/samam_distinct5_scratch

echo "=== Phase 1: Generate all images (fp32, skip existing) ==="
python $SCRIPT_DIR/gen_samam_images_phase1.py \
    --ckpt-dir $CKPT_DIR \
    --image-root $IMAGE_ROOT \
    --output-root $OUTPUT_ROOT \
    --image-size 512 \
    --max-src-per-style 30 \
    --style-names $STYLE_NAMES
echo "PHASE1_DONE=$(date -Iseconds)"
echo ""
echo "=== Phase 2: Evaluate all metrics (80 ckpts) ==="
python $SCRIPT_DIR/eval_samam_metrics_phase2.py \
    --image-root $IMAGE_ROOT \
    --output-root $OUTPUT_ROOT \
    --image-size 512 \
    --max-src-per-style 30 \
    --metric-batch-size 64 \
    --style-names $STYLE_NAMES
echo "PHASE2_DONE=$(date -Iseconds)"
echo "ALL_DONE"
INNEREOF
chmod +x $TRAIN_OUT/run_2phase.sh

# Launch with setsid (fully detached, survives SSH disconnect)
setsid bash $TRAIN_OUT/run_2phase.sh > $TRAIN_OUT/eval_2phase_v2.log 2>&1 &
sleep 5

echo "=== 2-phase v2 launched (setsid, fp32 fixed) ==="
echo "PID: $(pgrep -f run_2phase.sh | head -1)"
echo "Log: $TRAIN_OUT/eval_2phase_v2.log"
ps aux | grep -E "run_2phase|gen_samam" | grep -v grep | head -5
