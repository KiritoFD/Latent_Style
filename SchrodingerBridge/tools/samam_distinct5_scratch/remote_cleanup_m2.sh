#!/usr/bin/env bash
# M2: Reorganize remote I drive experiments into baselines/samam/ours structure
# All moves are within same filesystem (instant rename), no copy needed
set -uo pipefail

LOG=/mnt/i/Github/Latent_Style/_cleanup_m2.log
RESULTS=/mnt/i/Github/Latent_Style/_cleanup_m2.results
echo "=== M2 Reorg Start: $(date -Iseconds) ===" > "$LOG"
echo "" > "$RESULTS"

moved_count=0
move_dir() {
    local src="$1"
    local dst_parent="$2"
    local reason="$3"
    if [ ! -d "$src" ]; then
        echo "SKIP (not exists): $src" >> "$LOG"
        return 0
    fi
    mkdir -p "$dst_parent"
    local base=$(basename "$src")
    if [ -e "$dst_parent/$base" ]; then
        echo "SKIP (dst exists): $dst_parent/$base" >> "$LOG"
        return 0
    fi
    mv "$src" "$dst_parent/"
    if [ -e "$dst_parent/$base" ]; then
        echo "OK: $src -> $dst_parent/$base | $reason" >> "$RESULTS"
        moved_count=$((moved_count + 1))
    else
        echo "FAIL: $src" >> "$RESULTS"
    fi
}

ROOT=/mnt/i/Github/Latent_Style
BL_EVAL=$ROOT/Related_Works/baseline_pipeline/results

# ==================== Create new structure ====================
mkdir -p $ROOT/exp_baselines
mkdir -p $ROOT/exp_samam/training
mkdir -p $ROOT/exp_samam/eval
mkdir -p $ROOT/exp_ours/phase2
mkdir -p $ROOT/exp_ours/recent

# ==================== Move baselines (12 + SaMST) ====================
echo "--- Group A: Baselines ---" >> "$LOG"

# 12 baseline eval results from Related_Works/baseline_pipeline/results/
for d in cut s2wat sdedit_str_0p10 sdedit_str_0p20 sdedit_str_0p35 sdedit_str_0p40 \
         sdturbo seedream45_api styleid samst; do
    move_dir "$BL_EVAL/$d" "$ROOT/exp_baselines" "baseline_eval"
done

# SaMST training experiments (the ones with real content >200K)
for d in samst_distinct5_512_wsl_stepalign40_remote_20260605_r1 \
         samst_latent_distinct5_512_convergence_20260606_180529 \
         samst_latent_distinct5_512_convergence_20260606_214051 \
         samst_latent_distinct5_512_samecost_20260606_034941 \
         samst_latent_distinct5_512_samecost_20260606_041227 \
         samst_latent_distinct5_512_samecost_20260606_145824 \
         samst_latent_distinct5_512_samecost_20260606_172021; do
    move_dir "$BL_EVAL/$d" "$ROOT/exp_baselines" "samst_training"
done

# zimage_turbo and flux2_klein (empty but valid baseline placeholders)
for d in zimage_turbo flux2_klein; do
    move_dir "$BL_EVAL/$d" "$ROOT/exp_baselines" "baseline_placeholder"
done

# ours_pareto_probe_4_epoch_0001 (small probe, move to ours/recent)
move_dir "$BL_EVAL/ours_pareto_probe_4_epoch_0001" "$ROOT/exp_ours/recent" "ours_probe"

# ==================== Move SaMam training and eval ====================
echo "--- Group B: SaMam ---" >> "$LOG"

# SaMam main training (44G)
move_dir "$BL_EVAL/samam_distinct5_512_scratch_7k_250eval_remote" "$ROOT/exp_samam/training" "samam_main_training_44G"

# SaMam large training experiments (>100M, real training)
for d in samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag \
         samam_distinct5_512_mamba_b6_20k_remote_wsl_20260601_1900 \
         samam_distinct5_512_mamba_b8_20k_remote_wsl_20260601_1910 \
         samam_distinct5_512_mamba_b8_seg250_remote_wsl_20260601_1935 \
         samam_distinct5_remote_wsl_batch_probe_20260601_1840 \
         samam_latent_distinct5_512_convergence_20260606_222608 \
         samam_latent_distinct5_512_convergence_20260607_002420 \
         samam_latent_distinct5_512_convergence_20260607_011328 \
         samam_latent_distinct5_512_samecost_20260606_133730 \
         samam_latent_distinct5_512_samecost_20260606_155105 \
         samam_latent_distinct5_512_samecost_20260606_162933 \
         samam_latent_legacy256_probe4 \
         samam_256_faithful_p8_remote; do
    move_dir "$BL_EVAL/$d" "$ROOT/exp_samam/training" "samam_historical_training"
done

# ==================== Move ours: aaai2027_phase2_* and recent ====================
echo "--- Group C: Ours ---" >> "$LOG"

# aaai2027_phase2_* (valid experiments, ~28)
for d in $(ls -d $ROOT/exp/aaai2027_phase2_* 2>/dev/null); do
    move_dir "$d" "$ROOT/exp_ours/phase2" "aaai2027_phase2"
done

# Recent ours: 620_spatial_bridge, inmortal-exp, highres, phase2_eval_rgbcal
for d in 620_spatial_bridge inmortal-exp highres phase2_eval_rgbcal; do
    move_dir "$ROOT/exp/$d" "$ROOT/exp_ours/recent" "ours_recent"
done

# Also move the all620.json and 620_t5_base_multimodal_train.log
for f in all620.json 620_t5_base_multimodal_train.log; do
    if [ -f "$ROOT/exp/$f" ]; then
        mv "$ROOT/exp/$f" "$ROOT/exp_ours/recent/"
        echo "OK: moved file $f" >> "$RESULTS"
        moved_count=$((moved_count + 1))
    fi
done

# ==================== Rename experiments/ to experiments_historical/ ====================
echo "--- Group D: Historical ---" >> "$LOG"
if [ -d "$ROOT/experiments" ]; then
    mv "$ROOT/experiments" "$ROOT/experiments_historical"
    echo "OK: experiments -> experiments_historical" >> "$RESULTS"
    moved_count=$((moved_count + 1))
fi

# ==================== Move Related_Works/runs/ useful content ====================
echo "--- Group E: runs ---" >> "$LOG"
# Keep only useful runs (cut_5x5, sdedit_multi, sdturbo_5x5, s2wat_bs1_safe_e2000_full_eval, hf_snapshots, benchmark_logs, server_new_baselines)
# Move them to exp_baselines/_auxiliary_runs/
mkdir -p $ROOT/exp_baselines/_auxiliary_runs
for d in cut_5x5 sdedit_multi sdturbo_5x5 s2wat_bs1_safe_e2000_full_eval benchmark_logs server_new_baselines; do
    move_dir "$ROOT/Related_Works/runs/$d" "$ROOT/exp_baselines/_auxiliary_runs" "auxiliary_run"
done

# hf_snapshots is CLIP model cache (4.9G), keep in eval_cache location
# Leave cyclegan_5x5 empty dir alone (already deleted smoke version)

# ==================== Summary ====================
echo "" >> "$LOG"
echo "=== M2 Reorg End: $(date -Iseconds) ===" >> "$LOG"
echo "Moved: $moved_count directories" >> "$LOG"
echo ""
echo "=== Summary ==="
cat "$RESULTS" | wc -l
echo "moved directories"
echo ""
echo "=== New structure ==="
ls -la $ROOT/exp_baselines/ | head -3
echo "..."
ls $ROOT/exp_baselines/ | wc -l
echo "baselines"
ls $ROOT/exp_samam/training/ | wc -l
echo "samam training"
ls $ROOT/exp_ours/phase2/ 2>/dev/null | wc -l
echo "ours phase2"
ls $ROOT/exp_ours/recent/ 2>/dev/null | wc -l
echo "ours recent"
