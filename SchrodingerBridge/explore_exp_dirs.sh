#!/bin/bash
# explore_exp_dirs.sh
# Explore experiment output directories on remote Windows server (via WSL)
# Goal: locate all method generation-image directories for CLIP-T / MUSIQ / ART-FID metrics.
# Read-only: no files are modified.

ROOT="/mnt/i/Github/Latent_Style"
SB_EXP="${ROOT}/SchrodingerBridge/exp"
BV2="${SB_EXP}/baseline_v2"

# Collected summary rows (METHOD|PATH|PNG_COUNT)
SUMMARY=""

inspect_dir() {
    local label="$1"
    local dir="$2"
    echo "=========================================="
    echo "METHOD: ${label}"
    echo "PATH:   ${dir}"
    if [ ! -d "$dir" ]; then
        echo "STATUS: NOT FOUND"
        echo ""
        SUMMARY="${SUMMARY}${label}|${dir}|NOT_FOUND\n"
        return
    fi
    echo "STATUS: EXISTS"
    local png_count
    png_count=$(find "$dir" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l | tr -d ' ')
    echo "PNG_COUNT: ${png_count}"
    echo "SAMPLE_FILES:"
    find "$dir" -maxdepth 1 -name "*.png" -type f 2>/dev/null | sort | head -3 | sed 's/^/  /'
    if [ -d "$dir/step_000001/images" ]; then
        local sub_count
        sub_count=$(find "$dir/step_000001/images" -maxdepth 1 -name "*.png" -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "STEP_000001_IMAGES: YES (${sub_count} pngs)"
    else
        echo "STEP_000001_IMAGES: NO"
    fi
    SUMMARY="${SUMMARY}${label}|${dir}|${png_count}\n"
    echo ""
}

echo "#############################################"
echo "# 512 RESOLUTION METHODS"
echo "#############################################"
echo ""

# 512 baselines: primary location is baseline_v2/images/{method}/
for method in adain identity samst sdedit_str0.10 sdedit_str0.20 sdedit_str0.35 sdedit_str0.40 sdturbo styleid cut samam seedream wct_vgg19; do
    inspect_dir "512_${method} [images]" "${BV2}/images/${method}"
done

# 512 SaMam: also has a dedicated eval dir under exp_samam
inspect_dir "512_samam [curve_eval_hf_750_batched/step_020000]" "${ROOT}/exp_samam/training/samam_distinct5_512_scratch_7k_250eval_remote/curve_eval_hf_750_batched/step_020000/images"

# 512 baselines: secondary location baseline_v2/eval/{method}/images/
echo "# 512 eval/{method}/images variants (secondary)"
for method in adain identity samst sdedit_str0.10 sdedit_str0.20 sdedit_str0.35 sdedit_str0.40 sdturbo styleid cut samam seedream wct wct_vgg19; do
    inspect_dir "512_${method} [eval/images]" "${BV2}/eval/${method}/images"
done

echo ""
echo "#############################################"
echo "# 256 RESOLUTION METHODS"
echo "#############################################"
echo ""

# 256 train-free / trained baselines (exp_baseline_256)
inspect_dir "256_adain"  "${ROOT}/exp_baseline_256/adain/step_000001/images"
inspect_dir "256_wct"    "${ROOT}/exp_baseline_256/wct/step_000001/images"
inspect_dir "256_samst"  "${ROOT}/exp_baseline_256/samst/step_000001/images"

# 256 SaMam
inspect_dir "256_samam"  "${ROOT}/exp_samam/eval_256/samam_final_20k_256/step_020000/images"

# 256 Our model candidates (latent256 e10 + pixel256 e3)
inspect_dir "256_our_latent256_e10 [clean_base_v2]"       "${SB_EXP}/clean_base_v2/full_eval/epoch_0010/images"
inspect_dir "256_our_latent256_e10 [clean_base_v2_22cuts]" "${SB_EXP}/clean_base_v2_22cuts/full_eval/epoch_0010/images"
inspect_dir "256_our_latent256_e10 [clean_base]"          "${SB_EXP}/clean_base/full_eval/epoch_0010/images"
inspect_dir "256_our_pixel256_e3 [clean_base epoch_0008]" "${SB_EXP}/clean_base/full_eval/epoch_0008/images"
inspect_dir "256_our_pixel256_e3 [clean_base epoch_0009]" "${SB_EXP}/clean_base/full_eval/epoch_0009/images"

# Pixel256 e3 candidates from epoch_0003 search
inspect_dir "256_our_e3 [620_spectral_v11_ll10_hh20 epoch_0003]"          "${SB_EXP}/620_spectral_v11_ll10_hh20/full_eval/epoch_0003/images"
inspect_dir "256_our_e3 [tuning_deepdive/f1_repro_e2 epoch_0003]"         "${SB_EXP}/tuning_deepdive/f1_repro_e2/full_eval/epoch_0003/images"
inspect_dir "256_our_e3 [625_fc_sb/fc_sb_t1_full_curriculum epoch_0003]"  "${SB_EXP}/625_fc_sb/fc_sb_t1_full_curriculum/full_eval/epoch_0003/images"
inspect_dir "256_our_e3 [p3_remote_10h/r1_baseline_long epoch_0003]"      "${SB_EXP}/p3_remote_10h/r1_baseline_long/full_eval/epoch_0003/images"

echo ""
echo "#############################################"
echo "# CONFIG INSPECTION (pixel vs latent space)"
echo "#############################################"
for cfg in \
    "${SB_EXP}/clean_base_v2/config.json" \
    "${SB_EXP}/clean_base/config.json" \
    "${SB_EXP}/clean_base_v2_22cuts/config.json" \
    "${SB_EXP}/620_spectral_v11_ll10_hh20/config.json" \
    "${SB_EXP}/tuning_deepdive/f1_repro_e2/config.json" \
    "${SB_EXP}/625_fc_sb/fc_sb_t1_full_curriculum/config.json" \
    "${SB_EXP}/p3_remote_10h/r1_baseline_long/config.json"; do
    echo "--- ${cfg} ---"
    if [ -f "$cfg" ]; then
        grep -E '"(space|work_space|input_space|latent|pixel|vae|image_size|resolution|in_channels)"' "$cfg" 2>/dev/null | head -15 | sed 's/^/  /'
    else
        echo "  CONFIG NOT FOUND"
    fi
    echo ""
done

echo ""
echo "#############################################"
echo "# SEARCH FOR pixel256 / pixel DIRECTORIES"
echo "#############################################"
echo "find ${SB_EXP} -maxdepth 4 -type d -iname '*pixel*':"
find "${SB_EXP}" -maxdepth 4 -type d -iname "*pixel*" 2>/dev/null | sed 's/^/  /'
echo ""
echo "find ${ROOT} -maxdepth 3 -type d -iname '*pixel*256*':"
find "${ROOT}" -maxdepth 3 -type d -iname "*pixel*256*" 2>/dev/null | sed 's/^/  /'
echo ""
echo "find ${SB_EXP} -maxdepth 4 -type d -iname '*e3*' or '*epoch_0003*':"
find "${SB_EXP}" -maxdepth 4 -type d \( -iname "*epoch_0003*" -o -iname "*_e3_*" -o -iname "*_e3" \) 2>/dev/null | sed 's/^/  /'
echo ""

echo ""
echo "#############################################"
echo "# TEST SET PATHS"
echo "#############################################"
for t in "${ROOT}/SchrodingerBridge/test" "/mnt/i/wikiart_distinct5_samam_512_classview/test" "/mnt/i/wikiart_distinct5_samam_512_classview"; do
    echo "TEST: ${t}"
    if [ -d "$t" ]; then
        echo "  EXISTS"
        ls "$t" 2>/dev/null | head -10 | sed 's/^/    /'
    else
        echo "  NOT FOUND"
    fi
    echo ""
done

echo ""
echo "#############################################"
echo "# SUMMARY TABLE"
echo "#############################################"
echo ""
printf "%-50s | %-95s | %s\n" "METHOD" "PATH" "PNG_COUNT"
printf "%s-+-%s-+-%s\n" "$(printf '%0.s-' {1..50})" "$(printf '%0.s-' {1..95})" "$(printf '%0.s-' {1..10})"
echo -e "$SUMMARY" | while IFS='|' read -r method path count; do
    [ -z "$method" ] && continue
    printf "%-50s | %-95s | %s\n" "$method" "$path" "$count"
done

echo ""
echo "#############################################"
echo "# DONE"
echo "#############################################"
