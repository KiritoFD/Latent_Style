#!/bin/bash
# Remote cleanup script for inmortal-exp directory
# ONLY deletes .pt checkpoints, preserves all eval data and logs
set -e

BASE="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp"
FREED=0
COUNT=0

del_pts() {
    local dir="$1"
    if [ -d "$dir" ]; then
        for pt in "$dir"/epoch_*.pt; do
            if [ -f "$pt" ]; then
                sz=$(stat -c%s "$pt" 2>/dev/null || echo 0)
                rm -f "$pt"
                FREED=$((FREED + sz))
                COUNT=$((COUNT + 1))
            fi
        done
    fi
}

del_pts_except() {
    local dir="$1"; shift
    if [ -d "$dir" ]; then
        for pt in "$dir"/epoch_*.pt; do
            if [ ! -f "$pt" ]; then continue; fi
            local keep=0
            for pat in "$@"; do
                case "$(basename "$pt")" in
                    "$pat") keep=1; break ;;
                esac
            done
            if [ $keep -eq 0 ]; then
                sz=$(stat -c%s "$pt" 2>/dev/null || echo 0)
                rm -f "$pt"
                FREED=$((FREED + sz))
                COUNT=$((COUNT + 1))
            fi
        done
    fi
}

echo "=== Phase1: Delete all .pt from drop experiments ==="

# Full deletion (negatives with no future value)
del_pts "$BASE/aaai2027_inmortal_xpred_phighpass_seed42_b28"
del_pts "$BASE/aaai2027_inmortal_xpred_kmanifold_phighpass_seed42_b32"
del_pts "$BASE/aaai2027_inmortal_xpred_bary_seed42_b40"
del_pts "$BASE/aaai2027_inmortal_xpred_kmanifold_pattn_aniso_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_k_spatial_seed42_b32"
del_pts "$BASE/aaai2027_inmortal_k_spatial_seed42_b44"
del_pts "$BASE/aaai2027_inmortal_k_spectral_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_p_highpass_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_p_highpass_seed42_b32"
del_pts "$BASE/aaai2027_inmortal_xpred_kmanifold_pattn_edgegated_anisostokes_queue_from_e13_seed42_b8a2"

# from_e13 cleanup - these are all drop/hold
for d in \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_trust_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleasewide_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4twostage_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clampreleaselatewide_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_knee_e13_carriergate_injection_seed42_b8a2 \
    aaai2027_inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2; do
    del_pts "$BASE/$d"
done

# Carriergate hold4mid (keep eval, delete ckpt)
del_pts "$BASE/aaai2027_inmortal_hold4mid_e8_spatial_carriergate_bodydecoder_seed42_b8a2"
del_pts "$BASE/aaai2027_inmortal_hold4mid_e8_spatial_carriergate_decoder_seed42_b8a2"
del_pts "$BASE/aaai2027_inmortal_knee_e13_spatial_carriergate_bodydecoder_seed42_b8a2"

# Obsoleted hold experiments
del_pts "$BASE/aaai2027_inmortal_xpred_kmanifold_pmod_seed42_b32"
del_pts "$BASE/aaai2027_inmortal_xpred_bary_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_xpred_queue_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_xpred_structot_seed42_b16"
del_pts "$BASE/aaai2027_inmortal_xpred_teacher_endpoint_seed42_b16"

echo "=== Phase2: Trim intermediate ckpt from keep experiments ==="

# k_spatial: keep e6 only
del_pts_except "$BASE/aaai2027_inmortal_k_spatial_seed42_b16" "epoch_0006.pt"
# k_manifold: keep e6 only
del_pts_except "$BASE/aaai2027_inmortal_k_manifold_seed42_b16" "epoch_0006.pt"
# k_spectral: keep e2 only
del_pts_except "$BASE/aaai2027_inmortal_k_spectral_seed42_b12" "epoch_0002.pt"
# xpred_kmanifold: keep e7 only
del_pts_except "$BASE/aaai2027_inmortal_xpred_kmanifold_seed42_b32" "epoch_0007.pt"
# pattn_base: keep e6 only
del_pts_except "$BASE/aaai2027_inmortal_xpred_kmanifold_pattn_seed42_b16" "epoch_0006.pt"
# pattn_stokes_base: keep e3 only
del_pts_except "$BASE/aaai2027_inmortal_xpred_kmanifold_pattn_stokes_seed42_b16" "epoch_0003.pt"
# pattn_queue: keep e6 only
del_pts_except "$BASE/aaai2027_inmortal_xpred_kmanifold_pattn_queue_seed42_b16" "epoch_0006.pt"

# from_e13 keepers: keep only eval data (delete all ckpt)
for d in \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamp_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamprelease_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4wide_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4mid_reseed_from_e13_seed42_b8a2 \
    aaai2027_inmortal_xpred_kmanifold_pattn_anisostokes_queue_clamphold4slowmid_reseed_from_e13_seed42_b8a2; do
    del_pts "$BASE/$d"
done

FREED_MB=$((FREED / 1024 / 1024))
echo ""
echo "Done. Deleted $COUNT checkpoint files, freed approximately ${FREED_MB} MB"
