#!/bin/bash
# Phase 2 remote exp/ root cleanup
BASE="/mnt/i/Github/Latent_Style/SchrodingerBridge/exp"

del_pts_in() { for d in "$@"; do rm -f "$BASE/$d"/epoch_*.pt 2>/dev/null; done; }
del_dir()    { for d in "$@"; do rm -rf "$BASE/$d" 2>/dev/null; done; }

echo "=== 1. Delete smoke tests entirely ==="
del_dir _remote_smoke_distinct5_512_ema_baseline_b48_vlen005 \
        _remote_smoke_distinct5_512_ema_baseline_b64_vlen005 \
        _remote_smoke_distinct5_512_ema_baseline_b80_vlen005 \
        _codex_smoke \
        _dryrun_mse_controls_remote \
        _smoke_mse_controls \
        _launchers

echo "=== 2. Delete obsolete probes ==="
del_dir fisher_memory_consumer_probe_mds02 \
        fisher_operator_consumer_probe_smoke \
        fisher_operator_tokenizer_probe_smoke \
        fisher_style_backbone_probe \
        fisher_style_memory_adapter_probe \
        memory_contentdir_backbone_probe_smoke \
        memory_contentdir_backbone_probe \
        memory_residual_backbone_probe_smoke \
        memory_residual_backbone_probe \
        reference_memory_generation_probe_smoke \
        reference_memory_generation_probe_direct_smoke \
        reference_memory_generation_probe \
        reference_memory_generation_probe_full \
        router_aware_backbone_probe_smoke \
        router_aware_backbone_probe \
        style_adapter_distributional \
        style_embedding_distill \
        style_embedding_mainline_calibration \
        style_measure_aligned_adapter_probe \
        style_measure_aligned_backbone_probe \
        style_memory_bank_adapter_probe_smoke \
        style_memory_bank_adapter_probe \
        style_memory_bank_adapter_route_probe \
        style_memory_bank_probe_smoke \
        style_memory_bank_probe \
        style_memory_residual_adapter_probe \
        style_memory_typed_adapter_probe \
        style_representation_formation_probe \
        style_tokenizer_projector_refit \
        style_tokenizer_projector_smoke \
        style_tokenizer_vocab_refit_smoke \
        style_tokenizer_vocab_refit \
        tokenizer_adain_gate_calibration_smoke \
        tokenizer_adain_gate_calibration_smoke_g56 \
        tokenizer_adain_gate_calibration \
        tokenizer_adain_texture_gate_calibration_rerun \
        tokenizer_adain_texture_gate_calibration_rowlocal \
        tokenizer_adain_texture_gate_calibration \
        tokenizer_adain_texture_gate_smoke \
        tokenizer_bandgate_calibration_smoke \
        tokenizer_bandgate_calibration \
        tokenizer_bandgate_status \
        tokenizer_metric_mds_init \
        tokenizer_phase1 \
        tokenizer_prototype_carrier_calibration \
        tokenizer_prototype_carrier_smoke \
        tokenizer_prototype_carrier_smoke2 \
        tokenizer_stat_reader_probe \
        tokenizer_stat_reader_smoke \
        tokenizer_stat_vocab_probe \
        tokenizer_texton_carrier_calibration \
        typed_memory_backbone_probe_smoke \
        typed_memory_backbone_probe \
        typed_uniform_memory_backbone_probe

echo "=== 3. Delete unused tmp_debug dirs ==="
for d in tmp_debug_eval_{one,two,three,four,five,six,seven}; do
    rm -rf "$BASE/$d" 2>/dev/null
done
rm -f "$BASE"/tmp_debug_run_eval_init*.log 2>/dev/null

echo "=== 4. Clean .pt from completed review/mainline packets (keep eval data) ==="
del_pts_in aaai2027_executor_promotion_h_e1_seed42_b44 \
           aaai2027_mainline_h_softterm16_sem012_seed42_b44 \
           aaai2027_mainline_h_softterm18_sem010_seed42_b44 \
           aaai2027_mainline_h_softterm18_sem012_seed42_b44 \
           aaai2027_pairing_cache_h_randompair_seed42_b44 \
           aaai2027_projection_count_h_sem32_seed42_b44 \
           aaai2027_longer_train_f_seed42_b44_e8 \
           aaai2027_longer_train_k_seed42_b44_e8 \
           aaai2027_path_stability_probe_h_base_seed42_b44_e1

echo "=== 5. Clean .pt from distinct5 variants a-m (eval data captured in CSV) ==="
VARIANTS="baseline_direct_atom_residual variant_a_class_prototypes variant_b_global_vq variant_c_content_guided_spatial variant_d_vq_content_guided variant_e_latent_prototype_ot_queue variant_j_aux_hard_swd_queue_e3 variant_k_content_adaptive_vq_queue_e3 variant_l_content_adaptive_annealed_queue_e3 variant_m_style_gated_content_router_e3"
for v in $VARIANTS; do
    del_pts_in "distinct5_512_ema_${v}_b44_remote"
done

echo "=== 6. Clean .pt from path_kinetic (keep eval) ==="
del_pts_in aaai2027_path_kinetic_h_base_seed42_b44 \
           aaai2027_path_kinetic_h_base_seed42_b44_k000 \
           aaai2027_path_kinetic_h_base_seed42_b44_k025
rm -rf "$BASE/aaai2027_path_kinetic_h_base_seed42_b44_interrupted_20260603_1449" 2>/dev/null

echo "=== 7. Delete obsolete archive/analysis dirs ==="
rm -rf "$BASE/aaai2027_path_kinetic_packet" 2>/dev/null
rm -rf "$BASE/analysis" 2>/dev/null
rm -rf "$BASE/anchor_speed" 2>/dev/null
rm -rf "$BASE/baseline_from_scratch" 2>/dev/null
rm -rf "$BASE/baseline_repro" 2>/dev/null
rm -rf "$BASE/clean_baseline" 2>/dev/null
rm -rf "$BASE/concept_atom_m02_protocol" 2>/dev/null
rm -rf "$BASE/concept_atoms_mainline" 2>/dev/null
rm -rf "$BASE/concept_atoms_wsl_smoke" 2>/dev/null
rm -rf "$BASE/diagnostics" 2>/dev/null
rm -rf "$BASE/fisher_operator_consumer_probe" 2>/dev/null
rm -rf "$BASE/fisher_operator_consumer_probe_mds02" 2>/dev/null
rm -rf "$BASE/fisher_operator_tokenizer_probe" 2>/dev/null
rm -rf "$BASE/fisher_operator_tokenizer_probe_debug" 2>/dev/null
rm -rf "$BASE/freeze_stage" 2>/dev/null
rm -rf "$BASE/frontier" 2>/dev/null
rm -rf "$BASE/full_eval" 2>/dev/null
rm -rf "$BASE/inference" 2>/dev/null
rm -rf "$BASE/k2_repro" 2>/dev/null
rm -rf "$BASE/pareto_break_phase1_concept_ceiling" 2>/dev/null
rm -rf "$BASE/phase1_stylepush" 2>/dev/null
rm -rf "$BASE/phase1_variance" 2>/dev/null
rm -rf "$BASE/representation" 2>/dev/null
rm -rf "$BASE/rigid_spiral" 2>/dev/null
rm -rf "$BASE/sadd_exact_e3_saddsrc_8ep_20260528_231954" 2>/dev/null
rm -rf "$BASE/sadd_repro_38f_8ep_20260528_225252" 2>/dev/null
rm -rf "$BASE/speed_ladder" 2>/dev/null
rm -rf "$BASE/style_tokenizer_vocab_refit" 2>/dev/null
rm -rf "$BASE/tokenizer" 2>/dev/null
rm -rf "$BASE/tri_level_style" 2>/dev/null
rm -rf "$BASE/tri_level_style_ladder" 2>/dev/null
rm -rf "$BASE/multistep_mse" 2>/dev/null
rm -rf "$BASE/diffeomorphic_tangent_sweep" 2>/dev/null
rm -rf "$BASE/factorized_feature_status" 2>/dev/null
rm -rf "$BASE/factorized_tokenizer_status" 2>/dev/null

echo "=== 8. Clean wikiart stress .pt files ==="
del_pts_in wikiart_stress1_Color_Field_Painting__High_Renaissance__Mannerism_Late_Renaissance__Pop_Art__Realism_variant_f_b44_remote \
           wikiart_stress2_Abstract_Expressionism__Baroque__Cubism__Northern_Renaissance__Post_Impressionism_variant_f_b44_remote

echo "=== 9. Clean vae_backend probes ==="
del_dir vae_backend vae_backend_256_mse_controls vae_backend_256_probe vae_backend_256_sdxl_plain4_base

echo "=== 10. Clean concept_atom and budget experiments (.pt only, keep eval) ==="
del_pts_in distinct5_512_ema_baseline_direct_atom_residual_b40_remote \
           distinct5_512_ema_baseline_direct_atom_residual_b48_remote \
           distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote \
           distinct5_512_ema_variant_g_stratified_prototype_ot_queue_e3_b44_remote \
           distinct5_512_ema_variant_h_hard_explore_queue_e3_b44_remote \
           distinct5_512_ema_variant_i_dual_target_mix_queue_e3_b44_remote

echo "=== 11. Clean remaining logs ==="
rm -f "$BASE"/aaai2027_a2_softening_full_eval_repair.log 2>/dev/null
rm -f "$BASE"/aaai2027_a2_softening_full_eval_repairer.log 2>/dev/null

# Final size check
echo ""
echo "=== DONE ==="
du -sh "$BASE" 2>/dev/null
ls "$BASE" 2>/dev/null | wc -l
echo "directories remaining"
