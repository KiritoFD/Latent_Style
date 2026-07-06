# AAAI-27 Review Closure Status

Updated: 2026-06-08

This note maps the main concerns from [review.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/review.md) to the current paper-facing artifacts.

## W1. Strongest rows were not fully closed

Current status: `partially closed, materially improved`

- `LBM-Knee` now has:
  - transfer / all-pairs / identity replay in [paper_aaai2027.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/paper_aaai2027.tex)
  - target-pooled ArtFID in [aggregate_targetwise_artfid_fast_repro.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/lbm_knee_e13_artfid/aggregate_targetwise_artfid_fast_repro.json)
  - artifact-sensitive diagnostics in [distinct5_operating_point_selected_style_metrics.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_operating_point_selected_style_metrics.csv)
  - edge-purity diagnostics in [distinct5_operating_point_edge_purity.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_operating_point_edge_purity.csv)
  - non-CLIP probe in [distinct5_nonclip_style_probe.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.csv)
  - row-resampled stability in [distinct5_idt_bootstrap_extended.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_idt_bootstrap_extended.csv)
- `LBM-PS-v2` now has:
  - target-pooled ArtFID in [lbm_psv2_e13_artfid_fast_repro.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/lbm_psv2_e13_artfid_fast_repro.json)
  - artifact-sensitive diagnostics in [distinct5_operating_point_selected_style_metrics.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_operating_point_selected_style_metrics.csv)
  - non-CLIP probe in [distinct5_nonclip_style_probe.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.csv)
  - row-resampled stability in [distinct5_idt_bootstrap_extended.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_idt_bootstrap_extended.csv)

Open gap:

- no higher-grade blind preference / human-style closure yet

## W2. Variant definitions were too vague

Current status: `closed for the main paper-facing variants`

- hard variant wording now lives in [paper_aaai2027.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/paper_aaai2027.tex)
- exact reviewed configs:
  - [distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json)
  - [inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_anisostokes_queue_from_pattn_seed42_b8a2.json)
  - [inmortal_xpred_kmanifold_pattn_stokes002_from_pattn_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_stokes002_from_pattn_seed42_b16.json)

## W3. No main-paper qualitative figure

Current status: `closed`

- main-paper qualitative strip:
  - [fig_distinct5_qualitative_main.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_qualitative_main.png)
- generation script:
  - [scripts_gen_distinct5_qualitative_main.py](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/scripts_gen_distinct5_qualitative_main.py)

## W4. Seedream protocol unclear

Current status: `substantially closed`

- protocol note:
  - [2026-06-07-distinct5-seedream45-repaired750.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-distinct5-seedream45-repaired750.md)
- replacement map:
  - [2026-06-07-distinct5-seedream45-replacements.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-07-distinct5-seedream45-replacements.json)
- repaired assembly:
  - [assembly_manifest.json](/G:/GitHub/Latent_Style/Related_Works/baseline_pipeline/results/seedream45_api/distinct5_512_seedream45_windhub_20260607_repaired750/assembly_manifest.json)
- same-scope target-pooled ArtFID:
  - [aggregate_targetwise_artfid_fast_repro.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/local_eval/seedream_repaired750_artfid/aggregate_targetwise_artfid_fast_repro.json)

Open gap:

- no external or human blind preference score yet

## W5. Reproducibility was weak

Current status: `materially improved`

- active draft pointer:
  - [ACTIVE_DRAFT.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/ACTIVE_DRAFT.md)
- supplement:
  - [supplement_aaai2027.tex](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/supplement_aaai2027.tex)
  - [supplement_aaai2027.pdf](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/supplement_aaai2027.pdf)
- operating-point ledger:
  - [main_point_artifact_ledger.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/main_point_artifact_ledger.csv)
- shared manifest:
  - [operating_point_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/operating_point_manifest.csv)

Open gap:

- no single environment yaml exported yet

## Additional support added after the original review

- fixed-rule split support:
  - [2026-06-06-faraday-split1-paper-safe-packet.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-faraday-split1-paper-safe-packet.md)
  - [2026-06-06-faraday-split2-paper-safe-packet.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-06-faraday-split2-paper-safe-packet.md)
- held-out non-CLIP probe:
  - [distinct5_convnext_style_classifier_report.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_convnext_style_classifier_report.json)
  - [distinct5_nonclip_style_probe.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.md)
- extended row-resampled stability:
  - [distinct5_idt_bootstrap_extended.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_idt_bootstrap_extended.md)
- blind pairwise packet:
  - [blind_pairwise_v1/README.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/README.md)
  - [blind_pairwise_v1/exploratory_blind_audit.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/exploratory_blind_audit.md)

## Remaining highest-value gap

The single most valuable missing packet for a stronger acceptance case is still:

- higher-grade blind preference evidence
  - external VLM blind pairwise or human-style pairwise
