# Loss Pruning Phase 1

Date: 2026-06-10

This note records the first evidence-driven cleanup pass on the bridge loss stack.

## Deleted From Active Code

The following bridge-loss families were removed from active code paths:

- `content_anchor`
- `edge_anchor`
- `semantic_entropy`
- `divergence`
- `feature_riemannian`
- `kantorovich`
- `phase_separation`
- `fourier_phase_lock`
- `head_tax`
  - `w_head_color_tv`
  - `w_head_color_energy`
  - `w_head_amp_energy`
  - `w_warp_curl_reward`

The cleaned runtime also now keeps only the surviving proximal branch:

- keep:
  - `proximal_mode = crossattn_texture`
- retire from active runtime:
  - `highpass_residual`
  - `normfree_modulation`
  - `dualpath_texture`
  - `dualpath_spatialtexture`

These keys were also added to the retired-bridge-key ignore list in
[config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py), so older configs can still load without crashing, but the retired terms no longer participate in the current training objective.

## Why These Were Safe To Remove

1. They do not appear as positive paper-facing mechanism winners in:
   - [aaai2027_inmortal_results_master.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/aaai2027_inmortal_results_master.csv)
2. They do not appear as necessary components in the currently surviving frontier families:
   - `terminal_swd`
   - `xpred`
   - `K_manifold`
   - `crossattn_texture`
   - `anisotropic`
   - `stokes`
3. In the current config inventory, these deleted terms were either:
   - absent from active `aaai2027` configs
   - or present only as `0.0` legacy tokenizer-era placeholders

## What Was Explicitly Kept

The following mechanism families remain active because current evidence still says they matter:

- `terminal_swd`
- `w_kinetic`
- `w_curvature`
- `w_anisotropic_kinetic`
- `w_stokes_viscous`
- `w_style_energy_floor`
- `w_style_contrastive`
- `w_residual_style_direction`
- `w_spectral_amplitude`
- `w_generated_delta_diversity`
- `teacher_alignment`
- current proximal trust / clamp logic
- `crossattn_texture` late branch

## Current Interpretation

This is not the final cleanup pass.

It is the first low-risk pass that removes:

- long-zero terms
- no-evidence terms
- and loss branches that were only increasing conceptual and implementation complexity

without touching the currently evidence-bearing structure/style tradeoff families.
