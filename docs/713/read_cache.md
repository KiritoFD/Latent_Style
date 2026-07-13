# Read Cache

This caches the material already inspected so future sessions do not need to reread the same files.

## Paper Bundle

- `SchrodingerBridge/aaai2027_v4/paper.tex`
  - Presents WEAVE as latent Haar wavelet decoupling plus compact flow matching plus endpoint AdaIN.
  - Claims endpoint high-frequency style alignment is the primary style channel.
  - Describes final-step per-subband AdaIN on LH/HL/HH with LL excluded.
  - Important inconsistency to audit: paper line about RMSNorm conflicts with current code using `GroupNorm(1)`.

- `SchrodingerBridge/aaai2027_v4/README.md`
  - Paper bundle is self-contained.
  - Figure and table generation scripts live in the same folder.

## Architecture / Historical Docs

- `docs/621/architecture_audit.md`
  - Older 620 audit; useful for conceptual categories but not fully current.
  - Describes a DINO-patch style conditioner and richer block FiLM path that current `src/model.py` has since simplified.

- `docs/621/probe_design.md`
  - Good probe taxonomy: endpoint alpha, style sensitivity, layer statistics, style retention, attention patterns.
  - The probe design is still valid but needs updating to current T11 code paths.

- `SchrodingerBridge/mechanism_diagnosis/state/findings.jsonl`
  - M2: Flow-only CLIP-S=0.708, AdaIN-only=0.710, Full=0.727, synergy +0.017 over best individual.
  - M3: LH is primary style carrier, HL secondary, LL mostly content, HH frozen in the tested baseline.
  - M4: no AdaIN collapses style-transfer ratio; WCT improves spatial-fiber AdaIN only marginally; per-subband can be worse when it mismatches the training path.
  - Use these as mechanism evidence, not as final proof for every later T11 config.

## Current Code Paths

- `SchrodingerBridge/src/style.py`
  - External DINO patch/CLS input has been retired from the active conditioner.
  - `StyleConditioner` now projects learned `style_memory[style_id]` tokens.

- `SchrodingerBridge/src/model.py`
  - `forward()` decomposes content latent with single-level Haar DWT, stacks LL/LH/HL/HH, runs the shared backbone, then predicts subband velocities.
  - `enable_hh_head` is config-gated and commonly false; when false, HH is not transported by the ODE.
  - `integrate_transport()` runs spectral ODE steps and then applies endpoint AdaIN/WCT depending on config.
  - T11 / paper config uses `endpoint_adain_mode=per_subband_wct`, `endpoint_adain_only_last_step=true`, `endpoint_adain_scale_lh=0.3`, `endpoint_adain_scale_hl=0.3`, `endpoint_adain_scale_hh=0.5`, `endpoint_adain_scale_ll=0.0`.
  - Existing style-head code already supports the favored candidate:
    `style_velocity_head_enabled=true` plus `style_vhead_hf_nonzero_init=true` keeps LL conservative and uses nonzero style FiLM in LH/HL/HH heads.

- `SchrodingerBridge/src/blocks.py`
  - `_make_norm()` returns `GroupNorm(1, affine=False)`.
  - `ResidualBlock` has time AdaLN, self-attention, cross-attention, optional DWT-routed cross-attention, optional style AdaLN/Q-side AdaLN/channel gate paths.
  - T11 config keeps most learned style-injection extras off; cross-attention is the main learned style path.
  - DWT route only changes the cross-attn query features. The downstream velocity heads still read the same shared feature map `h`, so high-frequency-routed attention does not by itself force style influence to land in LH/HL/HH velocities.

- `SchrodingerBridge/src/flow.py`
  - Spectral FM loss uses `spectral_w_ll=0.3`, `spectral_w_lh=1.0`, `spectral_w_hl=1.0`, `spectral_w_hh=2.0` in the inspected T11 config.
  - HH loss only matters when the model predicts HH, i.e. when `enable_hh_head=true`.

## Evaluation Artifacts

- `SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/config.json`
  - Closest inspected config to `aaai_v4`.
  - Uses final-step per-subband WCT/AdaIN family.

- `SchrodingerBridge/exp/710_infra_t11_distinct5_5ep/full_eval_8step`
  - Has CLIP/LPIPS metrics and DINO metrics.
  - DINO-S is the primary style metric for decisions.

- `SchrodingerBridge/exp/710_baseline_weave_paper_reval`
  - Paper-style reevaluation artifact with DINO metrics.
  - Earlier quick averages showed higher DINO-S but lower DINO-C than the 5-epoch T11 artifact; compare only with matching evaluation protocol.

## Probe Artifacts Read

- `docs/713/probe_outputs/t11_ep5_style_path_n16_v2.md`
  - Established that endpoint WCT/AdaIN dominates LH/HL transfer and is the only HH mover.
  - Stronger endpoint scales were not automatically better.

- `docs/713/probe_outputs/t11_ep5_style_path_n32_gpu_pathsep.md`
  - Extended diagnosis on local GPU with path separation.
  - Learned path is active but not high-frequency-style-aligned: no-endpoint target style gives LH `0.0000`, HL `0.0104`, HH `0.0000`.
  - Style-swap velocity sensitivity is LL-heavy across time; at `t=0.5`, LL is about 12.8x LH and 12.5x HL.
  - Full cross-attn route is not a first fix because it increases content L2 without producing useful high-frequency transfer.
