# Status

## 2026-07-13

- Created 713 workspace.
- Cached previously read paper/code/probe materials.
- Set DINO-S as the primary style metric for all model decisions.
- Added `SchrodingerBridge/tools/probe_713_style_path.py`.
- Ran local CPU probe on `710_infra_t11_distinct5_5ep/epoch_0005.pt`.
- Key probe findings:
  - Endpoint alignment is the high-frequency style path: endpoint/flow delta is ~1.34x on LH, ~1.32x on HL, and effectively all HH movement because HH flow is frozen.
  - Learned velocity style-swap sensitivity is much larger in LL absolute terms than LH/HL, so unconstrained learned style injection risks pushing style pressure back into structure.
  - `per_subband_wct` and configured mode are identical for T11; `per_subband_adain` is slightly weaker but slightly more content-preserving in latent L2.
  - Increasing per-band endpoint strength worsened latent style-transfer ratios and content L2 in the probe; do not blindly raise endpoint scales.
  - `configured_hh_off` keeps LH/HL ratios but removes HH transfer and lowers latent content L2; useful as a DINO-S/DINO-C control.
- Next: create a high-frequency-only learned-style training candidate and a no-HH endpoint eval control.

## 2026-07-13 GPU Path Separation

- User clarified local GPU is also available; local probe ran on RTX 4070 Laptop GPU.
- Extended `SchrodingerBridge/tools/probe_713_style_path.py` with:
  - style-id / endpoint-style-latent path separation,
  - cross-attn off and full-cross-attn controls,
  - style-swap velocity time sweep at `t=0.1/0.5/0.9`.
- Ran:
  - `docs/713/probe_outputs/t11_ep5_style_path_n32_gpu_pathsep.json`
  - `docs/713/probe_outputs/t11_ep5_style_path_n32_gpu_pathsep.md`
- Key added findings:
  - With endpoint disabled, learned target-style path has LH `0.0000`, HL `0.0104`, HH `0.0000` latent style-transfer ratio.
  - Keeping target endpoint latent fixed while changing only `style_id` causes large output L2 shifts, so learned style path is active, but the high-frequency style ratios remain endpoint-dominated.
  - Cross-attn-off no-endpoint is not materially worse than learned-target no-endpoint for high-frequency style ratio; the learned cross-attn effect is not well aligned with target style statistics.
  - Full cross-attn no-endpoint increases content L2 (`0.399168`) without solving high-frequency style transfer.
  - Style-swap sensitivity remains LL-heavy across time: at `t=0.5`, LL `0.662210` vs LH `0.051794` and HL `0.052887`.
- Updated `theory_map.md` and `probe_plan.md`.
- Next: create candidate configs for HF-proximal learned style injection and endpoint HH-off eval control, then validate config loading.
