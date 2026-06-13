# 2026-06-14 Implementation Log

## Scope

Implemented the controlled-variable Fiber Bundle switches and the plot-update contract for the first two-day sweep:

- `model.solver_fiber_aligned=false` default-off Fiber-SDE gate-aligned noise path.
- `tokenizer_family=smoe_translator` default-off latent-only SMoE tokenizer.
- `semantic_supervision_family=fiberwise_swd` default-off loss-only fiberwise SWD.
- `tools/experiments/update_phase2_plot_points.py` plus `plot_points.csv` as the fixed homepage CLIP-style / LPIPS data source for completed experiments.

## Decisions

- Fiber-SDE only runs when `solver_family=solver_unsb_cycle` and `solver_stochastic_noise_scale>0`; legacy and deterministic behavior remains unchanged.
- Fiber gate source is only existing `StyleMaps.gate_16`. Gate handling is `sigmoid/clamp -> bilinear resize -> channel broadcast`; then per-sample RMS normalization preserves the configured sigma so isotropic and fiber-aligned controls differ by spatial direction rather than effective noise magnitude.
- SMoE keeps the current latent parser/PE/routing contract and initializes translation as exact identity (`I + delta`, delta zero). Low-rank mode is implemented as `I + A @ B`, also zero-initialized.
- Fiberwise SWD uses tokenizer routing attention `aux_16`; if missing under `fiberwise_swd`, training raises instead of silently falling back.
- Phase2 plot points are stored separately from the historical Distinct5 CSV and merged at render time.

## Next Queue

1. Use `k070 epoch_0003` as the deterministic topogate parent for Fiber-SDE because it is the current in-band structure point (`transfer LPIPS = 0.314618`).
2. Run Fiber-SDE eval-only matched scan from the same parent checkpoint:
   - deterministic parent
   - isotropic sigma `0.01, 0.02, 0.03, 0.05`
   - fiber-aligned sigma `0.01, 0.02, 0.03, 0.05`
   - optional low-noise health pair `0.005` is allowed only as a pre-scan check and is not the formal decision basis
3. After each eval output, update `plot_points.csv`, regenerate the homepage figures, and append matched deltas to `control_delta.csv`.
4. Launch SMoE-only training only after the Fiber-SDE scan decision note is written.

## Guardrails

- Remote eval/training cap remains `< 11.0 GiB`; any `> 11.3 GiB` is an exploded stop.
- No conclusion is made from absolute score only; Fiber-SDE conclusions compare candidate-minus-isotropic at the same sigma.
- DINO/VLM routes remain out of this queue.

## 2026-06-14 Homepage Figure Update

- Added the remote `k070` epoch `1-5` all-checkpoint curve to:
  - `curves/k070_epoch1_5_remote_clip_lpips_curve.csv`
  - `plot_points.csv`
- Added the active remote `pattn_enhanced_tok` epoch `1-10` all-checkpoint curve to:
  - `curves/pattn_enhanced_tok_epoch1_10_remote_clip_lpips_curve.csv`
  - `plot_points.csv`
- Regenerated the AAAI2027 page-1 summary figure:
  - `aaai2027/figures/fig_distinct5_page1_summary.png`
  - `aaai2027/figures/fig_distinct5_page1_summary.pdf`
- Current plotted `k070` read:
  - best transfer: `epoch_0001`, `0.672664 / 0.336344`, `style - IDT = +0.032743`
  - best LPIPS/structure point: `epoch_0003`, `0.671820 / 0.314618`, `style - IDT = +0.031900`
  - final retained point: `epoch_0005`, `0.671104 / 0.325637`, `style - IDT = +0.031183`
- Current plotted `pattn_enhanced_tok` read:
  - best transfer/all-pairs: `epoch_0002`, `0.673934 / 0.384340`, `style - IDT = +0.034013`
  - best all-pairs LPIPS: `epoch_0008`, `0.697299 / 0.358929` all-pairs and transfer `0.667859 / 0.361483`
  - latest settled before stop: `epoch_0010`, `0.670516 / 0.364172`, `style - IDT = +0.030595`
  - convergence read: e10 did not beat e2 style or e8 LPIPS; this line is held as non-promoted evidence while Fiber-SDE starts from the stronger `k070 epoch_0003` structure parent
- Label decision:
  - draw and connect every retained checkpoint
  - label only sparse key nodes on the page-1 panel to avoid collisions with the existing `K` and Lat-MAM labels

## 2026-06-14 Fiber-SDE Sigma 0.01 Matched Eval

- Parent: `k070 epoch_0003`, transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Isotropic control: transfer `0.671501 / 0.313795`, all-pairs `0.703024 / 0.311868`.
- Fiber-aligned candidate: transfer `0.671581 / 0.313762`, all-pairs `0.702954 / 0.311888`.
- Runtime observability:
  - isotropic: `solver_noise_scale=0.01`, `solver_fiber_gate_active=0.0`
  - fiber-aligned: `solver_noise_scale=0.01`, `solver_fiber_gate_active=1.0`, `solver_fiber_gate_mean≈0.652`
- Decision for this sigma: `inconclusive_tie`.
  - Transfer delta is slightly favorable: `+0.000080` style and `-0.000033` LPIPS.
  - All-pairs delta is slightly unfavorable: `-0.000070` style and `+0.000020` LPIPS.
  - The effect size is below a material threshold, so no promotion or rejection; continue the formal sigma sweep.

## 2026-06-14 Fiber-SDE Sigma 0.02 Matched Eval

- Isotropic control: transfer `0.672031 / 0.314990`, all-pairs `0.703432 / 0.313025`.
- Fiber-aligned candidate: transfer `0.671818 / 0.314936`, all-pairs `0.703320 / 0.313015`.
- Runtime observability:
  - isotropic: `solver_noise_scale=0.02`, `solver_fiber_gate_active=0.0`
  - fiber-aligned: `solver_noise_scale=0.02`, `solver_fiber_gate_active=1.0`, `solver_fiber_gate_mean≈0.652`
- Decision for this sigma: `conservative_not_promoted`.
  - Fiber-aligned reduces transfer LPIPS by `0.000054` and all-pairs LPIPS by `0.000010`.
  - It also lowers style by `0.000213` transfer and `0.000112` all-pairs, so the matched delta does not support gate-aligned noise as the style-injection mechanism at this sigma.

## 2026-06-14 Fiber-SDE Sigma 0.03 Matched Eval

- Isotropic control: transfer `0.673391 / 0.316894`, all-pairs `0.704514 / 0.314930`.
- Fiber-aligned candidate: transfer `0.673405 / 0.316883`, all-pairs `0.704633 / 0.314862`.
- Runtime observability:
  - isotropic: `solver_noise_scale=0.03`, `solver_fiber_gate_active=0.0`
  - fiber-aligned: `solver_noise_scale=0.03`, `solver_fiber_gate_active=1.0`, `solver_fiber_gate_mean≈0.652`
- Decision for this sigma: `marginal_positive_continue`.
  - Transfer delta favors fiber-aligned by `+0.000013` style and `-0.000010` LPIPS.
  - All-pairs delta favors fiber-aligned by `+0.000119` style and `-0.000068` LPIPS.
  - The best style in the Fiber-SDE scan so far is `sigma=0.03`, but LPIPS has risen to `0.3169`; continue `sigma=0.05` before closing the mechanism.

## 2026-06-14 Fiber-SDE Sigma 0.05 Matched Eval

- Isotropic control: transfer `0.675927 / 0.322953`, all-pairs `0.706639 / 0.320868`.
- Fiber-aligned candidate: transfer `0.675948 / 0.323189`, all-pairs `0.706763 / 0.321093`.
- Runtime observability:
  - isotropic: `solver_noise_scale=0.05`, `solver_fiber_gate_active=0.0`
  - fiber-aligned: `solver_noise_scale=0.05`, `solver_fiber_gate_active=1.0`, `solver_fiber_gate_mean≈0.652`
- Decision for this sigma: `style_upper_not_promoted`.
  - Style is the best in the scan, but transfer LPIPS rises to `0.323189`, and fiber-aligned adds `+0.000237` LPIPS versus isotropic.
  - Keep `sigma=0.05` as style-upper evidence only; it does not solve the `0.74 / 0.30` target.

## Fiber-SDE Closure

- Mechanism read: stochastic solver noise helps style more than deterministic parent, but the gain is small (`+0.0041` transfer style at `sigma=0.05`) and trades away structure.
- Gate-aligned noise read: not a strong positive mechanism. It is marginally favorable at `sigma=0.03`, but the effect is too small; at `sigma=0.05` it adds LPIPS cost.
- Stage decision: close Fiber-SDE eval-only scan as `not promoted as core`; retain `sigma=0.03` as balanced eval option and `sigma=0.05` as style-first diagnostic option.
- Next stage: start SMoE tokenizer training from the same `k070 epoch_0003` parent; keep solver/loss/topogate unchanged so tokenizer is the only core variable.

## SMoE Round Opened

- Config: `configs/aaai2027/phase2_smoe_translator_k070_e3_seed42_b12a1.json`.
- Parent: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Candidate switch: `model.tokenizer_family=smoe_translator`, `model.smoe_translation_rank=0`.
- Control discipline: no solver noise, no Fiberwise SWD, no DINO/VLM, no schedule changes.
- Remote eval discipline: full `CLIP-S + LPIPS` each epoch; use all-checkpoint curve for convergence and update the AAAI2027 page-1 plot before closure.

## SMoE Launch Health

- Launch time: `2026-06-14 04:42 Asia/Shanghai`.
- Task: `phase2-phase2_smoe_translator_k070_e3_seed42_b12a1-train`.
- Train PID: `456`.
- First GPU read: `6969 MiB / 12288 MiB`, `94%` utilization.
- The run is intentionally kept at inherited `b12a1` even though memory is below the preferred formal band, because changing batch would contaminate the tokenizer-only comparison.
- Smoke fix before launch: model/trainer structured-tokenizer routing now treats `smoe_translator` like `pure_latent_spatial` for legacy spatial-prior bypass and proximal structured-token selection.
