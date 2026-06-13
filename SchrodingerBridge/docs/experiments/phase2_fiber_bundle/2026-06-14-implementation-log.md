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

1. Close or explicitly stage-node the current `k070` parent.
2. Run Fiber-SDE eval-only matched scan from the same parent checkpoint:
   - deterministic parent
   - isotropic sigma `0.01, 0.02, 0.03, 0.05`
   - fiber-aligned sigma `0.01, 0.02, 0.03, 0.05`
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
- Added the active remote `pattn_enhanced_tok` epoch `1-9` all-checkpoint curve to:
  - `curves/pattn_enhanced_tok_epoch1_9_remote_clip_lpips_curve.csv`
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
  - latest settled: `epoch_0009`, `0.673337 / 0.384972`, `style - IDT = +0.033416`
  - convergence: not closed; `last_pareto = epoch_0008`, `tail_flat = false`
- Label decision:
  - draw and connect every retained checkpoint
  - label only sparse key nodes on the page-1 panel to avoid collisions with the existing `K` and Lat-MAM labels
