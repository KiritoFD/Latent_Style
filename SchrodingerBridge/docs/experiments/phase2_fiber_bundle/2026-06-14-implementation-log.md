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

## SMoE Epoch 1 Read

- Full eval completed at `2026-06-14 05:10:55 Asia/Shanghai`; training resumed into epoch 2.
- Training time: `1450.6s`; full-eval wall time: `242.7s` from the curve, `265.1s` from trainer log.
- Transfer: `0.672379 / 0.333173`; all-pairs: `0.703540 / 0.329736`; identity: `0.828184 / 0.315987`.
- Matched delta against `k070 epoch_0003`: transfer `+0.000558` style and `+0.018555` LPIPS; all-pairs `+0.000306` style and `+0.017186` LPIPS.
- Decision: continue. This is early negative-LPIPS evidence, not convergence; the best point is the newest checkpoint and cannot close the family.

## SMoE Runtime Guard Incident

- At `2026-06-14 05:13:32 Asia/Shanghai`, the first SMoE process was stopped by the runtime guard with `used=11694MiB cap=11000MiB`.
- Diagnosis: this was caused by an older off-plan `phase2_i2sb_topo_anchor_sigma0p25_seed42_b30a1` task that restarted concurrently and held about `8.8GiB`.
- Action: stopped I2SB PID `2221`, verified GPU idle at `388MiB / 12288MiB`, then relaunched the same SMoE config.
- Relaunch: `2026-06-14 05:21 Asia/Shanghai`, resumed from `epoch_0001.pt` at epoch 2/global step `1574`; health memory `6776MiB`.
- Decision: treat the incident as an orchestration fault. Do not change SMoE mechanism or batch for this lane; keep convergence reads tied to the recovered single-lane run.

## SMoE Second Guard And Task Quarantine

- At `2026-06-14 05:39:01 Asia/Shanghai`, the second SMoE attempt was stopped by the runtime guard with `used=11772MiB cap=11000MiB`.
- Diagnosis: another historical I2SB one-shot task, `phase2_i2sb_topo_anchor_sigma0p10_warm_vel2_seed42_b30a1`, started at `05:38:47` and held about `8.6GiB`.
- Action: stopped the I2SB run and disabled every scheduled task containing `i2sb`, including historical train tasks and stale curve/watch helpers.
- Verification: GPU returned to `364MiB / 12288MiB`; no Python process remained; no `i2sb` task had a future trigger time.
- Relaunch: `2026-06-14 05:47 Asia/Shanghai`, same SMoE config, resumed from `epoch_0001.pt` at epoch 2/global step `1574`; health memory `6828MiB`.
- Decision: this remains an orchestration fault. Keep all I2SB tasks quarantined until SMoE closes; do not change SMoE mechanism or batch because no clean single-lane e2 has been observed yet.

## SMoE Epoch 2 Read

- Full eval completed at `2026-06-14 06:17:22 Asia/Shanghai`; training resumed into epoch 3.
- Training time: `1468.3s`; full-eval wall time: `240.4s`.
- Transfer: `0.670478 / 0.331323`; all-pairs: `0.701436 / 0.327738`; identity: `0.825270 / 0.313400`.
- Style above IDT: transfer `+0.030557`; all-pairs `+0.021314`.
- Matched delta against `k070 epoch_0003`: transfer `-0.001343` style and `+0.016705` LPIPS; all-pairs `-0.001797` style and `+0.015189` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.007511`, `routing_entropy=1.663507`, `effective_experts=5.321320`, `spatial_abs=0.896592`.
- Plot update: added `SMoE e2` to `plot_points.csv` and regenerated the AAAI2027 page-1 figure.
- Decision: continue to e3. e2 is not a promotion candidate; it improves LPIPS relative to e1 but is still dominated by the parent and now has lower style than parent.

## SMoE Epoch 3 Read

- Full eval completed at `2026-06-14 06:46:41 Asia/Shanghai`; training resumed into epoch 4.
- Training time: `1461.7s`; full-eval wall time: `244.4s`.
- Transfer: `0.668568 / 0.323320`; all-pairs: `0.699963 / 0.321329`; identity: `0.825543 / 0.313364`.
- Style above IDT: transfer `+0.028647`; all-pairs `+0.019840`.
- Matched delta against `k070 epoch_0003`: transfer `-0.003253` style and `+0.008702` LPIPS; all-pairs `-0.003271` style and `+0.008779` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.009242`, `routing_entropy=1.579277`, `effective_experts=4.887480`, `spatial_abs=0.877825`.
- Plot update: added the e3 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Decision: continue only under the formal curve rule. e3 is a lower-LPIPS candidate Pareto point but remains dominated by the matched parent and shows a continuing style bleed.

## SMoE Epoch 4 Read

- Full eval completed at `2026-06-14 07:16:31 Asia/Shanghai`; training resumed into epoch 5.
- Training time: `1477.0s`; full-eval wall time: `259.9s`.
- Transfer: `0.669259 / 0.322103`; all-pairs: `0.700884 / 0.317942`; identity: `0.827383 / 0.301297`.
- Style above IDT: transfer `+0.029339`; all-pairs `+0.020762`.
- Matched delta against `k070 epoch_0003`: transfer `-0.002561` style and `+0.007485` LPIPS; all-pairs `-0.002349` style and `+0.005392` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.010881`, `routing_entropy=1.752495`, `effective_experts=5.804411`, `spatial_abs=0.871335`.
- Plot update: added the e4 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Decision: continue only under the formal curve rule. e4 is the closest SMoE structural point so far but is still lower-style and higher-LPIPS than the matched parent.
