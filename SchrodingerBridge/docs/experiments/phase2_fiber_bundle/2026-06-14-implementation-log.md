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

## SMoE Epoch 5 Read

- Full eval completed at `2026-06-14 07:47:53 Asia/Shanghai`; training resumed into epoch 6.
- Training time: `1564.0s`; full-eval wall time: `261.0s`.
- Transfer: `0.670998 / 0.326648`; all-pairs: `0.702158 / 0.323817`; identity: `0.826800 / 0.312494`.
- Style above IDT: transfer `+0.031077`; all-pairs `+0.022036`.
- Matched delta against `k070 epoch_0003`: transfer `-0.000822` style and `+0.012030` LPIPS; all-pairs `-0.001075` style and `+0.011268` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.012272`, `routing_entropy=1.802536`, `effective_experts=6.149995`, `spatial_abs=0.903199`.
- Plot update: added the e5 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Decision: continue only under the formal curve rule. e5 recovers style from e4 but gives back LPIPS and still does not beat the matched parent.

## SMoE Epoch 6 Read

- Full eval completed at `2026-06-14 08:19:03 Asia/Shanghai`; training resumed into epoch 7.
- Training time: `1558.2s`; full-eval wall time: `256.6s`.
- Transfer: `0.669504 / 0.335022`; all-pairs: `0.700534 / 0.330586`; identity: `0.824652 / 0.312843`.
- Style above IDT: transfer `+0.029583`; all-pairs `+0.020411`.
- Matched delta against `k070 epoch_0003`: transfer `-0.002316` style and `+0.020404` LPIPS; all-pairs `-0.002700` style and `+0.018037` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.013472`, `routing_entropy=1.453477`, `effective_experts=4.380065`, `spatial_abs=0.885638`.
- Plot update: added the e6 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Decision: continue under the formal curve rule. e6 is the first non-Pareto tail point; if the next retained checkpoints also fail to create a Pareto point, SMoE-only can close as negative evidence.

## SMoE Epoch 7 Read

- Full eval completed at `2026-06-14 08:49:20 Asia/Shanghai`; training resumed into epoch 8.
- Training time: `1511.5s`; full-eval wall time: `249.8s`.
- Transfer: `0.670730 / 0.333296`; all-pairs: `0.701995 / 0.328815`; identity: `0.827052 / 0.310890`.
- Style above IDT: transfer `+0.030809`; all-pairs `+0.021872`.
- Matched delta against `k070 epoch_0003`: transfer `-0.001090` style and `+0.018678` LPIPS; all-pairs `-0.001239` style and `+0.016266` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.014656`, `routing_entropy=1.694707`, `effective_experts=5.537481`, `spatial_abs=0.854391`.
- Plot update: added the e7 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `since_last_pareto=2`, `tail_flat=true`.
- Decision: continue. e7 is the second tail point after the e5 Pareto point, so SMoE-only cannot close until at least two more retained checkpoints fail to create a new Pareto point.

## SMoE Epoch 8 Read

- Full eval completed at `2026-06-14 09:19:29 Asia/Shanghai`; training resumed into epoch 9.
- Training time: `1514.0s`; full-eval wall time: `241.0s`.
- Transfer: `0.669985 / 0.317808`; all-pairs: `0.701901 / 0.315335`; identity: `0.829565 / 0.305445`.
- Style above IDT: transfer `+0.030065`; all-pairs `+0.021779`.
- Matched delta against `k070 epoch_0003`: transfer `-0.001835` style and `+0.003189` LPIPS; all-pairs `-0.001332` style and `+0.002785` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.015515`, `routing_entropy=1.681098`, `effective_experts=5.435514`, `spatial_abs=0.806139`.
- Plot update: added the e8 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `since_last_pareto=0`, `tail_flat=true`, and `epoch_0008` is a new candidate-curve Pareto point.
- Decision: continue. e8 materially improves structure versus e6/e7, but it is still lower-style and slightly higher-LPIPS than the parent, so SMoE-only remains unpromoted unless later epochs recover style at this structure level.

## SMoE Epoch 9 Read

- Full eval completed at `2026-06-14 09:48:45 Asia/Shanghai`; training resumed into epoch 10.
- Training time: `1465.7s`; full-eval wall time: `242.5s`.
- Transfer: `0.672774 / 0.327155`; all-pairs: `0.704251 / 0.322688`; identity: `0.830159 / 0.304821`.
- Style above IDT: transfer `+0.032853`; all-pairs `+0.024128`.
- Matched delta against `k070 epoch_0003`: transfer `+0.000953` style and `+0.012536` LPIPS; all-pairs `+0.001017` style and `+0.010138` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.016185`, `routing_entropy=1.583672`, `effective_experts=4.917878`, `spatial_abs=0.826611`.
- Plot update: added the e9 point to `plot_points.csv`, labeled it `e9 style`, and regenerated the AAAI2027 page-1 figure.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `best_in_newest_2=true`.
- Decision: continue. e9 is useful positive tokenizer evidence because it beats parent style under the matched control, but it is not promotable until a later checkpoint recovers LPIPS toward the e8/parent band.

## SMoE Epoch 10 Read

- Full eval completed at `2026-06-14 10:19:29 Asia/Shanghai`; training resumed into epoch 11.
- Training time: `1528.6s`; full-eval wall time: `257.4s`.
- Transfer: `0.670014 / 0.323925`; all-pairs: `0.701628 / 0.320149`; identity: `0.828084 / 0.305047`.
- Style above IDT: transfer `+0.030093`; all-pairs `+0.021506`.
- Matched delta against `k070 epoch_0003`: transfer `-0.001806` style and `+0.009307` LPIPS; all-pairs `-0.001605` style and `+0.007600` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.016836`, `routing_entropy=1.671797`, `effective_experts=5.370153`, `spatial_abs=0.841168`.
- Plot update: added the e10 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `since_last_pareto=0`, and `epoch_0010` is a new candidate-curve Pareto point.
- Decision: continue. e10 shows structure recovery from e9 but gives up the style lift; SMoE-only still needs a point that combines e9 style with e8/e10 structure before it can be promoted.

## SMoE Third Guard: Safe-Rescan Quarantine

- At `2026-06-14 10:36:08 Asia/Shanghai`, epoch 11 was stopped by the runtime guard with `used=11925MiB cap=11000MiB`.
- Diagnosis: old scheduled task `phase2_vel_tok32_safe_rescan_r1_seed42_b20a1` started at `10:35:51` and overlapped the SMoE lane.
- Action: killed the safe-rescan process, verified GPU idle at `533MiB / 12288MiB`, and disabled old safe-rescan plus structure/topogate scheduled tasks that can restart without explicit approval.
- Relaunch: `2026-06-14 10:46 Asia/Shanghai`, same SMoE task and config, resumed from `epoch_0010.pt` at epoch 11/global step `15740`; health memory `4549MiB`.
- Decision: the partial e11 before the guard has no eval/checkpoint and is invalid. Continue from e10 under the same matched-control contract; do not change mechanism parameters or batch.

## SMoE Epoch 11 Read

- Full eval completed at `2026-06-14 11:17:19 Asia/Shanghai`; training resumed into epoch 12.
- Training time: `1557.8s`; full-eval wall time: `253.4s`.
- Transfer: `0.669667 / 0.327548`; all-pairs: `0.701142 / 0.324272`; identity: `0.827041 / 0.311168`.
- Style above IDT: transfer `+0.029747`; all-pairs `+0.021019`.
- Matched delta against `k070 epoch_0003`: transfer `-0.002153` style and `+0.012929` LPIPS; all-pairs `-0.002091` style and `+0.011722` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.017423`, `routing_entropy=1.627435`, `effective_experts=5.111550`, `spatial_abs=0.896498`.
- Plot update: added the e11 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `last_pareto_epoch=epoch_0010`, `since_last_pareto=1`, `tail_flat=true`.
- Decision: continue. e11 is a clear non-Pareto tail point, but e10 reset the Pareto patience, so the SMoE-only family cannot close yet.

## SMoE Epoch 12 Read

- Full eval completed at `2026-06-14 11:48:03 Asia/Shanghai`; training resumed into epoch 13.
- Training time: `1535.2s`; full-eval wall time: `254.0s`.
- Transfer: `0.670048 / 0.331773`; all-pairs: `0.701154 / 0.328471`; identity: `0.825575 / 0.315265`.
- Style above IDT: transfer `+0.030127`; all-pairs `+0.021031`.
- Matched delta against `k070 epoch_0003`: transfer `-0.001772` style and `+0.017155` LPIPS; all-pairs `-0.002080` style and `+0.015922` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.017843`, `routing_entropy=1.635242`, `effective_experts=5.161444`, `spatial_abs=0.850654`.
- Plot update: added the e12 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `last_pareto_epoch=epoch_0010`, `since_last_pareto=2`, `tail_flat=true`.
- Decision: continue. e12 is the second post-e10 non-Pareto tail point and does not improve either the e9 style or the e8/e10 structure tradeoff.

## SMoE Epoch 13 Read

- Full eval completed at `2026-06-14 12:19:26 Asia/Shanghai`; training resumed into epoch 14.
- Training time: `1556.4s`; full-eval wall time: `268.5s`.
- Transfer: `0.671565 / 0.336139`; all-pairs: `0.702369 / 0.332186`; identity: `0.825586 / 0.316374`.
- Style above IDT: transfer `+0.031644`; all-pairs `+0.022246`.
- Matched delta against `k070 epoch_0003`: transfer `-0.000256` style and `+0.021521` LPIPS; all-pairs `-0.000865` style and `+0.019636` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.018194`, `routing_entropy=1.585930`, `effective_experts=4.921388`, `spatial_abs=0.836015`.
- Plot update: added the e13 point to `plot_points.csv` and regenerated the AAAI2027 page-1 figure; the point is unlabeled to keep the panel readable.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `last_pareto_epoch=epoch_0010`, `since_last_pareto=3`, `tail_flat=true`.
- Decision: continue to e14. e13 style is close to parent, but the LPIPS cost is too high and it is still non-Pareto.

## SMoE Epoch 14 Read

- Full eval completed at `2026-06-14 12:51:38 Asia/Shanghai`; training resumed into epoch 15.
- Training time: `1603.7s`; full-eval wall time: `266.5s`.
- Transfer: `0.672185 / 0.324834`; all-pairs: `0.703218 / 0.322686`; identity: `0.827351 / 0.314091`.
- Style above IDT: transfer `+0.032264`; all-pairs `+0.023095`.
- Matched delta against `k070 epoch_0003`: transfer `+0.000365` style and `+0.010216` LPIPS; all-pairs `-0.000015` style and `+0.010136` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.018548`, `routing_entropy=1.495014`, `effective_experts=4.496835`, `spatial_abs=0.819658`.
- Plot update: added the e14 point to `plot_points.csv`, labeled it `e14 pareto`, and regenerated the AAAI2027 page-1 figure.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `last_pareto_epoch=epoch_0014`, `since_last_pareto=0`, `tail_flat=true`.
- Decision: continue. e14 resets formal patience because it is a new candidate-curve Pareto point, but it is still not promotable against the matched parent because the LPIPS cost is about `+0.010`.

## SMoE Epoch 15 Read And Stop

- Full eval completed at `2026-06-14 13:23:45 Asia/Shanghai`; the remote training process was stopped after e15 eval artifacts were confirmed.
- Training time: `1597.6s`; full-eval wall time: `268.7s` from summary, `294.9s` from trainer log.
- Transfer: `0.671284 / 0.333647`; all-pairs: `0.702173 / 0.330398`; identity: `0.825728 / 0.317400`.
- Style above IDT: transfer `+0.031364`; all-pairs `+0.022050`.
- Matched delta against `k070 epoch_0003`: transfer `-0.000536` style and `+0.019029` LPIPS; all-pairs `-0.001061` style and `+0.017848` LPIPS.
- Runtime observability: `translation_delta_from_identity=0.018724`, `routing_entropy=1.541963`, `effective_experts=4.716263`, `spatial_abs=0.813047`.
- Plot update: added the e15 point to `plot_points.csv`, labeled it `stop e15`, and regenerated the AAAI2027 page-1 figure.
- Convergence read: `converged=false`, `best_epoch=epoch_0009`, `last_pareto_epoch=epoch_0014`, `since_last_pareto=1`, `tail_flat=true`.
- Decision: close as `cost_stopped_not_promoted`. The line is not automatically converged because e14 reset patience, but additional epochs are not worth the cost; do not launch `SMoE + fiberwise_swd` on this parent.

## K070_KIN070 Launch And Cost Stop

- Launch: `phase2-structure-k070-kin070-train` on the remote 3060 after local smoke passed.
- Mechanism delta: only `w_kinetic: 0.85 -> 0.70`; tokenizer, solver, topogate, appearance path, and dataset stayed unchanged.
- Runtime read: epoch `1/24` reached about `9%` after roughly `2.2min`, with throughput around `1.1 it/s`; projected full epoch time remained about `24-25min` before full eval.
- VRAM read: about `6.9 GiB / 12 GiB`, under the `<11.0 GiB` formal cap.
- Stop: process terminated before the first checkpoint/eval because the full-data training cost is not justified for this single knob.
- Eval/plot status: no `CLIP-S + LPIPS` point exists, so nothing is appended to `plot_points.csv` and the AAAI2027 page-1 figure is unchanged.
- Decision: `cost_stopped_no_eval`. This is a cost/value stop, not a model-quality conclusion. Future style-release training should use shorter virtual-length probes or be skipped in favor of eval-only tests unless there is stronger prior evidence.

## Short-Probe Replan

- Added `phase2_vel_tok32_safe_semantic_topogate_k070_kin070_vlen010_seed42_b12a1.json`.
- Probe contract:
  - mechanism stays fixed to `w_kinetic=0.70` behind the existing `k070` topology-release setting;
  - parent/eval surface stays inherited from the full-data `k070_kin070` config;
  - only `data.virtual_length_multiplier=0.10`, `training.num_epochs=6`, and the output root change.
- Rationale: screen the kinetic-release direction with retained per-epoch `CLIP-S + LPIPS` points before spending another full-data remote run.
- Manifest: `short_probe_manifest.csv` records the prepared probe and launch/eval roots.

## K070_KIN070_VLEN010 Short Probe And Stop

- Launch: `phase2-k070-kin070-vlen010-probe` on the remote 3060 with `virtual_length_multiplier=0.10`, `num_epochs=6`, `save_interval=1`, and full eval after every retained epoch.
- Parent/control: `k070 epoch_0003`; only the kinetic-release knob stayed changed through inherited `w_kinetic=0.70`.
- Smoke: local config smoke passed and wrote `eval/k070_kin070_vlen010_smoke.json`; DINO runtime was not required.
- Runtime: GPU stayed under cap; train peak was about `4.91 / 5.70-5.89 GiB`, so this was not a memory failure.
- Epoch 1: train `148.3s`, eval wall `250.0s`; transfer `0.672150 / 0.336911`, all-pairs `0.702885 / 0.332846`.
- Epoch 2: train `148.9s`, eval wall `248.8s`; transfer `0.674131 / 0.340593`, all-pairs `0.704226 / 0.337494`.
- Epoch 3: train `147.3s`, eval wall `248.0s`; transfer `0.673330 / 0.352338`, all-pairs `0.703133 / 0.347736`.
- Matched best read: e2 beats `k070 epoch_0003` by only `+0.002310` transfer style and `+0.000993` all-pairs style, while worsening LPIPS by `+0.025975` and `+0.024944`.
- Stop: after e3 eval settled, the remote process was terminated before continuing epoch 4; final status was `settled_no_live_process` and GPU returned to idle.
- Pullback: copied remote `clip_lpips_curve.csv`, per-epoch `summary.json`/`metrics.csv`, and training CSV into this round folder; no ckpts were pulled.
- Plot update: appended unlabeled e1-e3 full/transfer points to `plot_points.csv` with exact train seconds and regenerated the AAAI2027 page-1 figure.
- Decision: `cost_stopped_negative`. Do not spend a full-data lane on this kinetic-release setting; the style gain is too small for the LPIPS and wall-clock cost.
