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

## RGB Calibration Eval-Only Sweep And Stop

- Motivation: because full training probes were too slow for the observed returns, pivot to an eval-only decoded RGB style-affine calibration switch before spending more remote training budget.
- Implementation: added default-off `full_eval.postprocess_mode=style_rgb_affine` with strength, mean strength, std strength, and reference-limit controls. The transform is applied after VAE decode and before metric computation, so legacy checkpoints and training remain unchanged unless the eval config enables it.
- Parent/control: `k070 epoch_0003`, transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Remote runs: `phase2_eval_rgbcal_s025_k070_e3`, `phase2_eval_rgbcal_s050_k070_e3`, and `phase2_eval_rgbcal_s075_k070_e3`; all were eval-only on the remote 3060, with no ckpt pullback.
- Runtime: each pass took about `154-155s`; GPU returned to idle after the sweep, and no Python training/eval process remained active.
- `s025`: transfer `0.654740 / 0.308625`, all-pairs `0.688668 / 0.305492`; delta versus parent was transfer `-0.017080` style and `-0.005993` LPIPS, all-pairs `-0.014566` style and `-0.007058` LPIPS.
- `s050`: transfer `0.645430 / 0.328338`, all-pairs `0.679670 / 0.324097`; this loses more style and also loses structure versus the parent.
- `s075`: transfer `0.641211 / 0.352983`, all-pairs `0.675231 / 0.347546`; this is dominated by both parent and `s025`.
- Plot update: appended the three-point `rgbcal_k070_e3` trace to `plot_points.csv`, added `RGBCal` to the AAAI2027 page-1 panel, and wrote the fixed curve CSV at `curves/rgbcal_k070_e3_eval_only_curve.csv`.
- Decision: `cost_positive_quality_negative`. The cheap screen is useful negative evidence, but simple exposure/contrast/stat matching suppresses CLIP-style and should not be promoted or used as a training target without a stronger style-specific objective.

## Fiber-SDE Fine Style-Ceiling Eval-Only Scan

- Trigger: the user called out that training was too slow and not worth the observed marginal gains. The lane was switched to cheap-first execution: no long training, no local GPU deep eval, and only remote eval-only matched controls.
- Infra fix: `run_phase2_eval_only_override.py` and `launch_remote_phase2_eval_only_override.py` now accept `--seed`; default remains `-1`, so legacy calls are unchanged. `update_phase2_plot_points.py` now reads CSV/JSON via `utf-8-sig` so PowerShell-pulled artifacts with BOM do not silently drop rows.
- Configs added: `phase2_fiber_sde_{iso,fiber}_sigma0p04.json`, `phase2_fiber_sde_{iso,fiber}_sigma0p06.json`, and `phase2_fiber_sde_{iso,fiber}_sigma0p08.json`.
- Parent/control: `k070 epoch_0003`; parent transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Remote execution: six sequential eval-only tasks under `exp/inmortal-exp/phase2_fiber_sde_fine_k070_e3`; seed fixed to `42`; no ckpts were pulled; PNG grids were left on remote.
- Runtime: each eval stayed around `2.6 GiB` during generation/eval and returned to idle after completion, well under the `<11 GiB` formal cap.
- `sigma=0.04`: isotropic transfer `0.674624 / 0.319512`, all-pairs `0.705757 / 0.317374`; fiber transfer `0.674520 / 0.319587`, all-pairs `0.705680 / 0.317446`.
- `sigma=0.06`: isotropic transfer `0.677688 / 0.327456`, all-pairs `0.708146 / 0.325220`; fiber transfer `0.677541 / 0.327567`, all-pairs `0.708073 / 0.325322`.
- `sigma=0.08`: isotropic transfer `0.681007 / 0.339036`, all-pairs `0.710653 / 0.336767`; fiber transfer `0.681075 / 0.339063`, all-pairs `0.710641 / 0.336797`.
- Matched delta: fiber-aligned is worse than isotropic at `0.04` and `0.06`; at `0.08` it gains only `+0.000068` transfer style for `+0.000027` LPIPS and is still slightly worse on all-pairs.
- Plot update: appended the full fine SDE trace to `plot_points.csv`, labeled only `SDE s0.08 ceiling`, regenerated the AAAI2027 page-1 figure, and wrote `curves/fiber_sde_fine_k070_e3_eval_only_curve.csv`.
- Decision: `style_ceiling_not_promoted`. More inference noise buys style, but the style/LPIPS slope is too poor and remains far from `0.74`; do not launch a long training lane from this evidence alone.

## Topology-Release Eval-Only Scan

- Trigger: after SDE and training probes were cost-negative, test the other guide-aligned cheap knob: reduce semantic topology blending at inference only.
- Parent/control: `k070 epoch_0003`, trained with `semantic_self_topology_blend=0.7`; parent transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Configs added: `phase2_eval_topology_release_blend0p5_k070_e3.json`, `phase2_eval_topology_release_blend0p3_k070_e3.json`, and `phase2_eval_topology_release_blend0p0_k070_e3.json`.
- Remote execution: three sequential eval-only runs under `exp/inmortal-exp/phase2_topology_release_k070_e3`; seed fixed to `42`; generated PNG grids were not saved; no ckpts were pulled.
- Runtime: eval-only memory stayed around `2.7 GiB` and returned to idle after each point.
- `blend=0.5`: transfer `0.671887 / 0.314608`, all-pairs `0.703252 / 0.312524`.
- `blend=0.3`: transfer `0.671899 / 0.314675`, all-pairs `0.703265 / 0.312592`.
- `blend=0.0`: transfer `0.671696 / 0.314660`, all-pairs `0.703089 / 0.312572`.
- Plot update: appended the three-point `topology_release_k070_e3` trace to `plot_points.csv`, regenerated the AAAI2027 page-1 figure, and wrote `curves/topology_release_k070_e3_eval_only_curve.csv`.
- Decision: `flat_no_training_value`. This knob is not the bottleneck; lowering topology blend at inference has no material style response and should not receive a long training lane by itself.

## Appearance-Blend Eval-Only Scan

- Trigger: the user called out that training is too slow and not worth it. The next action stayed eval-only and targeted the style signal path rather than launching another training probe.
- Parent/control: `k070 epoch_0003`; parent transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Configs added: `phase2_eval_appearance_blend0p0_k070_e3.json`, `phase2_eval_appearance_blend0p5_k070_e3.json`, and `phase2_eval_appearance_blend1p0_k070_e3.json`.
- Remote execution: three sequential eval-only runs under `exp/inmortal-exp/phase2_appearance_blend_k070_e3`; seed fixed to `42`; generated grids disabled; no ckpts were pulled.
- Runtime: each point took about `151s`; health-window GPU memory was about `2.4 GiB` and the remote GPU returned to idle after completion.
- `blend=0.0`: transfer `0.671748 / 0.314596`, all-pairs `0.703189 / 0.312540`.
- `blend=0.5`: transfer `0.671748 / 0.314596`, all-pairs `0.703189 / 0.312540`.
- `blend=1.0`: transfer `0.671744 / 0.314595`, all-pairs `0.703187 / 0.312539`.
- Matched delta: every point is noise-level and slightly below the parent in CLIP-style; the best LPIPS change is only about `-0.000023`.
- Plot update: appended the three-point `appearance_blend_k070_e3` trace to `plot_points.csv`, regenerated the AAAI2027 page-1 figure, and wrote `curves/appearance_blend_k070_e3_eval_only_curve.csv`.
- Decision: `flat_no_training_value`. Do not train this isolated output appearance blend knob; the next candidate must be either eval-only or a much cheaper probe with a visible style response before full-data training.

## PC-Lowpass Eval-Only Scan

- Trigger: PC solver remained the cheapest unclosed recommendation in `FIBER_BUNDLE_DESIGN.md` and `guide_for_running_codex.md`, while training-side probes are too slow for the observed returns.
- Parent/control: `k070 epoch_0003`; parent transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Configs added: `phase2_eval_pc_lowpass_step0p03_k070_e3.json`, `phase2_eval_pc_lowpass_step0p06_k070_e3.json`, and `phase2_eval_pc_lowpass_step0p10_k070_e3.json`.
- Remote execution: three sequential eval-only runs under `exp/inmortal-exp/phase2_pc_lowpass_k070_e3`; seed fixed to `42`; generated grids disabled; no ckpts were pulled.
- Runtime: each point took about `149-152s`; health-window GPU memory was about `2.4 GiB` and the remote GPU returned to idle after completion.
- `step=0.03`: transfer `0.671606 / 0.313594`, all-pairs `0.703035 / 0.311723`.
- `step=0.06`: transfer `0.671214 / 0.312733`, all-pairs `0.702729 / 0.311048`.
- `step=0.10`: transfer `0.671096 / 0.311748`, all-pairs `0.702628 / 0.310271`.
- Matched delta: increasing PC correction monotonically improves LPIPS but lowers style; the strongest correction gains `-0.002870` transfer LPIPS but loses `-0.000725` transfer style.
- Plot update: appended the three-point `pc_lowpass_k070_e3` trace to `plot_points.csv`, regenerated the AAAI2027 page-1 figure, and wrote `curves/pc_lowpass_k070_e3_eval_only_curve.csv`.
- Decision: `structure_repair_not_style_path`. Do not train this isolated PC path; keep it as a future safety correction if a style-strong mechanism needs LPIPS repair.

## Latent-Affine Eval-Only Scan

- Trigger: decoded RGB calibration was style-negative, but the user's brightness/contrast concern remained valid. The next screen moved the same idea into VAE latent space before decode, where style statistics can shift without post-decoded RGB clipping artifacts.
- Implementation: added default-off `full_eval.latent_postprocess_mode=style_latent_affine` with strength, mean strength, std strength, and reference-limit controls. The transform is applied after `lgt.generation(...)` and before VAE decode; legacy eval and training are unchanged unless the config enables it.
- Parent/control: `k070 epoch_0003`; parent transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- Configs added: `phase2_eval_latent_affine_s025_k070_e3.json`, `phase2_eval_latent_affine_s050_k070_e3.json`, and `phase2_eval_latent_affine_s075_k070_e3.json`.
- Remote execution: three sequential eval-only runs under `exp/inmortal-exp/phase2_latent_affine_k070_e3`; seed fixed to `42`; generated image grids disabled; no ckpts were pulled.
- Runtime: each point took about `158-161s`; health-window GPU memory stayed around `2.7 GiB` and the remote GPU returned to idle after the sweep.
- `s0.25`: transfer `0.674868 / 0.310584`, all-pairs `0.707268 / 0.306689`; delta versus parent was transfer `+0.003047` style and `-0.004034` LPIPS, all-pairs `+0.004034` style and `-0.005861` LPIPS.
- `s0.50`: transfer `0.680303 / 0.322202`, all-pairs `0.712764 / 0.316212`; delta versus parent was transfer `+0.008483` style and `+0.007584` LPIPS, all-pairs `+0.009530` style and `+0.003662` LPIPS.
- `s0.75`: transfer `0.685444 / 0.344580`, all-pairs `0.717593 / 0.336945`; this is the current phase2 style ceiling but LPIPS cost rises quickly.
- Plot update: appended the three-point `latent_affine_k070_e3` trace to `plot_points.csv`, labeled `LatAff s0.50` and `LatAff s0.75`, regenerated the filtered WikiArt-5 AAAI2027 page-1 figure, and wrote `curves/latent_affine_k070_e3_eval_only_curve.csv`.
- Decision: `balanced_style_candidate`. This is the first cheap screen in this phase that gives a material style response without training. Next run should refine the `0.25-0.60` band and optionally pair the best latent-affine point with PC-lowpass as a structure safety check before any new long training lane.

## Filtered WikiArt-5 Homepage Figure

- Trigger: the existing page-1 figure mixed old `distinct5-512` / `1000-per-style` training points with the new WikiArt-5 full-train surface, making visual comparison invalid.
- Added `aaai2027/scripts_gen_wikiart5_page1_summary.py` and fixed table `aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
- The rebuilt figure contains `159` points: `1` IDT test-only reference, `1` Seedream test-only reference, `92` SaMAM WikiArt-5 points, `3` SaMST WikiArt-5 points, and `62` phase2 full-notest points.
- The plot uses transfer `CLIP-S - IDT` on the y-axis and `1 - LPIPS` on the x-axis. Old mixed-source points are excluded by construction.
- The legend is placed in the lower right as requested. The compatibility filenames `fig_distinct5_page1_summary.*` and `fig_distinct5_page1_summary_clip_delta_idt.*` now contain the same filtered WikiArt-5 panel.

## Latent-Affine Refinement And PC Safety Check

- Trigger: `s0.50` was a balanced candidate but the style/LPIPS slope around `0.25-0.75` was still undersampled. The follow-up stayed eval-only and tested `s0.35`, `s0.45`, `s0.60`, plus `s0.50+PC0.10` and `s0.75+PC0.10`.
- Remote execution: five sequential eval-only runs under `exp/inmortal-exp/phase2_latent_affine_refine_k070_e3`; seed fixed to `42`; generated grids disabled; no ckpts were pulled.
- Runtime: health-window GPU memory stayed around `2.8 GiB`, and the remote GPU returned to idle after the sweep.
- `s0.35`: transfer `0.676781 / 0.313606`, all-pairs `0.709329 / 0.308847`; improves both style and LPIPS versus parent.
- `s0.45`: transfer `0.679110 / 0.318818`, all-pairs `0.711609 / 0.313230`; current balanced frontier.
- `s0.60`: transfer `0.682390 / 0.330056`, all-pairs `0.714810 / 0.323339`; style rises but LPIPS cost starts to dominate.
- `s0.50+PC0.10`: transfer `0.680160 / 0.320104`, all-pairs `0.712667 / 0.314519`; PC repairs LPIPS slightly versus pure `s0.50`.
- `s0.75+PC0.10`: transfer `0.685304 / 0.343517`, all-pairs `0.717560 / 0.336053`; PC does not rescue high-strength LPIPS cost.
- Plot update: appended the five-point refine/PC traces to `plot_points.csv`, labeled only `LatAff s0.45`, regenerated the filtered WikiArt-5 AAAI2027 page-1 figure, and wrote `curves/latent_affine_refine_k070_e3_eval_only_curve.csv`.
- Decision: `balanced_frontier`. The affine path is useful as eval-time amplification but is not enough to reach `0.74`; next cheap screen should modify the style generation path rather than increasing affine strength.

## Proximal Texture Probe And Eval-Speed Fix

- Trigger: after mixed body+decoder actuation stayed flat, the next controlled
  training probe moved style actuation into a spatial proximal texture residual
  path while keeping tokenizer, solver, TopoGate, and parent fixed.
- Implementation: added/validated `proximal_mode=crossattn_texture` training
  support under `freeze_mode=injection_only`, including explicit
  `proximal_target` loss binding for the sampled bridge path.
- Infra fix: `append_training_log()` now writes `proximal_target` and defaults
  future unknown log columns to `0.0`, preventing a metric-column addition from
  crashing an epoch after the expensive training pass.
- Eval optimization: exposed the existing ONNX VAE decoder infrastructure to
  the formal `run_evaluation.py` path through default-off switches
  `full_eval_vae_onnx_decoder`, `full_eval_vae_onnx_tensorrt`, and
  `full_eval_vae_onnx_trt_cache_dir`.
- Robustness: ONNX decode now pads fixed-batch tail batches and falls back to
  diffusers decode if provider/shape execution fails, so accelerator mismatch
  cannot invalidate training.
- Remote setup: installed user-site `onnxruntime-gpu`, `onnx`, and
  `onnxscript` in the existing WSL Python and exported a matched
  `ema_b2_32` decoder for the current `32x32 -> 256` latent eval path.
- Remote probe: ONNXRuntime used `CUDAExecutionProvider,CPUExecutionProvider`;
  matched decoder averaged about `80 ms / batch2`.
- e1 replay after the fix: transfer `0.672447 / 0.312461`, wall `70.07s`,
  VAE decode `24.62s`, LANCET generation `12.06s`, metric loop `11.20s`.
- Resume: the active run
  `aaai2027_phase2_actuation_proximal_texture_k070_e3_b16a2bf16_vlen010`
  resumed from local `epoch_0001.pt` into epoch 2, with health-window VRAM
  around `7.5 GiB`.
- 2026-06-16 eval-speed follow-up: full transfer eval through `epoch_0006`
  remained stable but slow, `66.38-71.50s` per checkpoint for `600` transfer
  pairs. Best full-transfer point so far is `epoch_0006` at
  `0.673384 / 0.327238`.
- Added default-compatible `training.full_eval_output_subdir` so training-time
  fast curves can be stored separately from full curves. This avoids mixing eval
  contracts in `clip_lpips_curve.csv`.
- Exported a matched `ema_b16_32` ONNX VAE decoder and switched the active
  training-time eval contract to `full_eval_fast10`: transfer-only,
  deterministic `10` source samples per style (`200` transfer pairs),
  `target_chunk_size=5`, `vae_decode_batch_size=16`, no PNG/grid, and
  GPU-kept generated tensors.
- Backfilled `epoch_0001` through `epoch_0006` under `full_eval_fast10`.
  Runtime dropped to `28.76-29.10s` per checkpoint, about `2.3x` faster than
  the full transfer curve. Best fast10 point is `epoch_0006` at
  `0.680404 / 0.331394`.
- Restarted the remote WSL lane from local `epoch_0006.pt`; resume log confirms
  epoch 7 and `global_step=354`. Health-window VRAM is about `7.36 GiB`.
- Decision constraint: use `full_eval_fast10` for live convergence and early
  stop decisions, but run full transfer confirmation for the selected best and
  final checkpoints before closing or promoting the family.
- Live status at `epoch_0009`: fast10 transfer `0.680954 / 0.334124`,
  `29.01s` eval wall, convergence still false because the best/Pareto point is
  newest. Do not stop yet; continue while style improves within the accepted
  LPIPS budget.
- Live status at `epoch_0011`: fast10 transfer `0.680411 / 0.334782`; best
  remains `epoch_0009`, `since_best=2`, `converged=false`. The family is likely
  plateauing, but it must reach patience before closure.
- Closure: stopped after `epoch_0014` because the fast10 curve reached
  `converged=true` with `since_best=5`; best remained `epoch_0009`.
- Full-transfer confirmation: `epoch_0009` scored `0.674190 / 0.329931`, and
  final `epoch_0014` scored `0.673760 / 0.331171`.
- Decision: `converged_not_promoted`. Proximal texture gives a small style
  response but does not beat the R16 style frontier; use it as negative/weak
  evidence for local endpoint residuals, not as a promoted route.

## Generated-Delta Observability

- Trigger: the `fiber.md` diagnosis specifically points at generated-delta rank
  collapse and high off-diagonal cosine as the actuation bottleneck; later
  actuation experiments need this recorded, not inferred after the fact.
- Implementation: added default-off `full_eval.delta_observability` /
  `training.full_eval_delta_observability`. The evaluator groups generated
  latent deltas by source image across target styles and writes effective rank,
  off-diagonal cosine, delta RMS, and delta absolute mean to
  `summary.json -> settings.generated_delta_observability`.
- Validation:
  - local helper smoke on orthogonal toy deltas returned effective rank `4.0`
    and off-diagonal cosine `0.0`.
  - remote small eval smoke on proximal `epoch_0009` with `max_src_samples=1`
    wrote `effective_rank_mean=1.60` and `offdiag_cosine_mean=0.356` over five
    source groups.
- Decision: enable this switch for the next generated-style-section experiment
  and include it in closure criteria.

## Fast Eval Source-Latent Cache

- Trigger: live `CLIP-S + LPIPS` eval was still too slow for per-checkpoint
  convergence reads. The fast10 path already avoided PNG roundtrips and used a
  fixed-batch ONNX VAE decoder, but every checkpoint still re-encoded the same
  source images through the VAE and ran LPIPS in small chunks.
- Implementation:
  - Added default-off `full_eval.source_latent_cache` /
    `training.full_eval_source_latent_cache`.
  - Cache key includes source path, file size, mtime, VAE id, image size, and
    latent scale, so dataset or VAE changes do not silently reuse stale latents.
  - Cached latents are still passed through `LGTInference.inversion()` to keep
    future non-identity inversion compatible.
  - `eval_lpips_chunk_size <= 0` now means “use the current metric batch size”;
    this reduces fragmented LPIPS/VGG calls in fast live eval while allowing an
    explicit smaller chunk if VRAM requires it.
- Remote validation on WSL, proximal `epoch_0009`, transfer-only
  `max_src_samples=1` (`20` generated pairs):
  - Pass 1 rebuilt source-latent cache: outer wall `33.35s`, evaluator wall
    `20.20s`, cache build `1.67s`, metric loop `0.90s`.
  - Pass 2 loaded source-latent cache: outer wall `22.29s`, evaluator wall
    `12.41s`, cache load `0.06s`, metric loop `0.64s`.
  - Effective speedup after cache warmup: about `33%` by process wall and
    about `39%` by evaluator-internal wall on the small smoke.
- Active fast/live contract update: the phase2 proximal fast config now enables
  `full_eval_source_latent_cache=true` and uses
  `full_eval_lpips_chunk_size=0`. Keep full-transfer confirmation for closure;
  use this path only for live convergence curves unless explicitly promoted to
  the formal full-board eval contract.

## Pre-Decoder Style Section Probe

- Trigger: output-side delta-basis and proximal texture residual both produced
  only mild style lift. The current diagnosis is that style freedom is being
  compressed near `dec_out`, so the next controlled mechanism must act before
  the final output head rather than after it.
- Implementation:
  - Added default-off `model.style_delta_mode=predec_section`.
  - The module builds a style-conditioned low-rank feature section from decoder
    features `h` and injects it before `dec_out`.
  - `style_section_out` and the final style-weight layer are zero initialized,
    making the new path an initial no-op.
  - Existing `freeze_mode=injection_only` now includes only the new
    `style_section_*` modules when this mode is enabled.
  - Runtime/training observability reuses `last_style_delta_debug` and records
    `style_predec_section_abs`, RMS, relative RMS, rank, basis magnitude, and
    weight magnitude.
- Local validation:
  - compile smoke passed for `model.py`, `trainer.py`, `config_schema.py`,
    `run.py`, and `run_evaluation.py`.
  - random forward smoke produced finite output and
    `style_predec_section_active=1`, while initial `section_abs/rms=0`.
- Remote validation:
  - 2-step WSL training smoke resumed the `k070 epoch_0003` parent with
    `missing=10`, exactly matching the new zero-init section tensors.
  - `freeze_mode=injection_only` selected only the ten `style_section_*`
    parameter tensors.
  - smoke peak memory was about `2.05GB`.
- Formal launch:
  - run id
    `aaai2027_phase2_actuation_predec_section_k070_e3_b16a2bf16_vlen010`.
  - remote log `logs/predec_section_20260616_012934.log`.
  - first health check: `3041 / 12288 MiB`, util `89%`, power `137.73W`.
  - low VRAM is expected for this injection-only mechanism and is not treated
    as a failure.

## Fast Eval VAE-Skip And Batch Probe

- Trigger: the live fast10 `CLIP-S + LPIPS` path was still too slow for tight
  per-checkpoint convergence reads.
- Implementation:
  - Added default-on `full_eval.skip_diffusers_vae_when_onnx` /
    `training.full_eval_skip_diffusers_vae_when_onnx`.
  - When ONNX VAE decode is enabled, source latent cache is enabled, and no
    latent postprocess needs VAE encoding, `run_evaluation.py` defers and can
    fully skip loading the diffusers VAE.
  - Summary settings now record `skip_diffusers_vae_when_onnx` and
    `diffusers_vae_loaded`.
- Validation:
  - e6 pre-decoder fast10 eval confirmed
    `skip_diffusers_vae_when_onnx=true`,
    `diffusers_vae_loaded=false`, and
    `source_latent_cache_status=loaded`.
  - e6 timing: `wall_total=26.85s`, `lancet_generation=5.93s`,
    `vae_decode=8.41s`, `eval_metrics_loop=3.62s`.
  - The gain versus warmed e2-e5 was only about `0.3-0.5s`, so diffusers VAE
    cold load was not the main bottleneck.
- Negative batch probe:
  - Temporarily increased live eval generation batch from `8` to `10` and
    metric batch from `16` to `32` for e8.
  - It reduced generation chunks from `7` to `5`, but internal wall worsened to
    `30.12s` and trainer wall to `44.5s`.
  - Reverted to generation `8` and metric `16`.
- Operational fixes:
  - Fixed an e6 crash where summary observability referenced `vae` after the
    generation phase had deleted it; the flag is now cached before releasing
    generation models.
  - Manually re-ran e6 eval and refreshed the curve.
  - e9 was later confirmed present after curve refresh; the pre-decoder live
    curve is complete through the current retained checkpoint.
- Decision: keep the safe ONNX decode-only VAE skip. Do not increase live eval
  batch on the 3060 lane. For a material speedup, use a separate fixed fast5
  live contract or implement a persistent evaluator that avoids per-checkpoint
  process/model cold starts.

## Fast Eval In-Process Probe

- Trigger: trainer wall time for fast10 eval was about `39.6-39.9s`, while
  `run_evaluation.py` reported internal `wall_total` around `26.5-26.8s`.
  The suspected gap was Python subprocess/import cold start.
- Implementation:
  - Added default-off `training.full_eval_in_process`.
  - `run.py` can now call `utils.run_evaluation.main(argv)` directly with the
    exact same CLI argument list, while preserving the legacy subprocess path.
  - `run_evaluation.main` now accepts an optional argv list and derives
    explicit CLI flags from that list, so in-process calls do not accidentally
    let checkpoint config override passed eval arguments.
- Validation:
  - Local compile smoke passed for `run.py`, `run_evaluation.py`, and
    `config_schema.py`.
  - Remote compile smoke passed after applying the minimal patch on the dirty
    WSL worktree.
  - Negative live probe on pre-decoder e16: in-process mode was activated, but
    trainer eval wall worsened from `39.9s` at e15 to `74.1s` at e16.
- Decision:
  - Keep the code path as a default-off diagnostic switch.
  - Do not enable it for live training on the remote 3060 lane.
  - Active pre-decoder config was reverted to subprocess eval and resumed from
    e16. The next clean speed target is a true persistent evaluator or faster
    VAE decode backend, not in-process eval inside the training process.

## Fast Eval HF CLIP Processor-Skip Probe

- Trigger: e18 fast10 eval still spent about `26.6s` internal wall per retained
  checkpoint. The live curve needs cheaper checkpoint-by-checkpoint reads, but
  the previous in-process and batch-size probes were negative.
- Implementation:
  - Added default-off `training.full_eval_hf_clip_skip_processor`.
  - Added `run_evaluation.py --clip_hf_skip_processor`.
  - When enabled for HF CLIP, the evaluator skips loading `CLIPProcessor` and
    keeps using tensor-native preprocessing from the CLIP model config plus the
    standard CLIP mean/std. This evaluator never uses tokenizer/PIL processor
    paths for image metrics, so the metric definition is unchanged for the
    current `openai/clip-vit-base-patch32` setup.
  - Summary settings now record `clip_backend` and
    `clip_hf_skip_processor`.
- Validation on remote WSL, pre-decoder e18, exact fast10 contract
  (`200` transfer generated images, no image save, ONNX decoder b16):
  - Baseline official e18: `wall_total=26.60s`, `eval_total=9.78s`,
    `generation=5.94s`, `vae_decode=8.47s`.
  - Skip-processor probe: `wall_total=26.10s`, `eval_total=9.39s`,
    `generation=5.90s`, `vae_decode=8.41s`,
    `eval_metrics_loop=3.61s`.
  - `generation_batch_size=10` probe was negative:
    `wall_total=27.22s`; fewer generation chunks were offset by slower ONNX
    decode scheduling.
  - `metric_batch_size=32` probe was negative:
    `wall_total=26.35s`, `eval_metrics_loop=3.68s`.
  - `vae_decode_batch_size=32` is invalid for the current ONNX decoder because
    it was exported for fixed batch `16`; it falls back to diffusers VAE and
    must not be used as a speed path.
- Decision:
  - Enable `full_eval_hf_clip_skip_processor=true` for the current fast10
    pre-decoder config.
  - Keep generation batch `8`, metric batch `16`, and ONNX decode batch `16`.
  - For material speedup beyond this small `~0.5s/ckpt` gain, the next infra
    task should be either a b32/b40 ONNX decoder export matched to the live
    eval shape, or a true persistent evaluator process that keeps CLIP/LPIPS
    loaded across checkpoints without sharing CUDA state with training.

## Fast Eval Runtime Model Cache Switch

- Trigger: clean I2SB fast10 live eval remains about `26s/ckpt` internally:
  `generation ~=5.4s`, `vae_decode ~=8.5s`, and `eval_total ~=9.5s`.
  The remaining cost includes repeated CLIP, LPIPS, and ONNX Runtime decoder
  construction for every checkpoint.
- Implementation:
  - Added default-off `training.full_eval_runtime_model_cache`.
  - `run.py` passes `--runtime_model_cache` to `run_evaluation.py` when the
    switch is enabled.
  - `run_evaluation.py` now has a process-local cache for:
    - `LPIPS(net=vgg)` keyed by device.
    - HF/OpenAI CLIP keyed by source, device, cache dir, network flag, and HF
      processor-skip flag.
    - `ORTVAEDecoder` keyed by ONNX path, device id, TensorRT flag, and TRT
      cache dir.
- Guardrail:
  - The cache is only useful when eval is in-process or inside a future
    persistent evaluator. It intentionally does not change metrics, generated
    samples, CLIP preprocessing, LPIPS inputs, or VAE decode math.
  - A previous in-process-only probe was negative on the 3060 lane, so this
    switch must only be promoted with runtime-cache evidence, not by
    in-process mode alone.
- Validation:
  - Local `py_compile` passed for `config_schema.py`, `run.py`, and
    `run_evaluation.py`.
  - Remote WSL target files were copied and `py_compile` passed with
    `/usr/bin/python`.
- Remote speed A/B on clean I2SB e5, exact fast10 contract:
  - Baseline subprocess: outer wall `43.99s`, summary wall `29.06s`,
    `eval_total=11.65s`.
  - In-process cache warmup: outer wall `31.45s`, summary wall `27.40s`,
    `eval_total=10.87s`.
  - In-process cache hot: outer wall `28.41s`, summary wall `24.37s`,
    `eval_total=8.37s`.
  - The hot run logs confirmed runtime-cache hits for `ORTVAEDecoder`,
    `LPIPS`, and HF `CLIP`.
- Decision:
  - Enable `full_eval_in_process=true` and
    `full_eval_runtime_model_cache=true` for the current fast10 contract and
    future matched training-time evals.
  - Keep the switches default-off for untested eval contracts, especially any
    setup that saves PNGs, enables IntroStyle/ArtFID/KID, or changes decoder
    batch shape.

## 2026-06-16 Fast Eval Decode Recheck

- Trigger: user reported training-time eval was still too slow.
- Current blend0p25 fast10 profile through e6:
  - typical wall `26-29s/ckpt`, VAE decode `8.3-8.5s`, generation `5.3-5.7s`,
    metric loop about `3.8s`.
  - e4 was a slow outlier (`34.62s` wall), but the metric definition and
    sample contract were unchanged.
- Additional probes on blend0p25 e2:
  - deferred decode with existing b16 ONNX: wall `26.34s`, VAE decode `8.37s`.
    Not promoted because the current `generation_batch_size=8`,
    `target_chunk_size=5`, and fixed batch-16 decoder already create nearly
    packed decode calls.
  - newly exported fixed b32 ONNX decoder for latent `32x32`: wall `84.36s`,
    VAE decode `65.30s`. Rejected.
- Decision:
  - Keep `ema_b16_32/decoder.onnx`.
  - Do not keep a deferred-decode switch in the main config/code path because
    the probe showed no material speedup under the current packed b16 contract.
  - Use the earlier validated hot path for live eval:
    `full_eval_in_process=true` and `full_eval_runtime_model_cache=true`.
  - Required guardrail for the next run: after the first cached eval, check
    remote VRAM. If eval-model cache pushes the restored training process near
    the 11GB band, turn runtime cache off again and fall back to subprocess
    eval.
- Probe artifact:
  `docs/experiments/phase2_fiber_bundle/eval/speed_probes/20260616_eval_decode_inprocess_probe.json`.

## 2026-06-16 LANCET Runtime Cache For Training-Time Eval

- Trigger: user reported eval still too slow after the decoder recheck.
- Diagnosis:
  - `full_eval_runtime_model_cache` cached CLIP, LPIPS, and ORT decoder, but
    in-process eval still constructed a fresh `LGTInference` model for every
    retained checkpoint.
  - This does not dominate the packed fast10 decode path, but it is pure fixed
    overhead and becomes expensive when every checkpoint is evaluated during
    training.
- Implementation:
  - Added an architecture/solver signature to `LGTInference`.
  - Added `LGTInference.reload_checkpoint(...)` so same-run checkpoints reuse
    the constructed model and only reload weights.
  - Added a guarded `lgt_inference` entry to `run_evaluation.py`'s existing
    process-local runtime cache. If the signature changes, it falls back to a
    full rebuild instead of unsafe reuse.
  - Updated
    `configs/aaai2027/phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010.json`
    to the fast10 eval contract: b16 ONNX decoder, in-process runtime cache,
    source latent cache, CLIP processor skip, transfer-only, and no PNG/grid.
- Remote WSL smoke:
  - Pair:
    `aaai2027_phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010/epoch_0001.pt`
    then `epoch_0002.pt`.
  - Small transfer-only probe used `max_src_samples=1` per style, producing 20
    transfer images.
  - Cold first ckpt: wall `20.57s`, `load_lancet=1.09s`,
    `load_vae_onnx_decoder=1.18s`, `eval_total=12.58s`.
  - Warm second ckpt: wall `4.66s`, `reload_lancet=0.56s`,
    `load_vae_onnx_decoder=0.006s`, `eval_total=1.03s`.
  - Logs confirmed cache hits for LANCET reload, ORT decoder, LPIPS, and CLIP.
- Decision:
  - Promote LANCET runtime reuse for in-process training-time fast eval.
  - Keep all runtime caches default-off outside explicit configs; metric math
    and eval sample contract are unchanged.
- Probe artifact:
  `docs/experiments/phase2_fiber_bundle/eval/speed_probes/20260616_lgt_runtime_cache_probe.json`.
