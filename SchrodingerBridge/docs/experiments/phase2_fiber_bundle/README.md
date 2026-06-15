# Phase2 Fiber Bundle Experiment Ledger

This folder stores the controlled-variable Fiber Bundle sweep artifacts.

## Live Files

- `plot_points.csv`: fixed input table for the homepage CLIP-style / LPIPS progress plots. Every closed experiment must append or update its full/all-pairs and transfer rows here before the closure note is final.
- `SchrodingerBridge/aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`: fixed paper-facing page-1 plot table. It filters the homepage panel to the current WikiArt-5 full-train surface plus test-only references and intentionally excludes old `distinct5-512` / `1000-per-style` training points.
- `curves/`: raw per-run all-checkpoint CLIP-S / LPIPS curves copied from remote eval artifacts before they are normalized into `plot_points.csv`.
- `smoe_training_manifest.csv`: Round-2 SMoE-only launch and closure status.
- `short_probe_manifest.csv`: cost-controlled short virtual-length probes used to screen style-release ideas before any full-data relaunch.

## Current Homepage Overlay

- `k070` epoch `1-5`, `pattn_enhanced_tok` epoch `1-10`, Fiber-SDE `sigma=0.01/0.02/0.03/0.05` plus the fine `0.04/0.06/0.08` style-ceiling extension, SMoE epoch `1-15`, the short `k070_kin070_vlen010` kinetic-release probe epoch `1-3`, the eval-only `rgbcal_k070_e3` scan, the eval-only `topology_release_k070_e3` blend scan, the eval-only `appearance_blend_k070_e3` output-affine scan, the eval-only `pc_lowpass_k070_e3` solver scan, the eval-only `latent_affine_k070_e3` latent postprocess scan, and the `latent_affine_refine_k070_e3` narrow/PC follow-up are plotted on the AAAI2027 page-1 IDT/SaMAM/Seedream CLIP-S / LPIPS panel.
- The current page-1 panel is rendered from `aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`, not from the older mixed-source `distinct5` aggregate tables.
- The trace uses transfer `CLIP-S - IDT` on the y-axis and `1 - LPIPS` on the x-axis.
- All retained checkpoints are drawn and connected.
- Labels are sparse by design:
  - `k070`: `e1` and `e3 best LPIPS`
  - `pattn_enhanced_tok`: `e2 best style` and `e8 low LPIPS`
  - `smoe_translator_k070_e3`: `SMoE e1`, `SMoE e2`, `e9 style`, `e14 pareto`, and `stop e15`; other points are plotted but unlabeled to avoid collisions.
  - `fiber_sde_fine_k070_e3`: only `SDE s0.08 ceiling` is labeled; the full `0.04/0.06/0.08` trace is still plotted.
  - `latent_affine_k070_e3`: `LatAff s0.50` and `LatAff s0.75` are labeled; `s0.25` remains plotted as the balanced positive but unlabeled.
  - `latent_affine_refine_k070_e3`: `LatAff s0.45` is labeled as the current balanced frontier; PC combo points are plotted as a separate trace but unlabeled.
- Source curve:
  - [k070_epoch1_5_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/k070_epoch1_5_remote_clip_lpips_curve.csv)
  - [pattn_enhanced_tok_epoch1_10_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/pattn_enhanced_tok_epoch1_10_remote_clip_lpips_curve.csv)
  - [eval/fiber_sde_sigma0p01/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p01/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p01/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p01/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p02/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p02/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p02/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p02/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p03/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p03/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p03/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p03/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p05/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p05/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p05/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p05/fiber_aligned/summary.json)
  - [smoe_translator_k070_e3_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/smoe_translator_k070_e3_remote_clip_lpips_curve.csv)
  - [k070_kin070_vlen010_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/k070_kin070_vlen010_remote_clip_lpips_curve.csv)
  - [rgbcal_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/rgbcal_k070_e3_eval_only_curve.csv)
  - [fiber_sde_fine_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/fiber_sde_fine_k070_e3_eval_only_curve.csv)
  - [topology_release_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/topology_release_k070_e3_eval_only_curve.csv)
  - [appearance_blend_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/appearance_blend_k070_e3_eval_only_curve.csv)
  - [pc_lowpass_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/pc_lowpass_k070_e3_eval_only_curve.csv)
  - [latent_affine_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/latent_affine_k070_e3_eval_only_curve.csv)
  - [latent_affine_refine_k070_e3_eval_only_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/latent_affine_refine_k070_e3_eval_only_curve.csv)
- Fixed page-1 CSV:
  - [wikiart5_page1_clip_lpips_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv)
- Rendered page-1 figure:
  - [fig_distinct5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary.png) is the tracked compatibility filename and now contains the filtered WikiArt-5 page-1 panel.
  - `fig_wikiart5_page1_summary.*` is also written locally by the generator, but is ignored by the existing `aaai2027` figure policy.

## Current Fiber-SDE Matched Delta

- 2026-06-15 implementation audit: the existing Fiber-SDE rows below were produced before the gate definition was tightened. That implementation used `gate / gate_rms`, which changes the noise energy and makes the matched-control read less clean. The code now follows the controlled-variable spec exactly: `sigmoid/clamp -> bilinear resize -> channel broadcast -> noise * gate`, with no RMS renormalization and no learnable parameters.
- Decision status for the rows below is therefore `archived_rms_gate_evidence`. They remain useful as negative/weak evidence that stochastic inference alone has a poor style/LPIPS slope, but a new `gate_fixed` matched scan is required before making any final claim about fiber-aligned noise.
- `sigma=0.01` isotropic: transfer `0.671501 / 0.313795`, all-pairs `0.703024 / 0.311868`.
- `sigma=0.01` fiber-aligned: transfer `0.671581 / 0.313762`, all-pairs `0.702954 / 0.311888`.
- Decision: `inconclusive_tie`; transfer improves by `+0.000080` style and `-0.000033` LPIPS, while all-pairs is slightly worse by `-0.000070` style and `+0.000020` LPIPS.
- `sigma=0.02` isotropic: transfer `0.672031 / 0.314990`, all-pairs `0.703432 / 0.313025`.
- `sigma=0.02` fiber-aligned: transfer `0.671818 / 0.314936`, all-pairs `0.703320 / 0.313015`.
- Decision: `conservative_not_promoted`; fiber-aligned preserves slightly more structure but loses style against the matched isotropic control.
- `sigma=0.03` isotropic: transfer `0.673391 / 0.316894`, all-pairs `0.704514 / 0.314930`.
- `sigma=0.03` fiber-aligned: transfer `0.673405 / 0.316883`, all-pairs `0.704633 / 0.314862`.
- Decision: `marginal_positive_continue`; fiber-aligned is better than isotropic on both style and LPIPS, but the delta is still small.
- `sigma=0.05` isotropic: transfer `0.675927 / 0.322953`, all-pairs `0.706639 / 0.320868`.
- `sigma=0.05` fiber-aligned: transfer `0.675948 / 0.323189`, all-pairs `0.706763 / 0.321093`.
- Decision: `style_upper_not_promoted`; this is the style-first upper point of the scan, but it pays clear LPIPS cost and does not reach the `0.74 / 0.30` target.

## Current Fiber-SDE Fine Style-Ceiling Extension

- Trigger: full-data training probes were too slow for the observed marginal gains, so the next action was restricted to eval-only SDE inference on the existing `k070 epoch_0003` parent.
- Scope: matched isotropic vs fiber-aligned SDE at `sigma=0.04, 0.06, 0.08`, fixed seed `42`, fixed `solver_corrector_steps=2`, fixed `solver_corrector_step_size=0.06`, and no training.
- Best transfer style point: `sigma=0.08 fiber`, transfer `0.681075 / 0.339063`; matched isotropic was `0.681007 / 0.339036`.
- Best all-pairs style point: `sigma=0.08 isotropic`, all-pairs `0.710653 / 0.336767`; matched fiber was `0.710641 / 0.336797`.
- Matched fiber-aligned read: fiber is worse than isotropic at `0.04` and `0.06`; at `0.08` it gains only `+0.000068` transfer style while worsening LPIPS by `+0.000027`, and all-pairs is still slightly worse.
- Decision: `style_ceiling_not_promoted`. More SDE noise can buy style, but the slope is poor and LPIPS cost rises quickly; this does not justify any long training lane by itself.
- Operational decision: full-data training is no longer the default next action. New training mechanisms must first pass a cheap eval-only screen or a short virtual-length probe with a clear style delta and no SMoE-like LPIPS reopen.

## Current SMoE Closure

- Round 2 `smoe_translator_k070_e3` is closed as `cost_stopped_not_promoted`.
- Best SMoE style point was `epoch_0009`: transfer `0.672774 / 0.327155`, all-pairs `0.704251 / 0.322688`.
- Best late candidate-curve Pareto point was `epoch_0014`: transfer `0.672185 / 0.324834`, all-pairs `0.703218 / 0.322686`.
- Stop point was `epoch_0015`: transfer `0.671284 / 0.333647`, all-pairs `0.702173 / 0.330398`.
- Matched-control read at e15: transfer style `-0.000536` and LPIPS `+0.019029` versus `k070 epoch_0003`; all-pairs style `-0.001061` and LPIPS `+0.017848`.
- Decision: do not launch `SMoE + fiberwise_swd` on this parent. The tokenizer-only mechanism has not produced enough style gain for its training cost.
- 2026-06-15 diagnosis update: this closure is consistent with the actuation-bottleneck critique. SMoE increases tokenizer-side geometric freedom, but the current backbone/output path can still compress generated deltas into a low-rank residual direction. Any future SMoE relaunch must record generated-delta rank/off-diagonal cosine in addition to tokenizer observability, otherwise a tokenizer-only positive or negative result is under-explained.

## 2026-06-15 Theory/Implementation Guardrail

- The Fiber Bundle language is retained as an organizing hypothesis, not as a proof that the current `TopoGate` is a true Ehresmann connection. Current code constrains attention routing and solver noise heuristically; it does not prove a tangent-space direct-sum decomposition.
- Formal conclusions must use matched deltas only: same parent, same dataset, same eval seed, same inference steps, same sigma or same single changed switch.
- Do not combine I2SB, PnP, Fiber-SDE, SMoE, kinetic schedules, or fiberwise SWD in a first-pass mechanism test. Combination configs may exist for later integration, but they are not valid first-round evidence.
- PC lowpass is a structure repair tool after a style-strong candidate, not the primary style path, because its low-frequency projection can suppress style-bearing low-frequency statistics.
- Latent/RGB affine calibration is diagnostic and default-off. It can expose style-statistics gaps but should not be overclaimed as a local fiber operation unless region-wise behavior is measured.
- Infra cleanup note: `infra_cleanup_2026-06-15.md`.

## Current I2SB/PnP/Fiberwise Mixed Screen

- Run: `phase2_i2sb_pnp_fiber_sde_k070`.
- Closure note: `i2sb_pnp_fiber_sde_k070_closure.md`.
- Best transfer point: e1 `0.684073 / 0.394578`.
- Latest point: e2 `0.683612 / 0.407860`.
- Decision: `cost_stopped_mixed_negative`. The run is useful negative evidence that the mixed endpoint/PnP/fiberwise route is too structure-expensive, but it is not a valid single-mechanism conclusion.

## Current Style Overdrive Diagnostic

- Run: `style_overdrive_k070_e3`.
- Closure note: `style_overdrive_k070_e3_closure.md`.
- Best pure-overdrive transfer point: `s160` `0.683721 / 0.295983`.
- Best pure-overdrive balanced point: `s135` `0.678224 / 0.288947`.
- Best style-stat diagnostic with latent affine: `s160_lataff045` `0.686336 / 0.315394`.
- Decision: `diagnostic_only_not_promoted`. These points expose style headroom but rely on out-of-domain integration and/or metric-affecting calibration.

## Current Style-Release Cost Stop

- `k070_kin070` was launched as the next least-invasive training-side style-release probe after SMoE closure.
- It was stopped during epoch `1/24` at about `9%` progress because the projected full-data/full-length retrain cost was not justified for a single `w_kinetic: 0.85 -> 0.70` change.
- VRAM was healthy at about `6.9 GiB`, so the stop does not indicate a memory failure.
- No checkpoint or `CLIP-S + LPIPS` point exists; it is intentionally not plotted on the homepage figure.
- `k070_sp256` is deferred for the same cost reason, because increasing tokenizer spatial capacity is expected to be slower and less isolated than the kinetic probe.

## Current Short-Probe Policy

- Full-data retraining is no longer the default for single-knob style-release ideas.
- The first short probe is `k070_kin070_vlen010`: same `w_kinetic: 0.85 -> 0.70` mechanism, same parent/eval contract, but `virtual_length_multiplier=0.10` and `num_epochs=6`.
- Promotion rule: only consider a full-data follow-up if the short probe shows a clear positive transfer/all-pairs style delta without reopening the LPIPS cost seen in SMoE.
- Plot rule: append short-probe eval points to `plot_points.csv` only after remote `CLIP-S + LPIPS` artifacts exist; no synthetic or partial-training points are plotted.

## Current Short-Probe Closure

- `k070_kin070_vlen010` was stopped after epoch `0003` eval because the curve was already cost-negative.
- Per-epoch train time was `148.3s`, `148.9s`, and `147.3s`; each full eval was about `248-250s`, so each plotted point costs roughly `6.6min` even at `virtual_length_multiplier=0.10`.
- Best point was `epoch_0002`: transfer `0.674131 / 0.340593`, all-pairs `0.704226 / 0.337494`.
- Matched against `k070 epoch_0003`, best e2 delta was transfer `+0.002310` style with `+0.025975` LPIPS and all-pairs `+0.000993` style with `+0.024944` LPIPS.
- `epoch_0003` worsened to transfer `0.673330 / 0.352338` and all-pairs `0.703133 / 0.347736`.
- Decision: `cost_stopped_negative`. Do not run the full-data `k070_kin070` follow-up; the kinetic-release knob is not worth more remote training under this parent/eval contract.
- Next style-release work should prefer eval-only solvers or post-decode calibration first. Any new training-side knob must pass a short-probe threshold before a full lane is launched.

## Current RGB Calibration Eval-Only Closure

- `rgbcal_k070_e3` tested decoded RGB style-affine calibration from the matched `k070 epoch_0003` parent with strengths `0.25`, `0.50`, and `0.75`.
- Runtime was cheap enough for screening: each full `CLIP-S + LPIPS` pass took about `154-155s` and remote VRAM stayed in eval-only territory.
- Best structure point was `s025`: transfer `0.654740 / 0.308625`, all-pairs `0.688668 / 0.305492`.
- Matched against `k070 epoch_0003`, `s025` improved transfer LPIPS by `-0.005993` and all-pairs LPIPS by `-0.007058`, but lost `-0.017080` transfer style and `-0.014566` all-pairs style.
- Stronger calibration worsened the tradeoff: `s050` fell to transfer `0.645430 / 0.328338`; `s075` fell to transfer `0.641211 / 0.352983`.
- Decision: `cost_positive_quality_negative`. Keep the post-decode switch for reproducible negative evidence, but do not promote it or spend training budget on brightness/contrast-only alignment under this parent.

## Current Topology-Release Eval-Only Closure

- `topology_release_k070_e3` tested inference-only `semantic_self_topology_blend = 0.5, 0.3, 0.0` from the matched `k070 epoch_0003` parent.
- Parent/control: transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- `blend=0.5`: transfer `0.671887 / 0.314608`, all-pairs `0.703252 / 0.312524`.
- `blend=0.3`: transfer `0.671899 / 0.314675`, all-pairs `0.703265 / 0.312592`.
- `blend=0.0`: transfer `0.671696 / 0.314660`, all-pairs `0.703089 / 0.312572`.
- Decision: `flat_no_training_value`. Lowering the topology blend at inference does not move style materially; do not spend a training lane on further isolated topology-blend reduction under this parent.

## Current Appearance-Blend Eval-Only Closure

- `appearance_blend_k070_e3` tested output appearance affine blend values `0.0`, `0.5`, and `1.0` from the matched `k070 epoch_0003` parent.
- Parent/control: transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- `blend=0.0`: transfer `0.671748 / 0.314596`, all-pairs `0.703189 / 0.312540`.
- `blend=0.5`: transfer `0.671748 / 0.314596`, all-pairs `0.703189 / 0.312540`.
- `blend=1.0`: transfer `0.671744 / 0.314595`, all-pairs `0.703187 / 0.312539`.
- Decision: `flat_no_training_value`. The output appearance affine path is not the style bottleneck under this parent; do not spend a long training lane on this isolated knob.

## Current PC-Lowpass Eval-Only Closure

- `pc_lowpass_k070_e3` tested `solver_pc + latent_lowpass` correction step sizes `0.03`, `0.06`, and `0.10` from the matched `k070 epoch_0003` parent.
- Parent/control: transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- `step=0.03`: transfer `0.671606 / 0.313594`, all-pairs `0.703035 / 0.311723`.
- `step=0.06`: transfer `0.671214 / 0.312733`, all-pairs `0.702729 / 0.311048`.
- `step=0.10`: transfer `0.671096 / 0.311748`, all-pairs `0.702628 / 0.310271`.
- Decision: `structure_repair_not_style_path`. PC improves LPIPS but lowers style, so keep it only as a possible safety correction after a future style-strong mechanism, not as the primary style route.

## Current Latent-Affine Eval-Only Closure

- `latent_affine_k070_e3` tested latent-space style-affine postprocess strengths `0.25`, `0.50`, and `0.75` from the matched `k070 epoch_0003` parent.
- Parent/control: transfer `0.671820 / 0.314618`, all-pairs `0.703234 / 0.312550`.
- `s0.25`: transfer `0.674868 / 0.310584`, all-pairs `0.707268 / 0.306689`; this improves both style and LPIPS versus the parent.
- `s0.50`: transfer `0.680303 / 0.322202`, all-pairs `0.712764 / 0.316212`; this is the balanced style candidate with transfer style `+0.008483` and all-pairs style `+0.009530` versus the parent.
- `s0.75`: transfer `0.685444 / 0.344580`, all-pairs `0.717593 / 0.336945`; this is the current phase2 style ceiling but pays too much LPIPS to be a balanced promotion.
- Decision: `balanced_style_candidate`. Keep the switch default-off and run a narrower cheap screen around the `0.25-0.60` band before any training-side follow-up.

## Current Latent-Affine Refine Closure

- `latent_affine_refine_k070_e3` refined the positive band with `s0.35`, `s0.45`, and `s0.60`, then tested `s0.50+PC0.10` and `s0.75+PC0.10`.
- `s0.35`: transfer `0.676781 / 0.313606`, all-pairs `0.709329 / 0.308847`; this is structure-positive versus the parent.
- `s0.45`: transfer `0.679110 / 0.318818`, all-pairs `0.711609 / 0.313230`; this is the current balanced frontier.
- `s0.60`: transfer `0.682390 / 0.330056`, all-pairs `0.714810 / 0.323339`; style improves but LPIPS cost starts to dominate.
- `s0.50+PC0.10`: transfer `0.680160 / 0.320104`, all-pairs `0.712667 / 0.314519`; PC repairs LPIPS slightly versus pure `s0.50`, but not enough to be a new direction.
- `s0.75+PC0.10`: transfer `0.685304 / 0.343517`, all-pairs `0.717560 / 0.336053`; PC does not rescue the style-ceiling LPIPS cost.
- Decision: `balanced_frontier`. Keep `s0.45` as the clean eval-time style amplifier and use `s0.35` as the safe structure anchor; the next route to `0.74` needs a new style-generation mechanism, not more affine strength.

## Current Proximal Texture Closure

- `actuation_proximal_texture_k070_e3_b16a2bf16_vlen010` tested an isolated
  endpoint cross-attention texture residual from the matched `k070 epoch_0003`
  parent.
- Fast10 live convergence closed at `epoch_0014`: best and last Pareto remained
  `epoch_0009`, `since_best=5`, `tail_flat=true`, `converged=true`.
- Full-transfer confirmation best was `epoch_0009`: transfer
  `0.674190 / 0.329931`.
- Final `epoch_0014` confirmed at transfer `0.673760 / 0.331171`.
- Decision: `converged_not_promoted`. The mechanism is structure-safe and
  mildly style-positive versus parent, but it does not beat the R16 style
  frontier and is not enough for the Seedream/`0.74` style target.
- Next controlled direction: attack the generated-delta / `dec_out` rank
  bottleneck directly and log generated-delta rank plus off-diagonal cosine for
  every retained checkpoint.

## Current Delta-Rank Observability

- Added default-off eval switch `full_eval.delta_observability` /
  `training.full_eval_delta_observability`.
- When enabled, eval records generated latent deltas per source image across
  target styles and writes effective rank plus off-diagonal cosine into
  `summary.json -> settings.generated_delta_observability`.
- Remote smoke on proximal e9 with `max_src_samples=1` produced
  `effective_rank_mean=1.60` and `offdiag_cosine_mean=0.356` over five source
  groups, confirming the instrumentation is active and cheap enough for the
  next actuation probe.

## Plot Update Contract

Use `tools/experiments/update_phase2_plot_points.py` after each completed eval:

```bash
python SchrodingerBridge/tools/experiments/update_phase2_plot_points.py \
  --curve-csv <clip_lpips_curve.csv> \
  --family "FiberBundle" \
  --variant "<experiment_id>" \
  --trace-id "<experiment_id>" \
  --label-prefix "<short label>"
```

Then regenerate the homepage figures:

```bash
python SchrodingerBridge/aaai2027/scripts_gen_wikiart5_page1_summary.py --rebuild
```

Or pass `--render` to `update_phase2_plot_points.py`; it refreshes the filtered WikiArt-5 AAAI2027 page-1 summary figure. Do not use the older mixed-source `best.csv` / `fig_distinct5_all_points_big.csv` page-1 path for paper-facing claims on the new dataset.
