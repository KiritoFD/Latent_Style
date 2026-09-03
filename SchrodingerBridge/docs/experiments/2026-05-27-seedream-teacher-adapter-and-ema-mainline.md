# Seedream Teacher Adapter Side Probe And EMA Mainline

Date: 2026-05-27

## Scope

This note separates two threads that must not be mixed:

1. **Mainline unsupervised LANCET search**: Seedream is only a diagnostic golden reference.
2. **External-teacher side probe**: Seedream 4.5 generates pseudo-paired references on non-test images, then only `style_emb.weight` and optional `style_spatial_id_16` are fitted.

The second thread is not eligible for the main paper table. It can only answer whether the style-id adapter has enough capacity to absorb a strong teacher's style prior.

## End-To-End Script

Script:

```text
tools/experiments/run_seedream_adapter_probe.py
```

Local launcher:

```text
tools/experiments/start_seedream_adapter_probe_local.ps1
```

Pipeline:

1. Select non-test source images from `style_data/train`.
2. Call Windhub `v1/images/generations` with `doubao-seedream-4-5-251128`.
3. Save teacher images and `teacher_manifest.csv` under `exp/seedream_distill_adapter/.../teacher_refs`.
4. Encode source/teacher pairs with the checkpoint's VAE.
5. Freeze the LANCET backbone.
6. Train only `style_emb.weight` and optional `style_spatial_id_16`.
7. Save `style_adapter.pt`.
8. Run standard repo evaluation with `--style_adapter`.

API key handling:

- the script reads `WINDHUB_API_KEY`;
- the key is not written into config, manifest, logs, or adapter artifacts.
- the local launcher inherits `WINDHUB_API_KEY` from the shell and passes it only through the child-process environment.

Initial local launch:

```text
exp/seedream_distill_adapter/api_photo_train_t01_probe_20260527_023541
```

Status:

- stopped before training;
- Windhub returned HTTP 401 with "token expired";
- no teacher refs were generated.

Second local launch after adding API preflight:

```text
exp/seedream_distill_adapter/api_photo_train_t01_probe_20260527_025055
```

Status:

- stopped at the first preflight request;
- Windhub again returned HTTP 401 with "token expired";
- `teacher_manifest.csv` and `teacher_generation_config.json` were still written for failure provenance;
- no teacher image was generated, so adapter fitting and eval did not start.

Successful API smoke after replacing the expired key:

```text
exp/seedream_distill_adapter/api_photo_train_t01_probe_20260527_031932
```

Protocol:

- `16` requested teacher pairs from non-test `photo` train images;
- target styles: `Hayao`, `monet`, `vangogh`, `cezanne`;
- train only the external style adapter for `80` iterations;
- evaluate on the untouched standard `750` benchmark.

Result:

| teacher pairs | ok pairs | clip_style | content_lpips | clip_content |
|---:|---:|---:|---:|---:|
| 16 | 16 | 0.70347 | 0.50339 | 0.77278 |

Interpretation: this only proves the end-to-end path works. It is too small to support any claim about adapter capacity.

Larger side-probe after the smoke was judged too small:

```text
exp/seedream_distill_adapter/api_photo_train_t01_probe_20260527_032740
```

Protocol:

- `120` requested teacher pairs from non-test `photo` train images;
- `116` valid pairs after `4` transient connection resets;
- target-style coverage: `Hayao=29`, `monet=29`, `vangogh=30`, `cezanne=28`;
- train only the external style adapter for `160` iterations;
- evaluate on the untouched standard `750` benchmark.

Result:

| teacher pairs | ok pairs | clip_style | content_lpips | clip_content |
|---:|---:|---:|---:|---:|
| 120 | 116 | 0.70111 | 0.49938 | 0.77478 |

Interpretation: increasing the teacher set from `16` to `116` usable pseudo-pairs did not raise the benchmark style score. This argues that the current adapter-only path is not a hidden route to `clip_style > 0.72`; it is more likely a capacity/route limitation of the frozen style-id injection, or a mismatch between non-test Seedream supervision and the benchmark transfer geometry. This remains a side result only and must not be mixed into the main unsupervised LANCET table.

## Academic Status

This is acceptable only as:

- teacher-distilled adapter;
- style-embedding capacity probe;
- model compression / calibration side result;
- diagnostic evidence about what the native objective fails to learn.

It is not acceptable as:

- the main unsupervised LANCET result;
- evidence that native LANCET beats SaMST;
- a way to tune the official test set.

The clean protocol is:

- generate teacher refs only from non-test sources;
- train adapter only on those refs;
- evaluate the adapter on the untouched benchmark;
- label the method as `LANCET + Seedream-distilled Adapter`.

## EMA vs MSE: Current Answer

Current evidence supports a nuanced answer, not a simple "EMA wins".

### What EMA Is Better At

EMA avoids the MSE VAE's over-smoothed latent geometry and gives cleaner content-preserving branches:

- `ema_plain4_w20_anchor`: about `clip_style=0.7007`, `content_lpips=0.4215`.
- `ema_dynamic_guard_w28`: about `clip_style=0.7078`, `content_lpips=0.4477`.

This means EMA is useful when the goal is content safety, cleaner grids, and diagnosing style carriers without the MSE smoothing bias.

### Where EMA Fails

EMA has not yet produced the desired `clip_style > 0.72` while keeping LPIPS near or below `0.5` through safe operators.

Observed pattern:

- no-warp EMA dynamic/operator branches usually cap around `0.706-0.716`;
- stronger style pressure mostly buys style by worsening LPIPS;
- unsafe warp/t01-like routes can cross `0.72`, but do so through visible deformation and high LPIPS.

Therefore EMA is not currently "better than MSE/original VAE" for the final target. It is better as a diagnostic and content-safe backend, while the original VAE/t01 family still has the higher native style ceiling.

## How To Use EMA

Use EMA for controlled architecture diagnosis:

- keep warp off or nearly off;
- test where style enters the network, especially body/skip routing;
- measure Seedream-gap diagnostics: flat color flood, highpass mist, phase alignment, object-boundary leakage;
- reject variants that gain CLIP style only by increasing global texture fog or cross-edge color motion.

Do not use EMA for blind patch/SWD strength sweeps anymore. The repeated result is that more terminal style pressure mostly follows the same style-content tradeoff.

## Mainline Hypothesis

The remaining gap is not raw style strength. It is **semantic routing of style energy before decoder/skip fusion**.

Seedream looks strong because it performs style-dependent, region-organized repainting:

- Hayao can move more strongly while preserving object identity;
- Van Gogh/Cezanne can add texture without flooding all flat regions;
- high-frequency change is more phase-locked to object boundaries.

Our safe EMA variants mostly add local texture or palette pressure uniformly. The next mainline tests should therefore focus on body-level routing and style-conditioned residual bands, not on more endpoint SWD or unsafe warp.

## Mainline Style-Embedding Calibration

User question: can the whole training set improve `style_emb` without using Seedream or any external teacher?

Script:

```text
tools/experiments/run_style_embedding_mainline_calibration.py
```

This is deliberately separate from the Seedream side probe. It uses only native training latents and the current LANCET checkpoint as an internal geometry anchor.

Hypothesis:

- If `style_emb.weight` still has unused conditioning capacity, full-train SWD calibration should raise `clip_style` while the anchor prevents the frozen transport geometry from drifting.
- If it again lowers style or worsens LPIPS, the bottleneck is not the embedding table itself; it is the carrier path that consumes the style embedding.

Recipes:

| recipe | trainable | objective |
|---|---|---|
| `m00_emb_swd_anchor` | `style_emb.weight` | full-train SWD plus teacher endpoint anchor |
| `m01_embspatial_swd_anchor` | `style_emb.weight`, `style_spatial_id_16` | stronger geometry-preserving adapter branch |
| `m02_embspatial_highpass_style` | `style_emb.weight`, `style_spatial_id_16` | high-pass style SWD with lighter anchor |

The anchor is internal: the teacher endpoint is the same checkpoint before adapter fitting. Seedream remains diagnostic only and does not enter this training path.

Remote full-train result on `ema_transport_adain_w34_guard/epoch_0006.pt`:

Baseline anchor:

| checkpoint | clip_style | content_lpips | clip_content |
|---|---:|---:|---:|
| `ema_transport_adain_w34_guard e6` | 0.71343 | 0.49859 | not recorded here |

Full-train adapter results:

| recipe | trainable | clip_style | content_lpips | clip_content |
|---|---|---:|---:|---:|
| `m00_emb_swd_anchor` | style embedding | 0.71118 | 0.43530 | 0.82836 |
| `m01_embspatial_swd_anchor` | style embedding + spatial id | 0.71041 | 0.41618 | 0.84629 |
| `m02_embspatial_highpass_style` | style embedding + spatial id, high-pass SWD | 0.71073 | 0.40735 | 0.84967 |

Artifacts:

```text
exp/style_embedding_mainline_calibration/ema_transport_adain_w34_e6_fulltrain/mainline_style_emb_results.csv
```

Interpretation:

- all three full-train calibrations improve content preservation strongly;
- none improves style over the frozen checkpoint;
- adding `style_spatial_id_16` makes the adapter even more content-safe;
- high-pass SWD lowers the training high-frequency objective but still does not translate into benchmark `clip_style`.

Conclusion: full-training-set `style_emb` fitting is a useful **content-safe calibration / diagnostic** branch, but it is not the main route to `clip_style > 0.72`. The bottleneck is upstream of the embedding table or in the carrier that consumes it, not in underfit style-id vectors.

## Current Remote Mainline

Remote 3060 was restarted on:

```text
exp/vae_backend/ema_bodyresband
```

First visible row:

| variant | epoch | clip_style | content_lpips | EC |
|---|---:|---:|---:|---:|
| `ema_bodyresband_w32_guard` | 6 | 0.7044 | 0.4767 | 0.3686 |

This first row is not promising, but the run should finish before drawing the final conclusion for the body-residual-band route.

Final body-residual-band rows:

| variant | epoch | clip_style | content_lpips | EC |
|---|---:|---:|---:|---:|
| `ema_bodyresband_w32_guard` | 6 | 0.70436 | 0.47665 | 0.36862 |
| `ema_bodyresband_w32_guard` | 7 | 0.70186 | 0.49128 | 0.35705 |
| `ema_bodyresband_w32_guard` | 8 | 0.70448 | 0.48830 | 0.36048 |
| `ema_bodyresband_w36_style` | 6 | 0.70695 | 0.50059 | 0.35306 |
| `ema_bodyresband_w36_style` | 7 | 0.70513 | 0.51199 | 0.34411 |
| `ema_bodyresband_w36_style` | 8 | 0.70687 | 0.50931 | 0.34685 |

Conclusion: body-residual-band did not break the style ceiling. It is a negative branch unless later diagnostics show a clear bug in the implementation.

## Body Dual Residual Probe

The body-residual-band negative result is informative: pure high/band-pass residual removed the low-frequency body drift, but it also removed part of the bodyblend style carrier. The revised hypothesis is:

> bodyblend's positive style gain comes from two carriers: a small smooth semantic/body residual plus a phase-compatible mid/high texton residual. A single band-pass carrier is too weak; hard body replacement is too free.

Implementation:

- added `style_blender_mode="residual_dual"`;
- splits `painted_body - content_body` into:
  - bounded low/smooth residual for region-level style;
  - content-support and phase-gated mid residual for object-aligned style detail;
  - optional tiny high residual;
- writes body-carrier debug into `numeric_debug.jsonl`:
  - `body_blend_ratio`;
  - `body_dual_support_gate`;
  - `body_dual_phase_gate`;
  - `body_dual_low_delta`;
  - `body_dual_mid_delta`;
  - `body_dual_high_delta`.

Smoke variants:

| variant | status | peak VRAM | notes |
|---|---|---:|---|
| `ema_bodydual_w34_guard` | train_failed_step1 | 9515 MB | forward finite, but one non-finite gradient in dynamic output head; do not use as main readout yet |
| `ema_bodydual_w40_style` | train_ok | 10012 MB | stable 30-batch smoke; full 8-epoch run launched |

First smoke debug for `ema_bodydual_w40_style`:

| carrier | mean_abs / mean |
|---|---:|
| blend ratio | `0.5300` |
| support gate mean | `0.5972` |
| phase gate mean | `0.4573` |
| low residual mean_abs | `0.0602` |
| mid residual mean_abs | `0.0387` |
| high residual mean_abs | `0.0051` |

This is a live structural probe, not a parameter sweep. If full eval crosses `0.72` with LPIPS near `0.50`, the dual-carrier body residual becomes the next mainline. If it stays below `0.72`, the evidence says body-level style still needs a stronger semantic region router, not merely more residual energy.

Full body-dual result:

| variant | epoch | clip_style | content_lpips | EC |
|---|---:|---:|---:|---:|
| `ema_bodydual_w40_style` | 6 | 0.70930 | 0.52211 | 0.33897 |
| `ema_bodydual_w40_style` | 7 | 0.70435 | 0.52907 | 0.33170 |
| `ema_bodydual_w40_style` | 8 | 0.70619 | 0.52848 | 0.33299 |

Conclusion: the dual-carrier body residual is a negative mainline branch. It confirms that body-level residual energy by itself is not enough: the missing piece is stronger semantic/object-aware routing of where style energy is allowed to enter, not another scalar increase in style pressure.

## Full-Train Style Adapter Probe On EMA Frontier

Question: can the strong balanced EMA anchor be improved by freezing the backbone and calibrating only style-conditioning parameters on the whole train latent set, without any external teacher?

Anchor:

```text
exp/vae_backend/representative_ckpts/ema_dynamic_frontier_w32_e6_clip07093_lpips04690/epoch_0006.pt
```

Baseline anchor metric:

| clip_style | content_lpips | EC |
|---:|---:|---:|
| 0.7093 | 0.4690 | 0.3767 |

Implementation notes:

- `tools/experiments/run_style_embedding_distill.py` was repaired for this probe.
- The old script assumed a t01-style static `dec_out` hook and therefore failed on dynamic-head EMA checkpoints.
- The generic probe now optimizes the endpoint distribution through a differentiable Euler loop over `model.forward(...)`; it does not use `model.integrate(...)`, because the wrapper integration path is `@torch.no_grad()`.
- Evaluation now respects the checkpoint VAE via `--vae-model ema` instead of hard-coding `sd15/mse`.
- Training data: full `latent-256-sd15-ema` train latent folders for `photo`, `Hayao`, `monet`, `vangogh`, `cezanne`.

Remote/local result archive:

```text
exp/style_embedding_distill/ema_frontier_fulltrain_foreground_full
```

Results:

| recipe | trainable route | clip_style | content_lpips | clip_content | interpretation |
|---|---|---:|---:|---:|---|
| `d00_emb_only_swd_s4_it60` | `style_emb.weight` | 0.7024 | 0.6230 | 0.6851 | style drops and content collapses; embedding-only is not a hidden style lever |
| `d02_embspatial_swd_tv_grad_s8_it80` | `style_emb.weight + style_spatial_id_16` | 0.6767 | 0.2742 | 0.9202 | learns a very conservative content-safe prior, effectively de-stylizing |

Conclusion:

Full-train post-hoc style-embedding calibration is negative on the current EMA frontier. The failure mode is not "insufficient data"; using the whole train latent distribution still moves the model away from the desired style frontier. The spatial-id route has capacity, but its easiest optimum under SWD+TV+gradient guard is content preservation, not stronger organized style. This supports the revised mainline: the missing component is not a better global style vector, but a semantic/object-aware style actuator that can place style energy into the right regions without turning the whole endpoint into either texture fog or a content-safe identity map.

## Target-Style Stratification Rule

Do not collapse future results into one global average until each target style
has been checked separately. Hayao is now the priority failure case.

Current strongest content-safe point:

```text
exp/style_embedding_mainline_calibration/ema_transport_adain_w34_e6_fulltrain/m02_embspatial_highpass_style
```

Overall it looks excellent:

| clip_style | content_lpips | clip_content |
|---:|---:|---:|
| 0.71073 | 0.40735 | 0.84967 |

But the target-style breakdown shows the real bottleneck:

| target style | slice | clip_style | content_lpips | weak rows (`clip_style < 0.70`) |
|---|---|---:|---:|---:|
| Hayao | all | 0.64864 | 0.41960 | 80.7% |
| Hayao | cross only | 0.60516 | 0.44920 | 99.2% |
| cezanne | cross only | 0.70535 | 0.37673 | 45.0% |
| monet | cross only | 0.69586 | 0.37524 | 48.3% |
| vangogh | cross only | 0.72302 | 0.41200 | 39.2% |

Interpretation:

- Hayao is not just "harder" in a vague sense; it is the only target where the
  current actuator almost never produces a strong cross-style signal.
- The same latent/VAE/backbone family can already produce strong `vangogh`
  style, so there is not enough evidence to declare the VAE unusable.
- Hayao's carrier is likely contour simplification plus flat color-plane
  repainting. The current high-pass/texton routes mostly express brush texture,
  so they help `vangogh` more than Hayao.
- Post-hoc Hayao-only `style_emb` repair from m02 did not solve it:
  `m03_m02_styleboost_balanced`, `m04_m02_styleboost_loose`, and
  `m05_m02_midcolor_push` all remained near global `clip_style=0.710` and did
  not move the Hayao cross target enough.

Actionable protocol change:

1. Every full eval summary should expose `by_target_style` and
   `cross_by_target_style`.
2. Every experiment CSV that ranks candidate points should include Hayao cross
   `clip_style/content_lpips` or at least the weakest cross target.
3. A Hayao-weighted run is justified, but the weight must be diagnostic:
   increase Hayao sampling/style losses while protecting content/kinetic anchors.
   If Hayao still fails, the failure is architectural, not data exposure.
4. The next model branch should target Hayao's visual grammar directly:
   low-frequency flat-color repainting plus edge/contour alignment, rather than
   simply increasing generic SWD or high-pass texture pressure.

Implemented follow-up levers:

- `src/utils/run_evaluation.py` now writes `by_target_style`,
  `cross_by_target_style`, `by_source_style`, and `cross_by_source_style` into
  every new `summary.json`.
- `tools/experiments/run_style_embedding_mainline_calibration.py` now adds
  `hayao_cross_clip_style`, `hayao_cross_content_lpips`, and weakest-cross-target
  columns to future adapter summary CSVs.
- `BridgeConfig.target_style_loss_weights` allows style-specific weighting, but
  only for style/terminal terms in the OMF objective. Flow, kinetic, content
  anchor, and edge anchor are not upweighted. This is deliberate: if Hayao
  improves only when content anchors are also amplified, we have not learned a
  stronger Hayao style actuator.

New Hayao probes in `tools/experiments/run_vae_backend_256_probe.py`:

| variant | hypothesis |
|---|---|
| `ema_transport_texton_hayao_exposure_w34` | If Hayao is simply under-exposed, target sampling weights should lift Hayao cross style without architecture changes. |
| `ema_transport_texton_hayao_weighted_w34` | If exposure is insufficient but the actuator is capable, Hayao-only style/terminal loss weighting should lift Hayao while leaving content anchors stable. |
| `ema_transport_texton_hayao_flatcontour_w36` | If Hayao needs a different visual grammar, a low-frequency flat-color plus edge-contour emphasis should outperform generic texton pressure. |

## Region-Routed Painted Body Probe

The full-train style adapter result changes the next action: do not continue tuning the global style vector. The next mainline architecture probe is a region-routed body paint actuator.

Hypothesis:

> `ema_bodyblend_w28_guard` was the first structural positive because it lets style enter at the 16x16 body before decoder/skip fusion. Its failure mode is broad body replacement. `ema_bodydual_w40_style` then showed that raw residual band splitting is too weak. The missing operation is therefore not more style energy, but a style-conditioned spatial gate over the remapped painted-body delta.

Implementation:

- added `style_blender_mode="region_paint"`;
- compute the proven bodyblend-style `remapped = conv(norm(lerp(content_body, painted_body)))`;
- split `remapped - content_body` into smooth low, mid, and high residuals;
- learn a style-conditioned region gate from low-frequency body bins;
- apply the region gate to low residuals and region/support/phase gates to detail residuals;
- record debug tensors: `body_region_gate`, `body_region_support_gate`, `body_region_phase_gate`, `body_region_low_delta`, `body_region_mid_delta`, `body_region_high_delta`.

Remote probes:

| variant | purpose |
|---|---|
| `ema_bodyregion_w34_guard` | conservative guard: test whether region gating recovers bodyblend style while protecting LPIPS |
| `ema_bodyregion_w42_style` | style push: allow LPIPS near `0.50` only if region-routed body paint crosses `clip_style > 0.72` |

Launch files:

```text
run_ema_bodyregion_smoke_remote.cmd
run_ema_bodyregion_full_remote.cmd
```

Smoke:

| variant | status | peak VRAM | interpretation |
|---|---|---:|---|
| `ema_bodyregion_w34_guard` | `train_failed_step1` | 9566 MB | non-finite gradient in `output_head.weight_generator.0.weight`; same failure family as `ema_bodydual_w34_guard` |
| `ema_bodyregion_w42_style` | `train_ok` | 10046 MB | stable 30-batch smoke; full run launched |

Full `ema_bodyregion_w42_style` result:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.71308 | 0.54967 | 0.32112 |
| 7 | 0.70790 | 0.55496 | 0.31505 |
| 8 | 0.70850 | 0.55229 | 0.31720 |

Detailed epoch-6 breakdown:

| slice | clip_style | content_lpips | note |
|---|---:|---:|---|
| all pairs | 0.71308 | 0.54967 | below target and worse content than the EMA frontier |
| style-transfer ability | 0.69924 | 0.55404 | the apparent all-pair style is partly inflated by identity/art-to-art cases |
| photo-to-art | 0.67669 | 0.58873 | weak as a practical transfer model |
| identity reconstruction | 0.76844 | 0.45545 | identity/art-style rows remain easy and inflate the average |

Debug diagnosis:

- `body_region_gate` was effectively constant in the early numeric dump: mean `0.61076`, max `0.61083`.
- checkpoint inspection at epoch 6 showed the gate's last layer remained tiny: `region_gate_generator.2.weight` mean abs `0.00207`, max abs `0.00335`.
- the low body residual was large (`body_region_low_delta` mean abs around `0.107`), while the learned region gate did not spatially select where it should enter.

Conclusion: `region_paint` is negative as implemented. It did not become a semantic/object-aware router; it mostly acted as a scaled body rewrite, which preserved the bodyblend content damage while failing to raise style. The next architecture should not rely on an unconstrained learned region MLP to self-discover routing from the terminal SWD signal. It needs either a deterministic transport-confidence gate from semantic attention, a diversity/entropy constraint on region routing, or a separate objective that directly supervises region selectivity without external teacher images.

## Transport-Confidence Body Paint

The next probe replaced the learned region MLP with a deterministic confidence gate computed from the semantic transport matrix:

```text
row confidence = top1 - top2, optionally entropy-weighted
key uniqueness = inverse expected style-key load
transport gate = normalized confidence-to-gate map
```

Implementation:

- added `style_blender_mode="transport_paint"`;
- retained the proven painted-body carrier from `bodyblend`;
- split residual into low/mid/high bands;
- routed all bands through transport confidence;
- kept support/phase gates for detail bands.

Full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_bodytransport_w36_guard` | 6 | 0.71398 | 0.50967 | 0.35009 | LPIPS recovers vs `region_paint`, style still below target |
| `ema_bodytransport_w36_guard` | 7 | 0.71113 | 0.51841 | 0.34247 | worse |
| `ema_bodytransport_w36_guard` | 8 | 0.71270 | 0.51654 | 0.34457 | worse |
| `ema_bodytransport_w42_style` | 6 | 0.70067 | 0.55124 | 0.31443 | style-pressure branch collapses |
| `ema_bodytransport_w42_style` | 7 | 0.69860 | 0.55704 | 0.30945 | worse |
| `ema_bodytransport_w42_style` | 8 | 0.69974 | 0.55581 | 0.31082 | worse |

Interpretation:

- Transport confidence fixed the constant-gate pathology from `region_paint`.
- It recovered content relative to `region_paint` (`0.5097` vs `0.5497` LPIPS) without using external supervision.
- It did not raise style enough. The style-push branch getting worse means the failure is not insufficient terminal SWD.

New diagnosis:

The first `transport_paint` implementation multiplied the low-frequency body carrier by the same highpass content-support gate used for detail residuals. That is theoretically wrong for smooth semantic repainting: flat areas have low highpass support but still need visible style color and broad stroke organization. This explains the observed tradeoff: the model became safer, but it under-stylized semantic flats.

## Transport Low-Free Probe

Code change:

- added `style_blender_transport_low_use_support`;
- default keeps previous behavior;
- new variants set it to `False`, so:

```text
low_gate    = transport_confidence
detail_gate = transport_confidence * content_support * phase_alignment
```

Remote smoke:

| variant | status | peak VRAM | debug signal |
|---|---|---:|---|
| `ema_bodytransport_lowfree_w34_guard` | `train_ok` | 10575 MB | low gate mean `0.533`, low delta mean_abs `0.0716` |
| `ema_bodytransport_lowfree_sconv_w38_style` | `train_ok` | 10830 MB | low gate mean `0.640`, low delta mean_abs `0.0796`, mid delta mean_abs `0.0702` |

This confirms the carrier actually changed: the low-frequency style path is no longer being suppressed by highpass support, while detail residuals remain gated. Full remote run is launched under:

```text
exp/vae_backend/ema_bodytransport_lowfree
```

Eval plan: epochs `6/7/8`, same 750 protocol, target `clip_style > 0.72` with LPIPS allowed up to about `0.50`.

Partial full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_bodytransport_lowfree_w34_guard` | 6 | 0.71431 | 0.50034 | 0.35691 | best low-free point, but still below the style target |
| `ema_bodytransport_lowfree_w34_guard` | 7 | 0.71150 | 0.50873 | 0.34954 | worse |
| `ema_bodytransport_lowfree_w34_guard` | 8 | 0.71318 | 0.50679 | 0.35175 | does not recover epoch 6 |
| `ema_bodytransport_lowfree_sconv_w38_style` | 6 | 0.71401 | 0.52566 | 0.33869 | semantic-conv style carrier does not raise style and hurts content |
| `ema_bodytransport_lowfree_sconv_w38_style` | 7 | 0.71177 | 0.53220 | 0.33297 | worse |
| `ema_bodytransport_lowfree_sconv_w38_style` | 8 | 0.71213 | 0.53266 | 0.33280 | worse |

Interim conclusion:

Releasing the low-frequency carrier from highpass support is not enough. It makes the guarded branch content-competitive with the transport guard line, but the style ceiling remains around `0.714`. The semantic-conv carrier did not use the released low path more effectively; it mainly damaged content. This weakens the hypothesis that broad semantic flats were the only missing style path, and points back to the carrier itself: we need a different style actuator or objective geometry, not another gate on the same painted-body residual.

## Phase-Envelope SWD Probe

The next hypothesis comes from the Seedream diagnostic table rather than another gate sweep.

Observed diagnostic pattern:

- Seedream's all-pair `highpass_phase_cos` is about `-0.35`.
- Our strong branches are much more anti-phase: `t01_original_vae_e8` around `-0.81`, EMA dynamic/routed branches around `-0.72` to `-0.74`.
- Our highpass energy is not lower than Seedream; it is often higher.

Conclusion:

The active micro-SWD branch likely matches the signed high-pass phase of unpaired style images. That target phase is random with respect to the source content, so the loss can reward anti-phase texture. This is an OT-objective problem, not evidence that EMA VAE is invalid.

Code change:

- added `swd_signed_highpass_weight` and `swd_abs_highpass_weight`;
- default is unchanged: signed highpass weight `1.0`, abs highpass weight `0.0`;
- new variants set signed `0.0`, abs `1.0`, keeping Sobel magnitude and a weak Fourier phase lock to the source.

Remote smoke:

| variant | status | peak VRAM |
|---|---|---:|
| `ema_phase_envelope_w36_guard` | `train_ok` | 9840 MB |
| `ema_phase_envelope_w44_style` | `train_ok` | 9839 MB |

Full remote run launched:

```text
exp/vae_backend/ema_phase_envelope
```

Full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_phase_envelope_w36_guard` | 6 | 0.71174 | 0.48888 | 0.36378 | best branch; content-safe but style below ceiling |
| `ema_phase_envelope_w36_guard` | 7 | 0.70908 | 0.50392 | 0.35176 | worse |
| `ema_phase_envelope_w36_guard` | 8 | 0.70945 | 0.49909 | 0.35537 | content acceptable, no style gain |
| `ema_phase_envelope_w44_style` | 6 | 0.70675 | 0.56482 | 0.30756 | style-pressure branch collapses |
| `ema_phase_envelope_w44_style` | 7 | 0.70205 | 0.57078 | 0.30133 | worse |
| `ema_phase_envelope_w44_style` | 8 | 0.70269 | 0.56764 | 0.30382 | worse |

Conclusion:

Phase-envelope SWD is a negative style-ceiling branch. It supports the diagnosis
that "more high-frequency pressure" is not enough: the guarded variant protects
LPIPS but does not move `clip_style`, and the style-push variant harms both
metrics. The next mainline should inspect the style actuator and semantic/macro
transport geometry, not keep increasing global SWD pressure.

## Full-Train Style Adapter On Current Best Mainline

The first full-train style-adapter probe used the EMA frontier anchor
(`0.7093 / 0.4690`) and was negative. To rule out the possibility that the
adapter only works after a stronger body-transport backbone, a second probe is
now launched on the current best style point:

```text
exp/vae_backend/ema_bodytransport_lowfree/ema_bodytransport_lowfree_w34_guard/epoch_0006.pt
```

Anchor metric:

| clip_style | content_lpips | EC |
|---:|---:|---:|
| 0.71431 | 0.50034 | 0.35691 |

Remote task:

```text
LANCET_style_emb_bodytransport_fulltrain
```

Output:

```text
exp/style_embedding_distill/ema_bodytransport_lowfree_w34_fulltrain
```

Recipes:

| recipe | trainable route | reason |
|---|---|---|
| `d00_emb_only_swd_s4_it60` | `style_emb.weight` | direct test of whether the global style vector can move the frontier |
| `d02_embspatial_swd_tv_grad_s8_it80` | `style_emb.weight + style_spatial_id_16` | stronger spatial style prior with TV/gradient guard |

This remains unsupervised and uses the full `latent-256-sd15-ema` train latent
distribution. Seedream is not used as a training teacher.

Result:

| recipe | clip_style | content_lpips | clip_content | interpretation |
|---|---:|---:|---:|---|
| `d00_emb_only_swd_s4_it60` | 0.68548 | 0.60483 | 0.69254 | global embedding calibration sharply degrades both metrics |
| `d02_embspatial_swd_tv_grad_s8_it80` | 0.68625 | 0.62951 | 0.74694 | spatial style prior also degrades the frontier; no hidden gain |

Conclusion:

Full-train post-hoc `style_emb` tuning is negative on both the older EMA
frontier anchor and the current best body-transport anchor. This is stronger
evidence than the first probe alone: even when the backbone is already near the
best observed style point, moving only `style_emb.weight` or
`style_spatial_id_16` destroys the learned transport geometry instead of
raising `clip_style`. The mainline should not spend more GPU time on global
style-vector calibration unless the backbone exposes a new, explicitly
localized style actuator.

## Transport Moment Carrier

After the negative `style_emb` and phase-envelope probes, the next mainline
hypothesis is no longer "more style pressure". It is:

> the current body style carrier does not perform semantic channel-statistic
> transport, so it cannot express enough organized style before damaging
> content.

Important code-level correction:

The intended low-free body transport gate was:

```text
low_gate    = transport_confidence
detail_gate = transport_confidence * content_support * phase_alignment
```

but the implementation used `detail_gate = low_gate * phase_alignment`. When
`style_blender_transport_low_use_support=False`, this also removed the
content-support gate from mid/high detail. That means the previous low-free run
was not the clean hypothesis test described in the document.

Code changes:

- corrected `transport_paint` so detail always uses
  `transport * support * phase`;
- added `style_blender_mode="transport_adain"`;
- `transport_adain` uses the semantically painted body feature as a local
  moment field:

```text
target = (content_body - local_mean(content_body)) / local_std(content_body)
         * local_std(painted_body) + local_mean(painted_body)
```

and then applies the same corrected low/detail transport gates. This is
unsupervised; Seedream remains diagnostic only.

Planned remote run:

```text
exp/vae_backend/ema_transport_moment
```

| variant | purpose |
|---|---|
| `ema_bodytransport_lowfree_fixed_w34_guard` | rerun the intended low-free gate after fixing detail support |
| `ema_transport_adain_w34_guard` | conservative transport-conditioned local AdaIN carrier |
| `ema_transport_adain_w40_style` | style-push local moment carrier, allowing LPIPS near 0.50 only if style crosses 0.72 |

Remote smoke:

| variant | status | peak VRAM |
|---|---|---:|
| `ema_bodytransport_lowfree_fixed_w34_guard` | `train_ok` | 10125 MB |
| `ema_transport_adain_w34_guard` | `train_ok` | 10387 MB |
| `ema_transport_adain_w40_style` | `train_ok` | 10413 MB |

These fit the 10G-class budget and are ready for the full 8-epoch run.

Partial full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_bodytransport_lowfree_fixed_w34_guard` | 6 | 0.71294 | 0.50442 | 0.35332 | corrected detail support is more conservative and does not improve style |
| `ema_bodytransport_lowfree_fixed_w34_guard` | 7 | 0.71068 | 0.51199 | 0.34682 | worse |
| `ema_bodytransport_lowfree_fixed_w34_guard` | 8 | 0.71164 | 0.51005 | 0.34867 | worse |
| `ema_transport_adain_w34_guard` | 6 | 0.71343 | 0.49859 | 0.35772 | conservative local moment carrier; content-safe but still below style ceiling |
| `ema_transport_adain_w34_guard` | 7 | 0.71083 | 0.50747 | 0.35010 | worse |
| `ema_transport_adain_w34_guard` | 8 | 0.71221 | 0.50527 | 0.35235 | worse |
| `ema_transport_adain_w40_style` | 6 | 0.71429 | 0.52948 | 0.33609 | style pressure gives only a tiny gain and breaks the LPIPS target |
| `ema_transport_adain_w40_style` | 7 | 0.71076 | 0.53770 | 0.32858 | worse |
| `ema_transport_adain_w40_style` | 8 | 0.71199 | 0.53626 | 0.33018 | worse |

Interim conclusion: the detail-gate correction is not the missing style lever.
The earlier `0.71431 / 0.50034` unfixed low-free result was not being unfairly
suppressed by the implementation bug; if anything, the extra detail leakage gave
it a little style. The conservative AdaIN carrier is a cleaner content-safe
carrier (`0.71343 / 0.49859`) but still does not cross the style ceiling. The
`ema_transport_adain_w40_style` is the decisive negative test: the moment
carrier cannot tolerate stronger style pressure. It reaches only `0.71429`,
while LPIPS moves to `0.52948`.

Next action: move from body carrier tuning to the terminal OT target. The new
probe is `terminal_swd_mode="semantic_moment"`: match low-band region moments
and high-pass envelopes across quantile regions, but do not reward literal
unpaired high-frequency phase matching.

## Semantic Moment OT

Implementation:

- added `terminal_swd_mode="semantic_moment"`;
- matched low-band region mean/std and high-pass envelope mean/std by quantile
  region;
- did not use Seedream or any external teacher in training.

Remote run:

```text
LANCET_semantic_moment_full
exp/vae_backend/ema_semantic_moment
```

Full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_semantic_moment_adain_w30_guard` | 6 | 0.71325 | 0.49158 | 0.36263 | cleaner content-safe endpoint, but no style lift |
| `ema_semantic_moment_adain_w30_guard` | 7 | 0.71088 | 0.49955 | 0.35575 | worse |
| `ema_semantic_moment_adain_w30_guard` | 8 | 0.71210 | 0.49768 | 0.35770 | no recovery |
| `ema_semantic_moment_adain_w38_style` | 6 | 0.71441 | 0.53082 | 0.33518 | only a tiny style gain, with content failure |
| `ema_semantic_moment_adain_w38_style` | 7 | 0.71122 | 0.53777 | 0.32875 | worse |
| `ema_semantic_moment_adain_w38_style` | 8 | 0.71228 | 0.53623 | 0.33033 | no recovery |

Conclusion:

The semantic-moment target rejects another plausible theory: the style ceiling
is not mainly caused by raw SWD matching the wrong unpaired phase. Removing
literal phase pressure keeps the guarded branch content-safe but does not move
`clip_style`, while stronger pressure still follows the same style-content
tradeoff. Together with the full-train `style_emb` negative result, the current
evidence says that the missing component is a stronger localized style actuator,
not another endpoint loss scalar or post-hoc embedding fit.

## Sinkhorn Semantic Router

After `semantic_moment` failed to raise style, the next structural check was a
router-only hypothesis:

> if softmax semantic transport is smearing style, a near-doubly-stochastic
> Sinkhorn router with content-topology blend should recover object-level
> routing and improve style without content damage.

Variant:

```text
ema_sinkhorn_body_w28_guard
```

Smoke result:

| variant | status | peak VRAM |
|---|---|---:|
| `ema_sinkhorn_body_w28_guard` | `train_ok` | 8988 MB |

Full remote run:

```text
LANCET_semantic_router_sinkhorn
exp/vae_backend/ema_semantic_router
```

Full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_sinkhorn_body_w28_guard` | 6 | 0.70237 | 0.44125 | 0.39245 | excellent content/structure, style too weak |
| `ema_sinkhorn_body_w28_guard` | 7 | 0.70543 | 0.44364 | 0.39247 | slight style recovery, still far below target |
| `ema_sinkhorn_body_w28_guard` | 8 | 0.70493 | 0.44633 | 0.39030 | same direction |

Conclusion:

Sinkhorn routing is not the missing style lever either. It is almost the
opposite failure mode from the style-push branches: routing becomes
content/topology preserving enough to improve LPIPS and EC, but it suppresses
visible style. This is useful evidence. The next actuator cannot be "better
routing" alone. It needs a separated pair of fields:

1. a semantic router deciding **where** style may enter;
2. a style-amplitude/texton field deciding **how much** and **what local
   statistics** to inject inside the allowed regions.

The amplitude field must not be tied directly to transport confidence; otherwise
every safer router becomes an identity-preserving style suppressor.

## Transport-Amplitude Negative Result

Question: if semantic routing answers **where** style may enter, can a separate
local moment-envelope amplitude field answer **how much** style enters?

Implementation:

- added `style_blender_mode="transport_amp"`;
- `where_gate = transport_confidence(content, semantic_attn)`;
- `amp_gate = local_style_envelope(residual)`;
- low/mid/high AdaIN residuals are applied through `where_gate * amp_gate`
  plus content support and phase gates for detail bands.

Remote run:

```text
LANCET_transport_amp_full
exp/vae_backend/ema_transport_amp
```

Result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_transport_amp_w34_guard` | 6 | 0.71371 | 0.50176 | 0.35559 | near previous frontier, but still below target |
| `ema_transport_amp_w34_guard` | 7 | 0.71133 | 0.51023 | 0.34839 | worse |
| `ema_transport_amp_w34_guard` | 8 | 0.71233 | 0.50832 | 0.35024 | no recovery |
| `ema_sinkhorn_amp_w36_style` | 6 | 0.70525 | 0.51666 | 0.34087 | Sinkhorn router remains too conservative |
| `ema_sinkhorn_amp_w36_style` | 7 | 0.70614 | 0.51697 | 0.34109 | no style lift |
| `ema_sinkhorn_amp_w36_style` | 8 | 0.70698 | 0.51692 | 0.34153 | no style lift |

Conclusion: splitting transport permission from a scalar local amplitude is not
enough. The "what" field still lacks enough dimension; it is only a local
moment envelope. This is consistent with the negative full-train `style_emb`
tests: the global style vector exists, but the current actuator cannot convert
it into organized visible textons.

## Transport-Texton Mainline

New hypothesis: the next actuator must synthesize a local style residual field,
not merely scale an AdaIN envelope.

Implementation:

- added `style_blender_mode="transport_texton"`;
- `style_emb` conditions a small convolutional texton carrier over content
  band-pass features and local moment residuals;
- the carrier is decomposed into low/mid/high bands;
- semantic transport remains the `where` gate, while content support and phase
  gates guard the detail bands.

Smoke:

| variant | batch | status | peak VRAM |
|---|---:|---|---:|
| `ema_transport_texton_w34_guard` | 128 | `train_ok` | 10577 MB |
| `ema_transport_texton_w40_style` | 128 | `train_ok` | 10605 MB |

Full remote run:

```text
LANCET_transport_texton_full
exp/vae_backend/ema_transport_texton
```

Full result:

| variant | epoch | clip_style | content_lpips | EC | interpretation |
|---|---:|---:|---:|---:|---|
| `ema_transport_texton_w34_guard` | 6 | 0.71451 | 0.48261 | 0.36968 | best current balance; improves LPIPS over transport_amp while slightly lifting style |
| `ema_transport_texton_w34_guard` | 7 | 0.71200 | 0.49110 | 0.36233 | no better than e6 |
| `ema_transport_texton_w34_guard` | 8 | 0.71338 | 0.48886 | 0.36464 | stable but still below 0.72 |
| `ema_transport_texton_w40_style` | 6 | 0.71112 | 0.52287 | 0.33929 | stronger pressure hurts both style and content |
| `ema_transport_texton_w40_style` | 7 | 0.70586 | 0.53336 | 0.32938 | negative |
| `ema_transport_texton_w40_style` | 8 | 0.70811 | 0.53168 | 0.33162 | negative |

Conclusion:

`transport_texton` is the first positive actuator change after the scalar
amplitude gate: it gives the strongest balanced point so far at
`0.71451 / 0.48261`. The failure of the `w40` branch is important: the missing
style is not recovered by simply making the texton carrier stronger. The next
probe should test whether the current best backbone still has unused style
conditioning capacity by calibrating `style_emb` on the full training set, but
this should be treated as a diagnostic. Previous full-train `style_emb` runs on
other anchors were content-safe and style-negative.

This branch is still unsupervised. Seedream remains diagnostic only and is not
used as a training target.

## Texton Full-Train Style-Embedding Probe

Question: after `transport_texton_w34_guard/e6`, does the full training set let
`style_emb` unlock more visible texton specificity without changing the
backbone?

Remote run:

```text
LANCET_style_emb_texton_w34_fulltrain
exp/style_embedding_mainline_calibration/ema_transport_texton_w34_e6_fulltrain
```

Design:

- checkpoint: `ema_transport_texton_w34_guard/epoch_0006.pt`;
- latent set: full `latent-256-sd15-ema`;
- recipes: `m00_emb_swd_anchor`, `m01_embspatial_swd_anchor`,
  `m02_embspatial_highpass_style`;
- supervision: only in-dataset latent SWD plus teacher endpoint anchor; no
  Seedream or external model target.

Status: running on the remote 3060 at about `10.36GB` VRAM. This is a narrow
diagnostic. If it does not exceed the base `0.71451`, the conclusion is that
`style_emb` is not the missing mainline lever even after the texton actuator.
