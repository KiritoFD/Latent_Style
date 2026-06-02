# 2026-06-01 Main Table Gap Analysis

## Evaluation contract

Use the strict 750-image all-5x5 protocol as the headline protocol. The primary metrics are:

- `clip_style`: target style recognition, higher is better.
- `content_lpips`: content preservation, lower is better.
- `EC = clip_style * (1 - content_lpips)`: internal Pareto diagnostic only.

Do not use transfer-only subsets or quick/n6 runs for headline claims.

## Current confirmed LANCET point

Selected base:

- Run: `exp/wikiart512_ema_spectral_stat_full_adapt_e2_b48/full_eval_true_integrate_epoch0002_b4`
- `clip_style = 0.790531`
- `content_lpips = 0.300558`
- `clip_dir = 0.277411`
- `EC = 0.5529`

This is the current base for representation/tokenizer work because it has the best confirmed style-content balance among the full all-5x5 512 evaluations.

## Reproduced baselines

### SaMAM 512

Training time:

- 0 to 5k: about 2h06m
- 5k to 7k: about 50m46s
- 7k to 10k: 5302.35s = 1h28m22s
- 0 to 10k total: about 4h25m

Inference and evaluation:

- 8k/9k/10k generation: 2250 images in 480.1s
- Single checkpoint inference: about 160s / 750 images
- Single image inference: about 0.213s/img
- SB eval for 8k/9k/10k: about 107.8s total

Metrics:

| Step | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 5k | 0.791244 | 0.283292 | 0.5671 |
| 7k | 0.784850 | 0.246103 | 0.5918 |
| 8k | 0.787916 | 0.190641 | 0.6377 |
| 9k | 0.786826 | 0.166118 | 0.6559 |
| 10k | 0.785089 | 0.164336 | 0.6562 |

Conclusion: SaMAM-512 is currently ahead of LANCET on LPIPS. Its best style is at 5k, while LPIPS continues improving through 10k. LANCET is style-competitive with SaMAM-512 but not yet content-competitive.

### SaMAM 256

Training time:

- 0 to 15k: 8669.64s = 2h24m30s
- 17k to 25k: 5512.33s = 1h31m52s
- 15k to 17k exact wall time not found.

Metrics:

| Step | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 5k | 0.684885 | 0.534389 | 0.3188 |
| 10k | 0.687492 | 0.473146 | 0.3622 |
| 14k | 0.696867 | 0.436278 | 0.3928 |
| 17k | 0.695625 | 0.419127 | 0.4041 |
| 20k | 0.694062 | 0.409598 | 0.4098 |
| 25k | 0.693823 | 0.393958 | 0.4205 |

Conclusion: SaMAM-256 style plateaus around 14k. Continued training mostly buys LPIPS.

### SaMST

Historical strict 750 result:

- Inference: 39.826s / 750 images
- Single image inference: 0.0531s/img
- Training probe: 1 epoch / 5 styles / 16 imgs per style = 67.687s
- `clip_style = 0.7194`
- `content_lpips = 0.4664`
- `EC = 0.3839`

Current WikiArt512 SaMST training is incomplete:

| Target | Current epoch |
|---|---:|
| Realism | 30/30 |
| Impressionism | 30/30 |
| Post_Impressionism | 21/30 |
| Expressionism | 15/30 |
| Symbolism | 15/30 |

Conclusion: LANCET clearly beats the historical strict SaMST point on both primary metrics, but the current WikiArt512 SaMST 30-epoch result is not yet available.

## Current ranking

### Where LANCET is ahead

- Stronger than historical strict SaMST on `clip_style`: 0.7905 vs 0.7194.
- Stronger than historical strict SaMST on `content_lpips`: 0.3006 vs 0.4664.
- Much stronger than SaMAM-256 on style at comparable all-5x5 headline level.

### Where LANCET is behind

- Slightly behind SaMAM-512 best-style checkpoint: 0.7905 vs 0.7912.
- Behind SaMAM-512 on LPIPS:
  - vs 5k: 0.3006 vs 0.2833.
  - vs 10k: 0.3006 vs 0.1643.
- Current LANCET 512 training/inference timing needs a fresh same-protocol measurement before making speed claims against SaMAM-512.

## Improvement plan

The next improvements should optimize the observed gap, not chase raw style blindly.

1. Preserve the 0.790 style level while lowering LPIPS.
   - Acceptance gate should be OR-based: keep a probe if either `clip_style` improves meaningfully or `content_lpips` improves meaningfully without catastrophic collapse.
   - Full all-5x5 eval is required before promoting a result.

2. Target weak matrix cells rather than global pressure.
   - Known weak style cells from the 0.790 base: Realism source to artistic targets, especially Expressionism and Symbolism, plus Expressionism to Realism.
   - Known LPIPS pressure: Symbolism and Expressionism targets.
   - Avoid naive global style-budget increases because they tend to worsen LPIPS.

3. Representation/tokenizer direction.
   - Keep the spectral-stat tokenizer initialization as the only robustly confirmed tokenizer direction so far.
   - Complex pair/content spatial actuator variants have not earned promotion; prior pair-content release was negative on quick/n6 and should not be escalated.
   - Next tokenizer work should test whether source/content-conditioned execution budgets can reduce LPIPS in hard cells without adding noisy spatial actuators.

4. Baseline completion.
   - Resume SaMST-512 to 30/30 for all five target models before making final SaMST claims.
   - Measure current LANCET 512 inference and training wall time under the same 750-image protocol used for SaMAM-512.

## Paper table decision

The paper main table was updated to the current 512 reproduced protocol. The text now makes the claim precise:

- LANCET beats historical strict SaMST.
- LANCET is style-competitive with SaMAM-512.
- LANCET still trails SaMAM-512 on LPIPS, so the next technical goal is content-preserving style execution, not more raw style pressure.

## 2026-06-01 low-cell sampling probe

Run:

- Config: `configs/wikiart512_ema_lowcell_weighted_from_0790_e1_b48.json`
- Checkpoint: `exp/wikiart512_ema_lowcell_weighted_from_0790_e1_b48/epoch_0001.pt`
- Parent: `exp/wikiart512_ema_spectral_stat_full_adapt_e2_b48/epoch_0002.pt`
- Training: 1 epoch, 160 batches, batch 48, LR `5e-5`
- Runtime: 120.67s, 7680 samples, 63.64 samples/s
- VRAM: about 10.36GB on the remote 3060

Quick/n6 gate:

| Run | all clip_style | all LPIPS | transfer clip_style | transfer LPIPS |
|---|---:|---:|---:|---:|
| Base quick repeat | 0.798453 | 0.330614 | 0.789446 | 0.332046 |
| Base gain2 | 0.799093 | 0.334419 | 0.790322 | 0.335585 |
| Base gain4 | 0.799627 | 0.346881 | 0.791343 | 0.347640 |
| Base strength1.25 | 0.800678 | 0.377970 | 0.792385 | 0.379861 |
| Low-cell b48 | 0.781003 | 0.396649 | 0.775308 | 0.397369 |

Decision: reject. It fails the OR gate because neither primary metric improves. It should not receive full all-5x5 evaluation.

Interpretation:

- Reweighting weak source/target cells is not a representation improvement. It changes the empirical risk but does not provide a new executable degree of freedom for the model.
- The result worsened both style and LPIPS, which suggests the model used the biased batches to move the shared vector field away from the existing good basin.
- The weak cells need conditional execution capacity or a better tokenizer geometry, not just more gradient frequency.

Next action:

- Do not continue the low-cell sampling line.
- Keep `0.790/0.300` as the base.
- Next probe should introduce a bounded, source/content-conditioned execution budget that can reduce movement in content-sensitive cells while preserving the spectral-stat tokenizer. The acceptance gate remains OR-based, but any full promotion still requires strict all-5x5 evaluation.

## 2026-06-01 execution-budget probes

Existing remote budget probes were checked before adding new code.

### Results

| Run | all clip_style | all LPIPS | transfer clip_style | transfer LPIPS | Decision |
|---|---:|---:|---:|---:|---|
| Base quick repeat | 0.798453 | 0.330614 | 0.789446 | 0.332046 | reference |
| `tokenbudget_gradfix_tokonly` quick | 0.798214 | 0.299028 | 0.788999 | 0.300052 | promising quick only |
| `tokenbudget_gradfix_tokonly` full | 0.790876 | 0.306589 | -- | -- | not promoted; LPIPS worse than base full |
| `truegrad_tokenbudget_full` quick | 0.799008 | 0.309128 | 0.790271 | 0.310399 | style neutral, LPIPS quick gain |
| `truegrad_tokenbudget_full` full | 0.791317 | 0.315730 | -- | -- | not promoted; LPIPS too high |
| `metric_budget_decoder_safety125` quick | 0.783885 | 0.352117 | 0.777616 | 0.352621 | reject |
| `budget_only_safety125` quick | 0.784784 | 0.357282 | 0.778579 | 0.357363 | reject |

### Interpretation

Target-style-only budget has limited value. The best quick result (`tokenbudget_gradfix_tokonly`) showed that low/high gains can reduce LPIPS on a small sample, but the full all-5x5 result did not beat the selected base. The safety125 budget variants are clearly negative.

This supports the stronger representation claim:

- A target-only execution budget is still too close to a global style strength knob.
- The budget must be conditioned on the source/content side, because the same target style has different content-risk depending on the source domain and image structure.
- The next budget design should be pair/content-conditioned and bounded, with very few trainable parameters, rather than a free spatial actuator.

### Next probe specification

Add or reuse a budget interface with the following constraints:

- Inputs: target tokenizer metric fields plus source style ID or low-cost content statistics.
- Output: two bounded gains `[low_gain, high_gain]` consumed by the existing low/high residual path.
- Initialization: exact identity budget `[1, 1]`.
- Bound: narrow log span first, e.g. `exp(tanh(logit) * log(1.25))`.
- Training: freeze renderer; train only the budget head for a short run.
- Gate: keep if either quick `clip_style` improves or quick LPIPS improves without a style collapse; promote only after full all-5x5.

The important distinction is that this is an execution budget, not a style representation by itself. The style metric still comes from spectral/color/tokenizer fields; the budget controls how aggressively that style is executed for a particular content condition.

## 2026-06-01 inference timing audit

Same-machine WSL timing was added for LANCET using the strict 750 all-5x5 set and checkpoint `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`.

### Baseline generation timings

| Method | Protocol | Wall time | sec/img | Notes |
|---|---|---:|---:|---|
| SaMAM-512 step10000 | generate 750 | 173.92s | 0.232 | `eval_samam_checkpoint_curve.py`, mamba path |
| SaMST epoch15 | generate 750 | 126.22s | 0.168 | resized content to 512 before inference |
| LANCET | generate 750 | 131.02s | 0.175 | `target_chunk_size=5`, `vae_decode_batch_size=2`, save generated PNGs |

LANCET is now faster than SaMAM generation under this measured path and slightly slower than SaMST. This is generation-only, so it must not be compared to full eval time.

### LANCET full eval timing

Run:

- Output: `exp/timing_20260601/lancet_fulleval750_b2_tchunk5_vaebs2`
- Full wall time: 221.81s
- Internal `wall_total`: 218.05s
- `clip_style`: 0.792319
- `content_lpips`: 0.355036

Timing breakdown:

| Component | Seconds |
|---|---:|
| `decoded_to_cpu` | 111.76 |
| `lpips_generated` | 41.51 |
| `clip_generated` | 9.91 |
| `source_clip` | 9.85 |
| `lancet_generate` | 9.23 |
| `vae_decode` | 7.78 |
| `style_prototypes` | 6.12 |

Interpretation:

- The renderer is not the inference bottleneck: LANCET forward is about 9.2s for 750 outputs.
- Naive VAE compile paths are not promoted. Earlier ONNX/JIT/torch.compile VAE tests were slower end-to-end or added complexity without a win.
- The dominant cost is moving decoded 512 images to CPU for save/PIL metrics. This means further infra work should focus on structural evaluation changes: metric-on-GPU without PIL roundtrips, cached source/style features, and separating generation timing from full metric timing.
- `target_chunk_size=5` is useful for generation-only timing because it avoids repeated latent loads, but it does not make full eval faster than the older full eval path.

Decision:

- Keep `--generate-only` and target chunking as timing tools.
- Do not make JIT/ONNX VAE compile the default.
- Use normal full eval for headline metrics until a GPU metric path is proven bitwise or tolerance-equivalent.

## 2026-06-01 representation/actuation probe

Tooling was extended in `tools/probe_style_representation.py`:

- `tokenizer_code_pairs.csv`: pairwise tokenizer code distance/cosine.
- `generated_delta_stats.csv`: target-wise latent residual norm.
- `generated_delta_pairs.csv`: pairwise generated residual distance/cosine.
- Summary correlations between real latent style geometry, tokenizer code geometry, and generated residual geometry.

`tools/eval_wikiart512_latent.py` now also writes `generated_delta_diagnostics`
into `summary.json` for non-`--generate-only` runs. This makes residual-rank
tracking part of quick/full eval, not a separate one-off diagnostic.

Run:

- Checkpoint: `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`
- Data: `/mnt/f/wikiart_latents_512_ema`
- Output: `exp/probes_20260601/local_hist_e8_latent_tokenizer_delta20`
- Latent stats: 1000 samples per style.
- Delta probe: 20 content latents per style, all five target styles.

Results:

| Probe | Value |
|---|---:|
| tokenizer effective rank | 3.986 |
| tokenizer mean off-diagonal cosine | 0.015 |
| generated-delta effective rank | 3.324 |
| generated-delta mean off-diagonal cosine | 0.725 |
| corr latent L2 -> delta L2 | 0.823 |
| corr tokenizer L2 -> delta L2 | 0.426 |
| corr latent cosine -> delta cosine | 0.788 |
| corr tokenizer cosine -> delta cosine | 0.324 |

Smoke verification for eval integration:

- Output: `exp/timing_20260601/lancet_fulleval150_delta_diag`
- 150-transfer full eval smoke.
- `generated_delta_effective_rank = 3.325`
- `generated_delta_mean_offdiag_cos = 0.703`

Diagnosis:

- The current tokenizer codes are separated. Tokenizer rank collapse is not the first-order failure mode for this checkpoint.
- The generated residuals are still too aligned across target styles. This is an actuation/injection bottleneck: the renderer maps distinct controls into a low-rank shared edit direction.
- The residual geometry follows real latent style geometry more than tokenizer code geometry. This is a useful sign that the model sees data-domain style structure, but the injection path is not preserving enough controllable directions.

Next model implication:

- Do not spend the next run on a larger style embedding alone.
- Add or prioritize switches that change how controls are consumed: bounded source/content execution budget, multi-site token injection, or residual-rank diagnostics.
- Since SaMAM-512 wins mainly on LPIPS, the first switch should be content-risk-aware execution budget rather than more global style force.

## 2026-06-01 bounded execution budget implementation

Implemented switch:

- `model.execution_budget_mode`: `none`, `scalar`, or `low_high`.
- `model.execution_budget_hidden_dim`.
- `model.execution_budget_log_span`, default `log(1.25)`.
- `training.freeze_mode: budget_only`.

Design:

- Default is `none`, so existing checkpoints/configs keep historical behavior.
- `low_high` decomposes the predicted residual into a 3x3 average-pooled low
  component and a high residual, then applies bounded gains.
- The budget head input is the current style/time code plus cheap content
  statistics from the source latent.
- Final layer is zero-initialized, so initial gains are exactly one.

Smoke evidence:

- Config: `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_execution_budget_from_hist_e1.json`.
- Resume: `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`.
- Freeze mode selected exactly six trainable parameters:
  `execution_budget_head.0/1/3.{weight,bias}`.
- Short WSL run completed in `exp/local_wsl_wikiart512_execution_budget_from_hist_e1/epoch_0001.pt`.
  This used `virtual_length_multiplier=0.02`, 11 batches, and peak VRAM about
  `1.35/1.52GB` because only the budget head was trainable.
- 150-transfer eval output:
  `exp/timing_20260601/execution_budget_smoke_fulleval150`.

150-transfer comparison:

| Run | clip_style | content_lpips | delta rank | delta offdiag cosine |
|---|---:|---:|---:|---:|
| base 150 smoke | 0.802409 | 0.358092 | 3.325 | 0.703 |
| budget 11-batch e1 | 0.802382 | 0.358259 | 3.325 | 0.703 |

Decision:

- Wiring is valid and identity-safe.
- The 11-batch local result does not improve either metric or residual geometry,
  so it is not promoted.
- This does not fully falsify the budget idea; it says the narrow identity
  budget does not move enough under a tiny local run. If continued, run it as a
  longer remote 3060 probe or combine it with a residual-rank/LPIPS-aware loss.
- The next code-level probe should probably target how tokens enter the actuator
  rather than only scaling the final residual.

## 2026-06-01 multi-site style injection implementation

Implemented switch:

- `model.style_injection_mode`: `none`, `body`, `decoder`, or `body_decoder`.
- `model.style_injection_hidden_dim`.
- `model.style_injection_scale`.
- `training.freeze_mode: injection_only`.

Design:

- Default is `none`, preserving old behavior.
- `body` injects a zero-initialized feature bias after semantic body routing.
- `decoder` injects before decoder-side `NormFreeModulation`.
- `body_decoder` enables both.
- The injection head input is the current style/time code plus cheap content
  statistics from the source latent.
- Final layer is zero-initialized. Local check with the historical e8 checkpoint
  gave max output difference `0.0` before training.

Probe:

- Config: `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_style_injection_from_hist_e3.json`.
- Resume: `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`.
- Freeze mode selected 12 trainable injector parameters.
- Three short local epochs completed in
  `exp/local_wsl_wikiart512_style_injection_from_hist_e3`.
- Peak VRAM about `1.93/2.07GB`.

150-transfer comparison:

| Run | clip_style | content_lpips | delta rank | delta offdiag cosine |
|---|---:|---:|---:|---:|
| base 150 smoke | 0.802409 | 0.358092 | 3.325 | 0.703375 |
| injection e1 | 0.802445 | 0.358082 | 3.335 | 0.703844 |
| injection e3 | 0.802399 | 0.358237 | 3.344 | 0.702829 |

Decision:

- This switch affects generated-delta geometry more than the final residual
  budget, so it is a better actuation-side candidate.
- The 150-smoke primary metrics are neutral, not a promotion.
- Next useful run: longer remote 3060 injection probe or an injection probe with
  an explicit residual-geometry objective. Full 750 eval is not justified until
  a quick gate shows either a primary metric improvement or a clearer residual
  geometry gain without LPIPS damage.

## 2026-06-01 generated-delta diversity objective

Implemented optional bridge switches:

- `w_generated_delta_diversity`
- `generated_delta_diversity_margin`

The loss groups non-identity samples by target style and penalizes positive
cosine between target-wise mean generated residuals. It is designed to test the
actuation bottleneck found by the tokenizer/delta probes: tokenizer codes are
well separated, but generated residuals are too aligned.

Short WSL probes from the historical e8 checkpoint:

| Run | clip_style | content_lpips | clip_dir | delta rank | delta offdiag cosine | wall |
|---|---:|---:|---:|---:|---:|---:|
| base 150 smoke | 0.802409 | 0.358092 | 0.370576 | 3.325 | 0.703375 | 74.83s |
| injection e3 | 0.802399 | 0.358237 | 0.370410 | 3.344 | 0.702829 | 72.77s |
| injection + delta diversity 0.05 | 0.802397 | 0.358240 | 0.370265 | 3.344 | 0.702397 | 73.64s |
| injection + delta diversity 0.50 | 0.802388 | 0.358267 | 0.370028 | 3.345 | 0.698567 | 73.16s |

Decision:

- Keep the switch as an experimental diagnostic only.
- Do not promote it to the clean baseline: it improves residual geometry but
  does not improve `clip_style`, `clip_dir`, or LPIPS on the quick gate.
- The next model change should increase the executable representation capacity
  or change the consumer path, not simply increase the weight on residual
  orthogonality.

## 2026-06-01 residual variance decomposition

`tools/eval_wikiart512_latent.py` now includes
`generated_delta_variance_decomposition` in `summary.json`.

Base 150-transfer rerun:

- Output: `exp/timing_20260601/lancet_fulleval150_delta_var`
- Checkpoint: `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`

Metrics stayed aligned with prior smoke:

| clip_style | content_lpips | clip_dir | delta rank | delta offdiag cosine |
|---:|---:|---:|---:|---:|
| 0.802358 | 0.358096 | 0.370340 | 3.325 | 0.703379 |

Variance decomposition:

| Term | Ratio |
|---|---:|
| target_between_ratio | 0.028285 |
| source_style_between_ratio | 0.143721 |
| source_image_between_ratio | 0.950058 |
| source_target_pair_between_ratio | 0.175145 |
| target_after_source_image_ratio | 0.566358 |

Reading:

- Generated residuals are mostly source-image/content-driven in raw variance.
- Target style is still a strong factor after source-image mean removal.
- The immediate bottleneck is not only tokenizer representation and not only
  residual geometry. The missing object is an executable target-style carrier
  that is not drowned out by content-conditioned motion, plus a content gate to
  keep LPIPS low.

Next experiment should therefore combine:

- a target carrier consumed at multiple blocks;
- a content/source gate controlling residual amplitude or low/high bands;
- the existing residual diagnostics as the quick promotion gate.

## 2026-06-01 carrier/gate injection probe

Implemented switch:

- `model.style_injection_form`: `mixed` or `carrier_gate`.
- `model.style_injection_gate_log_span`.

`carrier_gate` keeps `style_injection_mode=body_decoder` but separates the
inputs:

```text
style_code + time_code -> carrier bias
content latent stats   -> bounded gate
feature injection      = tanh(carrier) * exp(tanh(gate) * gate_log_span)
```

Probe:

- Config: `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_carrier_gate_from_hist_e3.json`
- Resume: `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt`
- Freeze mode: `injection_only`
- Trainable params: carrier/gate modules at body and decoder sites.
- Short local WSL training completed in about 33s wall for three tiny epochs.

150-transfer comparison:

| Run | clip_style | content_lpips | clip_dir | delta rank | delta offdiag cosine | target ratio | source-image ratio | target after source-image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base delta-var | 0.802358 | 0.358096 | 0.370340 | 3.325 | 0.703379 | 0.0283 | 0.9501 | 0.5664 |
| carrier_gate e3 | 0.802273 | 0.358190 | 0.369542 | 3.349 | 0.702816 | 0.0283 | 0.9500 | 0.5669 |

Decision:

- Keep the implementation as an ablation switch.
- Do not promote it. It improves residual-rank diagnostics but regresses primary
  metrics and does not raise raw target variance.
- The next candidate needs a stronger target carrier, likely a residual-basis or
  block-token stream, not only channel-bias injection.
