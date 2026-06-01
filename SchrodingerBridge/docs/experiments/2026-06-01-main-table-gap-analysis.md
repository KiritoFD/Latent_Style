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
