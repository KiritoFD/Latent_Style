# Rebuttal Remaining Experiments Plan

**Date:** 2026-07-16  
**Inputs:** `batch1_summary.md`, `WEAVE/docs/reproduction/artfid_d5_audit.md`, and the current paper draft.  
**Principle:** run only experiments that close a concrete evidence gap. Do not continue generic parameter or architecture sweeps.

## Decision Summary

Batch 1 is sufficient to support the source-aligned endpoint claim, but it is not sufficient to support the current AdaIN, reference-pool, early-stop, or no-HH claims as written.

| Priority | Item | Action | Why it is required |
|---|---|---|---|
| P0 | ArtFID manifest audit | Keep only packets that resolve `750/750` to the D5 manifest; regenerate StyleAligned/Z-STAR only if their direct ArtFID comparison is essential. | Current audit mixes Distinct5 source images for WEAVE with Random20 sources for StyleAligned/Z-STAR. |
| P0 | AdaIN D1 control | Rerun D1 with `model.endpoint_adain_scale=0.0`, verify generated-image hashes differ from D0, then evaluate. | The current script writes a top-level key and leaves `model.endpoint_adain_scale=2.0` active. |
| P0 | Reference-pool B1 | Resample one fixed without-replacement pool per target style per iteration, shared by all requests to that style; report CI of the mean paired margin. | Existing B1 resamples a new subset for every row and does not test a reference-pool choice. |
| P0 if early stop remains a contribution | Canonical no-stop curves | Train canonical seed 42 and seed 123 to all 15 epochs with active stopping disabled; apply the frozen gate retrospectively. | Their present "oracle" is censored at the active stopping epoch. |
| P1 if promoting HH | HH seed replication | Run learned HH head with seed 7 under the same 15-epoch no-stop protocol; inspect selected outputs and measure the final board. | D5 improves both style metrics at seed 42 but costs a small amount of DINO-C/LPIPS. |
| Writing only | Method/ablation wording | Remove the claims that AdaIN has been shown causal, that HH is negative, that `lambda_LL=0.3` is a sharp sweet spot, and that early stop is architecture-independent. | Batch 1 contradicts or weakens each claim. |

## Evidence Already Sufficient

### Source-aligned endpoint

D4 is strong matched evidence. Removing source alignment lowers the best DINO-S from `0.4917` to `0.4894`, worsens LPIPS from `0.2595` to `0.3138`, and makes the frozen internal gate stop eight epochs before the DINO-S oracle. This is the core mechanistic ablation for the paper.

### LL weighting

D3 is a weak but usable directional result: `lambda_LL=1.0` peaks at `0.4910` versus `0.4917` for the production setting. Do not state that the LL weight is decisive or uniquely optimal. Keep the gradient-routing explanation, but frame the weight as a modest preconditioner.

### Current HH result

D5 changes the decision boundary:

| Variant | DINO-S | CLIP-S | DINO-C | LPIPS |
|---|---:|---:|---:|---:|
| Baseline, epoch 4 | 0.4917 | 0.7127 | 0.8104 | 0.2595 |
| Learned HH head, epoch 4 | 0.4930 | 0.7164 | 0.8061 | 0.2670 |

The HH head improves both reported style scores, but has a small content cost. It does **not** improve content preservation. The current paper must not claim that a matched HH-head ablation was negative. Whether to promote D5 is a model-selection decision; do not tune `w_hh` before obtaining a seed-7 replication.

## P0-A: Canonical ArtFID Audit

### Goal

Make every ArtFID bar in the paper comparable by using the same D5 request manifest, target reference pool, feature extractor, resize path, and LPIPS implementation.

### Existing-packet decision (2026-07-16 audit)

The archived `D5-512/stylealigned` and `D5-512/zstar` packets each contain 750
images, but only `15/750` filenames resolve to the current Distinct5 source
set. In contrast, the WEAVE packet resolves `750/750`. The two archived
diffusion-baseline packets therefore use the older Random20 source manifest and
are not directly comparable to the current D5 board. They must not be cited as
direct ArtFID bars in the main paper. Regeneration is optional rather than a
submission blocker; retain these values only in a clearly labelled historical,
non-comparable record if needed.

### Required methods

- IDT, TGT, WEAVE, and SaMam are currently canonical and can remain in the
  direct chart. Seedream may remain only when its exact request manifest is
  verified.
- StyleAligned and Z-STAR require regeneration before direct comparison. Do
  not block submission on that regeneration: omit them from the direct chart
  otherwise.

### Protocol

1. Freeze `canonical_d5_pairs.csv`: 750 source-target requests, five target styles, 150 requests per target style.
2. Use the same source image for every method request. Do not accept another dataset merely because each style has 30 images.
3. Confirm `750/750` filename-to-source matches before computing metrics.
4. Report per-target FID, source LPIPS, and ArtFID, followed by their mean over five targets.
5. Preserve TGT separately as an empirical control. Its random-exemplar uncertainty is already available in `WEAVE/results/tgt_artfid_random_stability.json`; it is not an absolute ArtFID bound.
6. Record method commit/config, output directory, evaluator hash, and file count.

### Acceptance

- No direct ArtFID comparison mixes the current Random20 and Distinct5 source manifests.
- The main figure receives only the canonical values.
- ArtFID is described as a composite plausibility diagnostic, with raw FID and LPIPS components in the supplement.

## P0-B: Correct D1 AdaIN Ablation

The current script contains this ineffective override:

```python
base_inf["endpoint_adain_scale"] = 0.0
```

The active configuration key is nested:

```python
base_inf["model"]["endpoint_adain_scale"] = 0.0
```

### Protocol

1. Generate D0 and corrected D1 from the same epoch-4 checkpoint and canonical D5 manifest.
2. Save resolved overrides with the output packet.
3. Compare SHA-256 hashes of at least 20 corresponding PNG files and report mean pixel/latent difference. This proves the switch reached the inference path.
4. Evaluate DINO-S, CLIP-S, DINO-C, LPIPS, and ArtFID components.

### Interpretation rule

Only after this corrected run may the paper state whether stepwise AdaIN is causal. If D1 remains nearly identical, remove AdaIN from the claimed active mechanism rather than trying to explain it away.

## P0-C: Correct B1 Reference-pool Sensitivity

### Invalid aspect of the current implementation

`expB_reference_margin.py` samples a distinct reference subset inside the per-row loop. This changes the pool for every source image. The intended question is whether a **style reference pool** changes the board-level conclusion.

### Correct protocol

For pool sizes `m=8` and `m=16`, repeat at least 1,000 times:

1. For each target style, draw one size-`m` subset without replacement.
2. Reuse that subset for all 150 requests to the target style.
3. Compute board-level paired mean margin
   `mean(DINO-S(WEAVE) - DINO-S(IDT))`.
4. Report mean, standard deviation, 95% interval, minimum, and fraction of draws with positive **board mean** margin.
5. Separately report the per-request positive fraction. It is diagnostic, not the criterion for the board-level claim.

### Permitted claim

Use “positive mean paired margin is stable under target-reference-pool resampling” only if the 95% interval of the board mean is positive. Do not use “consistently beats IDT” for individual requests: the current full-pool result is positive for about 69.5% of pairs, not all pairs.

## P0-D: Early-stop Validation or Claim Removal

The present seed-42 and seed-123 runs end at their gate event. Their reported oracle is therefore the best **observed** epoch, not the best epoch of the intended 15-epoch trajectory.

### If retaining the early-stop contribution

1. Run canonical seed 42 and seed 123 for 15 epochs with `internal_early_stop_enabled=false`.
2. Keep the exact current threshold and probe configuration frozen.
3. Evaluate all 15 checkpoints and compute oracle epoch, selected epoch, DINO-S regret, and content metrics at both points.
4. Recompute the seed-7 gate using the same relative rule. If it still does not fire, report this explicitly and do not claim unconditional robustness.

### If avoiding this work

Retain the probe as a diagnostic figure, but remove the practical early-stop contribution from the abstract/conclusion and call it a canonical-run observation.

## P1: HH Model Decision

Do this only if the D5 HH-head checkpoint is being considered as the final method.

1. Repeat D5 with seed 7, full 15 epochs and frozen selection rule.
2. Compare D5 and no-HH at their internally selected epochs on the full D5 board.
3. Inspect a fixed qualitative panel before changing the main checkpoint.
4. If the style gain repeats with bounded content cost, promote D5 and update parameters, training time, main table, ArtFID, radar, and method text together.
5. If it does not repeat, retain no-HH but describe HH as an unresolved trade-off, not a rejected path.

## Explicitly Not Needed Now

- More generic `lambda_LL`, alpha, or scale sweeps.
- Multi-level Haar/db2 runs, unless the paper continues to claim measured Haar optimality.
- A human study.
- A full 3x3 seed x probe-batch grid.
- New architecture exploration beyond the one HH replication above.

## Execution Order

1. Canonical ArtFID audit.
2. Correct D1 and B1; both are inexpensive and remove invalid evidence.
3. Decide whether early stop remains a contribution. If yes, run canonical no-stop seed curves.
4. Decide whether D5 is the final model. If yes, run one paired seed replication.
5. Rewrite paper tables and claims only after these choices are resolved.
