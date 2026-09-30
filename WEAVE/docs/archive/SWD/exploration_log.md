# Semantic SWD Exploration Log

Date: 2026-07-07

## User Goal

Inspect existing data and code, fix the model-side use of semantic SWD, run new outputs, compute MUSIQ, and compare against the `aaai2027_v4` main table. Post-processing is allowed as a later knob, but semantic SWD is a core method/theory issue.

## Baseline Table Context

From `aaai2027_v4/paper.tex` main table:

- WEAVE D5: CLIP-S `0.7213`, LPIPS `0.2868`, MUSIQ `35.31`
- WEAVE P256: CLIP-S `0.6826`, LPIPS `0.2031`, MUSIQ `45.68`
- WEAVE R5: CLIP-S `0.7434`, LPIPS `0.2904`, MUSIQ `31.72`
- SaMAM D5: MUSIQ `51.17`
- Seedream D5: MUSIQ `69.51`
- SD-Turbo D5: MUSIQ `60.72`

Interpretation: MUSIQ is the weak axis for WEAVE-style outputs, so semantic SWD should be judged by whether it improves perceptual naturalness without destroying directionality.

## Step 1: Re-examined SWD Formulation

Initial implementation work had already added cross-attention-guided SWD support in:

- `src/losses620.py`
- `src/spectral_losses620.py`
- `src/blocks620.py`
- `src/model620.py`
- `src/spectral_bridge620.py`

The important theoretical correction was that guidance should be used as sampling mass over spatial locations. It should not multiply latent features before projection, because feature scaling changes the empirical distribution being measured.

Current guided branch:

- Reads `model.last_cross_attn_guidance`.
- Normalizes it to a positive spatial weight map.
- Uses deterministic weighted quantile sampling before sliced projections.

This matches the method story: cross-attention routing identifies where the model edits content; SWD then matches distributions on those local edit regions.

## Step 2: Found a Training-Path Mismatch

The first suspicion was that training might use `losses620.py` rather than `spectral_losses620.py`. That was false for current configs.

Actual config:

- `model.contract_family = "620_spectral_ode"`
- Trainer uses `SpectralODEObjective620` from `src/spectral_losses620.py`

So the problem was not the wrong objective class.

## Step 3: Found the Real Reference-Latent Bug

`SpectralODEObjective620.compute()` read:

```python
style_latent = conditioning.get("target_style_latent")
```

But `AdaCUTLatentDataset.__getitem__()` returns:

```python
{
  "content": content,
  "target_style": target_style,
  "target_style_id": target_style_id,
  "source_style_id": content_style_id,
}
```

There is no `target_style_latent` key in the normal packed latent batch. Therefore spectral training passed `style_latent=None` into the model. This meant the cross-attention tokens came only from class-level `style_id` memory, not from the sampled reference latent.

Fix:

- In `src/spectral_losses620.py`, when `target_style_latent` is absent, default `style_latent = target_style`.

## Step 4: Added Reference-Latent Style Tokens to Spectral Bridge

`SpatialBridge620` already had an intrinsic style path for target latents. `SpectralODEBridge620` did not.

Fix in `src/spectral_bridge620.py`:

- Added `style_condition_source` and `use_intrinsic_style`.
- If `style_condition_source` is one of `latent`, `target_latent`, `target_style_latent`, or `target_dino_patches`, and a tensor `style_latent` is provided, encode it through a small CNN.
- Pool to `16 x 16` style features.
- Project to local style tokens.
- Use the pooled feature as `style_global`.
- Fall back to `StyleConditioner620(style_id)` when no tensor reference is available.

This gives semantic SWD a real local reference signal instead of class-only memory.

## Step 5: Added a Probe Tool

Added `scripts/probe_semantic_swd_batch.py`.

Purpose:

- Build the real dataset.
- Build the real trainer/model/objective.
- Move one real batch to device.
- Run `loss_fn.compute()`.
- Print `swd_guidance_active`, `style_latent_conditioning_active`, guidance-map shape/mean/std, and selected losses.

Verified for `semantic_swd_ref_guided_cons5`:

- `swd_guidance_active = 1.0`
- `style_latent_conditioning_active = 1.0`
- `last_cross_attn_guidance.shape = [4, 1, 64, 64]`
- `last_cross_attn_guidance.std > 0`

Verified for `semantic_swd_ref_global_clean5`:

- `style_latent_conditioning_active = 1.0`
- `swd_guidance_active = 0.0`

This is the intended ablation pair.

## Step 6: Cleaned Execution Path

Problem: `src/exp/*` contains many historical source snapshots. Normal training should not import them, but they pollute search and can cause accidental stale-code runs.

Fix in `src/trainer.py`:

- Added `_assert_active_source_modules`.
- For `620_spectral_ode`, it checks `config_schema`, `model`, `trainer`, `spectral_bridge620`, `spectral_losses620`, `blocks620`, `style_encoder620`, and `utils.training`.
- For `620_spatial_bridge`, it checks the corresponding spatial modules.
- If any active module is loaded from `src/exp/*`, training raises immediately.
- Training logs now print active source module paths.

Fix in `src/utils/training.py`:

- Added `spectral_losses620.py` and `spectral_bridge620.py` to source snapshots.
- Removed stale `lancet_backbone.py`, `lancet_blocks.py`, and `lancet_runtime.py` from source snapshots.

## Step 7: Found an Observability Bug

During the 5-epoch guided run, `logs/training_20260707_163520.csv` showed:

- `swd_guidance_active = 0.0`
- `swd_guidance_mean = 0.0`
- `swd_guidance_std = 0.0`

This contradicted the real-batch probe.

Root cause:

- `TRAIN_LOG_COLUMNS` contained the SWD guidance columns.
- `append_training_log()` did not write `swd_guidance_active`, `swd_guidance_mean`, or `swd_guidance_std` into `row_map`.
- Missing keys defaulted to `0.0`.

Fix:

- Added those three fields to `row_map` in `src/utils/training.py`.

Status:

- The just-completed 5-epoch CSV still contains the old false zeros because the logging fix happened after the run.
- Future runs should log the guidance fields correctly.
- The probe remains the reliable activation proof for the completed run.

## Step 8: Ran New Guided Config

Created:

- `configs/semantic_swd_musiq/semantic_swd_ref_guided_cons5.json`
- `configs/semantic_swd_musiq/semantic_swd_ref_global_clean5.json`

Ran:

```powershell
python src\run.py --config configs\semantic_swd_musiq\semantic_swd_ref_guided_cons5.json
```

Completed:

- Checkpoint: `exp/semantic_swd_ref_guided_cons5/epoch_0005.pt`
- Full eval summary: `exp/semantic_swd_ref_guided_cons5/full_eval/epoch_0005/summary.json`
- Curve summary: `exp/semantic_swd_ref_guided_cons5/full_eval/curve_summary.json`

Full eval metrics:

- Transfer CLIP-S: `0.6923`
- Transfer CLIP-S delta over fixed IDT: `0.0524`
- Transfer LPIPS: `0.4140`
- All-pairs CLIP-S: `0.7203`
- All-pairs CLIP-S delta over fixed IDT: `0.0803`
- All-pairs LPIPS: `0.4015`
- Identity CLIP-S: `0.8321`
- Identity LPIPS: `0.3513`

Training loss trend:

- Epoch 1 loss: `6.0911`, SWD: `0.4102`
- Epoch 5 loss: `5.0873`, SWD: `0.2949`

MUSIQ for this new run:

- Pending. Do not reuse old `semantic_swd_guided_cons5` MUSIQ as evidence for the new reference-latent mechanism.

## Step 9: Current Interpretation

The method story should not be "we added a generic SWD regularizer." The correct story is:

The model builds a local style-edit field through target-reference cross-attention. Semantic SWD uses that field as an empirical transport mass, so endpoint distribution matching is concentrated where the bridge actually edits content. This makes SWD a geometry-aware terminal constraint rather than a global texture-matching penalty.

However, the completed guided run does not yet prove a MUSIQ gain. The CLIP-S is close to table WEAVE D5 (`0.7203` vs `0.7213` all-pairs), but LPIPS is worse (`0.4015` vs `0.2868`). MUSIQ must be recomputed before this becomes a paper claim.

## Next Actions

1. Rerun a short guided config after the logging fix to confirm CSV `swd_guidance_active = 1.0`.
2. Compute MUSIQ for `exp/semantic_swd_ref_guided_cons5/full_eval/epoch_0005/images`.
3. Run `semantic_swd_ref_global_clean5` and compare against guided with the same reference-latent token path.
4. If guided improves MUSIQ without unacceptable CLIP/LPIPS loss, update method as "cross-attention-routed empirical SWD."
5. If guided does not improve MUSIQ, present it as an ablation and look at post-processing/endpoint fiber control separately.

## Session 2 (2026-07-07 evening): SWD weight + patch SWD mechanism

### MUSIQ sweep results (class-memory path restored)

| run | SWD weight | patch mode | CLIP-S | LPIPS | MUSIQ |
|---|---|---|---|---|---|
| swd_cm_cons5 | 8 | off | 0.7216 | 0.3342 | 40.84 |
| swd_cm_softer5 | 8 | off (floor0.7/pow0.4) | 0.7230 | 0.3343 | 40.56 |
| swd_cm_strongswd5 | 12 | off | 0.7228 | 0.3151 | 42.95 |
| ref_guided_cons5 (regressed) | 8 | off (intrinsic CNN) | 0.7203 | 0.4015 | 39.33 |

Key findings:
1. Reference-latent intrinsic-CNN style path REGRESSED all metrics vs class-memory. Reverted.
2. SWD weight is the dominant MUSIQ lever: 8->12 gave strict Pareto win (MUSIQ +2.1, CLIP +0.001, LPIPS -0.019).
3. Softer guidance (floor/power) is neutral-to-negative for MUSIQ.

### Mechanism diagnosis

Active SWD (`_sliced_wasserstein`) projects individual latent pixels (4-dim channel
vectors) onto random directions and matches sorted quantiles. This matches only the
per-pixel COLOR/TONE marginal histogram — the sort destroys all spatial arrangement,
so it carries zero local-texture information. MUSIQ is a no-reference perceptual metric
that rewards natural texture/sharpness, which pixel-marginal SWD cannot target. The
config declared `swd_patch_sizes:[1,3,5,9]` but the active objective ignored it.

### Mechanism implemented: multi-scale patch SWD

`_patch_swd` in src/spectral_losses620.py: im2col (F.unfold) lifts each spatial location
to a (C*k^2)-dim local texture vector before slicing. patch=1 reduces to pixel SWD.
Cross-attn guidance map is downsampled to the unfolded grid and reused as empirical
sampling mass. Config knobs: swd_patch_mode ("off"|"multi"), swd_patch_sizes,
swd_patch_weights. Compile + real-batch probe verified (swd_guidance_active=1.0,
class-memory path, loss_swd finite).

### Batch 2 (running)
- swd_cm_w16: weight 16, patch off (push the confirmed lever)
- swd_cm_patch12: weight 12, patch multi [1,3,5]
- swd_cm_patch_w16: weight 16, patch multi [1,3,5]
Target to beat: strongswd5 MUSIQ 42.95 / LPIPS 0.3151.
