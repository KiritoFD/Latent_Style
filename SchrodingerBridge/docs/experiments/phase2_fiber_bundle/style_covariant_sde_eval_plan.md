# Style-Covariant SDE Eval-Only Plan

Date: 2026-06-16

## Why This Probe Exists

Recent phase-2 evidence has already closed the plain stochastic route:

- isotropic latent noise can raise style, but the LPIPS slope is poor
- gate-supported noise is better than raw global highpass noise, but still not target-facing
- residual-envelope noise is the cleanest noise basis tried so far, but still not enough

The remaining open question is narrower:

> Is the failure caused mainly by "noise magnitude" or by the fact that the injected latent noise basis is spectrally wrong for the target style?

The attached theory note proposes replacing white noise with **style-covariant colored noise** built from target-style Fourier amplitude and randomized phase. This is worth a cheap eval-only probe because it changes only the inference noise basis, not the trained model.

## Current Read

This proposal is plausible as an empirical probe, but it should **not** be treated as a proved theorem yet.

What is likely true:

- Our data already supports the claim that plain isotropic noise is too unconstrained.
- Our data also supports the claim that structured support/basis matters more than just increasing `sigma`.
- The proposal is aligned with the current retention policy: `rough_probe_before_reintegration`.

What is **not** yet established:

- In VAE latent space, "amplitude = style, phase = structure" is only a heuristic. It is exact neither mathematically nor empirically in the current codebase.
- A per-target-image style latent would contaminate the closed-set `style_id` benchmark surface. For Distinct5-style evaluation, the noise prior must come from a **canonical per-style bank**, not the paired target image itself.

## Clean Experimental Contract

### Parent

- Primary parent: `i2sb_orthogonal_lowanchor050_k070_e3::epoch_0009`
  - current retained style-first train-time point
  - transfer `0.701429 / 0.372203`
- Optional diagnostic parent after primary closure:
  - `i2sb_latent_slerp_k070_e3::epoch_0002`
  - style-peak diagnostic only

### Solver

- `solver_family=solver_i2sb`
- same checkpoint
- same eval dataset
- same inference steps
- same random seed protocol

### Control Arms

For each tested `sigma`, compare:

1. deterministic parent (`sigma=0`)
2. isotropic Gaussian noise
3. style-covariant noise

No other mechanism change is allowed in the same sweep.

## Noise Source Definition

### Allowed source for this benchmark

- build a **per-style latent amplitude template** from the current style bank
- do not use the paired target image latent directly during the main Distinct5 board run

### Recommended first implementation

- encode a small fixed bank of target-style reference images into latents
- average their FFT amplitude per style
- at inference time:
  - sample random phase
  - reconstruct noise with that style amplitude
  - standardize per sample/channel

This keeps the probe compatible with the current closed-set `style_id` contract.

## Minimal Switch Surface

Add only default-off eval switches:

- `model.i2sb_noise_family = gaussian | style_covariant`
- `model.i2sb_style_noise_bank_root = ""`
- `model.i2sb_style_noise_bank_limit = 16`
- `model.i2sb_style_noise_phase_seed = 0`
- `model.i2sb_style_noise_amplitude_power = 1.0`
- `model.i2sb_style_noise_use_gate = true | false`

All defaults should preserve legacy behavior.

## Implementation Notes

### Important constraint 1

Inject the new noise **before** the existing gate/basis postprocessing, so the comparison stays local:

- first choose the raw noise family
- then apply the existing `i2sb_fiber_project_noise` / `i2sb_fiber_aligned_noise` path if enabled

### Important constraint 2

For this probe, turn off RMS renormalization in the gate path if possible:

- current `_i2sb_fiber_aligned_noise()` still supports `gate / gate_rms`
- that changes effective energy and makes interpretation noisier
- use plain clamped gate multiplication for the cleanest first read

### Important constraint 3

Record observability:

- `style_noise_family`
- `style_noise_bank_active`
- `style_noise_amp_mean`
- `style_noise_amp_std`
- `style_noise_phase_seed`
- `style_noise_post_std`
- existing gate stats

## Sweep

### First-pass cheap sweep

Run only:

- `sigma = 0.2`
- `sigma = 0.5`
- `sigma = 0.8`
- `sigma = 1.2`

If runtime or quality degrades too sharply, stop after `0.8`.

### Success criterion

We do **not** need a final winner immediately.

A positive first-pass read is any matched point where style-covariant noise beats isotropic noise at the same `sigma` by:

- higher transfer CLIP-S at same or lower LPIPS, or
- materially lower LPIPS at near-equal style

### Failure criterion

Close as negative if:

- every matched `sigma` is dominated by isotropic control, or
- any gain is within noise floor while runtime/complexity increases, or
- only style rises but LPIPS reopens back into the same damaged band as earlier raw Fiber-SDE lines

## Expected Outcomes

### If positive

Then the key problem was not "SDE itself is wrong", but "the latent noise prior was spectrally mismatched".

Next step:

- promote to a slightly richer eval-only scan
- compare bank-mean amplitude vs sampled-reference amplitude
- then decide whether training-time adaptation is warranted

### If negative

Then we should stop spending effort on latent-domain stochastic basis hacks and move the next constraint/basis experiment closer to:

- decoder-aware metrics
- RGB/high-frequency decoded residual basis
- explicitly learned local actuator basis

## Decision

This idea is worth doing as a **small eval-only probe**.

It is **not** yet worth reopening a long training lane or rewriting the main model around the theory claim.
