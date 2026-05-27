# Phase-Envelope SWD

Date: 2026-05-27

## Diagnosis

The current failures do not prove that the VAE is unusable. They show that the
unpaired style objective is asking the model to match the wrong high-frequency
quantity.

Seedream 4.5 is used only as a diagnostic reference. Its image-space gap table
shows that our stronger branches often have enough or even excessive high-pass
energy, but that the high-pass residual is much more anti-phase with the source
structure:

| method | highpass_delta_energy | highpass_phase_cos | output_grad_mean |
|---|---:|---:|---:|
| `seedream45_golden` | 0.1011 | -0.3524 | 0.3467 |
| `t01_original_vae_e8` | 0.1346 | -0.8097 | 0.2722 |
| `ema_dynamic_guard_w28_e6` | 0.1464 | -0.7325 | 0.3476 |
| `ema_routed_w36_texton_e6` | 0.1517 | -0.7222 | 0.3853 |
| `ema_routed_w44_stylepush_e6` | 0.1506 | -0.7433 | 0.4093 |

Interpretation: the bottleneck is not simply "not enough style energy". The
model is spending high-frequency energy in a sign/phase direction that does not
respect the content manifold.

## Mathematical Failure

The active micro-SWD branch uses signed high-pass patches:

```text
high_pass = z - lowpass(z)
micro_features = [high_pass, sobel_magnitude(high_pass)]
```

For unpaired style transfer, the target image's signed high-pass phase is not a
valid target for the source image. A brush stroke edge in a randomly paired
style image has no reason to share the source object's phase. Matching signed
high-pass distributions therefore injects a hidden contradiction:

```text
match style high-frequency statistics
and
match random target high-frequency signs
```

The model can lower SWD by producing anti-phase residuals or texture mist. This
can raise some style proxies while damaging visible structure.

## Proposed Objective

Separate high-frequency style into:

- **envelope/amplitude**, which should be matched to the style distribution;
- **phase/sign**, which should remain tied to the source structure.

The new micro-SWD feature replaces the signed high-pass term with an absolute
high-pass envelope:

```text
micro_features = [abs(high_pass), sobel_magnitude(high_pass)]
```

and keeps a small source phase regularizer:

```text
L = terminal_swd(abs/highpass_envelope) + lambda_phase * FourierPhaseLock(pred, content)
```

This is still fully unsupervised. It uses no Seedream teacher signal in the
training path. Seedream only supplied the diagnostic evidence that the old loss
was optimizing the wrong high-frequency variable.

## Implementation

Added bridge config knobs:

```text
swd_signed_highpass_weight
swd_abs_highpass_weight
```

Default behavior is unchanged:

```text
swd_signed_highpass_weight = 1.0
swd_abs_highpass_weight = 0.0
```

The new variants set:

```text
swd_signed_highpass_weight = 0.0
swd_abs_highpass_weight = 1.0
```

## Test

Remote smoke on the 12G machine:

| variant | status | peak VRAM |
|---|---|---:|
| `ema_phase_envelope_w36_guard` | `train_ok` | 9840 MB |
| `ema_phase_envelope_w44_style` | `train_ok` | 9839 MB |

Full run launched under:

```text
exp/vae_backend/ema_phase_envelope
```

Eval plan: epochs `6/7/8`, standard 750 protocol. Success requires
`clip_style > 0.72` with `content_lpips` preferably below `0.50`. Even if the
score does not cross the target, the follow-up diagnostic must check whether
`highpass_phase_cos` moves closer to Seedream without losing output gradient.
