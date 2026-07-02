# DualPath SpatialTexture Trigger

Date: 2026-06-09

This note originally defined the trigger for promoting:

- `inmortal_knee_e13_spatial_carriergate_bodydecoder_qedgegated_dualpath_spatial_seed42_b8a2`

That promotion has now already happened.

The note is now the closure/readout anchor for that handoff.

## Promotion outcome

The previous `dualpath_texture` line drained.

The prepared `dualpath_spatialtexture` follow-up was launched on the remote `3060`.

Observed launch behavior:

- first-health passed
- runtime memory stayed safely below the formal cap
- training completed with checkpoints through `epoch_0012`

## Current read after promotion

The new line has now produced an early fresh-eval curve, not just a train-health signal.

Current fresh-eval rows include:

- `epoch_0001`
- `epoch_0002`
- `epoch_0003`
- `epoch_0004`
- `epoch_0005`
- `epoch_0006`
- `epoch_0007`
- `epoch_0008`
- `epoch_0009`
- `epoch_0010`
- `epoch_0012`

Current curve shape:

- transfer style remains tightly bounded around:
  - `0.6916 to 0.6929`
- `LPIPS` rises from about:
  - `0.401` to `0.440`

## Current implication

So far, this promotion has **not** shown that:

- a broader spatial late branch is enough to reopen target-specific style

It has shown that:

- the family remains stable
- the family remains relatively clean
- but it still appears trapped in the same conservative low-style basin

## Current decision

Use this line as:

- an active piece of negative-to-mixed evidence against
  - `branch capacity alone solves the ceiling`

Do not yet treat it as:

- a style-ceiling rescue

Only revise that read if later non-CLIP / qualitative evidence clearly overturns the current early-curve pattern.
