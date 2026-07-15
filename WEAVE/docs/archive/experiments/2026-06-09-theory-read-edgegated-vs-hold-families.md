# Theory Read: EdgeGated vs Hold Families

Date: 2026-06-09

This note records the current theory-facing interpretation from the latest local non-CLIP evidence.

It is not a paper claim yet.
It is a mechanism read used to decide what kind of next line is justified.

## Scope

Compared families:

- `Hold4TwoStage`
- `Knee + spatial carrier body+decoder + edge-gated structure leash`

Read stack:

- local best-few `IntroStyle + DINO`
- local CPU-only `VLM`
- remote `CLIP/LPIPS` retained-point curves

## Stable read on Hold-family continuation

The `Hold4TwoStage` family is now negative enough to close as a direction.

Why:

- it was originally attractive because it looked like a geometry-first anchor that might later reopen style
- but the later reopen never became convincing on any serious read:
  - not on local `IntroStyle`
  - not on local `DINO`
  - not on the growing CPU-only `VLM` set

Latest local CPU-only `VLM` evidence:

- `Hold4TwoStage_e12`
  - `20 / 534` candidate overall wins
- `Hold4TwoStage_e02`
  - `13 / 480` candidate overall wins
- `Hold4TwoStage_e20`
  - `16 / 466` candidate overall wins

The more important point is not just low win-rate.

It is that even after hundreds of cases:

- `LBM-Knee` still keeps the stronger structure-side mean
- `Seedream` still dominates style and overall judgment
- `Hold4TwoStage` does not emerge as a hidden target-style winner

Theory implication:

- schedule shaping and release smoothing alone are not enough
- once style has been suppressed into a geometry-safe basin, later continuation is not reliably recovering target-specific style
- therefore the bottleneck is not just optimization patience
- it is missing representational capacity or missing style-targeted mechanism

## Why EdgeGated is still alive

The `EdgeGated` family is still not promotable.

But unlike `Hold4TwoStage`, it is still theory-positive enough to keep informing the next mechanism.

Latest CPU-only `VLM` evidence:

- `EdgeGated_e01`
  - `7 / 157` candidate overall wins
- `EdgeGated_e03`
  - `13 / 265` candidate overall wins
  - `41` structure subwins
  - `51` artifact-control subwins
- `EdgeGated_e12`
  - `7 / 156` candidate overall wins

This means:

- the family is still far from beating `Seedream`
- but it is not a dead branch
- it repeatedly improves the *kind* of things that a stricter structure leash should improve:
  - local structure judgment
  - local artifact cleanliness

Theory implication:

- explicit structure leashing is not useless
- it is doing something real
- but what it is doing is mostly:
  - preventing failure
  - cleaning geometry
  - cleaning artifacts
- it is not yet reopening enough target-specific style energy

So the current failure is not:

- `structure leash is wrong in principle`

It is:

- `structure leash alone cannot carry the missing style recovery burden`

## Combined mechanism read

Across these families, the current mechanism picture is:

1. `Hold`-type schedules can preserve or freeze geometry.
2. `EdgeGated`-type leashes can improve structure and artifacts.
3. Neither one is enough to recover the missing target-specific style on its own.

This is important because it narrows the bottleneck.

The bottleneck now looks less like:

- `need a better training schedule`

and more like:

- `need a later, more target-specific style reinjection path`

## What this says about the active qedgegated+pattn line

The active line is:

- `qedgegated + crossattn_texture proximal`

Why it is the right current bet:

- `qedgegated` keeps the stronger, more selective structure leash hypothesis alive
- `crossattn_texture` is the first explicit attempt to move the missing burden away from the carrier branch

So this line is the correct next test of the current theory:

- keep structure discipline
- but stop asking the same spatial carrier path to do all the style work

## If the active line still fails

If `qedgegated+pattn` still closes as:

- style-up only on cheap `CLIP`
- weak or flat on `IntroStyle`
- worse than `LBM-Knee` on `DINO`
- and clearly below `Seedream` on `VLM`

then the next conclusion should be stronger:

- the current style reinjection path is still too weak or too entangled with transport

At that point the next family should not be:

- more hold variants
- more release variants
- more plain spatial-carrier micro-adjustments

It should be:

- a stronger late style-recovery head
- or a more explicit target-style residual branch
- or another mechanism that increases target-specific style capacity without handing back the structure collapse

## Current operational conclusion

Use the current read as:

- `Hold4TwoStage`: closed negative
- `EdgeGated`: not promotable, but still informative
- `qedgegated+pattn`: the correct current live test of the narrowed theory
