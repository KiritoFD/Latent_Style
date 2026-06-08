# IntroStyle Mainline Switch

Date: 2026-06-08

Decision:

- `raw CLIP-S` is no longer the paper-facing main style metric
- `IntroStyle` becomes the preferred paper-facing style axis for the next round
- the current Distinct5 non-CLIP image classifier remains useful, but only as a fallback / supporting style signal
- `DINO` remains a structure axis, not a main style axis

## Why switch

Current project risks:

- `CLIP-friendly stylization`
- `no-op / existing-art bias`
- high score from palette / semantics rather than real style-ID transfer

Research read:

- `raw DINO` is not a good replacement main style metric
- the strongest direction is:
  - `IntroStyle`
  - then `style classifier / deception-style probes`
  - then `CSD`
  - then `raw CLIP-S`
  - then `raw DINO`

Operational interpretation:

- `CLIP-S` stays in the inner loop because it is cheap
- `IntroStyle` should be used for:
  - shortlisted points
  - abnormal points
  - paper-facing operating points
  - theory-sensitive geometry anchors

## New style-evaluation stack

### Fast screening

- `CLIP-S + LPIPS`

### Paper-facing style evidence

Primary:

- `IntroStyle`

Fallback / supporting:

- Distinct5 non-CLIP image classifier
  - target-style accuracy
  - target probability
  - target-source margin

### Structure axis

- `LPIPS`
- `DINO structure`

### Artifact / visual axis

- visual comparison against `Seedream`
- explicit failure tags:
  - semantic drift
  - layout drift
  - texture hierarchy missing
  - palette mismatch
  - artifact / repetition

## Immediate implications for current points

`Hold4Mid e8`:

- still treat as `geometry anchor`
- it should not be called a style winner unless it also clears:
  - `IntroStyle delta-IDT`
  - `IntroStyle style margin`

`AnisoStokesQueue e13` and `Pattn+Stokes002 e13`:

- remain the current headline style-side candidates
- should be first in line for IntroStyle audit once the evaluator is available

## Immediate implementation plan

1. find or stage an `IntroStyle` inference path that can run offline on Distinct5-512
2. define a held-out style bank per target style
3. for each selected point, compute:
   - `target_style_score`
   - `style_margin`
   - `delta_idt_style`
4. combine with:
   - `LPIPS`
   - `DINO structure`
   - visual comparison to `Seedream`

## Bottom line

From this point forward:

- `CLIP-S` is a fast-screen heuristic
- `IntroStyle` is the intended main style judge
- `DINO` is the structure judge
- geometry-anchor points must be audited with stronger style evidence before they influence paper claims
