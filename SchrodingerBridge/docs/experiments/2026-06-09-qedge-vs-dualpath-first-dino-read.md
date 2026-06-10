# QEdge vs DualPath First DINO Read

Date: 2026-06-09

This note records the first image-backed local `DINO` read for the current two live successor families.

Compact table:

- [first_bestfew_dino_compare_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/first_bestfew_dino_compare_20260609.csv)
- [first_bestfew_dino_compare_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/first_bestfew_dino_compare_20260609.md)

## Newly landed structure rows

`QEdgePattn e01`:

- source-aligned pairs:
  - `745`
- image-backed local `DINO structure`:
  - `0.02635`

`DualPath e01`:

- source-aligned pairs:
  - `745`
- image-backed local `DINO structure`:
  - corrected `0.02635`

Reference anchor:

- `LBM-Knee full750 DINO = 0.02171`

## Immediate read

The corrected structure-side comparison is:

- `LBM-Knee` best
- `QEdgePattn e01` and `DualPath e01` now effectively tie on the first landed bestfew `DINO`
- later `DualPath e09` currently drifts slightly worse:
  - `0.02742`

So the corrected first image-backed `DINO` read does **not** say:

- `QEdgePattn e01` has a decisive structure lead over `DualPath e01`

Instead it says:

- both families pay a real global-structure price relative to `LBM-Knee`
- and the later style-leaning `DualPath e09` point currently pays a little more

## Why this matters

This is useful because the current local `VLM` read had been suggesting:

- `DualPath e01` looks cleaner / safer / more stable

while the corrected first `DINO` read now says:

- `QEdgePattn e01` and `DualPath e01` are basically tied under the first landed global-geometry read

These are not contradictory.

They imply that the current families are trading off **different notions of structure**:

1. `VLM` is rewarding:
   - visible cleanliness
   - artifact control
   - and broad perceptual plausibility

2. `DINO` is rewarding:
   - source-side global feature geometry
   - even if the result is less visually clean or more stylized

## Current theory implication

The current bottleneck is therefore sharper than just:

- `more style without losing structure`

It is now:

- `more target-specific style`
- without losing:
  - `global source geometry` under `DINO`
  - and without losing:
    - `local cleanliness / artifact control` under `VLM`

That makes the current `dualpath_spatialtexture` follow-up more justified, not less.

Why:

- `DualPath e01` no longer looks strictly worse than `QEdgePattn e01` on the corrected first landed `DINO`
- `DualPath` still looks slightly cleaner on the current direct local `VLM` mean scores

So the next useful family should try to:

- keep the perceptual cleanliness of `DualPath`
- while recovering some of the structure-side advantage that `QEdgePattn` still keeps under `DINO`

## Current decision

Do not promote either family yet.

Current read is:

- `QEdgePattn`
  - tied first image-backed `DINO` at `e01`
  - weaker `LPIPS`
  - no direct blind overall wins so far

- `DualPath`
  - better local `VLM` quality shape
  - tied first landed `DINO` at `e01`
  - slightly worse later `DINO` at `e09`
  - still too weak on target-specific style recovery

This keeps the current remote strategy unchanged:

- let the live `dualpath_texture` line finish
- then decide on `dualpath_spatialtexture` using:
  - the current `VLM`
  - the current first `DINO`
  - and any later landed bestfew structure rows

## Direct blind VLM side-read

A separate direct local blind packet is now running:

- `QEdgePattn e01 vs DualPath e01 vs Seedream`

Current interim summary:

- [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv)

Current read on the first `91` cases:

- `Seedream = 91 / 91` overall wins
- `DualPath e01 = 0 / 91`
- `QEdgePattn e01 = 0 / 91`

But current mean local scores still lean slightly toward `DualPath e01`:

- `DualPath e01`
  - style `1.978`
  - structure `3.275`
  - artifact `2.275`
- `QEdgePattn e01`
  - style `1.901`
  - structure `3.231`
  - artifact `2.209`

So the combined read is now:

- `QEdgePattn e01`
  - tied first image-backed `DINO`
- `DualPath e01`
  - a still-small current blind `VLM` style-mean edge
- `Seedream`
  - still dominating both in the direct head-to-head blind comparison

