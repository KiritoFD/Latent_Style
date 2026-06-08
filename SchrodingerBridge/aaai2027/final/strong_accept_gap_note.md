# Strong-Accept Gap Note

Updated: 2026-06-08

## What is now genuinely hard evidence

- `IDT` no-op failure is closed on Distinct5-WikiArt.
- `LBM-Knee` is no longer only a CLIP-S/LPIPS point:
  - target-pooled ArtFID
  - artifact-sensitive diagnostics
  - non-CLIP ConvNeXt style probe
  - row-resampled stability over IDT
  - main-paper qualitative strip
- `LBM-PS-v2` is clearly framed as a style ceiling rather than a universal winner.
- `Seedream-4.5` protocol and repaired-750 assembly are documented and tied to the same target-pooled ArtFID path.
- two fixed-rule follow-up splits (`split1`, `split2`) now support the claim that the unchanged-image pathology is not Distinct5-only.
- a reproducibility supplement, artifact ledger, parameter-count script, and local analysis requirements snapshot now exist.

## What is still missing if the goal is truly strong accept

- higher-grade blind preference evidence
  - human pairwise
  - or an external VLM blind pairwise judge
- if possible, multi-seed support for `LBM-Knee`

## Current best paper-facing story

The strongest defensible story is now:

> IDT calibration exposes a real no-op failure in art-to-art style transfer. LBM is a compact style-ID latent transport family with selectable operating points. `LBM-Knee` is the main closed Pareto point under artifact-sensitive, non-CLIP, and qualitative checks; `LBM-PS-v2` is the explicit style ceiling; `Seedream-4.5` remains the stronger external large-prior reference on target-style recognition and identity preservation.

## Recommendation on next work

If more time exists, the only next experiment with clearly higher value than more writing is:

1. score the prepared blind pairwise packet
2. report only the smallest clear result table
3. do not expand method lines further

The blind packet is no longer empty:

- a blinded protocol bundle exists
- an exploratory internal blind rubric audit exists

What is still missing is an evaluator that is stronger than the paper authoring model itself.
