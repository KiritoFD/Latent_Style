# 2026-06-16 Performance Status

## Canonical CSV And Figure

- Plot data:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
- Page-1 figure data:
  `aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
- Current rendered figure:
  `aaai2027/figures/fig_wikiart5_page1_summary.pdf`.
- Per-run curves:
  `docs/experiments/phase2_fiber_bundle/curves/`.
- Per-run eval records:
  `docs/experiments/phase2_fiber_bundle/eval/`.

## Current Transfer Frontier Read

| line | best transfer CLIP-S | LPIPS | status |
|---|---:|---:|---|
| clean absolute I2SB sigma0p02 | `0.709094` | `0.490233` | style-positive, structure too damaged |
| clean absolute I2SB sigma0p01 | `0.713162` | `0.590598` | strongest style, unacceptable LPIPS |
| I2SB blend0p25 | `0.694567` | `0.415258` | closed negative; scalar shrink suppresses style |
| content-anchor I2SB | `0.703953` | `0.458607` | closed negative; anchor remains coupled |
| orthogonal low/high I2SB | `0.705847` | `0.451386` | live; e1 improves matched anchors, e2 retreats to `0.699997 / 0.420951` |
| latent affine s0.75 | `0.685444` | `0.344580` | in-band diagnostic, not enough style |
| SMoE tokenizer | `0.672774` | `0.327155` | stable structure, style bottleneck unchanged |

## What Worked

- SDE/I2SB consistently breaks the ODE style ceiling: style enters the
  `0.70+` band, unlike the ODE/tokenizer-only lines around `0.67-0.68`.
- Training-time eval is now fast enough for every retained checkpoint on
  remote WSL: recent fast10 runs complete in roughly `26-29s` wall with cached
  CLIP/source/reference features.
- `orthogonal_lowhigh` is a cleaner intervention than content-anchor:
  e1 gives better style and better LPIPS than content-anchor e1, with the
  runtime observability proving the endpoint projection switch is active.

## What Failed Or Is Not Promoted

- `blend0p25` failed because scalar interpolation suppresses style and
  structure together.
- `content_anchor` failed because lowpass/edge anchor losses did not create a
  true orthogonal split; the best style points still stayed at LPIPS
  `0.445-0.459`, and e7 collapsed to `0.683597`.
- SMoE and topology release are not enough by themselves: they preserve
  structure but do not solve style actuation.

## Next Queue

- Let `orthogonal_lowhigh` reach e3, then stop if style keeps retreating and
  LPIPS is still above `0.38`.
- If orthogonal remains only partial positive, test fiber-directed/topogated
  SDE noise as a solver-only matched eval: same absolute I2SB parent, same
  sigma, noise masked by existing gate only.
- Keep all DINO/VLM-heavy work after the clean geometry/SDE probes unless a
  matched control specifically requires it.
