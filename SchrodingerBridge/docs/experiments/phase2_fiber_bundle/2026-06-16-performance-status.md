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
| orthogonal low/high I2SB | `0.705847` | `0.451386` | closed partial positive; e4 improves structure to `0.698245 / 0.390826` but style retreats |
| I2SB fiber-directed noise | `0.706816` | `0.489969` | closed negative; active gate but no matched Pareto gain |
| I2SB latent slerp path | `0.712038` | `0.476511` | running positive; e2 beats clean I2SB e2 by `+0.002944` style and `-0.013722` LPIPS; e14 gives structure-side Pareto `0.686199 / 0.357695` |
| latent affine s0.75 | `0.685444` | `0.344580` | in-band diagnostic, not enough style |
| SMoE tokenizer | `0.672774` | `0.327155` | stable structure, style bottleneck unchanged |

## What Worked

- SDE/I2SB consistently breaks the ODE style ceiling: style enters the
  `0.70+` band, unlike the ODE/tokenizer-only lines around `0.67-0.68`.
- Training-time eval is now fast enough for every retained checkpoint on
  remote WSL: recent fast10 runs complete in roughly `26-29s` wall with cached
  CLIP/source/reference features.
- `orthogonal_lowhigh` is a cleaner intervention than content-anchor:
  e1 gives better style and better LPIPS than content-anchor e1, and e4 moves
  LPIPS near `0.39`. Runtime observability proves the endpoint projection
  switch is active.
- `latent_slerp` is the first path-geometry intervention with a clean matched
  I2SB Pareto gain: e2 reaches `0.712038 / 0.476511`, improving both style and
  LPIPS versus clean I2SB e2. It is not closed or promoted yet because LPIPS is
  still high. The later curve is not flat negative: e10 recovers to
  `0.701837 / 0.385366`, dominating the earlier e7 structure-side point, and
  e14 reaches `0.686199 / 0.357695`. Current read: slerp separates the style
  peak and structure peak instead of solving both at once.

## What Failed Or Is Not Promoted

- `blend0p25` failed because scalar interpolation suppresses style and
  structure together.
- `content_anchor` failed because lowpass/edge anchor losses did not create a
  true orthogonal split; the best style points still stayed at LPIPS
  `0.445-0.459`, and e7 collapsed to `0.683597`.
- SMoE and topology release are not enough by themselves: they preserve
  structure but do not solve style actuation.
- `orthogonal_lowhigh` is not promoted because its best style point remains
  high-LPIPS and its best structure point loses too much style. It is a useful
  partial positive, not the target frontier.
- I2SB fiber-directed noise is not promoted. Runtime observability proves the
  gate was active, but the matched e2/e5 controls show no useful Pareto delta.

## Next Queue

- Do not spend a training lane on topogated Brownian noise alone.
- Let `latent_slerp` run to the formal tail rule before deciding whether to
  use its e2 checkpoint as an integration parent.
- If `latent_slerp` closes positive-but-high-LPIPS, next intervention should
  combine path geometry with an explicit structure restraint rather than add
  more actuator capacity.
- Keep all DINO/VLM-heavy work after the clean geometry/SDE probes unless a
  matched control specifically requires it.
