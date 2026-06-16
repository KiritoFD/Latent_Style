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
| I2SB latent slerp path | `0.712038` | `0.476511` | closed partial positive; e2 beats clean I2SB e2 by `+0.002944` style and `-0.013722` LPIPS; e28 gives LPIPS floor `0.682638 / 0.352726`, but style decays |
| I2SB slerp + orthogonal low/high | `0.704828` | `0.446676` | closed negative; e15 reaches `0.678109 / 0.350421` but only as LPIPS-only style collapse |
| I2SB low-anchor0.50 | `0.711470` | `0.472991` | running partial positive; e5 reaches `0.702532 / 0.393892`, close to short gate but not in-band |
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
  LPIPS versus clean I2SB e2. The full e1-e28 curve confirms the mechanism is
  not promotable alone: e28 reaches the best LPIPS `0.352726`, but only at
  `0.682638` style. Current read: slerp separates the style peak and structure
  cooling tail instead of solving both at once.

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
- I2SB latent-slerp is not promoted as a standalone model. It has a real e2
  matched gain, but after that 26 later checkpoints fail to recover or exceed
  the style frontier. The automatic joint Pareto tracker is misleading here
  because late checkpoints keep making low-style LPIPS-only improvements.
- I2SB slerp+orthogonal-lowhigh is not promoted. It confirms hard lowpass
  endpoint anchoring suppresses the low-frequency/color component of style:
  e1 loses `0.007210` CLIP-S versus latent-slerp e2 while improving LPIPS, and
  e15 improves LPIPS only by falling to `0.678109` style.

## Completed / Effective / Pending

| lane | done | effect | next read |
|---|---|---|---|
| Infra cleanup | yes | eval steady-state is about `25s` per fast10 checkpoint; cold restart eval is about `40s` | keep cached in-process eval, avoid unnecessary restarts |
| Fiber / SDE noise scan | yes | isotropic/fiber noise did not create target-facing Pareto gain | do not spend more lanes on noise-only scans |
| SMoE tokenizer | yes | preserves structure but stays near ODE style ceiling | not the current bottleneck |
| I2SB clean absolute | yes | proves SDE style shock, but LPIPS too high | use as style-force reference, not as promoted model |
| I2SB blend/content-anchor | yes | scalar/lowpass anchors suppress or couple style | closed negative |
| I2SB orthogonal low/high | yes | partial structure restraint, style still retreats | useful ingredient for combo |
| I2SB latent-slerp | yes, e1-e28 | small matched style+LPIPS gain at e2; later structure cooling only | combine with explicit structure projection |
| I2SB slerp+orthogonal | yes, e1-e16 | closes the structure gap to `0.350421` LPIPS but suppresses style to low `0.68` | closed negative; use as evidence against hard lowpass endpoint replacement |
| I2SB low-anchor0.50 | running, e1-e7 | restores style relative to hard lowpass anchor; e5 is `0.702532 / 0.393892` | continue; if tail falls below 0.700 style, close partial-positive and scan anchor strength |

## Next Queue

- Do not spend a training lane on topogated Brownian noise alone.
- Do not continue latent-slerp alone. It is closed as
  `partial_positive_not_promoted`.
- Next intervention should keep absolute endpoint style force and make the
  structure correction local or semantic rather than replacing the full
  endpoint lowpass. Candidate controls: partial lowpass anchor, chroma/style
  low-frequency preservation, or an absolute-endpoint structure loss.
- Keep all DINO/VLM-heavy work after the clean geometry/SDE probes unless a
  matched control specifically requires it.
