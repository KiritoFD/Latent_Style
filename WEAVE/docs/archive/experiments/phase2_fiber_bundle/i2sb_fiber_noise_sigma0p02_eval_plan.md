# I2SB Fiber-Directed Noise Eval: sigma0p02

## Purpose

Test the next clean geometry hypothesis after `orthogonal_lowhigh` closed as
partial positive: keep the absolute endpoint style force, but restrict the
Schrodinger bridge Brownian term to the existing tokenizer/topogate fiber mask.

## Controlled Change

- Eval-only override:
  `configs/aaai2027/phase2_i2sb_fiber_noise_sigma0p02_eval.json`.
- Parent checkpoints:
  clean absolute I2SB sigma0p02 e2 and e5 from
  `exp/aaai2027_phase2_i2sb_clean_k070_e3_sigma0p02_b8a2_vlen010/`.
- Changed only:
  `model.i2sb_fiber_aligned_noise=true`,
  `model.i2sb_fiber_noise_rms_normalize=true`.
- Unchanged:
  endpoint parameterization remains `absolute`; no `orthogonal_lowhigh`; no
  content-anchor loss; no tokenizer, backbone, appearance, PC corrector, or
  training schedule change.

## Implementation Contract

- The gate source is only cached `StyleMaps.gate_16`.
- Gate processing is `sigmoid/clamp -> bilinear resize -> channel broadcast`.
- RMS normalization keeps effective sigma comparable to isotropic I2SB so the
  conclusion is about noise direction, not reduced noise magnitude.
- Legacy behavior is unchanged unless `i2sb_fiber_aligned_noise=true`.

## Matched Controls

- Clean absolute I2SB sigma0p02 e2:
  transfer `0.709094 / 0.490233`.
- Clean absolute I2SB sigma0p02 e5:
  transfer `0.704671 / 0.408530`.
- Orthogonal low/high e4:
  transfer `0.698245 / 0.390826`.

## Decision Rule

- Positive:
  same checkpoint improves LPIPS versus clean isotropic while retaining most
  of the clean style, or improves style at comparable LPIPS.
- Strong positive:
  `CLIP-S >= 0.705` with LPIPS `<=0.40`, or `CLIP-S >=0.700` with LPIPS
  `<=0.38`.
- Negative:
  style drops into the `0.69` band without beating `orthogonal_lowhigh` on
  LPIPS, or runtime observability shows the fiber gate was not active.

## Live Log

- 2026-06-16: added default-off I2SB fiber-noise switch and eval-only override.
  Next action is schema/import smoke, remote WSL eval of clean e2/e5, pull only
  CSV/JSON outputs, append to the AAAI2027 page-1 plot, then close or promote.
- 2026-06-16: schema/compile smoke passed for `config_schema.py`, `model.py`,
  and `lancet_runtime.py`.
- Remote WSL eval-only execution:
  e2 and e5 clean absolute I2SB checkpoints were evaluated with
  `--force-regen`, seed `42`, no checkpoint pullback, and no generated PNG
  grids.
- Runtime observability:
  both summaries report `i2sb_fiber_noise_requested=1.0`,
  `i2sb_fiber_noise_active=1.0`, gate mean/rms about `0.710`, and
  `i2sb_fiber_noise_rms_normalize=1.0`.

| checkpoint | clean control | fiber-noise eval | matched read |
|---|---:|---:|---|
| e2 | `0.709094 / 0.490233` | `0.706816 / 0.489969` | `-0.002278` style, `-0.000264` LPIPS |
| e5 | `0.704671 / 0.408530` | `0.703904 / 0.409715` | `-0.000767` style, `+0.001185` LPIPS |

- Decision:
  `closed_negative_directional_noise`. The implementation is active and the
  matched control is clean, but gate-directed Brownian noise does not improve
  the Pareto direction. It slightly damps style and does not materially protect
  structure. Do not launch a training lane for this mechanism.
- Plot update:
  appended the e2/e5 transfer points to `plot_points.csv` and regenerated the
  AAAI2027 WikiArt-5 page-1 figure.
- Next implication:
  the bottleneck is less likely to be solved by noise direction alone. The next
  mechanism should alter the style actuation path itself, preferably a direct
  decoder/fiber-section injection that can raise style without relying on
  stochastic variance.
