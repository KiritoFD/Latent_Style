# Closure: I2SB Mask-Aware Fiber Projection on lowanchor0.50 e9

## Result

Closed as `partial_positive_not_promoted`.

The gate-aware variant fixes part of the raw hard-projection failure: compared with the raw global latent low/high projector, it preserves more style and reduces the sigma0.5 LPIPS explosion. It still does not create the target-facing Pareto point because style gains require LPIPS to move far out of the current structure band.

## Matched Metrics

Parent checkpoint:
`../exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`

Parent transfer metric:
`CLIP-S=0.701428824365`, `LPIPS=0.372202562000`

| sigma | transfer CLIP-S | transfer LPIPS | delta CLIP-S vs parent | delta LPIPS vs parent | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.690047 | 0.349598 | -0.011382 | -0.022604 | LPIPS-only, style suppressed |
| 0.2 | 0.702386 | 0.396835 | +0.000957 | +0.024632 | near-parent style, mild LPIPS cost |
| 0.3 | 0.712721 | 0.439208 | +0.011292 | +0.067006 | useful style lift, too much structure cost |
| 0.4 | 0.719342 | 0.482121 | +0.017914 | +0.109918 | best style in scan, not structure-safe |
| 0.5 | 0.718477 | 0.521621 | +0.017048 | +0.149418 | style saturates while LPIPS keeps worsening |

Matched raw global projection comparisons:

- `sigma=0.0`: gate-aware is `+0.002336` CLIP-S and `-0.008568` LPIPS versus raw global projection.
- `sigma=0.5`: gate-aware is `+0.014917` CLIP-S and `-0.070604` LPIPS versus raw global projection.

## Runtime Observability

All five evals have valid gate activation:

- `i2sb_fiber_project_endpoint_active=1`
- `i2sb_fiber_project_noise_active=1`
- `i2sb_fiber_project_use_gate=1`
- `i2sb_fiber_project_gate_active=1`
- mean gate is stable around `0.7086-0.7090`

## Decision

The gate is a real improvement over global latent highpass projection, but the intervention remains an eval-only diagnostic rather than a promotable model:

- low sigma is structure-friendly but loses or barely changes style;
- medium sigma gives the first clean `0.71+` style lift from this parent, but LPIPS rises to `0.439+`;
- high sigma reaches `0.719` style, still below the `0.74` target, with LPIPS `0.48-0.52`.

Conclusion: the gate is useful as a support mask, but raw latent highpass Brownian noise is still not aligned with decoded image structure. The next mechanism should keep the gate support and replace the noise/projector direction with a decoder-aware or learned local fiber basis.

## Artifacts

- Curve: `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_gate_lowanchor050e9_sigma_scan.csv`
- Eval roots: `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_gate_lowanchor050e9_sigma*/`
- Plot CSV: `docs/experiments/phase2_fiber_bundle/plot_points.csv`
