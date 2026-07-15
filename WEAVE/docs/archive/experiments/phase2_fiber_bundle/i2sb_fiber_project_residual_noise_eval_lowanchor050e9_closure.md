# Closure: I2SB Residual-Aligned Fiber Noise on lowanchor0.50 e9

## Result

Closed as `weak_positive_not_promoted`.

Residual-envelope noise improves the style/structure slope versus gated highpass noise at matched sigma. Residual-direction noise is too conservative and gives back too much style. Neither variant reaches the target-facing `0.74 / 0.30` direction.

## Matched Metrics

Parent checkpoint:
`../exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`

Parent transfer metric:
`CLIP-S=0.701428824365`, `LPIPS=0.372202562000`

| variant | sigma | CLIP-S | LPIPS | delta CLIP-S vs gated control | delta LPIPS vs gated control | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| residual envelope | 0.3 | 0.711790 | 0.415683 | -0.000931 | -0.023525 | weak positive |
| residual envelope | 0.4 | 0.716145 | 0.452353 | -0.003197 | -0.029768 | weak positive, best residual point |
| residual envelope | 0.5 | 0.714700 | 0.487241 | -0.003777 | -0.034379 | saturation, not better than sigma0.4 |
| residual direction | 0.3 | 0.701911 | 0.396173 | -0.010810 | -0.043036 | too conservative |
| residual direction | 0.4 | 0.705169 | 0.418271 | -0.014173 | -0.063850 | too conservative |

## Runtime Observability

All runs are valid implementation evidence:

- `i2sb_fiber_project_gate_active=1`
- `i2sb_fiber_project_residual_active=1`
- envelope rows have `i2sb_fiber_project_residual_mode_envelope=1`
- direction rows have `i2sb_fiber_project_residual_mode_direction=1`

## Decision

The residual-envelope result is useful but not promotable. It supports the theory that the noise basis should follow the model's learned endpoint residual envelope instead of naked latent highpass noise. It does not solve the target because the absolute best point remains only `0.716145 / 0.452353`, still far from the desired `0.74 / 0.30`.

The residual-direction result is negative: forcing all stochasticity along the residual direction protects LPIPS better, but it collapses the style lift almost back to the parent.

## Next

Keep the residual-envelope idea as a component, but do not spend more eval-only scans on scalar sigma. The next mechanism should make the endpoint residual itself stronger and more local, likely by adding a small trained local fiber basis or decoder-aware residual head while the main backbone is frozen.

## Artifacts

- Curve: `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_residual_noise_lowanchor050e9_scan.csv`
- Eval roots: `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_residual_noise_lowanchor050e9_*`
- Plot CSV: `docs/experiments/phase2_fiber_bundle/plot_points.csv`
