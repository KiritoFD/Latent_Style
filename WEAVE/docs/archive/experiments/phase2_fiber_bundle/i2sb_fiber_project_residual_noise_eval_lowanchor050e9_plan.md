# I2SB Residual-Aligned Fiber Noise Eval: lowanchor0.50 e9

## Purpose

The mask-aware Fiber-SDE scan showed that `StyleMaps.gate_16` is a useful support mask, but random latent highpass noise still raises LPIPS too quickly. This eval-only follow-up keeps the same parent and gate support, then replaces the raw highpass noise direction with endpoint-residual-aware noise.

No training, loss, tokenizer, attention, endpoint projection, eval board, or parent checkpoint changes are allowed in this scan.

## Parent And Controls

- Parent checkpoint:
  `../exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Parent metric:
  transfer `CLIP-S=0.701428824365`, `LPIPS=0.372202562000`
- Matched highpass-gated control:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_gate_lowanchor050e9_sigma_scan.csv`

## Switches

Common:

- `solver_family=solver_i2sb`
- `transport_prediction_mode=endpoint`
- `i2sb_fiber_project_endpoint=true`
- `i2sb_fiber_project_noise=true`
- `i2sb_fiber_project_use_gate=true`
- `i2sb_fiber_project_kernel=5`

Variants:

- `i2sb_fiber_project_noise_mode=residual_envelope`
- `i2sb_fiber_project_noise_mode=residual_direction`

## Scan

| Config | Sigma | Role |
| --- | ---: | --- |
| `phase2_eval_fiber_project_resenv_sigma0p3_lowanchor050e9.json` | 0.3 | envelope at first useful style point |
| `phase2_eval_fiber_project_resenv_sigma0p4_lowanchor050e9.json` | 0.4 | envelope at previous best style point |
| `phase2_eval_fiber_project_resenv_sigma0p5_lowanchor050e9.json` | 0.5 | envelope saturation check |
| `phase2_eval_fiber_project_resdir_sigma0p3_lowanchor050e9.json` | 0.3 | direction at first useful style point |
| `phase2_eval_fiber_project_resdir_sigma0p4_lowanchor050e9.json` | 0.4 | direction at previous best style point |

## Decision Rule

- Positive: improves CLIP-S/LPIPS slope versus the matched highpass-gated control at the same sigma.
- Weak positive: preserves most of the style lift while reducing LPIPS materially.
- Negative: reduces style without enough LPIPS recovery, or keeps the same LPIPS blow-up.
- Invalid: `i2sb_fiber_project_gate_active` or `i2sb_fiber_project_residual_active` is not `1`.

## Outputs

- Curve: `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_residual_noise_lowanchor050e9_scan.csv`
- Eval roots: `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_residual_noise_lowanchor050e9_*`
- Closure: `docs/experiments/phase2_fiber_bundle/i2sb_fiber_project_residual_noise_eval_lowanchor050e9_closure.md`
