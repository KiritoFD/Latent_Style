# I2SB Gate-Local Head Adapter: lowanchor0.50 e9

## Purpose

The eval-only Fiber-SDE scans show:

- raw latent highpass noise raises style but damages LPIPS too quickly;
- `StyleMaps.gate_16` is a useful support mask;
- residual-envelope noise improves the matched CLIP-S/LPIPS slope but remains eval-only and not strong enough.

This training lane keeps the parent model fixed and trains a small decoder-side local fiber head so the endpoint residual itself can become stronger and more local.

## Control

- Parent checkpoint:
  `../exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Parent transfer metric:
  `CLIP-S=0.701428824365`, `LPIPS=0.372202562000`
- Eval-only residual-envelope sigma0.4 diagnostic:
  `CLIP-S=0.716145075262`, `LPIPS=0.452353101900`

## Switches

Training change:

- `style_delta_mode=head_adapter`
- `style_head_adapter_scale=0.08`
- `style_head_adapter_force_highpass=true`
- `style_head_adapter_use_gate=true`
- `freeze_mode=injection_only`

Eval/solver setting:

- `solver_family=solver_i2sb`
- `i2sb_fiber_project_endpoint=true`
- `i2sb_fiber_project_noise=true`
- `i2sb_fiber_project_use_gate=true`
- `i2sb_fiber_project_noise_mode=residual_envelope`
- `bridge_sigma=0.4`

## Guardrails

- Only `style_head_adapter_*` should be trainable.
- `style_head_adapter_gate_active` must be `1` in runtime observability.
- Do not promote if style lift comes only with LPIPS above the residual-envelope diagnostic without a matched style gain.
- Do not continue past 10 epochs unless the newest 2 retained checkpoints are still setting transfer Pareto points.

## Outputs

- Run: `exp/aaai2027_phase2_i2sb_gate_head_adapter_lowanchor050e9_s008_b8a2_vlen010`
- Config: `configs/aaai2027/phase2_i2sb_gate_head_adapter_lowanchor050e9_s008_b8a2_vlen010.json`
- Curve target: `docs/experiments/phase2_fiber_bundle/curves/i2sb_gate_head_adapter_lowanchor050e9_s008_curve.csv`
