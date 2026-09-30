# I2SB Mask-Aware Fiber Projection Eval: lowanchor0.50 e9

## Purpose

The raw latent low/high hard projection scan is closed negative: global lowpass anchoring suppresses style at `sigma=0.0`, while highpass Brownian noise explodes LPIPS at larger sigma.

This follow-up keeps the same parent, solver, checkpoint, and fast10 transfer eval contract. It changes only the projection support:

- use `StyleMaps.gate_16` from the existing structured tokenizer/topogate path;
- endpoint projection anchors source lowpass mainly where `gate` is low;
- endpoint remains free where `gate` is high;
- noise is highpass-projected and then multiplied by `gate`;
- no new learnable parameters, no training, no loss change.

## Parent And Controls

- Parent checkpoint:
  `exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Matched parent metric:
  transfer `CLIP-S=0.701428824365`, `LPIPS=0.372202562000`
- Raw global hard projection control:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_lowanchor050e9_sigma_scan.csv`

## Switches

- `solver_family=solver_i2sb`
- `transport_prediction_mode=endpoint`
- `i2sb_fiber_project_endpoint=true`
- `i2sb_fiber_project_noise=true`
- `i2sb_fiber_project_kernel=5`
- `i2sb_fiber_project_use_gate=true`

## Scan

| Config | Sigma | Role |
| --- | ---: | --- |
| `phase2_eval_fiber_project_gate_sigma0p0_lowanchor050e9.json` | 0.0 | endpoint mask projection only |
| `phase2_eval_fiber_project_gate_sigma0p2_lowanchor050e9.json` | 0.2 | conservative gated stochasticity |
| `phase2_eval_fiber_project_gate_sigma0p3_lowanchor050e9.json` | 0.3 | intermediate style/structure slope check |
| `phase2_eval_fiber_project_gate_sigma0p4_lowanchor050e9.json` | 0.4 | intermediate style/structure slope check |
| `phase2_eval_fiber_project_gate_sigma0p5_lowanchor050e9.json` | 0.5 | compare against raw projection sigma0.5 |

## Decision Rule

- Positive: beats the raw global projection at matched sigma, or improves parent style without pushing LPIPS beyond the current tolerated style-first band.
- Negative: gate-aware projection still kills style at `sigma=0.0` and/or still drives LPIPS explosion at `sigma=0.2/0.5`.
- If `fiber_project_gate_active` is not `1`, discard the run as invalid implementation evidence.

## Outputs

- Eval roots:
  `docs/experiments/phase2_fiber_bundle/eval/aaai2027_eval_fiber_project_gate_lowanchor050e9_sigma*/`
- Curve:
  `docs/experiments/phase2_fiber_bundle/curves/i2sb_fiber_project_gate_lowanchor050e9_sigma_scan.csv`
- Closure:
  `docs/experiments/phase2_fiber_bundle/i2sb_fiber_project_gate_eval_lowanchor050e9_closure.md`
