# Closure: I2SB Gate-Local Head Adapter on lowanchor0.50 e9

## Result

Closed as `negative_diagnostic_not_promoted`.

The gate-local head adapter confirms that a tiny frozen-backbone local head can add style force, but it does so by moving the model into the same high-LPIPS region as the eval-only Fiber-SDE scans. It does not solve the style/structure split.

## Matched Control

Parent checkpoint:
`../exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`

Parent transfer metric:
`CLIP-S=0.701428824365`, `LPIPS=0.372202562000`

Eval-only residual-envelope sigma0.4 control:
`CLIP-S=0.716145075262`, `LPIPS=0.452353101900`

## Online Eval Curve

| epoch | transfer CLIP-S | transfer LPIPS | delta CLIP-S vs parent | delta LPIPS vs parent | note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.714002 | 0.470880 | +0.012573 | +0.098678 | immediate style lift, too much LPIPS |
| 2 | 0.713545 | 0.473220 | +0.012116 | +0.101018 | no style gain |
| 3 | 0.716935 | 0.473270 | +0.015506 | +0.101068 | best style |
| 4 | 0.716681 | 0.472890 | +0.015252 | +0.100687 | style plateau |
| 5 | 0.715643 | 0.472854 | +0.014214 | +0.100651 | style regression |
| 6 | 0.716439 | 0.472667 | +0.015010 | +0.100465 | no style breakthrough |

Best style checkpoint:
`epoch_0003`, `CLIP-S=0.716934573650`, `LPIPS=0.473270221350`.

## Runtime Observability

Implementation checks are valid:

- `freeze_mode=injection_only` trained only the 8 `style_head_adapter_*` tensors.
- `style_head_adapter_gate_active=1` for every retained eval checkpoint.
- `style_head_adapter_rel_rms` grew from `0.0297` to `0.0534`, while style stopped improving after e3.
- Full eval ran in-process after every retained checkpoint; no post-hoc-only evaluation was used.

## Decision

Do not promote this branch.

Compared with the residual-envelope eval-only control, the trained adapter gains only `+0.000790` CLIP-S at the best style point but costs another `+0.020917` LPIPS. Compared with the parent, it gains `+0.015506` CLIP-S but costs `+0.101068` LPIPS. This is not a useful target-facing move toward `0.74 / 0.30`.

The result also falsifies the idea that a small local decoder-side head is sufficient to turn hard Fiber-SDE noise into safe style actuation. The head learns to amplify the same unsafe fiber directions rather than creating an orthogonal structure-preserving fiber basis.

## Theory Note

The pure hard low/high Orthogonal Fiber-SDE described in the strong-projection proposal was already tested as an eval-only scan. It lifted style only by pushing LPIPS far out of band. The gate-aware and residual-envelope follow-ups improved the slope but did not reach a promotable point. This closure keeps the conclusion narrow: low/high latent projection is a useful diagnostic, not yet a valid geometric projection of decoded image structure.

The next step should not be another scalar sigma or another small output head. The next clean mechanism should move the hard projection closer to the decoded/RGB metric or learn a fiber basis with an explicit structure-null constraint.

## Artifacts

- Config: `configs/aaai2027/phase2_i2sb_gate_head_adapter_lowanchor050e9_s008_b8a2_vlen010.json`
- Eval curve: `docs/experiments/phase2_fiber_bundle/i2sb_gate_head_adapter_lowanchor050e9_eval/clip_lpips_curve.csv`
- Eval summaries: `docs/experiments/phase2_fiber_bundle/i2sb_gate_head_adapter_lowanchor050e9_eval/epoch_*/summary.json`
- Plot CSV: `docs/experiments/phase2_fiber_bundle/plot_points.csv`
