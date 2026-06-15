# Actuation Delta Basis Probe

Date: 2026-06-15

## Goal

Test the `fiber.md` actuation-bottleneck diagnosis with the smallest mechanism that directly touches the final delivered residual.

The previous `spatial_carrier_gate` probe was safe but flat, which suggests that feature-level carrier modulation is swallowed before `dec_out`. This probe keeps tokenizer, solver, losses, TopoGate, appearance alignment, overdrive, and metric postprocess fixed, then adds only a zero-init style-conditioned low-rank delta basis after `dec_out`.

## Controlled Delta

- Parent: `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config: `configs/aaai2027/phase2_actuation_delta_basis_k070_e3_vlen010.json`
- Only mechanism delta: `model.style_delta_mode="basis"`.
- Delta rank: `4`.
- Delta scale: `0.15`.
- Initialization: style weight head final layer is zero-initialized, so the side delta is initially zero and parent behavior is preserved.
- Freeze: `training.freeze_mode="injection_only"` trains only the new delta-basis branch.
- Tokenizer: unchanged.
- Solver: unchanged `euler_legacy`.
- Corrector: `solver_corrector_mode="none"`.
- Overdrive: disabled.
- Metric postprocess: disabled.

## Observability

Training logs and eval summaries must include:

- `style_delta_basis_active`
- `style_delta_basis_rank`
- `style_delta_basis_abs`
- `style_delta_weight_abs`
- `style_delta_side_abs`
- `style_delta_side_rms`
- `style_delta_scale`

If `style_delta_weight_abs` remains near zero after e1/e2, the branch is not learning. If `style_delta_side_abs` grows while CLIP-S is flat, the delivered residual is moving in the wrong direction.

## Schedule

- Short controlled probe: `virtual_length_multiplier=0.10`, `num_epochs=10`.
- Save/eval every epoch.
- Remote 3060 VRAM target: under `11.0 GiB`; reduce batch/accum only if needed, not mechanism parameters.

## Decision Rule

- Promote to a full-length convergence run only if transfer/all-pairs CLIP-S improves over the matched parent/carrier control while LPIPS remains in band.
- If e1/e2 LPIPS jumps materially, inspect delta scale and highpass behavior before continuing.
- If the branch moves (`style_delta_side_abs > 0`) but metrics stay flat, archive as evidence that output-side low-rank bases are still too weak or underconstrained.
- If the branch does not move, fix trainable scope/initialization rather than treating the result as theory evidence.

## Launch Log

Pending.

## Running Eval Curve

Pending.

## Closure Decision

Pending.
