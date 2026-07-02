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
- Style-priority tolerance: because the target is to close the style gap against
  Seedream-like baselines, transfer LPIPS up to about `0.35` is acceptable if
  transfer CLIP-S moves meaningfully upward.

## Decision Rule

- Primary objective: maximize transfer CLIP-S. LPIPS is a budget constraint, not
  the ranking target for this lane.
- Promote to a full-length convergence run if transfer CLIP-S improves over the
  matched parent/carrier control while LPIPS stays below the current style-first
  budget, approximately `0.35`.
- If e1/e2 LPIPS jumps above the style-first budget, inspect delta scale and
  highpass behavior before continuing.
- If the branch moves (`style_delta_side_abs > 0`) but metrics stay flat, archive as evidence that output-side low-rank bases are still too weak or underconstrained.
- If the branch does not move, fix trainable scope/initialization rather than treating the result as theory evidence.

## Launch Log

- 2026-06-15 15:55 remote WSL formal run started with
  `configs/aaai2027/phase2_actuation_delta_basis_k070_e3_b32bf16_vlen010.json`.
- Launcher: detached `setsid nohup`, exact PID tracking, no broad `pkill` after launch.
- Training command root: `/mnt/i/Github/Latent_Style/SchrodingerBridge`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_delta_basis_k070_e3_b32bf16_vlen010`.
- Log:
  `logs/phase2_actuation_delta_basis_k070_e3_b32bf16_vlen010.launch.log`.
- PID at launch: `560`.
- Batch/perf lane: `batch_size=32`, `accumulation_steps=1`, bf16 AMP,
  channels-last, compile off.
- Rationale: the b32 bf16 benchmark was the fastest stable tested lane after
  rejecting b64 as too close to VRAM pressure.

### Throughput / VRAM

- Short benchmark, b12 no-AMP channels-last:
  `36` steps, `432` samples, `56.1s` compute, about `7.7 samples/s`,
  peak about `2.40/2.64 GiB`.
- Short benchmark, b32 bf16 channels-last:
  `36` steps, `1152` samples, `86.7s` compute, about `13.3 samples/s`,
  peak about `6.29/8.62 GiB`.
- Formal epoch 1:
  `1888` samples, `149.9s` epoch wall, `12.60 samples/s`,
  peak `6.29/8.90 GiB`.
- Local GPU dashboard `http://127.0.0.1:8085/stream` was checked against
  remote `nvidia-smi`; samples agree. GPU utilization drops during eval are
  expected because full eval has generation, decode, CPU copy, and metric
  phases rather than one continuous training kernel stream.

## Running Eval Curve

Curve CSV:
`docs/experiments/phase2_fiber_bundle/eval/actuation_delta_basis_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall |
|---|---:|---:|---:|
| 1 | 0.672703 | 0.331056 | 241.56s |
| 2 | 0.673375 | 0.337056 | 204.90s |
| 3 | 0.673705 | 0.341082 | 204.67s |
| 4 | 0.673966 | 0.344454 | 207.46s |
| 5 | 0.673851 | 0.344927 | 203.85s |

Target-wise transfer at epoch 1:

| target style | transfer CLIP-S | transfer LPIPS |
|---|---:|---:|
| Early_Renaissance | 0.663090 | 0.296517 |
| Impressionism | 0.676368 | 0.305315 |
| Minimalism | 0.716824 | 0.440360 |
| Rococo | 0.660588 | 0.295209 |
| Ukiyo_e | 0.646647 | 0.317880 |

Matched comparison to previous carrier-gate e1 transfer point:

- Carrier-gate e1: `0.671841 / 0.314633`.
- Delta-basis b32 e1: `0.672703 / 0.331056`.
- Delta: style `+0.000862`, LPIPS `+0.016423`.
- Delta-basis b32 e2: `0.673375 / 0.337056`.
- Delta-basis b32 e3: `0.673705 / 0.341082`.
- Delta-basis b32 e4: `0.673966 / 0.344454`.
- Delta-basis b32 e5: `0.673851 / 0.344927`.
- Interim read under style-priority tolerance: the output-side branch is not
  dead and remains within the acceptable LPIPS band, but the style lift is only
  a slow thousandths-level climb while LPIPS steadily consumes structure budget.
  Best point in this short lane is e4. e5 slightly regresses in transfer CLIP-S
  while LPIPS rises. Under the Seedream-oriented style-first target this is not
  a structural failure, but it is too weak to close the style gap.

### Infra Note

2026-06-15: training accumulation collected `last_style_delta_debug`, but
`src/utils/training.py` had a fixed `TRAIN_LOG_COLUMNS` schema that omitted the
`style_delta_*` fields. The schema has been patched and covered by
`tests/test_training_log_schema.py`; current already-running epochs cannot
retroactively recover those per-epoch training means. Eval summaries remain
valid for CLIP-S/LPIPS convergence decisions.

## Closure Decision

Open. Continue at least through e3/e4 before deciding. With Seedream-style
structure tolerance, `LPIPS <= 0.35` is in band; closure is driven primarily by
transfer CLIP-S slope. If transfer CLIP-S stays nearly flat while LPIPS consumes
the whole 0.35 budget, archive as weak actuation rather than stopping merely
because LPIPS is above the previous 0.31 structure-preserving target.

2026-06-15 update: original run reached e5 and showed e4 as the best transfer
style point (`0.673966 / 0.344454`). e5 regressed to `0.673851 / 0.344927`.
A brief resume attempt from e5 was stopped and excluded: after the logging-schema
hot sync, optimizer state resume reported a parameter-group mismatch, so it is
not a clean continuation. The conclusion for the original e1-e5 segment is
`weak positive actuation, insufficient for Seedream target`; next action should
increase style actuation strength or change mechanism, while still evaluating to
true convergence for any promising Seedream-oriented lane.
