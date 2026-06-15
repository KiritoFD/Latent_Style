# Actuation Head Adapter Probe

Date: 2026-06-16

## Goal

Test the `fiber.md` diagnosis that style freedom is being swallowed by the
shared final `dec_out` convolution. Prior output-basis, pre-decoder section,
proximal texture, mixed body+decoder, and generated-delta diversity lanes were
safe but style-flat. This lane adds a small style-conditioned residual head in
parallel with `dec_out`, so style-specific output corrections do not need to
pass through the shared final convolution.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Primary config:
  `configs/aaai2027/phase2_actuation_head_adapter_k070_e3_b16a2bf16_vlen010.json`.
- Held fixed:
  tokenizer, solver, TopoGate, losses, data, parent checkpoint,
  `freeze_mode=injection_only`, b16 accumulation-2 training lane, and fast10
  transfer-only eval contract.
- Only model delta:
  `model.style_delta_mode=head_adapter`.
- Head adapter:
  `h -> 3x3 hidden -> style FiLM -> SiLU -> zero-init 3x3 latent residual`,
  added after the shared transport head.
- Initial settings:
  `style_head_adapter_hidden_dim=48`,
  `style_head_adapter_scale=0.20`,
  `style_head_adapter_force_highpass=false`.

## Decision Rule

- Primary metric: transfer CLIP-S.
- LPIPS budget: up to about `0.35` is acceptable only if style rises.
- Positive mechanism evidence:
  - e1/e2 should show non-zero `style_head_adapter_abs/rms`;
  - transfer CLIP-S should exceed the predec-family plateau near `0.68013`;
  - style gain must be larger than the current fast10 noise band, not just
    fourth-decimal drift.
- Negative evidence:
  - adapter RMS grows but style stays flat, meaning final-head capacity is not
    the active bottleneck;
  - LPIPS rises materially without style gain, meaning the branch is injecting
    off-target residual energy.

## Eval Infra Contract

- `full_eval_fast10`, transfer-only, `10` source samples per style.
- In-process eval, runtime model cache, source latent cache.
- ONNX VAE decode batch `16`.
- Metric scheduling: `metric_batch_size=50`, `lpips_chunk_size=16`.
- No generated PNG/grid during training-time eval.

## Launch Log

- 2026-06-16 06:15 remote WSL s0.20 run started.
- PID: `17667`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_head_adapter_k070_e3_b16a2bf16_vlen010`.
- Trainable scope confirmed:
  `trainable_count=8`, only `style_head_adapter_*`.
- Resume check:
  `missing=8` new head-adapter params; parent proximal keys pruned by trainer;
  final load `unexpected=0`.
- Stopped after e5 because style regressed while adapter RMS grew.
- 2026-06-16 06:33 remote WSL s0.05 scale-control run started.
- PID: `17948`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_head_adapter_s005_k070_e3_b16a2bf16_vlen010`.
- Trainable scope confirmed:
  `trainable_count=8`, only `style_head_adapter_*`.
- Stopped after e3 because style remained below the plateau while adapter RMS
  grew to `0.101`.

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_head_adapter_k070_e3_b16a2bf16_vlen010/full_eval_fast10/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | adapter abs | adapter rel RMS | note |
|---|---:|---:|---:|---:|---:|---|
| e1 | 0.679295 | 0.316571 | 64.59s | 0.010285 | 0.058994 | cold eval |
| e2 | 0.679957 | 0.318438 | 22.15s | 0.028615 | 0.161904 | best style, still below plateau |
| e3 | 0.679187 | 0.317511 | 22.34s | 0.026786 | 0.151246 | style regresses |
| e4 | 0.678691 | 0.316283 | 22.93s | 0.031265 | 0.182433 | off-target residual |
| e5 | 0.678111 | 0.316513 | 19.21s | 0.034916 | 0.194066 | stopped |

Eval infra read: the new in-process cache is active on this restarted lane.
After e1, reference/source cache status becomes `memory_loaded`, and eval total
falls from `35.10s` cold to about `8.5-9.3s` hot; e5 wall is `19.21s`.

Scale-control s0.05 local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_head_adapter_s005_k070_e3_b16a2bf16_vlen010/full_eval_fast10/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | adapter abs | adapter rel RMS | note |
|---|---:|---:|---:|---:|---:|---|
| e1 | 0.679605 | 0.315740 | 63.69s | 0.005318 | 0.029981 | cold eval |
| e2 | 0.679741 | 0.316290 | 22.01s | 0.016540 | 0.087514 | best style, below plateau |
| e3 | 0.679675 | 0.316862 | 21.87s | 0.019373 | 0.101071 | stopped |

## Closure Decision

Closed as `negative_not_promoted` for both scale settings.

- Best s0.20 point: e2, transfer `0.679957 / 0.318438`.
- Best s0.05 point: e2, transfer `0.679741 / 0.316290`.
- Both are below the predec-family plateau (`~0.68013`), while adapter RMS is
  non-zero and growing.
- Mechanism is active, so this is not a dead-branch implementation failure.
- Scale does affect damage: s0.05 has lower LPIPS cost than s0.20, but it still
  does not produce style lift.
- Interpretation: simply adding a parallel style-conditioned output residual
  gives the model actuator capacity, but the supervised signal routes it toward
  off-target residual energy. The next mechanism should change the training
  target/path geometry rather than just increasing output-head freedom.
- Next candidate: a conservative I2SB/endpoint path test with current TopoGate
  and no output residual adapter, because `fiber.md` identifies straight-line
  ODE/path averaging as the remaining theoretical failure mode after actuator
  probes.
