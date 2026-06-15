# Actuation Mixed Body+Decoder Probe

Date: 2026-06-15

## Goal

Test the fiber-bundle diagnosis that the style residual bottleneck is not only
late output rank, but where style enters the transport computation. R16 showed
that a larger output-side basis is active but does not create a meaningful
style breakthrough. This lane moves actuation into the body+decoder feature
path while keeping the rest of the experiment fixed.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config:
  `configs/aaai2027/phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010.json`
- Mechanism delta:
  `model.style_injection_mode=body_decoder`,
  `model.style_injection_form=mixed`.
- Explicitly disabled:
  `model.style_delta_mode=none`, `model.style_delta_scale=0.0`.
- Held fixed: tokenizer, solver, losses, TopoGate, output appearance head,
  k070 e3 parent, `freeze_mode=injection_only`, b32/bf16/channels-last
  throughput lane, `virtual_length_multiplier=0.10`, and training-time
  transfer-only CLIP-S/LPIPS eval.

## Decision Rule

- Primary: transfer CLIP-S, style-first toward Seedream.
- LPIPS budget: values near `0.35` are acceptable if style rises clearly.
- Matched control: compare against k070 e3 parent and the closed R16/S030
  actuation-only lanes, not against unrelated datasets or external boards.
- Positive evidence requires transfer CLIP-S to exceed the R16 e2 full-board
  closure (`0.674395 / 0.352223`) by a meaningful margin and avoid a pure
  LPIPS-cost trade.
- If e1/e2 immediately flatten around k070/R16 levels, inspect runtime
  observability for mixed injection magnitude before interpreting as a theory
  failure.

## Launch Log

- 2026-06-15 21:41 remote WSL formal run started.
- PID: `4786`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010`.
- Remote log:
  `logs/phase2_actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010.launch.log`.
- Remote sync note: remote git worktree is dirty, so launch used targeted file
  sync for `src/config_schema.py`, `src/run.py`, `src/model.py`,
  `src/trainer.py`, `src/utils/run_evaluation.py`,
  `src/utils/inference.py`, and the new config.
- Pre-launch remote smoke: `status=ok`; tokenizer `pure_latent_spatial`,
  solver `euler_legacy`, transport `velocity`, no DINO runtime required.
- 35s health check:
  - active `python src/run.py --config ...mixed_bodydecoder...`
  - GPU sample `10891 / 12288 MiB`, util `1%`, power `116 W`
  - dataset `wikiarts_5_full_notest_latents_ema/train`
  - formal freeze log:
    `Freeze mode=injection_only | trainable_count=12`
  - trainable tensors are only `body_style_injector.*` and
    `decoder_style_injector.*`
  - parent load: `loaded=282`, `missing=12`; missing keys are the new
    zero-init mixed injector parameters and are expected.

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_mixed_bodydecoder_k070_e3_b32bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | transfer-only |
|---|---:|---:|---:|---:|
| e1 | 0.672607 | 0.334239 | 90.19s | yes |
| e2 | 0.673771 | 0.350758 | 87.27s | yes |
| e3 | 0.673342 | 0.352926 | 87.43s | yes |
| e4 | 0.673095 | 0.352603 | 91.26s | yes |
| e5 | 0.673355 | 0.352523 | 86.89s | yes |
| e6 | 0.672937 | 0.350028 | 85.20s | yes |
| e7 | 0.672841 | 0.350166 | 85.18s | yes |
| e8 | 0.672608 | 0.350489 | 89.68s | yes |
| e9 | 0.672805 | 0.351310 | 85.69s | yes |
| e10 | 0.673065 | 0.352618 | 85.15s | yes |
| e11 | 0.672882 | 0.350660 | 82.15s | yes |
| e12 | 0.672977 | 0.352374 | 82.63s | yes |
| e13 | 0.672790 | 0.350746 | 83.34s | yes |
| e14 | 0.672662 | 0.351001 | 83.56s | yes |
| e15 | 0.672634 | 0.350382 | 83.55s | yes |

## Closure Decision

Closed as `converged_not_promoted`.

- Best style point: `epoch_0002`, transfer `0.673771 / 0.350758`.
- Formal convergence: `true`, `since_best=13`, `since_last_pareto=9`,
  `tail_flat=true`.
- Matched read: the best point does not beat the R16 full-board frontier
  (`0.674395 / 0.352223`) and is far below the style-priority Seedream target.
- Interpretation: moving style actuation into body+decoder feature injection
  was safe but did not break generated-delta collinearity. Archive as negative
  evidence and do not continue this lane.

Closure note:
`docs/experiments/phase2_fiber_bundle/actuation_mixed_bodydecoder_k070_e3_closure.md`
