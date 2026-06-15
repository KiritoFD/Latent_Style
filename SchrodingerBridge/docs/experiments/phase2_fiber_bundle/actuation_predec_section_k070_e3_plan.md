# Actuation Pre-Decoder Style Section: k070 e3

## Purpose

Test the Fiber-Bundle diagnosis that the current decoder/output path collapses
different style sections into nearly collinear generated latent deltas. This
probe attacks the generated-delta / `dec_out` bottleneck directly while keeping
the parent, tokenizer, solver, losses, TopoGate, and eval contract fixed.

## Controlled Change

- Parent checkpoint:
  `../exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Mechanism:
  `model.style_delta_mode=predec_section`.
- The new module applies a style-conditioned low-rank feature section before
  `dec_out`, not as a post-`dec_out` latent side residual.
- Initialization is no-op:
  `style_section_out` is zero initialized, so the parent behavior is preserved
  before training.
- Trainable scope:
  `freeze_mode=injection_only`, restricted to `style_section_*` modules.
- Unchanged:
  tokenizer family, structured tokenizer, solver, bridge objective/losses,
  TopoGate, output appearance alignment, proximal residuals, dataset, and
  pairing cache.

## Configs

- Main throughput config:
  `configs/aaai2027/phase2_actuation_predec_section_k070_e3_b32bf16_vlen010.json`.
- Active remote safety config:
  `configs/aaai2027/phase2_actuation_predec_section_k070_e3_b16a2bf16_vlen010.json`.

## Eval Contract

- Training-time eval subdir: `full_eval_fast10`.
- Transfer-only, `10` source samples per style.
- ONNX VAE decoder: `../eval_cache/vae_onnx/ema_b16_32/decoder.onnx`.
- Source latent cache enabled.
- `CLIP-S + LPIPS` every retained checkpoint.
- Generated-delta observability enabled for every checkpoint:
  effective rank, off-diagonal cosine, delta RMS/abs.

## Decision Rule

- Primary target: increase transfer `CLIP-S`, style-first toward Seedream and
  the `0.74 / 0.30` objective.
- LPIPS budget: up to the Seedream-like `~0.35` band is acceptable if style
  clearly moves; large LPIPS cost without style gain is negative evidence.
- Mechanism evidence requires one of:
  - higher transfer style than proximal/R16 family with comparable or
    explainable LPIPS cost;
  - increased generated-delta effective rank or lower off-diagonal cosine that
    correlates with style lift.
- Stop only by the established fast-curve convergence rule, not by a single
  early point.

## Remote Run

- Run id:
  `aaai2027_phase2_actuation_predec_section_k070_e3_b16a2bf16_vlen010`.
- Launch time: `2026-06-16 01:29` remote WSL.
- Remote launcher log:
  `logs/predec_section_20260616_012934.log`.
- 2-step smoke before formal launch:
  - strict-compatible parent resume worked with `missing=10`, matching the new
    zero-init section parameters.
  - `freeze_mode=injection_only` selected only the ten `style_section_*`
    parameter tensors.
  - peak smoke memory was about `2.05GB`.

## Initial Health

- First formal health check:
  GPU `3041 / 12288 MiB`, util `89%`, power `137.73W`.
- Low VRAM is expected because this is an injection-only/pre-dec section probe;
  it is not a reason to stop.

## Live Fast10 Curve

`full_eval_fast10` is transfer-only, so all-pairs equals transfer in the live
curve.

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | note |
| --- | ---: | ---: | ---: | --- |
| e1 | 0.680106 | 0.315591 | 31.83s | source latent cache rebuilt |
| e2 | 0.680117 | 0.315599 | 27.32s | cache loaded |
| e3 | 0.680121 | 0.315600 | 27.17s | current best style |
| e4 | 0.680102 | 0.315604 | 27.27s | flat |
| e5 | 0.680106 | 0.315573 | 27.13s | flat |
| e6 | 0.680070 | 0.315573 | 26.85s | ONNX decode-only VAE skip active |
| e7 | 0.680062 | 0.315583 | 26.68s | ONNX decode-only VAE skip active |
| e8 | 0.680074 | 0.315569 | 30.12s | temporary 10/32 eval batch, reverted |

Read so far: the pre-decoder section has not produced style lift by e8. The
curve is effectively flat around `0.6801 / 0.3156`; continue only under the
formal patience rule while watching whether the section magnitude and generated
delta rank start moving.

## Eval Infra Notes

- Added default-on `skip_diffusers_vae_when_onnx` for training-time eval:
  when the source latent cache is complete and ONNX decode is enabled,
  `run_evaluation.py` skips loading the diffusers VAE. The summary records
  `skip_diffusers_vae_when_onnx` and `diffusers_vae_loaded`.
- e6 confirmed the fast path:
  `diffusers_vae_loaded=false`, `source_latent_cache_status=loaded`.
- Timing read on e6:
  `wall_total=26.85s`, `lancet_generation=5.93s`,
  `vae_decode=8.41s`, `eval_metrics_loop=3.62s`.
- A batch-size probe changed live eval to generation `10` and metric `32` for
  e8. It reduced generation chunks from `7` to `5`, but wall time worsened to
  `30.12s` internal / `44.5s` trainer wall. Reverted to generation `8` and
  metric `16`.
- Decision: keep the VAE-skip path because it is safe and slightly positive;
  do not increase live eval batch on this 3060 lane. The next meaningful eval
  speed step is a separate fixed `fast5` convergence contract or a persistent
  evaluator, not larger batches.
- Incident: e6 initially failed after generation because summary observability
  referenced `vae` after it had been deleted. Fixed by caching
  `diffusers_vae_loaded_for_generation` before releasing generation models, then
  manually re-ran e6 eval and refreshed the curve.
- Incident: e9 eval was interrupted while reverting the negative batch probe.
  It must be manually re-run before closure so every retained checkpoint has
  `CLIP-S + LPIPS`.
