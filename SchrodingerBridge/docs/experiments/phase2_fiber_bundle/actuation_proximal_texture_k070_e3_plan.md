# Actuation Proximal Texture Probe

Date: 2026-06-15

## Goal

Test the `fiber.md` diagnosis that style must enter a spatially resolved fiber
section rather than only scaling a late low-rank residual or a global
body/decoder bias. The proximal texture branch uses the existing structured
tokenizer spatial map as a cross-attention texture source and writes a residual
directly at the endpoint.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`
- Config:
  `configs/aaai2027/phase2_actuation_proximal_texture_k070_e3_b16a2bf16_vlen010.json`
- Mechanism delta:
  `model.proximal_mode=crossattn_texture`.
- Explicitly disabled:
  `model.style_injection_mode=none`, `model.style_delta_mode=none`,
  `model.style_delta_scale=0.0`.
- Held fixed: tokenizer, solver, losses, TopoGate, k070 e3 parent,
  `freeze_mode=injection_only`, effective b32 via b16 accumulation-2,
  `virtual_length_multiplier=0.10`, and training-time transfer-only
  CLIP-S/LPIPS eval.

## Implementation Guard

- `proximal_attn_out` is zero-initialized so the new path starts as exact
  identity / zero residual.
- `freeze_mode=injection_only` now includes `proximal_attn_q/k/v/out` and
  `proximal_style_tokens`; without this fix the probe would silently not train
  the intended mechanism.
- `bridge.proximal_target_weight=0.35` binds the trainable final endpoint to
  the style target. The base flow loss remains fixed to the frozen parent, so
  this loss is scoped to the new proximal residual path.
- Initial clamp: `proximal_clamp_ratio=0.85`, held for 2 epochs and released to
  `1.20` across 8 epochs. This prevents random or early texture bursts from
  dominating LPIPS while still allowing style-first movement.

## Decision Rule

- Primary: transfer CLIP-S, style-first toward Seedream.
- LPIPS budget: up to about `0.35` is acceptable if style rises clearly.
- Positive evidence requires beating the closed R16 full-board point
  (`0.674395 / 0.352223`) and the running mixed-bodydecoder frontier with a
  nontrivial style margin.
- If the first two epochs are flat, inspect `proximal_residual_abs`,
  `proximal_clamp_scale`, and `proximal_to_transport_ratio` before closing.
- Do not combine latent-affine amplification in training; it remains an
  eval-time screen only.

## Launch Log

- 2026-06-15 23:03 remote first b16a2 launch passed backward but crashed at
  epoch-end logging because `TRAIN_LOG_COLUMNS` included `proximal_target`
  while `append_training_log()` did not populate that key. Fixed by adding the
  row-map entry and a default-safe row writer.
- The initial b32 lane had already been rejected for eval/training memory
  pressure (`11838/12288 MiB` health read). The active lane is therefore
  `b16a2`, preserving effective batch size without changing the mechanism.
- Eval infra optimization:
  - Added default-off `full_eval_vae_onnx_decoder`,
    `full_eval_vae_onnx_tensorrt`, and
    `full_eval_vae_onnx_trt_cache_dir` switches.
  - Exposed the ONNX VAE decoder path through `run.py` and
    `run_evaluation.py`.
  - Added fixed-batch padding to `ORTVAEDecoder` so batch-2 ONNX can handle
    the final batch-1 tail.
  - Added ONNX-decode failure fallback to diffusers decode so a bad accelerator
    path cannot crash the training lane.
- Remote WSL eval accelerator setup:
  - Installed user-site `onnxruntime-gpu`, `onnx`, and `onnxscript` on the
    existing `/usr/bin/python` environment.
  - Exported matched decoder
    `../eval_cache/vae_onnx/ema_b2_32/decoder.onnx`.
  - Probe: `CUDAExecutionProvider,CPUExecutionProvider`, fixed batch `2`;
    decode `32x32 -> 256` averaged about `80 ms / batch2`.
- 2026-06-15 23:37 manual e1 eval replay succeeded with the 32x32 ORT decoder.
- 2026-06-15 23:38 training resumed from local `epoch_0001.pt` and entered
  epoch 2; health check showed about `7.5 GiB` VRAM and live GPU load.
- 2026-06-16 00:10 eval-speed fix:
  - Paused after `epoch_0006` finished full transfer eval.
  - Added default-compatible `training.full_eval_output_subdir` so training
    can keep a fast convergence curve separate from the earlier full curve.
  - Exported `../eval_cache/vae_onnx/ema_b16_32/decoder.onnx` and switched the
    active training-time eval contract to `full_eval_fast10`.
  - `full_eval_fast10` uses transfer-only, deterministic `10` source samples
    per style (`200` transfer pairs total), `target_chunk_size=5`,
    `vae_decode_batch_size=16`, no generated PNG/grid, and GPU-kept decoded
    tensors.
  - Backfilled `epoch_0001` through `epoch_0006` under the same fast10 contract
    before resuming training from local `epoch_0006.pt` to epoch 7.

## Running Full Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/clip_lpips_curve.csv`

Contract: transfer-only, default `30` source samples per style (`600` transfer
pairs), ONNX VAE `batch=2`.

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | transfer-only |
|---|---:|---:|---:|---:|
| 1 | 0.672447 | 0.312461 | 70.07s | yes |
| 2 | 0.671846 | 0.313756 | 71.50s | yes |
| 3 | 0.672420 | 0.314827 | 67.21s | yes |
| 4 | 0.672949 | 0.322180 | 66.58s | yes |
| 5 | 0.671740 | 0.328618 | 66.38s | yes |
| 6 | 0.673384 | 0.327238 | 69.65s | yes |

Timing breakdown for e1:

- `wall_total=70.07s`
- `vae_decode=24.62s`
- `lancet_generation=12.06s`
- `eval_metrics_loop=11.20s`
- ONNX decoder:
  `/mnt/i/Github/Latent_Style/eval_cache/vae_onnx/ema_b2_32/decoder.onnx`

## Running Fast10 Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/clip_lpips_curve_fast10.csv`

Contract: transfer-only, deterministic `10` source samples per style (`200`
transfer pairs), ONNX VAE `batch=16`. This is the active training-time
convergence curve from epoch 7 onward. Do not compare absolute values directly
against the earlier full curve; compare trends within each contract.

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | decode | transfer-only |
|---|---:|---:|---:|---:|---:|
| 1 | 0.679886 | 0.315508 | 28.88s | 8.41s | yes |
| 2 | 0.679237 | 0.317352 | 28.91s | 8.41s | yes |
| 3 | 0.679731 | 0.318296 | 29.10s | 8.42s | yes |
| 4 | 0.680348 | 0.326056 | 28.94s | 8.42s | yes |
| 5 | 0.678679 | 0.332862 | 28.97s | 8.42s | yes |
| 6 | 0.680404 | 0.331394 | 28.76s | 8.45s | yes |
| 7 | 0.680526 | 0.332695 | 29.29s | 8.49s | yes |
| 8 | 0.680202 | 0.334218 | 29.12s | 8.47s | yes |
| 9 | 0.680954 | 0.334124 | 29.01s | 8.48s | yes |
| 10 | 0.680468 | 0.334741 | 28.97s | 8.46s | yes |
| 11 | 0.680411 | 0.334782 | 33.81s | 8.87s | yes |
| 12 | 0.680278 | 0.335167 | 29.66s | 8.51s | yes |
| 13 | 0.680467 | 0.335323 | 29.01s | 8.47s | yes |
| 14 | 0.680484 | 0.335496 | 28.70s | 8.49s | yes |

Speed decision:

- Full e1-e6 wall: `66.38-71.50s`.
- Fast10 e1-e14 wall: usually `28.70-29.66s`, with e11 at `33.81s`.
- Practical speedup: about `2.3x` per retained checkpoint while keeping a
  fixed transfer-only convergence surface.

## Closure Decision

Closed as `converged_not_promoted`.

- Fast10 convergence at `epoch_0014`: best and last Pareto remained
  `epoch_0009`; `since_best=5`, `tail_flat=true`, `converged=true`.
- Full transfer confirmation:
  - best fast checkpoint `epoch_0009`: `0.674190 / 0.329931`
  - final checkpoint `epoch_0014`: `0.673760 / 0.331171`
- Matched read: the confirmed best gains about `+0.002370` CLIP-S against the
  parent but costs `+0.015313` LPIPS, and it does not beat the R16 style
  frontier.

Decision note:
`docs/experiments/phase2_fiber_bundle/actuation_proximal_texture_k070_e3_closure.md`
