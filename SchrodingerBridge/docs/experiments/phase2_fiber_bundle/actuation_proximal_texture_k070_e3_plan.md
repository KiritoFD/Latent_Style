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

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_proximal_texture_k070_e3_b16a2bf16_vlen010/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | transfer-only |
|---|---:|---:|---:|---:|
| 1 | 0.672447 | 0.312461 | 70.07s | yes |
| 2 | 0.671846 | 0.313756 | 71.50s | yes |
| 3 | 0.672420 | 0.314827 | 67.21s | yes |

Timing breakdown for e1:

- `wall_total=70.07s`
- `vae_decode=24.62s`
- `lancet_generation=12.06s`
- `eval_metrics_loop=11.20s`
- ONNX decoder:
  `/mnt/i/Github/Latent_Style/eval_cache/vae_onnx/ema_b2_32/decoder.onnx`

## Closure Decision

Pending. e1-e3 are structure-safe but not style-positive versus the
parent/frontier; continue to convergence and judge by the all-ckpt transfer
curve. If the next retained points remain below the matched parent/style
frontier, close as negative and return to a cheaper style-actuation screen.
