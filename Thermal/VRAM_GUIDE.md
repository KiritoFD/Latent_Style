# VRAM Guide (Latent_Style / Thermal)

This document summarizes where GPU memory is used and how key config knobs affect VRAM usage. It is based on the current code in `Thermal/src`.

---

## Quick Summary (Biggest VRAM Drivers)

1. Batch size (`training.batch_size`)
   Linear increase in activations and loss buffers.

2. SWD loss buffers
   `loss.max_samples`, `loss.num_projections`, and number of `loss.swd_scales` dominate temporary tensors and LUT size.

3. Model size and activations
   `model.base_channels`, `num_encoder_blocks`, `num_decoder_blocks` increase UNet parameters and intermediate activations.

4. Gradient checkpointing and AMP
   `training.use_gradient_checkpointing` reduces activation memory.
   `training.use_amp` (bfloat16) roughly halves activation memory.

5. Style classifier extras
   Spectrum/Gram features add extra tensors; `style_classifier_input_size` controls their size.

6. Preloading latents to GPU
   `training.preload_data_to_gpu` (in `utils/dataset.py`) can reserve large VRAM.

---

## VRAM Components by Pipeline Stage

### 1) Model + Optimizer

- Model parameters: UNet + optional style classifier.
- Gradients + AdamW states: roughly 3x-4x parameter size.

Key knobs:
- `model.base_channels`
- `model.num_encoder_blocks`
- `model.num_decoder_blocks`
- `loss.use_style_classifier` (adds classifier params)

---

### 2) Activations (Forward/Backward)

Most memory during training is activation storage.

Key knobs:
- `training.batch_size`
- `training.use_amp` (reduces memory)
- `training.use_gradient_checkpointing` (reduces activations, increases compute)
- `training.ode_integration_steps` (more steps -> more forwards)
- `training.accumulation_steps` (does not increase per-step activation memory)

---

### 3) SWD / LUT Memory (GeometricFreeEnergyLoss)

The multi-scale SWD loss uses LUT caches and per-batch projection buffers.

LUT memory (persistent)

Approximate GPU LUT size:
```
bytes ~= num_styles * max_samples * num_projections * 4 * num_scales
```

Key knobs:
- `model.num_styles`
- `loss.max_samples`
- `loss.num_projections`
- `loss.swd_scales` (length = num_scales)

Per-batch SWD buffers (temporary)

Per-scale buffers scale with:
```
B * max_samples * num_projections
```

Key knobs:
- `training.batch_size`
- `loss.max_samples`
- `loss.num_projections`
- `loss.swd_scales` (each scale adds a pass)

---

### 4) Style Classifier Memory

The classifier can be content-agnostic by using:
- stats (mean/std)
- Gram matrix
- spectrum bins

Memory grows with input feature size:
- `loss.style_classifier_input_size` controls spatial size of classifier input.
  - Default 8 keeps Gram + FFT small.
- `loss.style_classifier_use_spectrum` uses FFT -> extra buffers.
- `loss.style_classifier_use_gram` adds Gram matrix feature tensor.

---

### 5) Dataset / Latents on GPU

`utils/dataset.py` supports preloading all latents to GPU:
- `training.preload_data_to_gpu: true`
  -> Large persistent VRAM use (entire dataset).

If you see a sudden big VRAM jump, this is often the cause.

---

## Config Knobs and VRAM Impact

High Impact
- `training.batch_size` (linear)
- `loss.max_samples` (linear, affects SWD + LUT size)
- `loss.num_projections` (linear, affects SWD + LUT size)
- `loss.swd_scales` (more scales = more buffers and LUTs)
- `model.base_channels` (larger UNet)
- `training.preload_data_to_gpu` (loads entire dataset on GPU)

Medium Impact
- `training.use_amp` (reduces memory ~2x)
- `training.use_gradient_checkpointing` (reduces activations)
- `training.ode_integration_steps` (more steps = more forwards)
- `loss.style_classifier_use_spectrum`
- `loss.style_classifier_input_size`

Low Impact
- `training.accumulation_steps` (no per-step activation increase)
- `loss.style_classifier_use_stats`
- `loss.style_classifier_use_gram`
- logging / metrics

---

## Common VRAM Spikes and Fixes

Spike after enabling classifier
- Cause: FFT/Gram on large input size
- Fix:
  - Lower `loss.style_classifier_input_size`
  - Disable `loss.style_classifier_use_spectrum` first

Spike after changing SWD config
- Cause: `max_samples` / `num_projections` too large
- Fix:
  - Reduce `loss.max_samples` to 4096 or 2048
  - Reduce `loss.num_projections` to 32
  - Use fewer `loss.swd_scales`

Spike on data load
- Cause: `training.preload_data_to_gpu = true`
- Fix:
  - Set to `false` (default)

---

## Practical VRAM Tuning Sequence

1. Lower `loss.max_samples` (fastest relief)
2. Lower `training.batch_size`
3. Reduce `loss.num_projections`
4. Reduce number of `loss.swd_scales`
5. Enable `training.use_gradient_checkpointing`
6. Lower `loss.style_classifier_input_size`
7. Disable `loss.style_classifier_use_spectrum`

---

## Notes

- AMP reduces VRAM but not LUT cache sizes.
- Gradient checkpointing reduces activations, not parameter or LUT memory.
- LUT size grows with `num_styles`, even if batch size is small.
