# Latent AdaCUT: Model and Training Design (Current)

## 1. Scope and Goal

This document explains the current model/training design in `Cycle-NCE/src`:

- how style is injected into latent space;
- how losses are scheduled over long runs (300 epochs);
- why image quality can degrade (fog/checkerboard-like artifacts);
- why current learning rate is chosen and what stability risks remain.

The target here is practical: keep style transfer effective while preventing structure collapse and visual artifacts.

## 2. Model Architecture (Core Principle)

Main file: `src/model.py`

### 2.1 Input/Output Domain

- Input and output are latent tensors (`[B, 4, 32, 32]` for SD-style 256px latent grid).
- The network predicts a residual latent delta and outputs:
  - `pred = content + delta`.

This keeps content identity easier to preserve than direct latent synthesis.

### 2.2 Style Injection Pathways

The model has both global and spatial style paths:

- Global style code:
  - `style_ref -> style_enc -> style_proj`
  - `style_id -> style_emb`
  - mixed by `style_mix_alpha`.
- Spatial style maps:
  - reference-style spatial features (`32x32`, `16x16`)
  - style-id spatial priors (`style_spatial_id_32`, `style_spatial_id_16`).

Injection points:

- pre-block and in-block injection at 32/16 scales;
- decoder spatial injection (`style_spatial_dec_gain_*`);
- optional texture head and force path.

This is why style can be injected without reference image at inference: style-id priors are learnable.

### 2.3 Frequency Bias and Output Control

Current model includes:

- optional high-frequency bias on delta (`use_delta_highpass_bias`);
- style gate floor / style force path controls;
- bilinear upsampling and optional blur in style/downsample branches.

The intent is to avoid pure low-frequency color-shift shortcuts and keep local texture changes.

## 3. Objective Design (Current)

Main file: `src/losses.py`

### 3.1 Teacher-Student Setup

Two forward passes per batch:

- Teacher: uses style reference (`style_ref=target_style`).
- Student: deployment path (`style_ref=None`, style-id only).

This trains the inference path directly instead of relying only on reference-guided outputs.

### 3.2 Distill and Style Closure

- Distill:
  - now supports low-pass-only distillation (`distill_low_only=true`),
  - and cross-domain-only aggregation (`distill_cross_domain_only=true`).
- Code loss:
  - teacher output code -> reference code,
  - student output code -> style-id prototype.

This keeps style conditioning active and reduces identity collapse.

### 3.3 Structure Constraints (Reworked)

`cycle` and `struct` now share the same configurable alignment form:

- loss type: `l1` or `mse`;
- low-pass blend strength: `[0, 1]` numeric parameter.

Config keys:

- `cycle_loss_type`, `cycle_lowpass_strength`
- `struct_loss_type`, `struct_lowpass_strength`

Extra structure terms:

- edge term (`w_edge`, Sobel magnitude),
- cycle edge blending (`cycle_edge_strength`),
- delta TV penalty (`w_delta_tv`) to suppress periodic artifacts.

### 3.4 NCE and Scheduling

- NCE is optional and ramped in with warmup+ramp.
- All major structure terms are scheduled by `_ramp_weight`.

This is critical: style must establish first, then structure regularization takes over.

## 4. Why Quality Degrades (Fog/Artifacts)

Observed failure modes were consistent with design tradeoffs:

1. Over-constrained structure stack:
   - cycle + struct + edge + NCE + TV all high/early
   - pushes model to safe, low-variance outputs (fog/softness).
2. Low-pass-dominant constraints:
   - if low-pass strength is high, high-frequency details are weakly protected.
3. Strong MSE structure terms:
   - MSE can over-penalize deviations and bias toward conservative smooth outputs.

Mitigation already applied:

- reduced structure weights and delayed warmup/ramp in `config.json`;
- switched distill to low-pass-only + cross-domain only;
- reduced TV strength and low-pass blend strength.

## 5. Learning Rate Assessment (Current 300-Epoch Config)

Config: `src/config.json`

- `learning_rate = 1.5e-4` (reduced from `2.0e-4`);
- cosine decay to `5e-6`;
- `grad_clip_norm = 1.0`;
- AdamW + bf16 + TF32.

### 5.1 Is it likely to explode?

Numerical explosion risk is low because:

- no adversarial discriminator instability here;
- gradient clipping is enabled;
- long cosine decay lowers step size over time.

### 5.2 Real risk at this LR

The practical risk is not NaN/overflow but early optimization bias:

- with large batch (`128`) and no explicit LR warmup,
- optimizer can settle into conservative local minima before style path matures.

So the LR choice is "safe but still needs schedule discipline", not "unsafe runaway".

## 6. Current 300-Epoch Strategy (Rationale)

In `src/config.json`:

- style phase first:
  - style terms active from start (`gram/code/push/distill`);
- structure phase later:
  - `struct/edge` warmup+ramp delayed;
  - `cycle/nce` warmup+ramp delayed further.

This prioritizes learning "how to change style" before enforcing strict structure consistency.

## 7. Practical Monitoring Checklist

Per epoch, watch these together (not single scalar only):

- `distill`, `code`, `gram`, `push`
- `cycle`, `struct`, `edge`, `delta_tv`, `nce`
- effective weights: `w_cycle_eff`, `w_struct_eff`, `w_edge_eff`, `w_nce_eff`

And in full eval:

- `photo->Hayao classifier_acc` (style transfer direction)
- `clip_style` and `clip_content`
- visual collage for fog/checkerboard artifacts.

If transfer collapses:

- reduce structure stack first (weights or warmup timing),
- do not immediately increase all style terms simultaneously.

## 8. Recommended Next Iteration

For the next main run:

- keep current `1.5e-4` LR and 300-epoch schedule;
- run until first 2 full-eval checkpoints (epoch 50, 100) before judging;
- if fog persists:
  - lower `cycle_lowpass_strength`,
  - reduce `w_delta_tv`,
  - reduce early `w_struct/w_edge` further before touching style terms.

