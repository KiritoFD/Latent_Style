# Latent Style Ablation Plan

## Unified Setup
- Dataset/split: fixed
- Seed: fixed to `42` for all ablation runs
- Eval: keep training's built-in full-eval behavior from config
- Strength sweep: run separately after selecting top variants (`0.2/0.4/0.6/0.8/1.0`)

## Three-Stage Schedule
- Stage S (screening): `50` epochs
- Stage V (validation): `150` epochs
- Stage F (final): `300` epochs

---

## A. Semigroup x Struct (Core Question)
Goal: whether semigroup can replace part of structure regularization.

- A1: `w_semigroup=0.0`, `w_struct=1.0` (BASE)
- A2: `w_semigroup=0.15`, `w_struct=1.0`
- A3: `w_semigroup=0.15`, `w_struct=0.5`
- A4: `w_semigroup=0.15`, `w_struct=0.0`
- A5: `w_semigroup=0.0`, `w_struct=0.0` (extreme control)

---

## B. Style Loss Composition
Goal: test necessity of Gram / Moment.

- B1: Gram + Moment (BASE)
- B2: Gram only (`w_color_moment=0`)
- B3: Moment only (`w_stroke_gram=0`)
- B4: No style loss (`w_stroke_gram=0`, `w_color_moment=0`)

---

## C. Injection Location
Goal: validate late-injection behavior with current gate-based implementation.

Mapped to real keys:
- `model.inject_gate_hires`
- `model.inject_gate_body`
- `model.inject_gate_decoder`

Groups:
- C1: ALL = `(1.0, 1.0, 1.0)`
- C2: BODY_ONLY = `(0.0, 1.0, 0.0)`
- C3: DEC_ONLY = `(0.0, 0.0, 1.0)`
- C4: LAST_2_STAGE (approx) = `(0.0, 1.0, 1.0)`

---

## D. Train/Inference Steps
Goal: verify 1-step trainability.

Your code uses `loss.train_num_steps_min/max`, so we set both:
- D1: `min=max=1`
- D2: `min=max=2` (current)

---

## E. Delta Regularization
Goal: impact of cheap stabilizers.

- E1: `w_delta_tv + w_delta_l2` (BASE)
- E2: no `w_delta_tv`
- E3: no `w_delta_l2`
- E4: no `w_delta_tv` and no `w_delta_l2`

---

## Key Outputs
- Table 1: A (semigroup x struct)
- Table 2: B (style loss composition)
- Table 3: C (injection location)
- Fig 1: strength sweep curves (selected best 2-3 runs)
- Fig 2: speed-quality Pareto
