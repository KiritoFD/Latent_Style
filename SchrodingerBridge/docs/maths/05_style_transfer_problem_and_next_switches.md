# Style Transfer Problem, LANCET Design, and Next Switches

Date: 2026-06-01

This note resets the research question before the next model change. SaMAM is a
background baseline now: it should keep saving and evaluating every 250 steps
until convergence, then its time and metrics are recorded. The active design work
is on LANCET.

## 1. Problem Definition

Style transfer is not only "increase CLIP style while keeping LPIPS low". That
metric pair is useful, but incomplete. A model can improve both by mostly
preserving the input image and shifting global color statistics. This is a valid
metric point, but it is not necessarily a strong style transfer point.

For LANCET, the intended problem is:

```text
given content latent z0 and target style id s,
learn an executable style control c_s and a latent vector field v_theta
so that z1 = z0 + integral v_theta(z_t, t, c_s) dt
matches the target style distribution without destroying content topology.
```

The key objects are therefore:

- the style representation `c_s`;
- the renderer/vector field `v_theta`;
- the metric that says whether generated samples match the target style
  distribution;
- the content-preserving mechanism that prevents trivial style over-painting.

The current 512 evidence says that SaMAM-like baselines may reach very low LPIPS
with weak visible geometry change. That makes it important to separate three
questions:

1. Is the style distribution separable in the dataset?
2. Does the tokenizer expose that separability in code space?
3. Does LANCET execute the code as a visible latent edit?

## 2. Current LANCET Design

The current model is a style-id conditioned latent editor:

```text
style_id -> tokenizer -> style_code
z0, t, style_code -> LANCET -> velocity / endpoint residual
```

The tokenizer is the representation module. LANCET is the actuator. The two
must be evaluated separately:

- frozen LANCET + train tokenizer tests whether better controls exist inside the
  current actuator landscape;
- frozen tokenizer + train LANCET tests whether the actuator can execute a fixed
  representation;
- joint training tests the final system, but cannot by itself explain failure.

The strongest current 512 line is the spectral-stat/tokenizer adapted model
around `clip_style ~= 0.79` and `LPIPS ~= 0.30`. This is the base to improve, not
the older 0.71/0.46 line.

## 3. Why It Works

LANCET works when three conditions hold:

1. The target style set has measurable distributional gaps in latent space,
   color space, or frequency space.
2. The tokenizer maps style ids to controls whose pairwise geometry is not
   collapsed.
3. The renderer has an injection path that can turn code differences into
   latent residual differences.

Terminal SWD gives distribution pressure; kinetic/content mechanisms limit
excessive movement; semantic routing and skip routing decide where the style
residual can enter.

This explains why simply increasing tokenizer size is not enough. A larger code
can still collapse if LANCET only reads a small low-rank direction, or if the loss
can be minimized through color shifts that never require geometry.

## 4. Known Weak Points

The current weak points are testable:

- **Representation collapse**: tokenizer code cosine is high, effective rank is
  low, or atom usage is flat/unused.
- **Metric under-determination**: global CLIP/SWD improves while frequency or
  class-conditional probes show no separation.
- **Injection bottleneck**: tokenizer codes are separated, but generated latent
  deltas remain similar across target styles.
- **VAE decode bottleneck**: full eval under-uses GPU because generated targets
  are decoded one style at a time.

## 5. Non-Training Probes

Use `SchrodingerBridge/tools/probe_style_representation.py` before launching a
new expensive run. It reports:

- per-style latent mean/std/covariance trace;
- low/mid/high FFT amplitude and high/low ratio;
- pairwise latent mean L2/cosine;
- optional tokenizer style-code norm, nearest-code cosine, and effective rank
  from a checkpoint.

Interpretation:

- If real style latents are close, the dataset split is weak and model gains will
  look like metric hacking.
- If real latents are separated but tokenizer codes are collapsed, fix the
  tokenizer.
- If tokenizer codes are separated but generated deltas are similar, fix
  injection/actuation.

## 6. Switches Added For Next Runs

These switches are intended for controlled ablation, not all-at-once tuning.

### SWD Metric

`swd_distance_mode` now supports:

- `sort`: the existing sliced Wasserstein path;
- `soft_cdf` / `cdf`: a soft-CDF distance that gives smoother gradients and can
  be more stable than sorting near repeated values.

`swd_use_dilated_projections` and `swd_projection_dilation` are now active in the
projection convolution path.

### Full Eval Inference

Full eval can now batch target styles:

- `target_chunk_size=1`: legacy behavior;
- `target_chunk_size>1`: run several target styles through LANCET and VAE in one
  chunk;
- `vae_decode_batch_size`: controls VAE decoder chunk size independently.

This is for inference throughput only. It should not change metrics except for
normal floating-point nondeterminism.

### Compile/Cache

`torch_compile` remains optional. If used, set a persistent
`torch_compile_cache_dir` so Inductor/Triton caches survive restarts. It should
be treated as an infra experiment, not a model improvement.

### Tokenizer Geometry

Tokenizer switches are now explicit and default to the historical behavior:

- `tokenizer_field_dropout_p`: diagnostic dropout over identity/texture/geometry
  fields. If it improves robustness, the tokenizer was over-relying on one
  field; if it hurts immediately, the field split is already capacity-limited.
- `tokenizer_code_l2_norm` and `tokenizer_code_scale`: separate code direction
  from code amplitude. This tests whether LANCET reads style mostly as direction
  or as norm.
- `tokenizer_atom_topk`: sparse concept-atom usage.
- `tokenizer_atom_hard_eval`: deterministic hard atom selection at inference.

These are representation probes, not final claims. Keep the OR gate: a switch is
worth retaining if it substantially improves either style strength or LPIPS
without catastrophic regression in the other.

## 7. Next Experimental Order

1. Let SaMAM segmented b8 finish enough 250-step checkpoints to see convergence
   direction; record wall time, inference time, and metrics.
2. Run non-training probes on the distinct-5 latent set and the current LANCET
   checkpoint.
3. If tokenizer code rank/cosine is bad, test representation switches first.
4. If tokenizer geometry is good but deltas are similar, test injection switches.
5. Only after the probe identifies the bottleneck, run the next LANCET training
   experiment from the `0.79 / 0.30` base.
