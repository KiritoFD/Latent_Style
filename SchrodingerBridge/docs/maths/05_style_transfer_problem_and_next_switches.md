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
- optional generated-delta residual geometry when `--delta-probe` is enabled:
  per-target residual norm, target-pair residual L2/cosine, residual effective
  rank, and correlations against latent/tokenizer pair geometry.

Interpretation:

- If real style latents are close, the dataset split is weak and model gains will
  look like metric hacking.
- If real latents are separated but tokenizer codes are collapsed, fix the
  tokenizer.
- If tokenizer codes are separated but generated deltas are similar, fix
  injection/actuation.

The 2026-06-01 local WSL probe on
`exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt` gives the third case:

- tokenizer effective rank: `3.986`;
- tokenizer mean off-diagonal cosine: `0.015`;
- generated-delta effective rank: `3.324`;
- generated-delta mean off-diagonal cosine: `0.725`;
- correlation from real latent style geometry to generated delta geometry:
  `0.823` by L2;
- correlation from tokenizer code geometry to generated delta geometry: `0.426`
  by L2.

This points to an actuation/injection bottleneck rather than a simple tokenizer
rank collapse. The model reads the five style controls, but the executed latent
residuals remain too co-linear.

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

`--generate-only` is now available for fair generation-speed comparisons. It
runs LANCET plus VAE decode and optional image saving, but skips CLIP/LPIPS,
style prototypes, source feature caching, and latent delta metrics. Use it when
comparing against baselines whose reported time is generation-only.

Current timing evidence on the 750 all-5x5 set:

- LANCET generate-only with `target_chunk_size=5` and `vae_decode_batch_size=2`:
  `131.02s / 750`.
- LANCET full eval with the same chunking: `221.81s / 750`.
- Full eval is dominated by decoded image transfer/PIL-compatible metric
  plumbing, not the LANCET vector field.

### Compile/Cache

`torch_compile` remains optional. If used, set a persistent
`torch_compile_cache_dir` so Inductor/Triton caches survive restarts. It should
be treated as an infra experiment, not a model improvement.

Naive VAE decoder compile paths (`torch.compile`, TorchScript JIT, ONNX CUDA)
have not shown an end-to-end win in the measured 750-eval setting. They should
remain optional flags until a run proves faster without changing metrics.

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

Given the current probe, the next promoted switches should target execution:

- add a generated-delta-rank diagnostic to quick eval before full promotion;
- test bounded pair/content execution budgets first, because LPIPS is the main
  gap against SaMAM-512;
- test multi-site token injection only with an ablation that reports whether
  generated-delta off-diagonal cosine decreases;
- reject pure tokenizer-size increases unless they improve generated-delta
  geometry, not only tokenizer-code geometry.

### Execution Budget Switch

`execution_budget_mode` is now a model switch:

- `none`: default historical behavior.
- `scalar`: one bounded residual gain per sample.
- `low_high`: split the predicted latent residual into a 3x3 average-pooled
  low-frequency component and a high-frequency residual, then apply two bounded
  gains.

The budget head consumes:

```text
[style_code + time_code, content mean/std/abs/highpass-energy/RMS]
```

It is identity-initialized: the final linear layer is zero, so initial gains are
exactly `1.0`. The gains are bounded by
`exp(tanh(logit) * execution_budget_log_span)`. The default log span is
`log(1.25)`, so the first probe cannot become a hidden global style-strength
knob.

Training support:

- `freeze_mode: budget_only` freezes tokenizer and LANCET, and trains only
  `execution_budget_head.*`.
- Probe config:
  `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_execution_budget_from_hist_e1.json`.
- Smoke run from `exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt` loaded the
  old checkpoint non-strictly, selected exactly six trainable budget parameters,
  and completed one short epoch.

150-transfer smoke after an 11-batch budget-only local run was still near
identity:

| Run | clip_style | LPIPS | delta rank | delta offdiag cosine |
|---|---:|---:|---:|---:|
| base smoke | 0.802409 | 0.358092 | 3.325 | 0.703 |
| budget 11-batch e1 | 0.802382 | 0.358259 | 3.325 | 0.703 |

This verifies wiring and suggests that a narrow final-residual budget alone is
too weak in a tiny run. Do not promote it yet. The next stronger probe should
change where style controls enter the actuator, or add a residual-geometry-aware
training signal to make the budget do measurable work.

### Multi-Site Style Injection Switch

`style_injection_mode` is now a model switch:

- `none`: default historical behavior.
- `body`: inject a zero-initialized style/content feature bias after semantic
  body routing.
- `decoder`: inject before decoder-side modulation.
- `body_decoder`: enable both sites.

The injection head consumes the same cheap conditioning family as the budget
head:

```text
[style_code + time_code, content mean/std/abs/highpass-energy/RMS]
```

The final layer is zero-initialized, so old checkpoints remain exactly
identity-safe. Training support:

- `freeze_mode: injection_only` freezes tokenizer and LANCET, and trains only
  `body_style_injector.*` / `decoder_style_injector.*`.
- Probe config:
  `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_style_injection_from_hist_e3.json`.

150-transfer local probe:

| Run | clip_style | LPIPS | delta rank | delta offdiag cosine |
|---|---:|---:|---:|---:|
| base smoke | 0.802409 | 0.358092 | 3.325 | 0.703375 |
| injection e1 | 0.802445 | 0.358082 | 3.335 | 0.703844 |
| injection e3 | 0.802399 | 0.358237 | 3.344 | 0.702829 |

Reading:

- Multi-site injection moves generated-delta geometry more than the final
  residual budget.
- The change is not yet a primary-metric win. `clip_style` and LPIPS remain
  effectively neutral on the 150 smoke.
- This is a better candidate than final residual budget for a longer remote
  probe, but it should be paired with a geometry-aware gate or loss before full
  promotion.

### Generated-Delta Diversity Loss

`w_generated_delta_diversity` is now an optional loss switch. It operates on the
executed residual, not on the tokenizer code:

```text
delta_i = v_theta(z_i, t=1, target_style_i)
mean_delta_s = mean(delta_i | target_style_i = s, source_style_i != s)
loss = mean ReLU(cos(mean_delta_s, mean_delta_r) - margin)^2
```

The intended probe is narrow: if tokenizer codes are already separated but
generated residuals are co-linear, add direct gradient pressure to make different
target styles execute as different residual directions.

Local WSL 150-transfer gate from the historical e8 checkpoint:

| Run | clip_style | LPIPS | delta rank | delta offdiag cosine |
|---|---:|---:|---:|---:|
| base smoke | 0.802409 | 0.358092 | 3.325 | 0.703375 |
| injection e3 | 0.802399 | 0.358237 | 3.344 | 0.702829 |
| injection + delta diversity 0.05 | 0.802397 | 0.358240 | 3.344 | 0.702397 |
| injection + delta diversity 0.50 | 0.802388 | 0.358267 | 3.345 | 0.698567 |

Reading:

- The loss is wired correctly: increasing the weight monotonically reduces
  target-wise residual co-linearity.
- The primary metrics do not improve; `clip_style`, `clip_dir`, and LPIPS all
  move slightly in the wrong direction at `0.50`.
- This falsifies the simplest version of "make residual directions more
  orthogonal and metrics will follow" for the frozen small-injector setting.

Decision:

- Keep the switch as a diagnostic/probe, default `0.0`.
- Do not promote it to the baseline.
- The useful next step is not stronger geometry regularization on the same tiny
  injector; it is a richer executable representation or a consumer path with
  enough capacity to translate separated controls into visible style changes.

### Residual Variance Decomposition Probe

Quick eval now also reports `generated_delta_variance_decomposition`. On the
historical e8 base, 150 transfers give:

| Decomposition term | Ratio |
|---|---:|
| target style between | 0.0283 |
| source style between | 0.1437 |
| source image between | 0.9501 |
| source-target pair between | 0.1751 |
| target after source-image removal | 0.5664 |

This is the strongest current diagnosis:

- The raw vector field is content dominated.
- Style is a conditional residual factor, not the dominant generated movement.
- The model can distinguish targets after content effects are removed, but the
  visible execution is mostly governed by source-image geometry.

Next switch direction:

```text
target_style_id -> target_carrier tokens / residual basis
content latent  -> content_gate / execution budget
LANCET blocks consume both, not a single blended style_code
```

This is stricter than simply adding a larger tokenizer: the carrier must be
read as an executable style direction, while the gate should control LPIPS risk.

### Carrier/Gate Injection Probe

`style_injection_form` is now a model switch:

- `mixed`: existing injection, one MLP consumes `[style_code, content_stats]`.
- `carrier_gate`: target/time style code produces a carrier bias; content stats
  produce a bounded gate. The final injected feature is
  `tanh(carrier(style_code)) * exp(tanh(gate(content)) * gate_log_span)`.

This tests whether explicitly separating "what target style wants to execute"
from "how much this content can tolerate" helps target style survive the
content-dominated vector field.

Short local WSL probe:

```text
config: configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_carrier_gate_from_hist_e3.json
checkpoint: exp/local_wsl_wikiart512_hist_b32_e8/epoch_0008.pt
freeze_mode: injection_only
```

150-transfer gate:

| Run | clip_style | LPIPS | clip_dir | delta rank | offdiag cosine | target ratio | source-image ratio | target after source-image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.802358 | 0.358096 | 0.370340 | 3.325 | 0.703379 | 0.0283 | 0.9501 | 0.5664 |
| carrier_gate e3 | 0.802273 | 0.358190 | 0.369542 | 3.349 | 0.702816 | 0.0283 | 0.9500 | 0.5669 |

Decision:

- The factorized injection path is wired and slightly improves residual rank.
- It does not improve primary metrics and barely moves raw target variance.
- Do not promote as-is. The target carrier is still too weak relative to the
  content/source-image residual basis.

The next stronger version should not only bias features. It should create a
target residual basis or target-specific operator/token stream that contributes
directly to the decoded residual, while the content gate bounds its execution.
