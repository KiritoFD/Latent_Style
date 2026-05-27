# Style Tokenizer Theory Agenda

## Purpose

This document summarizes the current tokenizer state and defines the theory
problems that must be solved before more backbone sweeps are justified.

The target remains:

```text
clip_style > 0.72
content_lpips preferably < 0.50
Hayao must become a clean, strong visual style rather than a weak average slice.
```

The current working assumption is that the backbone is good enough to pause
large backbone changes. The immediate research object is the style tokenizer:
what style representation it should learn, what supervision identifies that
representation, and how to verify it as a component.

## Current Training Signal

The tokenizer refit path does not use paired image targets, CLIP loss,
Seedream outputs, or any external teacher. The active supervision is internal
and latent-space based.

For each target style `s`, the optimizer samples:

```text
content latent x0: random latent from the union of all style pools
target latent y_s: random latent from the target-style pool
style_id: s
```

The current remote data root is:

```text
I:\Github\Latent_Style\latent-256-sd15-ema
```

with the observed latent counts:

| split | count |
|---|---:|
| photo | 6187 |
| Hayao | 1752 |
| monet | 972 |
| vangogh | 600 |
| cezanne | 850 |

The loss used by `run_style_embedding_mainline_calibration.py` is:

```text
L =
  w_swd * SWD(highpass(pred), highpass(target_style_latent))
+ w_anchor * ||pred - teacher_pred||_2^2
+ w_grad * gradient_cosine_loss(pred, content)
+ w_tv * TV(pred - content)
+ w_token_l2 * ||token_vocab - token_vocab_0||_2^2
+ w_projector_l2 * ||projector - projector_0||_2^2
```

where:

- `SWD` is the only positive style-distribution signal;
- `teacher_pred` is the frozen checkpoint endpoint for the same content and
  style id, used as a content-preserving anchor;
- gradient and TV terms are geometry/smoothness guards;
- token/projector L2 terms are trust-region regularizers;
- full evaluation metrics are not used for training.

The current `m12/m13` projector route freezes:

```text
backbone weights
style_emb
style_spatial_id_16
```

and trains only:

```text
style_tokenizer.grammar_vocab.weight
style_tokenizer.band_vocab.weight
style_tokenizer.code_projector
```

The number `120` in logs such as `iter=120/120` is not a data count. It is
the number of optimization iterations for one target style. For `m12`, each
target style receives `120` iterations with batch size `14`, so one style sees
about `1680` sampled content latents and `1680` sampled target-style latents,
with replacement.

## Tested So Far

### 1. Clean No-Prior Tokenizer Backbone

Runs:

```text
ema_style_vocab_neutral_w34
ema_style_vocab_neutral_w36_stylepush
```

These removed manual per-style grammar/band priors and kept style exposure
balanced. Result:

| run | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|
| `neutral_w34` epoch 8 | 0.707817 | 0.514850 | 0.643154 | 0.566445 |
| `neutral_w36_stylepush` epoch 8 | 0.708146 | 0.519977 | 0.645181 | 0.570138 |

Finding: the tokenizer can create nonzero Hayao and Cezanne fields without
manual priors, but it does not become a full style vocabulary.

### 2. Component Scorecard

The tokenizer scorecard measures:

- effective rank;
- active non-photo style rows;
- field-to-actuator sensitivity;
- downstream style/LPIPS gates.

Current clean tokenizer score:

| run | grammar active | band active | coverage |
|---|---:|---:|---:|
| `neutral_w34` | 2 / 4 | 2 / 4 | 0.500 |
| `neutral_w36_stylepush` | 2 / 4 | 2 / 4 | 0.500 |

Finding: coverage is the primary component failure. Monet and Van Gogh remain
near neutral in the named token fields.

### 3. Frozen Vocabulary-Only Refit

Runs:

```text
m10_token_vocab_swd_anchor
m11_token_vocab_stylepush
```

These froze the backbone, `style_emb`, and `style_spatial_id_16`, then trained
only `grammar_vocab` and `band_vocab`.

| recipe | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|
| `m10_token_vocab_swd_anchor` | 0.710066 | 0.466699 | 0.618145 | 0.517782 |
| `m11_token_vocab_stylepush` | 0.710138 | 0.466697 | 0.618121 | 0.517815 |

Finding: LPIPS is good, but style does not move enough. Style-push does not
materially improve style.

Direct tensor diff showed:

```text
grammar_vocab delta: 0 for every style
band_vocab delta: mainly Cezanne, tiny elsewhere
```

### 4. Gradient Audit

The gradient audit measured whether the objective actually sends gradient to
the tokenizer rows.

Under the `m10` objective:

| style | grammar grad norm | band grad norm |
|---|---:|---:|
| Hayao | 1.66e-05 | 3.85e-04 |
| Monet | 0.00e+00 | 3.38e-04 |
| Van Gogh | 0.00e+00 | 4.31e-04 |
| Cezanne | 0.00e+00 | 3.45e-04 |

Finding: the current grammar fields are mostly non-executable. They can be
logged, but the objective cannot use them as a strong training handle.

### 5. Tokenizer Projector Route

Completed test:

```text
m12_token_projector_swd_anchor
m13_token_projector_stylepush
```

This keeps the backbone frozen but lets the named tokenizer fields generate
a residual style-code delta through `style_tokenizer.code_projector`.

Full-eval result:

| recipe | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|
| `m12_token_projector_swd_anchor` | 0.709745 | 0.430403 | 0.614650 | 0.482358 |
| `m13_token_projector_stylepush` | 0.709595 | 0.434844 | 0.622817 | 0.488738 |

Finding: the projector route affects the endpoint and improves LPIPS, but it
does not improve global style beyond the vocabulary-only result and Hayao
remains the weakest target. This is a negative result for `cat+project` as a
final tokenizer design. It confirms that simply giving the fields a route back
into the anonymous `style_code` is not enough to create a disentangled metric
space.

Component diagnosis:

| recipe | grammar active | band active | erank g/b | coverage | component |
|---|---:|---:|---:|---:|---:|
| `m12_token_projector_swd_anchor` | 2 | 2 | 0.384 / 0.599 | 0.500 | 0.445 |
| `m13_token_projector_stylepush` | 2 | 2 | 0.377 / 0.629 | 0.500 | 0.449 |

Metric-space diagnosis also failed the axioms:

| recipe | identity-low | grammar-high | grammar-abs-high | band-energy | all-full |
|---|---:|---:|---:|---:|---:|
| `m12` Spearman | 0.000 | 0.139 | 0.321 | -0.103 | -0.055 |
| `m13` Spearman | 0.000 | 0.139 | 0.406 | -0.236 | -0.042 |

Cross-field covariance stayed high:

```text
m13 identity/grammar: 0.748
m13 identity/band:    0.730
m13 grammar/band:     0.973
```

Verdict: the projector route is useful evidence, not the solution. The next
mainline tokenizer experiment should hard-bind fields to operators and measure
whether the field geometry becomes isometric to the real style distributions.

## Core Theoretical Problems

### Problem 1: Identifiability

The tokenizer fields are named, but names alone do not make them identifiable.
If the loss only says "match the target style distribution", several different
field configurations can explain the same SWD decrease.

Required solution:

```text
field value must correspond to a measurable operator effect
```

This means each field needs both:

- a data-derived statistic that identifies it;
- an actuator path that makes changing the field alter the output.

### Problem 2: Executability

The gradient audit shows that `grammar_vocab` is mostly not executable in the
current architecture. This is a concrete failure, not a vague "tokenizer not
large enough" issue.

Required solution:

```text
every tokenizer field must have nonzero Jacobian to the endpoint
```

The projector route is one test. If it fails, the next step is not another
loss-weight sweep; it is an operator change, especially flat-plane and contour
operators for Hayao.

### Problem 3: Style Is Not One Axis

Hayao is not a weaker Van Gogh. It requires:

```text
flat color planes
clean contours
high-frequency suppression
restrained local textons
```

Van Gogh and Cezanne need more texton and mid/high-band structure. Therefore a
single scalar style strength is insufficient.

Required solution:

```text
style code = operator coordinates, not strength coefficient
```

### Problem 4: SWD Alone Under-Specifies the Tokenizer

SWD is a good style-distribution signal, but it does not directly say which
tokenizer field should explain the style difference. It can improve style while
leaving named fields unused.

Required solution:

Use deterministic, training-data-derived auxiliary targets that do not import
external model supervision:

| field group | possible internal statistic |
|---|---|
| palette | low-frequency latent mean/covariance |
| flatness | highpass energy, local variance entropy, plane-size statistics |
| contour | latent/image gradient concentration and edge-band consistency |
| band allocation | multiscale energy ratios and patch covariance spectrum |
| transport softness | SWD/OT entropy, attention margin, match uniqueness |

These statistics should supervise tokenizer fields, not final image labels.

## Axiomatic Definition Of A Good Tokenizer

In this project the style tokenizer is not a static attribute vector. It is the
coordinate system for the vector-field operator used by the ODE. A useful
tokenizer therefore needs three properties:

1. **Algebraic orthogonality.** Named fields must not collapse into the same
   Jacobian direction. A grammar update should not be explainable by an
   identity update.
2. **Operator-isomorphic binding.** A field should map to a specific operator
   family, preferably linearly. The old `cat(identity, grammar, band) ->
   projector -> style_code` route is useful as a diagnostic but violates this
   principle because it re-mixes the fields before the backbone sees them.
3. **Measure separability.** The field should be identified by a matching
   statistic: low-frequency SWD for identity, high-frequency SWD for grammar,
   and multiscale energy ratios for band allocation.

The direct implementation route is:

| field | physical meaning | operator binding | identifying statistic |
|---|---|---|---|
| `identity` | zero-order color/light statistics | `1x1` pointwise channel mixing and bias | low-frequency SWD |
| `grammar` | local stroke and contour geometry | depthwise `3x3` spatial kernels | high-frequency / abs-high SWD |
| `band_logits` | frequency energy valve | direct low/mid/high residual gains | target style energy ratios |

This is now represented in code by `dynamic_style_operator_mode =
"factorized_token"`: the output head consumes `StyleTokenFields` directly
instead of only consuming the anonymous `style_code`.

## Metric-Space Diagnostic

The tokenizer must be tested as a representation before it is used as a
performance knob. The diagnostic script:

```powershell
python tools\experiments\diagnose_style_token_metric_space.py `
  --checkpoint exp\vae_backend_256_probe\ema_style_vocab_neutral_w36_stylepush\epoch_0008.pt `
  --latent-root I:\Github\Latent_Style\latent-256-sd15-ema `
  --out-dir exp\diagnostics\style_token_metric_space\neutral_w36_epoch8
```

computes:

- real style distribution distances for full / low / high / abs-high branches;
- target style low/mid/high energy ratios;
- token-space distances for identity / grammar / band / all fields;
- Spearman and Pearson correlation between token distances and data distances;
- field cross-covariance and effective rank.

The key readout is not a single score. The expected alignment is:

```text
identity_token_l2  <-> data_low_swd
grammar_token_l2   <-> data_high_swd and data_abs_high_swd
band_token_l2      <-> data_log_band_energy_l2
all_token_l2       <-> data_full_swd
```

If these correlations are weak, the tokenizer is not a disentangled metric
space even if a downstream eval score is acceptable.

### Problem 5: Component Metrics Must Precede Average Score

A run with good average LPIPS and mediocre style can still be a tokenizer
failure if the vocabulary does not cover all styles. Conversely, a run may be
useful even before it beats the full metric if it proves better field
executability.

Required solution:

Every tokenizer experiment should report:

```text
coverage
effective rank
per-style field norms
field Jacobian / gradient audit
Hayao cross-target metrics
first-grid visual check
global clip_style / LPIPS
```

## Proposed Research Plan

### Stage A: Finish Projector Test

Question:

```text
Does a tokenizer-owned style-code projector unlock unused backbone capacity?
```

Result:

The projector gave no global style gain and only a small Hayao lift
(`0.6147 -> 0.6228`) while preserving low LPIPS. This is not a performance
solution. It mainly proves that the fields can influence the endpoint if routed
through `style_code`, but that route destroys the desired field semantics.

### Stage B: Add Data-Derived Field Targets

Do not use external generated references as training targets. Compute style
statistics from the training latent/style image pools and use them to make
token fields identifiable.

Candidate losses:

```text
band target loss: token band_gains should match target style multiscale energy ratios
flatness target loss: flatness/highfreq suppression should match plane/variance statistics
coverage loss: non-photo styles should not collapse to neutral rows
orthogonality loss: fields should not all encode the same style axis
Jacobian floor: small sampled changes in active fields should measurably alter endpoints
```

### Stage C: Add Missing Operators If Needed

Projector training did not make Hayao visually clean. The immediate operator
test is the hard-bound dynamic head:

```text
dynamic_style_operator_mode = "factorized_token"
```

Candidate missing operators if hard binding still fails:

- flat-plane repaint / low-variance region simplification;
- contour-preserving edge-band branch;
- high-frequency suppression gated away from true semantic edges;
- texton injection that is style-specific rather than globally mid-band.

### Stage D: Spiral Back To Backbone

Only after the tokenizer has measurable coverage and executability should the
backbone be trained again. The next backbone run should consume the improved
tokenizer, then the tokenizer should be refined again against the improved
backbone.

## Current Task List

1. Finish `m12/m13` projector refit and run full eval.
2. Run the component scorecard on `m12/m13`.
3. Run gradient/Jacobian audit on `m12/m13`.
4. Inspect first-grid and Hayao grid, not only global metrics.
5. If projector helps, design field-statistic supervision.
6. If projector fails, implement explicit flat-plane / contour operators.
7. Log every step in `docs/logs/experiment_ledger.md`.
