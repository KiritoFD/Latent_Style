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
The route is now retired from the runnable tokenizer implementation: `StyleTokenizer`
returns independent fields only, and the old `cat -> code_projector -> style_code`
path is kept only as a historical negative result in this document.

### Stage B: Add Data-Derived Field Targets

Do not use external generated references as training targets. Compute style
statistics from the training latent/style image pools and use them to make
token fields identifiable.

Candidate component diagnostics, not active training losses:

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

First result from the hard-bound output-head-only run:

```text
ema_style_vocab_factorized_w36 epoch8:
  clip_style = 0.665982
  content_lpips = 0.324816
  EC = 0.449660
  Hayao clip_style = 0.586460

ema_style_vocab_factorized_w40_stylepush epoch8:
  clip_style = 0.665615
  content_lpips = 0.323082
  EC = 0.450567
  Hayao clip_style = 0.584838
```

Interpretation: this is not a style solution. It proves the fields can preserve
structure extremely well, but binding them only at the terminal output head
turns the model into a conservative near-identity vector field. The next
operator test is `dynamic_style_feature_operator`: the same independent
`StyleTokenFields` are injected into decoder features before the output head,
so the tokenizer controls an active vector-field actuator rather than only the
last residual projection.

That feature-level operator was also negative:

```text
ema_style_vocab_factorized_feature_w36 epoch8:
  clip_style = 0.664501
  content_lpips = 0.327682
  EC = 0.446756
  Hayao clip_style = 0.580768
```

This falsifies the "actuator placement alone" hypothesis. The feature operator
increased motion but degraded tokenizer metric-space alignment.

Correction after visual review: these factorized runs are also visually
unacceptable because they preserve structure by turning the endpoint into a
low-contrast, hazy, near-identity field. They must not be used as the new
baseline even though LPIPS is low. The style-normal anchor is still the
adapter-calibrated `m02_embspatial_highpass_style` result:

| anchor | clip_style | content_lpips | EC |
|---|---:|---:|---:|
| `m02_embspatial_highpass_style` | 0.71073 | 0.40735 | 0.84967 |

Therefore the next theory step is not to add more loss terms to the main OMF
objective. Tokenizer research should first be constrained to this style-normal
level: any tokenizer route that drops visible style toward the factorized
`0.665` regime is a negative result, regardless of LPIPS.

### Stage D: Spiral Back To Backbone

Only after the tokenizer has measurable coverage and executability should the
backbone be trained again. The next backbone run should consume the improved
tokenizer, then the tokenizer should be refined again against the improved
backbone.

## Current Task List

1. Keep the style-normal anchor active: `m02_embspatial_highpass_style`
   (`0.71073 / 0.40735 / 0.84967`). Do not promote factorized output/feature
   runs; they are hazy negative controls.
2. Do not modify the main OMF loss yet. First answer whether tokenizer changes
   are necessary and executable on top of the style-normal anchor.
3. Treat tokenizer as a component problem: measure coverage, effective rank,
   field response, gradient/Jacobian to endpoint, per-style field norms, and
   Hayao cross-target metrics.
4. If tokenizer is changed, use adapter-level or routing-level probes that keep
   the m02 visual style gate. Required gate: no global clip_style collapse below
   about `0.705`, no hazy first-grid regression.
5. Only after a tokenizer route preserves style should backbone training be
   restarted with that route.
6. Log every step in `docs/logs/experiment_ledger.md`.

## Completed Tokenizer Probe: Band-Gate Coordinates

Decision: the tokenizer step must not touch the output head and must not add
main OMF losses. The safe execution surface already exists in
`StyleBlender._style_texton_band_allocation`: `style_tokens.band_gains` can
multiply the texton carrier's low, mid, and high bands.

One-line hypothesis:

```text
Freeze the texton backbone; train only tokenizer.band_vocab as the low/mid/high
texton energy valve.
```

This is a cleaner tokenizer test than the factorized output head because:

- it preserves the proven `style_emb + style_spatial_id_16 + transport_texton`
  style path;
- it gives `band_logits` an executable physical meaning;
- it cannot by itself replace the whole endpoint with a hazy near-identity
  output;
- it directly targets the observed per-style frequency mismatch, especially
  Hayao's need for flatter planes and less generic high-pass texture.

The implementation is intentionally separate from backbone training:

```text
tools/experiments/run_tokenizer_bandgate_calibration.py
```

It constructs a tokenizer-enabled copy of a texton checkpoint, loads the source
weights non-strictly, freezes every existing parameter, and trains only:

```text
style_tokenizer.band_vocab.weight
```

The first gate is visual/style preservation, not just LPIPS:

```text
clip_style must not collapse below ~0.705, first-grid must not become hazy.
```

Result on 2026-05-27:

| recipe | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `bg00_band_anchor` | 0.71289 | 0.44403 | 0.60185 | safe but style-neutral |
| `bg01_band_stylepush` | 0.71264 | 0.44406 | 0.60096 | safe but style-neutral |

Interpretation:

- Band-gate calibration did not reproduce the hazy factorized failure. It stays
  above the style gate and improves LPIPS relative to the texton source.
- It also does not raise style. The global style remains near `0.713`, and
  Hayao remains the weakest target near `0.601`.
- Therefore band-gate is a valid tokenizer coordinate and a content-safety
  valve, but it is too low-rank to be the primary style actuator.
- The active rollback anchor remains `m02_embspatial_highpass_style`
  (`0.71073 / 0.40735 / 0.84967`). Factorized output/feature routes are
  rejected as hazy negative controls and must not be promoted.

Revised rule: tokenizer work should change representation or routing only
after preserving this style-normal level. Do not increase scalar losses to hide
tokenizer weakness; if the first grid becomes foggy, stop and return to the
anchor.

## Next Probe: Tokenizer-Gated Transport-AdaIN

One-line hypothesis:

```text
Tokenizer should not replace the output head; it should act as a low-rank
valve over the proven m02 transport-AdaIN carrier.
```

Reasoning:

- The m02 anchor is style-normal and content-safe, but Hayao remains weak.
- Texton-only band gates are safe but too weak because the texton carrier is
  not the active m02 carrier.
- The dangerous factorized head failed because it replaced the endpoint
  operator and collapsed into a hazy near-identity map.
- Therefore the next executable tokenizer field should be attached to the
  m02 carrier itself: band tokens multiply transport-AdaIN low/mid/high
  residuals, while grammar tokens only suppress high-frequency drift in flat
  regions.

Implementation:

```text
model.style_token_adain_gate_enable = true
tools/experiments/run_tokenizer_adain_gate_calibration.py
```

The switch defaults to `false`, so previous checkpoints keep their original
behavior. The probe loads `ema_transport_adain_w34_guard/epoch_0006.pt`,
applies `m02_embspatial_highpass_style/style_adapter.pt`, freezes the backbone,
`style_emb`, and `style_spatial_id_16`, then trains only:

```text
style_tokenizer.grammar_vocab.weight
style_tokenizer.band_vocab.weight
```

Decision gate:

- reject if style drops below the m02 style-normal range or first-grid becomes
  foggy;
- keep only if global style rises toward `0.72+` or Hayao cross-style improves
  without LPIPS leaving the useful `0.47-0.50` band.

Result on 2026-05-27:

| recipe | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `ag00_m02_safe_gate` | 0.71076 | 0.40728 | 0.60489 | safe but style-neutral |
| `ag01_m02_style_gate` | 0.71061 | 0.40729 | 0.60514 | safe but style-neutral |

This confirms that tokenizer fields can be connected to the m02 carrier without
causing the hazy factorized failure, but the connection is too weak to solve
style. The current rollback state is:

1. The active style-normal anchor is `m02_embspatial_highpass_style`
   (`0.71073 / 0.40735 / 0.84967`).
2. No main OMF loss changes should be made to compensate for tokenizer weakness.
3. Hazy/de-stylized factorized output or feature operators are rejected even if
   LPIPS is low.
4. The next tokenizer step is a diagnostic, not a new training run: measure
   token field movement, endpoint sensitivity, and per-style carrier response
   around the m02 anchor.

## 2026-05-28 Revision: Reader Bottleneck After Stat Vocab

The m02 sensitivity diagnostic and stat-vocab probe sharpen the theory.

Measured fact:

```text
band_low is executable;
band_mid is weakly executable;
band_high is nearly dead;
grammar texture fields are mostly dead;
stat-initialized token fields preserve the m02 visual family but barely change
global style.
```

This separates two possible failures:

1. **Bad vocabulary values.** The token rows do not encode real style
   differences.
2. **Bad reader/executability.** The token rows encode differences, but the
   carrier cannot read them.

The stat-vocab probe attacks the first failure. It maps measurable training
latent statistics into token fields:

- low/mid/high energy ratios -> `band_logits`;
- flatness / high-frequency suppression -> grammar dimensions 1 and 7;
- contour concentration -> grammar dimension 2;
- mid/high texture energy -> grammar dimensions 5 and 6.

Results:

```text
sv00_stat_m02_conservative: 0.710740 / 0.407393, Hayao 0.605003
sv01_stat_m02_balanced:     0.710551 / 0.407434, Hayao 0.604945
```

Interpretation:

```text
The stat vocabulary is safe but not executable enough. The immediate bottleneck
is not token value quality; it is the reader path from token fields to the m02
transport-AdaIN operator.
```

The next minimal architectural move is therefore a zero-initialized
`token_reader`:

```text
[identity, grammar, band_logits] -> tiny MLP -> low/mid/high gain residual
```

This is deliberately smaller than the rejected factorized output-head route. It
does not replace the endpoint operator and starts exactly at m02 behavior
because the final layer is zero-initialized. The only question it tests is:

```text
Can the proven m02 carrier learn to read a measured tokenizer coordinate system?
```

First reader result:

```text
sr00_stat_reader_safe: 0.710698 / 0.405284, Hayao 0.604194
sr01_stat_reader_style: 0.710523 / 0.402585, Hayao 0.604742
```

This verifies that the reader can remain visually safe and slightly improve
LPIPS, but it still does not lift style. The conclusion is not "increase loss".
The conclusion is that the current tokenizer readout only controls a weak
amplitude valve over the m02 carrier. It has no proven mechanism for style
placement or texture selection.

User correction incorporated: return to the style-normal m02 anchor before any
next probe. Hazy/de-stylized factorized outputs are rejected, even when LPIPS is
low. The next tokenizer step should be diagnostic and architectural: identify
which token fields have nonzero endpoint Jacobian, which transport-AdaIN
residual bands carry visible style, and whether a field can be bound to a
specific existing operator without changing the main OMF loss.

Current compact registry:

```text
docs/logs/tokenizer_spiral_experiment_registry.md
docs/logs/tokenizer_spiral_experiment_registry.csv
```

## 2026-05-28 Operator-Binding Revision: g56 Texture Executability

The latest reader probe falsifies a weak hypothesis:

```text
tokenizer fields -> scalar low/mid/high gain reader
```

is not enough. It improves LPIPS by staying close to the m02 endpoint, but it
does not create visible style. The more useful read from the earlier endpoint
sensitivity diagnostic is:

```text
grammar[5] and grammar[6] are intended as mid-texton and high-texture fields,
but m02 does not execute them unless they are explicitly bound to a residual
operator.
```

One-line hypothesis:

```text
Bind grammar[5]/grammar[6] directly to the existing m02 transport-AdaIN
mid/high residual gains; if this creates mid/high endpoint motion without
fogging low bands, tokenizer can be trained with the backbone frozen.
```

This is a tokenizer/operator change, not a loss change:

- the main OMF loss remains untouched;
- the m02 transport-AdaIN endpoint remains the carrier;
- the switch defaults off;
- the new binding only multiplies existing mid/high residuals and cannot
  replace the whole output head.

Diagnostic script:

```text
tools/experiments/diagnose_m02_tokenizer_operator_binding.py
```

Decision rule:

1. If `band_low` remains the only strong response, the tokenizer is still a
   color/fog valve and this route should be rejected.
2. If `grammar_mid_texton` and `grammar_high_texture` produce comparable
   mid/high endpoint motion, run the next stage: freeze m02 and train only
   tokenizer fields through this binding.
3. If stat-token preview remains tiny even after the binding, the measured
   style coordinates are not aligned to the executable direction and need a
   better tokenizer statistic, not more scalar style pressure.

Result:

```text
grammar_mid_texton perturbation: endpoint RMS up to 0.00543
stat_vocab_preview:              endpoint RMS 0.00493-0.00603
grammar_high_texture perturbation: near 0.0002
```

The hypothesis is partially validated. The tokenizer has one newly executable
texture coordinate: `grammar[5] -> mid residual`. This is enough to run a
tokenizer-only training step. It also says `grammar[6] -> high residual` is
not yet a useful high-frequency actuator; the high residual path is too small
or too gated. The next run should not touch the backbone or OMF loss:

```text
ag02/ag03: freeze m02, train grammar+band only through g56 binding.
```

If these runs improve style without haze, the tokenizer/mainline spiral can
continue on this operator. If they stay neutral, the next architectural change
must create a stronger high-frequency carrier rather than increasing scalar
style pressure.

g56 tokenizer-only result:

```text
ag02_m02_g56_texture_anchor: 0.710955 / 0.407269, Hayao 0.605668
ag03_m02_g56_texture_push:   0.710725 / 0.407305, Hayao 0.605254
```

The result is safe but not sufficient. It validates that `grammar[5]` can be
made executable, but also shows that a mid-band gain valve cannot create the
missing style energy. The failed part is now sharper:

```text
Tokenizer has a stable mid-texton actuator, but the high-texture carrier is
too weak; pushing HP-SWD only changes tiny gains and does not move CLIP style.
```

Next theoretical move:

```text
create a real high-frequency/texton carrier first, then let tokenizer fields
select it; do not add more scalar style pressure to g56.
```

## 2026-05-28 Carrier Revision: Token-Selectable Texton Residual

The g56 result isolates the failure more tightly than the earlier tokenizer
collapse:

```text
grammar[5] can select an existing mid-band actuator, but the actuator itself
does not contain enough high-frequency style energy to move CLIP style.
```

The next change is therefore not a new loss and not a larger tokenizer. It is a
new executable carrier:

```text
content high band + AdaIN residual mid/high band -> zero-start texton mapper
```

and the tokenizer controls that carrier through the already named fields:

```text
grammar[5] -> mid texton emphasis
grammar[6] -> high texture emphasis
band_logits -> low/mid/high residual allocation
```

The important constraints are:

- default off in `ModelConfig`;
- initialized to zero so the m02 anchor remains the rollback point;
- when enabled, train only `style_tokenizer.grammar_vocab`,
  `style_tokenizer.band_vocab`, and `blender.token_texton_carrier_mapper`;
- keep the teacher anchor as m02 behavior and reject the route immediately if
  visual grids become hazy or de-stylized.

One-line hypothesis:

```text
If the tokenizer plateau is caused by a missing high-frequency carrier, a
zero-start token-selectable texton residual should raise CLIP style beyond ag02
without the factorized-token haze failure.
```

Planned tests:

```text
tc00_m02_texton_carrier_anchor: conservative carrier, expected non-hazy style gain.
tc01_m02_texton_carrier_push: stronger high branch, stress-test toward 0.72+.
```

Result:

```text
tc00_m02_texton_carrier_anchor: 0.710431 / 0.407304, Hayao 0.604767
tc01_m02_texton_carrier_push:   0.710621 / 0.406945, Hayao 0.605833
ag02_m02_g56_texture_anchor:    0.710955 / 0.407269, Hayao 0.605668
```

The route is safe but rejected as a mainline improvement. It is not the same
failure as the hazy factorized tokenizer: grids remain close to m02 and LPIPS
stays healthy. The failure is subtler:

```text
The added carrier learns a small residual around the same AdaIN target, but it
does not change the spatial/source distribution of style textons. It is an
amplitude-side perturbation, not a new OT-routed style basis.
```

Theory correction:

```text
The bottleneck is no longer "tokenizer cannot actuate" and is not solved by
"add more high-band residual." The tokenizer needs a carrier whose source is
the target style manifold itself: style-prototype/texton patches routed by the
semantic transport map, not content-high + AdaIN residual remapping.
```

Next design direction:

```text
Build a small style-prototype carrier from target-style latent patches or
training-set style centroids, inject it through semantic attention / support
gates, and keep the tokenizer responsible for choosing prototype frequency and
strength. This keeps Seedream out of training and uses only the internal style
measure.
```

## 2026-05-28 Prototype-Carrier Test

The texton-carrier result rules out a simple amplitude explanation:

```text
content-high + AdaIN residual -> trainable carrier
```

is safe but does not lift style. The next minimal structural hypothesis is:

```text
The missing carrier source is the style-routed semantic feature itself.
```

Implementation:

- keep the m02 backbone and m02 style adapter frozen;
- keep the main OMF / SWD objective unchanged;
- add a default-off, zero-start `style_token_prototype_carrier`;
- source its seed from `style_feat` inside `StyleBlender.forward`, i.e. the
  semantic-routed painted feature, not Seedream and not an external teacher;
- train only `style_tokenizer.grammar_vocab`,
  `style_tokenizer.band_vocab`, and `blender.token_prototype_carrier_mapper`.

One-line experiment logic:

```text
If the plateau is caused by residual source mismatch, a style-routed prototype
carrier should add visible textons while preserving the m02 content anchor.
```

Decision rule:

1. Reject immediately if the grid becomes hazy/de-stylized, even if LPIPS
   improves.
2. Keep only if it beats `ag02_m02_g56_texture_anchor` both visually and in
   `clip_style`.
3. If prototype RMS stays near zero, the issue is carrier gradient flow or
   zero-start mapper capacity.
4. If prototype RMS is nonzero but style is flat, the source is still not
   target-discriminative enough and the next step should be explicit
   style-prototype memory, not more loss pressure.

Planned runs:

```text
pc00_m02_prototype_carrier_anchor
pc01_m02_prototype_carrier_push
```

Result:

```text
pc00_m02_prototype_carrier_anchor: 0.710468 / 0.406890, Hayao 0.605174
pc01_m02_prototype_carrier_push:   0.710299 / 0.407741, Hayao 0.605220
ag02_m02_g56_texture_anchor:       0.710955 / 0.407269, Hayao 0.605668
```

Interpretation:

```text
The prototype branch is executable but not style-discriminative. Increasing
prototype-carrier energy moves metrics sideways or down, so the missing piece
is not residual amplitude and not generic style_feat routing.
```

This falsifies the current prototype-carrier hypothesis. The style source used
by `StyleBlender` is still too close to the learned m02 centroid, so the next
test must expose the model to a target-style source constructed from the
training-set style manifold itself.

## 2026-05-28 Style-Memory Bank Probe

One-line hypothesis:

```text
If m02 is style-limited because its learned style_spatial_id_16 is an averaged
style source, replacing part of that source with training-set body-feature
prototypes should lift CLIP style before any tokenizer/loss change.
```

This is a source-quality diagnostic, not external supervision:

- no Seedream/API data is used;
- the backbone remains frozen;
- the main OMF/SWD loss remains unchanged;
- `style_emb` is kept from the m02 adapter;
- only the adapter's `style_spatial_id_16` is edited, using body-level features
  extracted from the same latent training pools.

Two source constructions are planned:

```text
mb00_body_mean_blend25:
  style_spatial_id_16 <- 75% m02 spatial prior + 25% mean target-style body feature

mb02_body_exemplar_blend35:
  style_spatial_id_16 <- 65% m02 spatial prior + 35% high-texture exemplar body feature
```

Decision rule:

1. If both runs lower style or create semantic imprinting, the learned
   `style_spatial_id_16` is not the primary source bottleneck.
2. If the mean prototype is safe but weak while the exemplar is stronger, the
   next model change should be a multi-prototype memory bank selected by the
   tokenizer rather than a single centroid.
3. If either run approaches or exceeds `0.72` without LPIPS leaving the useful
   band, use that adapter as the next frozen source and return to tokenizer
   training.

Result:

```text
mb00_body_mean_blend25:     0.710612 / 0.407762, Hayao 0.604948
mb02_body_exemplar_blend35: 0.710516 / 0.409222, Hayao 0.606350
ag02_m02_g56_texture_anchor:0.710955 / 0.407269, Hayao 0.605668
```

Prototype statistics:

```text
mean prototypes:
  prototype highpass ~= 0.118-0.127
  old m02 spatial highpass ~= 0.668-0.685

high-texture exemplars:
  exemplar highpass ~= 0.725-0.735
  blended spatial highpass ~= 0.699-0.709
```

Interpretation:

```text
The mean prototype branch is too smooth and loses style. The high-texture
exemplar branch proves that a non-averaged style source can move Hayao upward,
but a single blended 16x16 style map damages LPIPS and does not improve global
style. Therefore the problem is not solved by replacing style_spatial_id_16.
```

Theory correction:

```text
Style source quality matters, but a single centroid or single exemplar is the
wrong abstraction. The next memory design, if pursued, must be a tokenizer-
selected multi-prototype bank with local routing, so flat Hayao regions can
select contour/plane prototypes while Van Gogh/Cezanne select texton prototypes.
```

Immediate decision:

```text
Do not promote mb00/mb02. Keep ag02/m02 as the active safe anchor. The next
mainline move should either build token-selected multi-prototype routing, or
return to a frozen-backbone style-embedding refit that deliberately allows
LPIPS to move toward 0.47-0.50 while guarding against visible haze.
```

## 2026-05-28 Reference-Memory Generation Probe

The single-map memory-bank result left one open question:

```text
Is the m02 plateau caused by the architecture being unable to use style source
features, or by the id-only style source being too centroid-like?
```

Implementation:

- keep the m02 checkpoint and `m02_embspatial_highpass_style` adapter frozen;
- do not change OMF/SWD loss and do not train any parameter;
- pass an internal target-style latent into the existing runtime as
  `target_style_latent`;
- evaluate the generated images through the normal reuse-generated full-eval.

Two selection rules were tested:

```text
rm00_random_ref1:
  use one random training-pool latent from the target style as the reference
  source.

rm01_lowfreq_match_k8:
  restrict to high-texture target-style candidates, then choose the one whose
  low-frequency latent descriptor best matches the source content.
```

Result:

```text
rm00_random_ref1:        0.715127 / 0.477313, Hayao 0.628046
rm01_lowfreq_match_k8:   0.715447 / 0.477220, Hayao 0.627415
m02 anchor:              0.710730 / 0.407350
ag02 tokenizer anchor:   0.710955 / 0.407269, Hayao 0.605668
```

Interpretation:

```text
The runtime can use explicit target-style source features. This falsifies the
strong claim that the backbone cannot express more style. The bottleneck is
the id-only/tokenizer style source and routing: replacing the centroid source
with a real internal style latent gives +0.0044 clip_style and +0.022 Hayao
style, while keeping LPIPS inside the accepted 0.47-0.50 band.
```

Selector correction:

```text
The low-frequency global selector barely improves over random. A single global
reference is not the right abstraction. The useful signal is the existence of
multiple real target-style sources; the selection must become local and
tokenizer-controlled, not one image-level choice.
```

Visual gate:

```text
The grids are not the rejected hazy/factorized failure, but max-clip rows show
reference-source imprint risk and LPIPS rises to about 0.477. Therefore this is
a strong diagnostic and design guide, not a final id-only paper protocol.
```

Next design:

```text
Build a tokenizer-selected multi-prototype source inside the model. The
prototype bank should be internal training-set style features; tokenizer fields
should choose local mixtures of flat/edge/texton prototypes. Success criterion:
keep the rm00/rm01 style lift without needing an external reference image at
test time, and avoid the single-map imprint failure.
```

## 2026-05-28 Id-Only Multi-Prototype Bank Adapter Probe

The reference-memory result motivated the smallest protocol-compatible
translation:

```text
Replace the single learned style_spatial_id_16 source with an adapter-side
style_memory_bank_16 containing multiple internal training-set prototypes per
style. At inference, select only by style id, with no reference image and no
loss/training change.
```

Three bank recipes were evaluated:

```text
bm00_hightex_k4_blend65:        0.710854 / 0.407380, Hayao 0.605454
bm01_diverse_k4_blend65:        0.710676 / 0.407408, Hayao 0.605344
bm02_hightex_k4_boost_blend75:  0.710693 / 0.407397, Hayao 0.605188
ag02 tokenizer anchor:          0.710955 / 0.407269, Hayao 0.605668
rm01 reference-memory diagnostic:0.715447 / 0.477220, Hayao 0.627415
```

The prototypes were not empty or too smooth:

```text
high-texture prototype highpass scores are about 0.73-0.77 across styles.
diverse prototypes change the grids slightly, so the adapter path is loaded.
```

Theory correction:

```text
The missing ingredient is not merely "more style prototypes exist." A global
style-id mixture is still a centroid operator. It destroys the sample-local
geometry that made reference-memory useful and collapses into the same m02
plateau.
```

New tokenizer/source requirement:

```text
Prototype selection must be local and conditional. Each spatial site or
semantic region needs to choose among flat, edge, and texton prototypes based on
content features plus style tokens. A single bank average per target style is
not an adequate style representation.
```

Decision:

```text
Reject bm00/bm01/bm02 as mainline. Do not increase blend, boost, or loss around
this static-bank design. The next valid route is a local prototype router or a
backbone actuator that consumes a prototype bank before style_spatial_id_16
collapses it into one map.
```

## 2026-05-28 Local Prototype Router Probe

The static bank failure suggested a sharper hypothesis:

```text
Maybe the bank failed only because prototypes were averaged globally. Use
content_feat_16 as queries and style prototype tokens as a dictionary, then
build a local routed style map before the actuator.
```

Implementation:

- keep m02 checkpoint and adapter frozen;
- no loss change and no training;
- store the same internal prototype bank in the adapter;
- when `style_memory_bank_route_strength > 0`, skip static bank averaging and
  compute content-conditioned token attention over `K * 16 * 16` style
  prototype tokens.

Results:

```text
br00_route_hightex_k4_s45: 0.710530 / 0.407402, Hayao 0.604825
br01_route_hightex_k4_s65: 0.710609 / 0.407408, Hayao 0.604584
ag02 tokenizer anchor:     0.710955 / 0.407269, Hayao 0.605668
rm01 diagnostic reference: 0.715447 / 0.477220, Hayao 0.627415
```

Theory correction:

```text
Local dictionary lookup is not sufficient when it is inserted only as a
replacement style_map for a frozen actuator. The downstream body blocks were
trained to interpret the old style_spatial_id_16 distribution, so they absorb
the routed bank into the same weak m02-style operating region.
```

New requirement:

```text
The prototype router must be part of the trainable actuator/backbone phase, or
it must enter as a separate residual/source field that the body actually learns
to use. A route-only adapter is too weak.
```

Decision:

```text
Reject br00/br01. Do not sweep route temperature or route strength. The next
spiral step should train a router-aware actuator while freezing the tokenizer
bank, or start a backbone phase with the local router enabled from the first
batch.
```

## 2026-05-28 Router-Aware Actuator and Memory-Residual Tests

The next two probes tested the previous requirement directly.

### Probe A: train the old routed style-map actuator

```text
ra00_route_actuator_s45_e2:
  fixed br00 local router source
  train body/blender/decoder/output head for 2 epochs
  result = 0.710336 / 0.435838, Hayao 0.614082
```

Interpretation:

```text
Training the consumer of the routed style map is not enough. Hayao improves,
which proves the route is not dead, but global style drops and LPIPS worsens.
The old style-map interface still entangles source texture with content drift.
```

### Probe B: bypass style-map collapse with a residual source field

Implementation:

```text
Keep the base style_spatial_id_16 path intact. Build a local routed prototype
source from the style memory bank, then inject it after the body paint/blender
as an explicit residual field:

  h_body <- h_body + tanh(phi(routed_memory, base_map, content)) * strength

This tests whether reference-memory worked simply because it supplied a real
style source outside the learned single-map prior.
```

Results:

```text
rs00_memory_residual_s22_e2:    0.707073 / 0.432237, Hayao 0.606759
rs01_memory_residual_hp_s32_e2: 0.707358 / 0.429501, Hayao 0.606127
ag02 tokenizer anchor:          0.710955 / 0.407269, Hayao 0.605668
```

Numeric debug confirmed the branch was active:

```text
style_memory_residual_delta mean_abs ~= 0.095 in smoke, finite_ratio = 1.0
```

Theory correction:

```text
The missing component is not just an additional style-energy source. An
untyped prototype residual becomes a cross-style average perturbation: it can
change the image safely, but it does not align the generated endpoint with the
target style measure. Highpass gating reduces content damage slightly but does
not create style semantics.
```

Updated tokenizer requirement:

```text
The prototype source must be typed before it enters the dynamics. Each memory
atom needs an explicit role such as flat color, edge geometry, texton texture,
or target-specific palette, and assignment must be style-discriminative rather
than only content-nearest. The next route should use semantic/contrastive
prototype assignment or field-wise OT alignment, not residual amplitude sweeps.
```

Decision:

```text
Reject ra00/rs00/rs01 as mainline improvements. Do not continue route strength,
residual strength, or highpass-kernel sweeps. The next valid tokenizer step is
to learn a style-field typed prototype vocabulary while keeping ag02/m02 as the
rollback anchor.
```

## 2026-05-28 Typed Prototype Memory Probe

The next hypothesis was narrower:

```text
Maybe untyped residual memory fails because the bank mixes incompatible roles.
Split each style bank into flat, edge, and texton atoms, then route each local
content region through typed prototype groups before injecting the residual.
```

Implementation:

- build `style_memory_bank_type_ids` and `style_memory_bank_type_logits` in the
  adapter;
- compute local content gates from highpass magnitude and edge support;
- mix routed flat/edge/texton memories into a residual source;
- train the actuator/backbone consumer for 2 epochs while freezing
  `style_emb`, `style_spatial_id_16`, and the tokenizer.

Results:

```text
rt00_typed_fet_s18_e2:        0.708062 / 0.432247, Hayao 0.608992
rt01_typed_fet_hp_s24_e2:     0.706646 / 0.431261, Hayao 0.607102
rt02_typed_uniform_s20_e2:    0.706897 / 0.427892, Hayao 0.607524
rt03_typed_uniform_hp_s24_e2: 0.707270 / 0.428961, Hayao 0.607934
ag02 tokenizer anchor:        0.710955 / 0.407269, Hayao 0.605668
rm01 diagnostic reference:    0.715447 / 0.477220, Hayao 0.627415
```

Numeric debug:

```text
style_memory_type_gates finite_ratio = 1.0
style_memory_type_gates mean ~= 0.3333
style_memory_type_gates max ~= 0.99996-0.99999
style_memory_residual_delta mean_abs ~= 0.060-0.069
```

Theory correction:

```text
Typed flat/edge/texton roles are not sufficient. The branch is active, and
uniform-within-type routing rules out a pure content-nearest lookup failure.
The missing condition is style-measure alignment: atoms are selected by local
feature statistics, but their assignment is not constrained to reduce the
target style distribution discrepancy.
```

Decision:

```text
Reject rt00/rt01/rt02/rt03 as mainline improvements. Do not continue typed
residual strength, highpass support, or uniform-routing sweeps. The next valid
route is to make prototype assignment itself style-discriminative, for example
by selecting atoms through internal OT/style-distance contribution or a
contrastive target-vs-other-style criterion before the dynamics consume them.
```

## 2026-05-28 Fisher Style-Measure Probe

The next test isolated the assignment hypothesis.

Hypothesis:

```text
The raw style-measure descriptor is too close to a shared body-feature
coordinate system. Projecting it into a Fisher discriminant subspace using only
internal style labels should produce atoms that are genuinely style-separable.
```

Implementation:

- keep the base `m02/ag02` dynamics unchanged;
- compute the existing low/high channel-stat style descriptor for each
  candidate body feature;
- standardize descriptors, estimate within-style and between-style scatter,
  and project to the top Fisher directions;
- select prototypes by contrastive own-style-vs-other-style margin in this
  Fisher space;
- test both untyped and flat/edge/texton typed variants through the existing
  memory-residual actuator consumer.

Measured separability:

```text
raw ma00/ma01/ma02 style_purity abs_mean: 0.0015-0.0044
mf00_fisher_k6_s20 abs_mean:             1.063
mf01_typed_fisher_k9_s22 abs_mean:       0.913
```

Generation results:

```text
rf00_fisher_stylepure_s20_e2: 0.707854 / 0.435530, Hayao 0.608229
rf01_typed_fisher_s22_e2:     0.706630 / 0.431558, Hayao 0.606712
ag02 tokenizer anchor:        0.710955 / 0.407269, Hayao 0.605668
rm01 diagnostic reference:    0.715447 / 0.477220, Hayao 0.627415
```

Theory correction:

```text
Style separability is necessary but not sufficient. Fisher projection proves
that the training pool contains discriminative style coordinates, but the
current residual-memory consumer does not implement a style operator. It
turns even cleanly selected atoms into generic content drift / texture
perturbation, so the failure is now localized to operator execution rather
than prototype discovery.
```

Updated requirement:

```text
Stop memory-residual variants. The next tokenizer stage must bind the
discriminative style coordinate directly to an executable operator: for
example low/mid/high band gains, a depthwise spatial kernel, a pointwise color
kernel, or a transport-AdaIN actuator with a measurable local Jacobian. Run an
operator-readout diagnostic before spending a full 750-image eval.
```

## 2026-05-28 Fisher Operator-Binding Update

The operator-readout requirement was tested directly.

Hypothesis:

```text
Fisher-discriminative style coordinates should not be injected as a residual
source. They should be sign/order aligned by physical mid/high energy and
written directly into executable tokenizer fields on the transport-AdaIN path.
```

Observed no-training readout:

```text
fo00_fisher_operator_readout:
  endpoint_delta_rms = 0.01036
  detail_over_low    = 1.160
  high_fraction      = 0.430
  grammar[5] perturb endpoint_rms ~= 0.00222
```

Interpretation:

```text
The coordinate-to-operator path is real. The endpoint motion is not dominated
by the low band, so Fisher-to-token binding is a valid actuator diagnostic.
```

Training result:

```text
fo01_fisher_operator_token_swd80: 0.710301 / 0.408750, Hayao 0.604612
ag02_m02_g56_texture_anchor:      0.710955 / 0.407269, Hayao 0.605668
```

Correction to the theory:

```text
Operator binding alone is too weak if the operator is only scalar band/grammar
allocation. The scalar gates are executable, but their Jacobian rank is too low
to express a new style measure. Global SWD can see the discrepancy, yet its
gradient barely changes these fields, so the branch stays near m02/ag02.
```

New requirement:

```text
The tokenizer must bind style coordinates to a higher-capacity operator while
preserving field semantics. The next route should keep grammar responsible for
spatial texture, but let it control a depthwise spatial kernel or compact
filter bank instead of only a scalar mid/high multiplier. Identity/band fields
can remain scalar guards.
```

## 2026-05-28 Depthwise Grammar Operator Update

The higher-capacity grammar operator was implemented as a fixed eight-kernel
3x3 depthwise filter bank. `grammar[8:15]` selects Laplacian, Sobel, diagonal,
and checker-style local filters on the transport-AdaIN detail residual. The
operator is deterministic and has no extra trainable residual source.

Readout:

```text
fo10_depthwise_operator_readout:
  endpoint_delta_rms = 0.01087
  detail_over_low    = 1.261
  high_fraction      = 0.432
```

Training:

```text
fo11_depthwise_filter_swd80: 0.710301 / 0.408772, Hayao 0.604718
fo01_scalar_gate_swd80:      0.710301 / 0.408750, Hayao 0.604612
ag02_m02_g56_anchor:         0.710955 / 0.407269, Hayao 0.605668
```

Correction:

```text
Increasing tokenizer operator capacity from scalar gates to a fixed spatial
filter bank is not sufficient under a frozen m02 backbone. The filter delta is
large enough to measure, but the endpoint style metric stays unchanged. Thus
the current bottleneck is not only tokenizer expressivity; it is the frozen
actuator/backbone not converting the new local operator into a style-measure
transport path.
```

Next spiral step:

```text
Freeze the tokenizer fields produced by fo11 and train only the
backbone/actuator consumer. The test is whether the backbone can learn to route
content and style features so the depthwise grammar operator becomes useful.
Reject immediately if LPIPS worsens without style gain or the grid becomes
fragmented.
```

## 2026-05-28 Fisher Depthwise Consumer Test

The next experiment is `fo12_depthwise_consumer_guard_e2`.

One-line model:

```text
An executable style operator is useless unless the transport backbone learns to
consume it; therefore freeze the fo11 tokenizer and train only the consumer
path.
```

This is a backbone step in the tokenizer/backbone spiral, not another tokenizer
value sweep. The tokenizer fields keep their current semantics:

- `band` remains a low/mid/high scalar guard;
- `grammar[5:7]` remains the scalar texture allocation;
- `grammar[8:15]` remains the deterministic depthwise filter-bank selector.

Only the consumer path is trainable:

```text
body_blocks, blender, skip_fusion, decoder_blocks, dec_post, dec_mod, output_head
```

Rejection criteria:

```text
Reject if clip_style does not beat ag02, or if LPIPS/grid quality worsens
without a style gain. A Hayao-only gain is diagnostic but not sufficient unless
global clip_style also moves.
```

Result:

```text
fo12_depthwise_consumer_guard_e2: 0.707308 / 0.433830, Hayao 0.609139
fo11_depthwise_filter_swd80:      0.710301 / 0.408772, Hayao 0.604718
ag02_m02_g56_anchor:              0.710955 / 0.407269, Hayao 0.605668
```

Theory update:

```text
The depthwise grammar operator is not silent: after full consumer-path training,
the weakest Hayao slice improves. The failure is that the full body/decoder
consumer has too many degrees of freedom, so the operator becomes generic
repaint/content drift instead of a constrained style actuator. The next
consumer experiment must be narrower: train the operator interface only
(`blender` first, then possibly `blender + output_head`) or add a small
zero-init operator gate. Do not continue broad body/decoder updates from this
branch.
```

## 2026-05-28 Narrow Consumer Interface Test

The next test is `fo14_depthwise_blender_only_e2`.

One-line model:

```text
If fo12's failure is over-capacity repaint drift rather than operator
uselessness, then training only StyleBlender should expose the depthwise grammar
operator with much smaller LPIPS damage.
```

This is deliberately not a loss change. The tokenizer stays frozen and keeps
the `fo11` semantics. The backbone body, skip path, decoder, and output head
also stay frozen. The only trainable surface is the operator mixing interface:

```text
trainable_name_patterns = ["blender"]
```

Decision rule:

```text
Promote if global clip_style improves over ag02, or if Hayao improves without
LPIPS/grid damage. Reject if it remains metric-flat; in that case the next
minimal interface is `blender + output_head` or a zero-init depthwise operator
gain, not broad body training.
```

Observed failure:

```text
fo14 failed with no grad_fn. The trainable `StyleBlender` named weights are not
in the active `transport_adain` branch. That branch is a deterministic
frequency-transport program controlled by style maps and token fields.
```

Theory update:

```text
The operator interface is not the `StyleBlender` module as a whole; it is the
specific scalar/vector actuator multiplying the token-derived transport terms.
For depthwise grammar, the smallest valid actuator is a learnable mid/high gain
inside `_style_token_depthwise_filter_delta`.
```

Next test:

```text
fo16_depthwise_gate_only_e2
Add `token_depthwise_filter_gate_logits` with two components: mid and high.
Initialize at zero, so the old path is exactly preserved. Train only this gate
from fo11 while allowing its missing checkpoint key explicitly. This is a real
narrow consumer step because the loss is directly connected to the depthwise
operator output.
```

Result:

```text
fo16_depthwise_gate_only_e2: 0.710135 / 0.408745, Hayao 0.604389
fo11_depthwise_filter_swd80: 0.710301 / 0.408772, Hayao 0.604718
ag02 tokenizer anchor:       0.710955 / 0.407269, Hayao 0.605668
```

Learned gate:

```text
token_depthwise_filter_gate_logits = [-0.2566, 0.0657]
with scale 0.75, mid gain ~= 0.812 and high gain ~= 1.049
```

Theory correction:

```text
The new gate is a real connected trainable actuator, unlike fo14, but two
global mid/high scalars are too low-rank. It mostly learns to suppress the mid
branch and barely lift the high branch, which is insufficient to change the
style measure. Do not continue scalar gate-only sweeps. The next minimal
interface must add either the output_head, where transported depthwise features
become latent delta, or a richer per-style/per-basis gate that remains
operator-bound.
```

## 2026-05-28 fo17 Narrow Output-Head Consumer

One-line hypothesis:

```text
fo16 reached the depthwise operator but could only apply two global scalars.
Adding only `output_head` gives that operator a narrow latent-delta decoder
without reopening fo12's broad body/decoder repaint path.
```

Configuration:

```text
source: fo11_depthwise_filter_swd80
frozen: fo11 tokenizer, style identity source, body, skip path, decoder
train:  token_depthwise_filter_gate_logits + output_head
LR:     output_head 5e-5, depthwise gate 50x multiplier
loss:   unchanged main OMF/SWD
```

Decision rule:

```text
Promote only if global clip_style beats ag02 without visible grid damage and
without LPIPS drifting toward fo12. If it is flat, output_head is not the
missing consumer. If it raises Hayao but drops global style, the depthwise
operator is still not style-discriminative enough and the next move should be
per-style/per-basis gating rather than broader training.
```

Result:

```text
fo17_depthwise_gate_head_e2:       0.710782 / 0.438083, Hayao 0.614275
ag02_m02_g56_texture_anchor:      0.710955 / 0.407269, Hayao 0.605668
fo16_depthwise_gate_only_e2:      0.710135 / 0.408745, Hayao 0.604389
fo12_depthwise_consumer_guard_e2: 0.707308 / 0.433830, Hayao 0.609139
```

Learned gate:

```text
token_depthwise_filter_gate_logits = [0.0162, 0.0123]
with scale 0.75, mid gain ~= 1.012 and high gain ~= 1.009
```

Theory correction:

```text
The output head is connected and can consume the depthwise operator, because
Hayao rises sharply. But the gate itself stays near identity and global style
does not beat ag02, while LPIPS worsens to the fo12-like drift band. Therefore
the current failure is not "no downstream consumer"; it is that the operator
is not style-discriminative enough. A shared output head learns target-specific
repaint rather than a clean style geometry. The next move should keep the
operator binding but make the gate richer and style-local: per-style/per-basis
depthwise gains or an explicit style-discriminative assignment over depthwise
bases.
```

## 2026-05-28 fo18 Style-Local Depthwise Basis Gate

One-line hypothesis:

```text
fo17 proved the downstream path can move, but it moved by repaint. A per-style
depthwise-basis gate should change the operator's style geometry directly while
preserving the frozen endpoint decoder.
```

Configuration:

```text
source: fo11_depthwise_filter_swd80
new parameter: token_depthwise_filter_style_basis_gate_logits[num_styles, basis_count]
frozen: fo11 tokenizer, style identity source, body, skip path, decoder, output_head
train:  token_depthwise_filter_style_basis_gate_logits only
loss:   unchanged main OMF/SWD
```

Decision rule:

```text
Promote only if global style beats ag02, or if Hayao improves without the
fo17 LPIPS drift. If flat, the existing fixed depthwise basis is not the right
style-discriminative alphabet; the next move must change assignment/source, not
increase scalar strength.
```

Result:

```text
fo18_depthwise_style_basis_gate_e2: 0.710152 / 0.408805, Hayao 0.604769
ag02_m02_g56_texture_anchor:       0.710955 / 0.407269, Hayao 0.605668
fo17_depthwise_gate_head_e2:       0.710782 / 0.438083, Hayao 0.614275
```

Learned gate:

```text
shape = [5, 8]
abs_mean = 0.1611
abs_max  = 0.4795
photo row stays zero
Hayao row = [-0.2890, -0.4140, 0.1270, 0.2541, 0.2792, -0.1566, 0.1301, 0.2890]
```

Theory correction:

```text
The style-local gate is connected and does not need the output head to receive
gradient. It learns nontrivial per-style depthwise-basis allocations while
keeping LPIPS near the safe fo11/ag02 band. The negative result is not silence;
it is alphabet insufficiency. The fixed depthwise bases are too weak or not
style-discriminative enough to move global CLIP style. The next valid route is
joint tokenizer grammar/band plus style-basis gate training, or replacing the
fixed depthwise basis with a learned style-discriminative operator alphabet.
Do not repeat scalar-strength sweeps, and do not reopen output_head/body as the
main path.
```

## 2026-05-28 fo19 Superseded And fo20 Operator-Alphabet Probe

fo19 assignment-only was superseded before launch. It still assumed the fixed
Sobel/Laplace basis was sufficient. The stronger inference from fo18 is that
the basis alphabet itself is too weak.

fo20 one-line hypothesis:

```text
fo18 proved style-local gates are alive, but fixed analytic 3x3 bases remain
metric-flat. Learn a style-local zero-mean high-pass operator alphabet while
keeping the endpoint decoder and backbone frozen.
```

Configuration:

```text
source: fo18_depthwise_style_basis_gate_e2 epoch_0002
train:  token_depthwise_filter_style_basis_delta,
        token_depthwise_filter_style_basis_gate_logits,
        style_tokenizer.grammar_vocab,
        style_tokenizer.band_vocab
freeze: style_emb, style_spatial_id_16, body, skip path, decoder, output_head
loss:   unchanged main OMF/SWD
LR:     grammar 4.5e-4, band 1.575e-4, style-basis gate 1.35e-3,
        basis delta 6.75e-4
constraint: each learned kernel delta is projected to zero mean before use,
            so the new alphabet can express high-pass brush geometry but not
            low-frequency repaint.
```

Theory gate:

```text
If fo20 improves global style or Hayao without LPIPS drift, the failure was the
fixed operator alphabet. If it remains flat, the bottleneck is upstream: the
style source/OT assignment is not producing enough discriminative style
coordinates for any small local operator to exploit.
```
