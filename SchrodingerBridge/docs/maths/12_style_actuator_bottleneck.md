# Style Actuator Bottleneck

Date: 2026-05-27

## Claim

The current `clip_style` ceiling near `0.714` is not primarily caused by:

- underfit `style_emb`;
- insufficient full-training-set exposure;
- terminal SWD matching the wrong unpaired phase;
- a weak semantic routing matrix alone.

The bottleneck is the **style actuator**: the mechanism that converts a style
condition into localized visible repainting before decoder/skip fusion.

## Evidence

| probe | best clip_style | content_lpips | result |
|---|---:|---:|---|
| full-train `style_emb` on `ema_transport_adain_w34_guard e6` | 0.71118 | 0.43530 | safer content, lower style |
| full-train `style_emb/spatial` high-pass fit | 0.71073 | 0.40735 | safer content, lower style |
| body-transport full-train adapter | 0.68625 | 0.62951 | destroys the frontier |
| `transport_adain_w34_guard` | 0.71343 | 0.49859 | clean carrier, still capped |
| `transport_adain_w40_style` | 0.71429 | 0.52948 | tiny style gain, content failure |
| `semantic_moment_w30_guard` | 0.71325 | 0.49158 | content-safe terminal OT, no style lift |
| `semantic_moment_w38_style` | 0.71441 | 0.53082 | same tradeoff |
| `sinkhorn_body_w28_guard` | 0.70543 | 0.44364 | routing becomes too conservative |
| `transport_amp_w34_guard` | 0.71371 | 0.50176 | scalar amplitude gate cannot break ceiling |
| `sinkhorn_amp_w36_style` | 0.70698 | 0.51692 | conservative routing plus amplitude still suppresses style |
| `transport_texton_w34_guard` | 0.71451 | 0.48261 | best balanced actuator so far, but still below target |
| `transport_texton_w40_style` | 0.71112 | 0.52287 | stronger texton pressure is negative |

## Interpretation

The experiments form a useful bracket:

- If style pressure is increased through terminal losses or moment matching, the
  model reaches only about `0.714` and pays with LPIPS.
- If semantic routing is made more topology-preserving, LPIPS and EC improve but
  style falls to about `0.705`.
- If only `style_emb` is moved after the backbone is fixed, the easiest optimum
  is content preservation or transport-geometry damage, not stronger style.

So the missing variable is not a scalar. It is a factorization error. The model
needs two independent fields:

```text
where_field  = semantic/object router
what_field   = local style amplitude and texton statistics
endpoint     = content + where_field * what_field
```

Current variants partially conflate these:

- transport confidence is used both as permission and amplitude;
- full-train style embeddings change a global condition but do not create a new
  local actuator;
- terminal OT losses ask for more style but cannot specify a physically valid
  insertion path.
- `transport_amp` separates permission and scalar amplitude, but the "what"
  field is still a local-moment envelope, not an expressive texton carrier.

## Next Architecture Hypothesis

The next mainline should expose a localized style-amplitude field that is
conditioned by `style_emb` but not numerically equal to transport confidence.

Minimum viable design:

1. Compute a conservative semantic router from transport confidence or Sinkhorn
   attention.
2. Predict a separate bounded amplitude field from style embedding plus local
   content/body features.
3. Apply amplitude only inside router support, but regularize its entropy and
   region contrast so it cannot collapse to a constant gate.
4. Keep high-frequency injection phase-locked to content edges, while allowing
   low/mid style statistics in smooth semantic regions.

The target is not "more style loss"; it is a better actuator basis. A successful
run should lift `clip_style` beyond `0.72` before LPIPS exceeds `0.50`.

## Current Mainline: Transport-Texton

After `transport_amp`, the remaining hypothesis is stricter: the model needs a
localized **texton residual generator**, not another scalar gate. The new
`style_blender_mode="transport_texton"` keeps the semantic router as the
`where_field`, but uses `style_emb` to condition a small convolutional carrier
over content band-pass features and local moment residuals. The carrier is then
decomposed into low/mid/high bands and applied only through transport,
content-support, and phase gates.

This is still unsupervised and does not use Seedream or any external teacher in
training. It tests whether the missing variable is the dimension of the visible
style actuator itself.

Result so far: the guarded texton carrier is a genuine improvement in the
balance region (`clip_style=0.71451`, `content_lpips=0.48261`), but it still
does not cross `0.72`. The style-push branch is negative, which refines the
theory: the issue is not only actuator dimension, but style specificity and
where-to-what decoupling. More pressure on the same carrier collapses toward
content damage instead of class-discriminative style.

Active diagnostic: calibrate `style_emb` on the full training set using the
best texton checkpoint. If this fails, then the remaining missing component is
not the style vector but the backbone's style-specific carrier basis.

## Hayao As A Style-Specific Failure Case

New rule: do not judge a run from `all_pairs_overview` alone. Every result must
be inspected by target style, especially with identity pairs removed. A model can
look close to the global style target while one style is functionally failing.

The strongest content-safe adapter point so far,
`m02_embspatial_highpass_style`, is the clearest example:

| target style | slice | clip_style | content_lpips | weak rows (`clip_style < 0.70`) |
|---|---|---:|---:|---:|
| Hayao | all | 0.64864 | 0.41960 | 80.7% |
| Hayao | cross only | 0.60516 | 0.44920 | 99.2% |
| cezanne | cross only | 0.70535 | 0.37673 | 45.0% |
| monet | cross only | 0.69586 | 0.37524 | 48.3% |
| vangogh | cross only | 0.72302 | 0.41200 | 39.2% |

This is not evidence that the VAE is unusable. It is evidence that the current
style actuator is asymmetric across style families. `vangogh` already crosses
the target style band with good LPIPS, while `Hayao` fails almost every
cross-style row. The failure is style-specific, not globally latent-specific.

Working hypothesis:

1. `vangogh`/`monet`/`cezanne` are mostly captured by local texture, color
   statistics, and medium-frequency textons. The current SWD/texton/AdaIN
   actuators can express those reasonably well.
2. `Hayao` is carried by a different visual grammar: clean contour lines, flat
   color regions, simplified shading, and large low-frequency color planes.
   Raising high-pass SWD or generic texture pressure does not create this
   grammar; it can even create broken texture fragments that score poorly and
   look visibly wrong.
3. Post-hoc `style_emb` tuning from the good m02 adapter did not lift Hayao:
   `m03`, `m04`, and `m05` all stayed near global `clip_style=0.710` and did
   not repair the Hayao cross-style weakness. This suggests the missing variable
   is not merely the style vector after the backbone is frozen.

Therefore the next Hayao mainline should not be "more of the same style loss."
It should test style-family-aware actuators:

- target-style sampling or style-loss weighting for Hayao, but applied mainly
  to style/terminal terms, not to content/kinetic anchors;
- a flat-region color-plane branch for low-frequency Hayao repainting;
- an edge/contour branch that aligns stylized boundaries with content edges
  instead of injecting generic high-frequency texture;
- target-style diagnostic columns in every eval summary:
  `by_target_style`, `cross_by_target_style`, `by_source_style`, and
  `cross_by_source_style`.

Success criterion for the Hayao branch: raise Hayao cross-style `clip_style`
first, then check that global `clip_style > 0.72` and LPIPS remains below about
`0.50`. A global average improvement that leaves Hayao cross-style near `0.60`
is a false positive.

Implemented diagnostic levers:

- target-style stratified eval summaries are now first-class outputs;
- style-embedding calibration summaries now surface Hayao cross performance;
- `target_style_loss_weights` can bias style/terminal OMF terms for Hayao
  without also scaling content/kinetic losses;
- three probes separate the hypotheses:
  exposure-only, Hayao style-loss weighting, and Hayao flat-contour grammar.

## 2026-05-27 Hayao Probe Result

The Hayao-targeted probes are now complete:

| probe | best global clip_style | best global LPIPS | Hayao cross clip_style | Hayao cross LPIPS | readout |
|---|---:|---:|---:|---:|---|
| exposure-only | 0.71161 | 0.48667 | about 0.638 | about 0.540 | more Hayao samples alone do not repair the style |
| style-weighted | 0.71223 | 0.52518 | about 0.645 | about 0.573 | stronger Hayao style pressure mostly damages content |
| flat-contour hand recipe | 0.71289 | 0.54001 | 0.64882 at e8 | 0.58292 at e8 | hand-coded low/high redistribution is still the wrong actuator |

This rules out the simplest explanations:

- Hayao is not merely under-sampled.
- Hayao is not fixed by larger scalar style loss.
- A manually chosen "more low, less high" recipe is too crude; it raises the
  visible edit by breaking the image instead of learning Hayao's actual grammar.

The current structural diagnosis is sharper: `transport_texton` has a
style-conditioned carrier content, but its low/mid/high carrier allocation is a
global recipe. This lets Van Gogh use the shared mid/high texture basis well,
while Hayao is forced to simulate flat color planes and clean contours with a
texture-like residual basis.

### Next Structural Hypotheses

1. **Global band recipe bottleneck.** The model must let `style_emb` choose the
   low/mid/high carrier energy. Hayao should learn higher low-frequency repaint
   allocation and lower high-frequency texture allocation; Van Gogh should be
   allowed to keep a stronger mid/high allocation.
2. **Style-code bandwidth bottleneck.** If style-conditioned allocation helps but
   remains weak, enlarge the style path (`style_dim`, style attention tokens, and
   texton carrier hidden width) before increasing any scalar loss.
3. **Carrier diagnostic first.** Every run using the new allocator must inspect
   `numeric_debug.jsonl` by target style. A successful architecture should show a
   visibly different Hayao band profile before we trust global CLIP gains.

Implemented next probes:

- `ema_transport_texton_alloc_w34`: isolated style-conditioned band allocation.
- `ema_transport_texton_alloc_hayao_w36`: allocation plus mild Hayao style
  pressure, deliberately weaker than the destructive weighted probe.
- `ema_transport_texton_alloc_cap_w34`: allocation plus larger style/code/carrier
  capacity to test whether the current model is too narrow.

## Style Embedding As A Separate Module

The style embedding should now be treated as a standalone actuator module, not
as an incidental checkpoint tensor. It has three responsibilities:

1. encode target-style identity in a separable latent control space;
2. drive the frozen backbone into a visible, style-specific response;
3. remain reusable as an external adapter without mutating the base checkpoint.

Therefore an adapter is acceptable only if it passes three gates.

### Gate A: Embedding Geometry

Measure `style_emb.weight` and `style_spatial_id_16` before any image eval:

- shape and nominal dimension (`style_emb_dim`);
- centered rank and entropy effective rank;
- off-diagonal cosine maximum/mean between style rows;
- norm balance by style;
- delta from the base checkpoint embedding.

For only five style domains, the centered rank cannot exceed four. A larger
nominal dimension such as 160 is useful only if it creates a larger margin in
the four available style directions. If `max_offdiag_cos` is close to one, the
style vector table is geometrically crowded even when the raw dimension is
large.

Current m02 evidence:

| adapter | style_emb shape | centered rank | effective rank | max offdiag cos | spatial effective rank | readout |
|---|---:|---:|---:|---:|---:|---|
| `m02_embspatial_highpass_style` | `5 x 160` | 4 | 2.318 | 0.9298 | 3.416 | nominally wide, but actual vector code is compressed |
| `m04_m02_styleboost_loose` | `5 x 160` | 4 | 2.653 | 0.9298 | 3.431 | style boost mostly changes Hayao norm, not pairwise margin |

This does **not** prove 160 dimensions are useless. It proves the current
adapter training objective is not using the 160-dimensional vector space
efficiently. The capacity question should be tested by effective rank and
response separation, not by raw dimension.

### Gate B: Frozen-Backbone Response

With the backbone frozen, feed the same content batch through all target style
ids and measure:

- pairwise output L2/cosine separation between target styles;
- delta low/mid/high energy ratios;
- SWD gain before vs. after transfer, including high-pass SWD gain;
- content damage proxies: latent MSE, delta TV, gradient cosine.

This gate answers whether a style embedding is merely separable as a tensor or
actually consumed by the current carrier. A good Hayao adapter should show a
different band profile from Van Gogh/Monet: stronger clean low-frequency color
plane response, controlled mid-band contour response, and weak noisy high-band
texture.

### Gate C: Standard Eval兑现

Only after Gate A/B should we trust full image metrics:

- global `clip_style > 0.72`;
- global `content_lpips < 0.50` if possible;
- Hayao cross-style `clip_style` must improve, otherwise global average is a
  false positive;
- per-target rows must be reported in every table.

Seedream remains diagnostic only: its outputs are useful for visual gap
analysis, but they should not define the main training target unless the run is
explicitly labeled as an external-teacher side experiment.

Implementation:

```text
tools/experiments/eval_style_adapter_quality.py
```

The script evaluates existing `style_adapter.pt` files as modules. It can reuse
known full-eval summaries, or generate a fresh full eval with `--run-full-eval`.

### Training And Reuse Policy

- Direct reuse is supported by `utils.run_evaluation --style_adapter` and the
  inference `style_adapter_path` path. This keeps the base checkpoint immutable.
- Post-backbone training is supported by
  `tools/experiments/run_style_embedding_mainline_calibration.py
  --init-style-adapter`.
- If Gate A shows low effective rank or high off-diagonal cosine, increasing
  `style_dim` is justified, but only together with an objective that explicitly
  improves style-code margins or response separation. Purely enlarging the
  table is not enough.
- If Gate B shows weak response separation despite good embedding geometry, the
  bottleneck is the carrier/backbone, not the adapter.

## Deterministic Code Or Style Distribution

The current adapter is deterministic:

```text
target_style_id -> style_emb[target_style_id], style_spatial_id_16[target_style_id]
```

This is the right **deployment and paper-eval interface** because it is
reproducible and makes one checkpoint/adaptor comparable to baselines. However,
it is not the right mathematical model of style. A style class is a measure, not
a point. If the adapter collapses that measure to a single centroid, it can
repeat the same failure mode as MSE: the class-average code preserves content
but loses style modes.

This is especially suspicious for Hayao. The style is not just "more texture";
it mixes several discrete grammar variables: flat color-plane palette, contour
ink strength, simplified shadow shape, and suppressed texture noise. One
deterministic vector can become a compromise that expresses none of these
strongly enough.

### Proposed Adapter Factorization

Use a distributional adapter during training, but keep the deterministic mean as
the default evaluation path:

```text
e_y = mu_y                         # deterministic eval
e_y = mu_y + U_y alpha + sigma_y eps   # distributional training / diversity
```

Where:

- `mu_y` is the stable style identity code;
- `U_y alpha` is a low-rank, style-specific mode code, preferably inferred from
  a sampled target-style reference latent rather than free noise;
- `sigma_y eps` is a small stochastic residual used only after proving the
  backbone response is stable under perturbations.

The stochastic part must be structured. Free Gaussian noise in the full
160-dimensional code is not acceptable: it will mostly test backbone chaos, not
style diversity. The random degrees of freedom should be low-rank and
band-aware. For Hayao, variance should primarily affect palette/contour modes,
while high-frequency random texture variance should be suppressed.

### Training Objective

The adapter should be trained as a conditional style measure without external
teacher contamination:

- deterministic mean path: keep `mu_y` strong enough to pass standard eval;
- sampled path: match generated endpoint distribution to target-style latent
  distribution using SWD/MMD/energy distance over style bands;
- content stability: penalize content/gradient variance across samples from the
  same content image;
- style diversity floor: require nonzero response variance only in allowed
  bands, not in structural edges;
- style-code margin: explicitly increase pairwise style margin or effective
  rank if the adapter remains compressed.

### Evaluation Requirements For A Distributional Adapter

Report two score lines:

1. **Mean adapter score**: deterministic `mu_y`, comparable to all baselines.
2. **Sampled adapter score**: mean and std over several seeds, used only as a
   diversity/robustness diagnostic.

The sampled path is successful only if:

- mean `clip_style` improves or stays comparable;
- LPIPS mean remains within budget and std is small;
- Hayao cross-style score improves;
- band energy variance is style-appropriate;
- sampled outputs do not create high-frequency fragmentation.

This makes the distribution useful as a training regularizer and diagnostic
without letting randomness become a metric loophole.
