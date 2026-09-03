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
