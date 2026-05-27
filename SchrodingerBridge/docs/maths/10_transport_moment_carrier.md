# Transport Moment Carrier

Date: 2026-05-27

## Diagnosis

The current evidence does not prove that the EMA VAE is unusable. It shows a
more specific failure: the style actuator is not expressing the right semantic
statistics.

Recent probes narrow the search space:

- full-train `style_emb` tuning is negative on both the EMA frontier and the
  current best body-transport checkpoint;
- phase-envelope SWD protects LPIPS but does not raise `clip_style`;
- transport-confidence body paint fixes the learned constant-gate pathology,
  but still caps around `clip_style = 0.714`;
- the implemented low-free transport path did not exactly match the intended
  hypothesis.

The intended low-free hypothesis was:

```text
low_gate    = transport_confidence
detail_gate = transport_confidence * content_support * phase_alignment
```

The code had:

```text
low_gate    = transport_confidence
detail_gate = low_gate * phase_alignment
```

when `style_blender_transport_low_use_support=False`. Therefore the previous
low-free result also released mid/high detail from content-support gating. This
is not the clean test we wanted.

## Corrected Gate

The first fix is to make detail routing independent of the low gate:

```text
low_gate    = transport_confidence
detail_gate = transport_confidence * content_support * phase_alignment
```

This keeps the broad semantic carrier free in flat regions while forcing texton
and edge detail to remain on structural support.

## New Carrier Hypothesis

The remaining bottleneck may be the carrier itself. Existing body-paint modes
mostly learn a residual:

```text
residual = remap(blend(content_body, painted_body)) - content_body
```

This residual is flexible but not explicitly a style-statistic transform. A
style transfer backbone often needs channel moment transport: palette, contrast,
and brush-energy statistics should move inside semantically matched regions,
while content phase remains anchored.

New carrier:

```text
painted_body = SemanticCrossAttention(content_body, style_prior)
mu_c, sigma_c = local_moments(content_body)
mu_s, sigma_s = local_moments(painted_body)
target = (content_body - mu_c) / sigma_c * sigma_s + mu_s
residual = target - content_body
```

Then decompose the residual into low/mid/high bands and use the corrected
transport gates:

```text
low_gate    = transport_confidence
detail_gate = transport_confidence * content_support * phase_alignment
```

This is still unsupervised and uses no Seedream teacher. Seedream only motivates
the diagnostic target: style changes should be region-organized, not global
embedding shifts or unstructured high-frequency pressure.

## Test Matrix

Remote output:

```text
exp/vae_backend/ema_transport_moment
```

Variants:

| variant | purpose |
|---|---|
| `ema_bodytransport_lowfree_fixed_w34_guard` | rerun the intended low-free gate after fixing detail support |
| `ema_transport_adain_w34_guard` | conservative transport-conditioned local moment carrier |
| `ema_transport_adain_w40_style` | style-push moment carrier; accepts LPIPS near 0.50 only if `clip_style > 0.72` |

Smoke on the remote 12G machine:

| variant | status | peak VRAM |
|---|---|---:|
| `ema_bodytransport_lowfree_fixed_w34_guard` | `train_ok` | 10125 MB |
| `ema_transport_adain_w34_guard` | `train_ok` | 10387 MB |
| `ema_transport_adain_w40_style` | `train_ok` | 10413 MB |

Full result:

| variant | epoch | clip_style | content_lpips | EC | note |
|---|---:|---:|---:|---:|---|
| `ema_bodytransport_lowfree_fixed_w34_guard` | 6 | 0.7129 | 0.5044 | 0.3533 | cleaner detail support, but below the unfixed `0.7143 / 0.5003` point |
| `ema_bodytransport_lowfree_fixed_w34_guard` | 7 | 0.7107 | 0.5120 | 0.3468 | worse |
| `ema_bodytransport_lowfree_fixed_w34_guard` | 8 | 0.7116 | 0.5101 | 0.3487 | worse |
| `ema_transport_adain_w34_guard` | 6 | 0.7134 | 0.4986 | 0.3577 | best content-safe moment carrier so far, but not above the old style ceiling |
| `ema_transport_adain_w34_guard` | 7 | 0.7108 | 0.5075 | 0.3501 | worse |
| `ema_transport_adain_w34_guard` | 8 | 0.7122 | 0.5053 | 0.3524 | worse |
| `ema_transport_adain_w40_style` | 6 | 0.7143 | 0.5295 | 0.3361 | tiny style gain, unacceptable content loss |
| `ema_transport_adain_w40_style` | 7 | 0.7108 | 0.5377 | 0.3286 | worse |
| `ema_transport_adain_w40_style` | 8 | 0.7120 | 0.5363 | 0.3302 | worse |

Interpretation: correcting the detail gate does not recover the missing
style. The earlier unfixed low-free path probably gained a little style by
letting mid/high detail leak beyond content support. That leak is not enough to
cross 0.72, but it explains why the corrected guard is safer-looking on paper
and lower-style in metrics. The conservative AdaIN carrier is content-safe and
slightly improves EC, but still does not break the style ceiling. The `w40`
AdaIN style-push branch confirms the ceiling: it only reaches `0.7143` while
LPIPS degrades to `0.5295`.

Success/falsification:

- if fixed low-free improves over `0.7143 / 0.5003`, the previous result was
  contaminated by the detail-gate bug;
- if AdaIN carrier improves style without LPIPS collapse, the missing lever is
  semantic channel moment transport;
- if both remain capped, the next hypothesis should move from body carrier
  design to the OT target itself, especially object-level macro matching.

The third case happened. The next probe should change the terminal objective:
match region-level palette/contrast/envelope statistics while refusing to match
the unpaired style image's literal spatial phase. This is a targeted response to
the Seedream-gap diagnostic: our high-pass energy is already high, but its phase
is less organized than Seedream's region-coherent repainting.
