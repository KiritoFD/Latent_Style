# Supplementary Material for WEAVE

**Status:** submission-facing supplement aligned with the current AAAI v4 paper.  
**Last updated:** 2026-07-13.  
**Scope:** protocol, metric semantics, radar visualization, mechanism probes, training/inference accounting, and known gaps.

This supplement is written to match the paper narrative: WEAVE is a compact style-transfer model whose main strength is the balance between style movement and content preservation. DINO-S is the primary style metric; CLIP-S is secondary.

---

## S1. Evaluation Protocol

### S1.1 Benchmarks

The main table reports three settings:

| Short name | Resolution | Role |
|---|---:|---|
| D5-512 | 512px | Primary Distinct5-WikiArt benchmark |
| P2A-256 | 256px | Photo-to-art transfer benchmark |
| R5-WikiArt | 512px | Broader WikiArt style-family benchmark |

All methods are evaluated on the same ordered source-target pair lists for each benchmark.

### S1.2 Metric directions

| Metric | Direction | Meaning |
|---|---|---|
| DINO-S | higher better | Primary style metric |
| CLIP-S | higher better | Secondary style metric |
| DINO-C | higher better | Content preservation |
| LPIPS | lower better | Content preservation |

Radar plots show `1-LPIPS` so that every radial axis is "higher is better".

### S1.3 Identity calibration

The identity row is not a decorative baseline. It is the no-op floor for style metrics and the content ceiling for content metrics.

For a target style `s`, the identity floor is the score of the unchanged source image when it is evaluated against the requested target style. A pooled table statement such as "clears the identity floor" means the pooled style score exceeds the Identity row on the same benchmark. It does **not** mean every single pair clears every local floor.

For content metrics, the identity row is a ceiling: LPIPS is 0 and DINO-C is 1 by construction.

---

## S2. Radar Figure Conventions

Figure 5 is a visual summary of Table 1, not a new metric.

| Region | Axes |
|---|---|
| Left | Train speed, infer speed |
| Upper-left | CLIP-S |
| Right | DINO-S |
| Bottom | DINO-C and `1-LPIPS` |

The P2A-256 style axes are placed at the edge of the style blocks, not in the center. The same metric is always shown in the same broad direction across datasets.

### S2.1 DINO-S scale

DINO-S is drawn with a **visual** broken scale only for Figure 5.

| Raw DINO-S position | Radar radius |
|---|---:|
| Weakest method on that axis | inner ring |
| Second-highest method | 0.84 |
| Highest method | 1.0 |

This choice expands the middle tier while still keeping the strongest method visible. The raw DINO-S values in Table 1 remain the numeric source of truth.

### S2.2 Other axes

CLIP-S, DINO-C, and `1-LPIPS` use per-axis `v/max`. Speed axes are log-inverted so smaller wall-clock time maps outward. Missing or non-comparable speed entries are left as gaps.

Latent-WCT is shown as a training-free analytic baseline, but its speed segment is omitted from the radar so the figure does not mix learned-model speed with an analytic transform.

### S2.3 Line hierarchy

Line width and opacity encode interpretive priority, not rank.

| Tier | Methods |
|---|---|
| Thickest | Ours |
| Thick | Seedream 4.5, Z-STAR |
| Medium | StyleAligned, SaMam |
| Thin / faint | Identity, Latent-WCT, StyleShot, CUT, SaMST, SD-Turbo |

---

## S3. Main Baselines

The key comparison on D5-512 is:

| Method | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Identity | 0.419 | 0.693 | 0.000 | 1.000 |
| Latent-WCT | 0.362 | 0.673 | 0.441 | 0.559 |
| SaMam | 0.477 | 0.582 | 0.243 | 0.812 |
| StyleAligned | 0.675 | 0.780 | 0.869 | 0.239 |
| Seedream 4.5 | 0.486 | 0.720 | 0.477 | 0.739 |
| Ours (WEAVE) | 0.4859 | 0.7075 | 0.2583 | 0.8287 |

Two points matter for the paper:

1. Latent-WCT shows that wavelet statistics alone do not explain the result.
2. StyleAligned reaches very strong raw DINO-S, but the content cost is severe.

The paper's claim is therefore not "best on every axis." It is "best balanced learned method, with strong style movement, strong content retention, and low cost."

---

## S4. Information Flow Diagnosis

### S4.1 Current diagnosis

The main bottleneck is not the loss direction. The target is already style-heavy in the high-frequency bands. The bottleneck is the **routing path** that lets the model actually read the right style signal.

Current picture:

`target style -> DWT HF bands -> pooled HF code -> HF velocity heads / residual delta -> LH/HL/HH`

The model also has a style-memory / cross-attention route:

`style_id -> style_memory -> cross-attn`

That branch is auxiliary. Probes show it is not the main source of style transfer.

### S4.2 What the probes showed

The useful fix was to connect the target HF latent to HF velocity prediction while keeping LL content-preserving. Direct spatial target-HF injection was too strong and leaked geometry.

| Run | DINO-S | DINO-C | LPIPS | Interpretation |
|---|---:|---:|---:|---|
| `target_hf_spatial_ft6` | 0.490074 | 0.404308 | 0.538240 | High style, but content collapse from spatial leak |
| `target_hf_delta_strong_ft6` | 0.487036 | 0.799077 | 0.295459 | First usable HF route |
| `target_hf_subband_ft6` | 0.488624 | 0.798123 | 0.296553 | Best current primary architecture |
| `target_hf_subband_texture_ft6` | 0.488420 | 0.798815 | 0.296046 | Safer alternate, slightly better balance |
| `target_hf_content_anchor_ft6` | 0.484393 | 0.795462 | 0.298162 | Safe, but not better than subband-only |

Interpretation:

- Raw spatial HF maps expose target coordinates and damage content.
- Pooled subband HF codes give the model the right style signal without leaking layout.
- Content-anchor placement is safe, but it does not beat the simpler subband route.

### S4.3 Cross-attention probe

Cross-attention gates are near-closed in the baseline probe runs, typically around `0.056-0.061`. Gradient mass on `style_memory` is among the weakest groups. This supports the claim that the branch is auxiliary rather than the primary style path.

### S4.4 Final architectural reading

The current structure is best read as:

1. **Endpoint AdaIN / HF statistics** inject style.
2. **Rectified-flow transport** moves content toward the style target.
3. **Cross-attention style memory** provides a lightweight domain handle.

The probes do not support the reverse story.

---

## S5. Training Objective and What It Means

The training target is not "original LL plus target style H" in a raw replacement sense. It is a structure-aligned target:

- `LL = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)`
- `LH/HL/HH = target_style` bands

So the target direction is correct: the loss asks for style in HF while preserving content structure in LL.

The problem was that the model did not always have a clean path to read the relevant target HF signal. In other words, the target was right, but the routing was incomplete.

One implementation detail matters: the `HH` target must also have a supervised output head. If the target band exists in the supervision but the model has no output path for it, the supervision is partially disconnected.

---

## S6. Training and Inference Accounting

Main configuration:

| Item | Value |
|---|---:|
| Training epochs | 10 |
| Hardware | 1x RTX 3060 12GB |
| Train time | 176.9 s = 2.95 min |
| Inference (750 pairs, generation only) | 94.6 s |
| Full metrics eval | about 2 min |
| ODE steps | 8 |
| Trainable parameters | 903K |

These numbers are important because they bound the paper's scaling argument. WEAVE is not only a quality result; it is a low-cost operating point.

Short probe runs are used to diagnose architecture. They are not the final claim of the paper. The final paper result uses the stable 10-epoch configuration.

---

## S7. Claims and Non-Claims

### What the paper claims

- WEAVE is the best balanced learned method on the current benchmark set.
- DINO-S is the primary style metric.
- Latent-WCT is insufficient on its own.
- StyleAligned shows that raw style score alone is not enough when content collapses.
- The 512px setting has better scaling headroom than the shorter / cheaper settings.

### What the paper does not claim

- WEAVE is not best on every single axis.
- WEAVE does not beat StyleAligned on raw DINO-S.
- The radar plot is not a new metric; it is a visual summary.
- The current packet does not include a completed human study.

---

## S8. Open Gaps

| Gap | Status |
|---|---|
| Human preference study | Not yet completed |
| Exhaustive HPO for heavy baselines | Not attempted |
| No-DWT matched 903K control under the final recipe | Not yet published |
| More aggressive single-axis style gain | Still content-limited |

These are real gaps, not hidden ones. The current paper is strongest when it is read as a balance paper, not as a winner-everywhere paper.

---

## S9. Cross-Reference Map

| Topic | Source |
|---|---|
| Main metrics | `aaai2027_v4/paper.tex`, Table 1 |
| Radar generation | `aaai2027_v4/make_radar_metric_blocks.py` |
| DINO values | `aaai2027_v4/fig_data/dino_main.json` |
| Mechanism probes | `docs/713/HF_ARCHITECTURE_PROBE_2026-07-13.md` |
| Diagnosis log | `docs/model_probe/HF_DELTA_DIAGNOSIS_2026-07-13.md` |
| Delivery summary | `docs/delivery/DELIVERY_SUMMARY.md` |

*End of supplement.*
