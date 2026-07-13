# Supplementary Material for WEAVE

**Status:** submission-facing supplement map aligned with the current AAAI v4 paper.
**Last updated:** 2026-07-13.
**Formal long source:** `aaai2027_v4/supplement.tex`.
**Scope:** evaluation protocol, metric semantics, radar conventions, artifact audit, mechanism probes, training/inference accounting, and next architecture plan.

This document is the compact reader-facing map. The full reproducibility ledger, commands, and figures are in `supplement.tex`.

---

## S1. Source of Record

Use this priority order when resolving conflicts:

| Topic | Source |
|---|---|
| Main paper claims and raw table values | `aaai2027_v4/paper.tex`, Table 1 |
| Full supplement source | `aaai2027_v4/supplement.tex` |
| Radar generation | `aaai2027_v4/make_radar_metric_blocks.py` |
| DINO sidecar | `aaai2027_v4/fig_data/dino_main.json` |
| HF-route probe metrics | `docs/model_probe/target_hf_delta_eval_summary.json` |
| HF-route diagnosis and handoff | `docs/713/HANDOFF_2026-07-13.md` |
| Method exploration history | `docs/713/METHOD_EXPLORATION_AND_CKPT_2026-07-13.md` |
| Timing | `docs/model_probe/generation_only_timing_summary.json` |
| Archived legacy docs | `docs/archive/713_external_legacy/` |

The project repository is an active research worktree, not a clean release tree. The paper-facing supplement therefore cites committed paper files, frozen run snapshots, and explicit JSON/Markdown sidecars rather than ambient filesystem state.

---

## S2. Evaluation Protocol

The main table reports three boards:

| Short name | Resolution | Role |
|---|---:|---|
| D5-512 | 512px | Primary Distinct5-WikiArt benchmark |
| P2A-256 | 256px | Photo-to-art transfer benchmark |
| R5-WikiArt | 512px | Broader WikiArt style-family benchmark |

All methods are evaluated on the same ordered source-target pair lists where outputs are available. Identity rows are retained.

Metric directions:

| Metric | Direction | Meaning |
|---|---|---|
| DINO-S | higher better | Primary style metric |
| CLIP-S | higher better | Secondary style metric |
| DINO-C | higher better | Content preservation |
| LPIPS | lower better | Content preservation |

The identity row is a no-op style floor for DINO-S/CLIP-S and a content ceiling for DINO-C/LPIPS. A method with high DINO-S but collapsed DINO-C or high LPIPS is not counted as a valid content-preserving style transfer result.

---

## S3. Radar Conventions

Figure 5 is a visual summary of Table 1, not a composite score.

| Region | Axes |
|---|---|
| Left | Train speed, infer speed |
| Upper-left | CLIP-S |
| Right | DINO-S |
| Bottom | DINO-C and `1-LPIPS` |

DINO-S uses a visual broken scale only in the radar: the weakest method on an axis is placed near the inner ring, the second-highest method maps to `0.84`, and the highest method maps to `1.0`. This expands the middle tier while still showing the strongest method. Raw DINO-S values in Table 1 remain the source of truth.

CLIP-S, DINO-C, and `1-LPIPS` use per-axis `v/max`. Speed axes are log-inverted so smaller wall-clock time maps outward. Latent-WCT is shown as an analytic baseline, but its speed segment is omitted to avoid mixing analytic transform cost with learned-model inference accounting.

Line width and opacity encode emphasis tier, not rank:

| Tier | Methods |
|---|---|
| Thickest | Ours |
| Thick | Seedream 4.5, Z-STAR |
| Medium | StyleAligned, SaMam |
| Thin / faint | Identity, Latent-WCT, StyleShot, CUT, SaMST, SD-Turbo |

---

## S4. Main Result Reading

The key D5-512 comparison is:

| Method | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| Identity | 0.419 | 0.693 | 0.000 | 1.000 |
| Latent-WCT | 0.362 | 0.673 | 0.441 | 0.559 |
| SaMam | 0.477 | 0.582 | 0.243 | 0.812 |
| StyleAligned | 0.675 | 0.780 | 0.869 | 0.239 |
| Seedream 4.5 | 0.486 | 0.720 | 0.477 | 0.739 |
| Ours (WEAVE) | 0.4859 | 0.7075 | 0.2583 | 0.8287 |

The paper claim is not "best on every axis." It is "best balanced learned method under the shared automatic protocol": strong first-tier style movement, strong content retention, low trainable parameter count, and low wall-clock cost.

Latent-WCT shows that wavelet statistics alone do not explain the result. StyleAligned shows that raw style similarity can be bought by severe content collapse.

---

## S5. Training Objective and Information Flow

The training target is not simply source reconstruction. It is a structure-aligned target:

```text
LL       = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)
LH/HL/HH = target_style bands
```

So the target direction is correct: the loss asks for style in high-frequency bands while protecting low-frequency content structure. The diagnosed bottleneck is the route by which the model reads image-specific target style.

Baseline route:

```text
style_id -> style_memory -> cross-attention
```

This branch is auxiliary. Probe runs show near-closed gates around `0.056-0.061` and weak gradient mass on the style-memory group.

Useful route:

```text
target image -> DWT HF -> pooled per-subband code -> HF residual velocity -> LH/HL/HH
```

Unsafe route:

```text
target image -> DWT HF spatial maps -> HF residual maps
```

The unsafe route raises style metrics but leaks target geometry and collapses content.

---

## S6. HF-Route Probe Ledger

All listed runs fine-tune from the same `brk_a_ll03_10ep` checkpoint family and evaluate on D5-512 with DINO-S as the primary style metric.

| Run | Route | DINO-S | DINO-C | CLIP-S | LPIPS | Off DINO-S | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `target_hf_delta_ft15_epoch0006` | HF delta, AdaIN 1.0 | 0.482656 | 0.791748 | 0.717485 | 0.295013 | 0.398592 | Path connected, modest. |
| `target_hf_delta_ft15_epoch0015` | HF delta, longer fine-tune | 0.482480 | 0.791313 | 0.718030 | 0.299062 | 0.398681 | Longer training alone does not solve it. |
| `target_hf_delta_strong_ft6` | stronger pooled HF delta | 0.487036 | 0.799077 | 0.717586 | 0.295459 | 0.401948 | First usable architecture improvement. |
| `target_hf_spatial_ft6` | raw spatial HF maps | **0.490074** | 0.404308 | **0.748291** | 0.538240 | n/a | Reject: geometry leak/content collapse. |
| `target_hf_subband_ft6` | per-subband pooled HF residual | **0.488624** | 0.798123 | **0.720880** | 0.296553 | 0.403917 | Primary current architecture probe. |
| `target_hf_subband_texture_ft6` | pooled HF + stationary texture stats | 0.488420 | **0.798815** | 0.719357 | **0.296046** | **0.404302** | Conservative alternate. |
| `target_hf_content_anchor_ft6` | content-energy placement residual | 0.484393 | 0.795462 | 0.717251 | 0.298162 | 0.399538 | Safe but not competitive. |
| `target_hf_subband_basis_ft6` | target-HF selects low-rank content-derived residual basis | 0.482840 | 0.793659 | 0.718310 | 0.297061 | 0.398561 | Reject: safe but underpowered. |

Main lesson: the network can use target-HF information, but only if target spatial coordinates are removed. Any architecture promoted from this probe family should also give HH a supervised output path when HH appears in the target.

---

## S7. Timing and Cost Accounting

Main configuration:

| Item | Value |
|---|---:|
| Training epochs | 10 |
| Hardware | 1x RTX 3060 12GB |
| Train time | 176.9 s = 2.95 min |
| Inference, 750 pairs, generation only | 94.6 s |
| Full metrics eval | about 2 min |
| ODE steps | 8 |
| Trainable parameters | 903K |

Generation-only timing excludes CLIP/LPIPS/DINO metric computation. This is the value used for fair inference-speed comparison in the main table.

Probe timing:

| Run | Wall total | Network generation | VAE decode |
|---|---:|---:|---:|
| `brk_a_ll03_10ep` | 94.63 s | 53.57 s | 39.36 s |
| `target_hf_subband_ft6` | 106.25 s | 65.25 s | 39.34 s |

The architecture probe adds modest network cost but keeps the same fixed VAE decode cost.

---

## S8. Next Architecture Plan

The next useful change is not more gate tuning or raw epoch extension. It is a safer, higher-capacity target-HF route:

```text
target image
  -> DWT HF
  -> compact per-subband style code
  -> orientation-specific HF residual depth
  -> energy-normalized LH/HL/HH velocity
```

Keep:

| Choice | Reason |
|---|---|
| Compact per-subband style codes | Current best route; keeps target coordinates disconnected. |
| Orientation-specific LH/HL/HH residuals | Stroke direction and texture statistics are band-specific. |
| Energy normalization vs current HF heads | Prevents style route from overwhelming content structure. |
| LL protection | Avoids buying DINO-S by spending the content budget. |

Avoid:

| Choice | Reason |
|---|---|
| Raw target HF maps | Already causes content collapse. |
| Global target-token fusion | Over-controls LL. |
| Stationary-stat multi-token widening | Tested after the main probe; worse than subband-only on DINO-S, DINO-C, CLIP-S, LPIPS, and off-DINO-S. |
| Low-rank content-derived basis | Safe, but weaker than subband-only; target-HF coefficient selection underuses image-specific HF style. |
| Treating CFG as proven content fix | Earlier CFG runs were confounded with style delta heads, DWT route, HH head, and larger gates. |

Any promoted architecture must be rerun on D5-512, P2A-256, and R5-WikiArt with DINO-S primary, CLIP-S secondary, and DINO-C/LPIPS used to reject content-collapse wins.

---

## S9. Claims and Non-Claims

Claims:

- WEAVE is the best balanced learned method on the current benchmark set.
- DINO-S is the primary style metric; CLIP-S is secondary.
- Latent-WCT is insufficient on its own.
- StyleAligned is a high-style but content-collapsing point, not a balanced transfer point.
- The 512px setting and target-HF route show better scale potential than the cheaper 256px operating point.

Non-claims:

- WEAVE is not best on every single axis.
- WEAVE does not beat StyleAligned on raw DINO-S.
- The radar plot is not a new metric.
- The current packet does not include a completed human preference study.
- The HF-route probe is not yet a full replacement for the main-table model until it is rerun under the complete protocol.

---

## S10. Open Gaps

| Gap | Status |
|---|---|
| Human preference study | Not completed |
| Exhaustive HPO for heavyweight baselines | Not attempted |
| No-DWT matched 903K control under final recipe | Not yet published |
| Full D5/P2A/R5 rerun for target-HF subband architecture | Next required experiment |
| Public cleanup of scratch build artifacts | Pending after supplement stabilization |

The strongest current story is a balance story, not a winner-everywhere story.
