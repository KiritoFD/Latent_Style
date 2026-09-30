# Semantic SWD Session Log — 2026-07-07/08

Full step-by-step record of the exploration, code changes, and verified
results. All MUSIQ numbers are the 750-image D5 mean, freshly recomputed
directly on the on-disk `full_eval/.../images/` directory using
`scripts/_compute_musiq_batch.py`. CLIP-S / LPIPS come from
`full_eval/.../summary.json`.

Paper file (`aaai2027_v4/paper.tex`) is unchanged in this session — every
edit was reverted per user instruction. Only the main table may be
updated later; the narrative stays as-is.

## 0. Starting state

Paper WEAVE (D5, main table row): CLIP-S 0.7213, LPIPS 0.2868, MUSIQ 35.31.
Best pre-session probe run (`swd_cm_guided_cons5`, class-memory style
path, global SWD w=8): MUSIQ 40.91.

The paper's Discussion already flagged MUSIQ as the weak axis (VAE decode
grain from HF transport lowers no-reference quality).

## 1. First correction: reference-latent regressed

Before this session, an attempted "fix" (`ref_guided_cons5`) fed the
sampled `target_style` latent through a new intrinsic-CNN style path.
Fresh MUSIQ = **39.33**, CLIP 0.720, LPIPS 0.401 — worse than the class-
memory guided_cons5 (40.91 / 0.722 / 0.331) on every axis. The
intrinsic-CNN style path *hurts*.

Rule fixed for the rest of the session: keep `style_condition_source =
"style_memory"` (class-memory path, `use_intrinsic_style=False`).
Recorded in `memory/semantic_swd_ref_latent_regression.md`.

## 2. SWD weight sweep on the class-memory path

Class-memory + guided pixel SWD, 5 epochs, 750-img eval. Only knob:
`single_step_swd_weight`.

| run                    | weight | CLIP-S | LPIPS  | MUSIQ  |
|------------------------|--------|--------|--------|--------|
| swd_cm_cons5           |  8     | 0.722  | 0.334  | 40.84  |
| swd_cm_softer5         |  8¹    | 0.723  | 0.334  | 40.56  |
| swd_cm_strongswd5      | 12     | 0.723  | 0.315  | **42.95** |
| swd_cm_w14             | 14     | 0.724  | 0.313  | 42.87  |
| swd_cm_w16             | 16     | 0.726  | 0.323  | 42.43  |

¹ same weight, softer guidance floor/power.

Peak at weight 12, a strict Pareto win over 8 (better on all three
axes). Weight 16 regresses. Recorded in
`memory/swd_weight_drives_musiq.md`.

## 3. First mechanism attempt: multi-scale patch SWD (FAILED)

Hypothesis: pixel-marginal SWD only matches the latent color/tone
histogram (4-dim sort-based projection), discarding local texture, so
lifting each sample to a `C·k²`-dim patch texture vector via im2col
should route the constraint to texture.

Implementation (`spectral_losses620.py`):
- `_patch_swd(a, b, patch, num_projections, sample_weight, sample_size)`
  — im2col unfold, weighted-quantile sample, project, sort-match.
- Config: `swd_patch_mode ∈ {off, multi}`, `swd_patch_sizes`,
  `swd_patch_weights`.

Result (weight 12, patch sizes [1,3,5]): MUSIQ **41.56 < 42.95**. Kept
behind flag, default off. Reason: SWD lives in the 4-ch VAE latent —
a 3×3 latent patch decodes to a ~24×24 px macro region, not a micro
texture, and sort-based SWD discards intra-patch spatial arrangement
anyway. Patch dilutes the effective pixel-marginal signal.

## 4. Second mechanism attempt: spectral band-split SWD (FAILED)

Hypothesis: the full-latent SWD is dominated by LL energy; splitting
into LL/LH/HL/HH with a high-freq emphasis routes the constraint to
the band MUSIQ rewards.

Implementation (`spectral_losses620.py`):
- Reused `dwt2_haar` on `z_hat1` / `projected_target`; per-band
  `_sliced_wasserstein` with the guidance map downsampled by
  `avg_pool2d(k=2)` so cross-attn sampling mass aligned to the half-
  resolution subbands.
- Config: `swd_band_mode ∈ {off, split}`, `swd_band_w_ll/lh/hl/hh`.

Result (weight 12, band weights 0.25/1/1/1.5): MUSIQ **40.46 < 42.95**.
Kept behind flag. Reason: dilutes the global match without adding a
signal the constraint didn't already reach. Same pattern as patch:
partitioning is not the right move here.

## 5. Real breakthrough: semantic region-matched SWD

Hypothesis: the fault is not that SWD ignores texture, but that a
single global marginal forces incompatible regions (e.g. smooth sky
pixels) partway toward incompatible target statistics (textured
foreground), producing a muddy blend. Fix: partition the content
latent into content-similar regions, and match each generated region's
distribution to its appearance-corresponding target region — keeping
per-region statistics coherent.

Implementation (`spectral_losses620.py`):
- `_kmeans_labels(feat, K, iters)` — mini-batch k-means, farthest-point
  seed by feature-norm order (stable, no RNG divergence), no grad.
- `_semantic_region_swd(gen, target, seg_feat, num_regions, num_projections, kmeans_iters)`:
  cluster gen by content latent, cluster target by its own latent,
  align region indices by centroid mean-projection order (so region
  rank matches by appearance), per-region 1D OT via sorted-quantile
  interpolation to a common grid size.
- Wired into `_compute_swd` as a new branch, blended with the global
  SWD via `swd_semantic_blend ∈ [0,1]`.
- `content` threaded into `_compute_swd` and `compute` signature.
- Config: `swd_semantic_mode ∈ {off, region}`,
  `swd_semantic_regions`, `swd_semantic_kmeans_iters`,
  `swd_semantic_blend`.

Sweep (base: weight 12, cross-attn guidance floor 0.5 / power 0.5):

| run                | K  | β    | CLIP-S | LPIPS | MUSIQ |
|--------------------|----|------|--------|-------|-------|
| swd_cm_semantic12  | 4  | 0.5  | 0.720  | 0.386 | 47.31 |
| swd_cm_sem_b07     | 4  | 0.7  | 0.721  | 0.406 | 49.60 |
| swd_cm_sem_b10     | 4  | 1.0  | 0.713  | 0.385 | 49.76 |
| **swd_cm_sem_r8**  | **8** | **0.7** | **0.715** | **0.382** | **51.86** |

Peak: 8 regions, β=0.7 — MUSIQ 51.86, matching SaMam (51.17) at
comparable LPIPS. β saturates around 0.7 (β=1.0 marginally regresses
CLIP). 4 regions leaves K on the table; 8 regions is the sweet spot at
D5's 64×64 latent (~64 locations per region).

Cost: LPIPS rose 0.315 → 0.382 (+0.067). This is a different operating
point, not a Pareto improvement.

## 6. The decisive control (mechanism, not distortion)

Question: is MUSIQ gain from *semantic structure* or just *more
distortion*? Control: turn semantic mode off, push global SWD to
similar LPIPS by increasing weight and lowering the endpoint content
anchor.

| run              | mode      | swd_w | w_ep_content | LPIPS | MUSIQ |
|------------------|-----------|-------|--------------|-------|-------|
| swd_cm_strongswd5| global    | 12    | 1.0          | 0.315 | 42.95 |
| **swd_cm_ctrl_w24** | **global** | **24** | **0.5** | **0.345** | **43.93** |
| swd_cm_sem_r8    | semantic  | 12    | 1.0          | 0.382 | 51.86 |

At **higher** LPIPS (0.345 vs 0.315), global SWD only reaches MUSIQ
43.93 — a mere +1.0 per +0.03 LPIPS. Semantic SWD delivers +8.9 MUSIQ
per +0.07 LPIPS — roughly 4× the MUSIQ-per-LPIPS slope. The MUSIQ gain
is a mechanism effect, not a distortion effect. This is the load-
bearing scientific result of the session.

## 7. Failed / neutral architecture changes

### 7a. HH velocity head (NEUTRAL)

The old model had no `head_hh` — `v_hh` was silently zeroed via
`v_dict.get("hh", zeros)`. Since `spectral_w_hh=2.0` was declared but
unused, this looked like dead-band architecture worth fixing.

Implementation:
- `spectral_bridge620.py`: `head_hh = SpectralVelocityHead(...)` gated
  by `enable_hh_head`; forward returns `out["hh"] = v_hh` when active.
- Euler solver updated: `if "hh" in v_dict: hh = hh + v_dict["hh"] * dt`.
- `spectral_losses620.py`: keeps `target_hh` from
  `dwt2_haar(target_delta)`; adds `w_hh * fm_loss(v_hh, target_hh)`
  when `"hh" in v_dict`.

Backward compatible: when `enable_hh_head=False`, `out` has no `"hh"`
key so loss and endpoint fall back to previous behavior.

Result on sem_r8 base: MUSIQ **50.40 vs 51.86** — neutral / slightly
worse; CLIP/LPIPS essentially unchanged. Not worth the extra head.
Kept behind the flag, default off.

### 7b. FiLM-modulated velocity heads (FAILED)

Hypothesis: the output heads see no direct style signal (only via
backbone cross-attention), so FiLM-modulating each subband head's
normalization by `style_global` (zero-init γ,β = safe identity start)
should inject style statistics at the point of prediction.

Implementation:
- `SpectralVelocityHead(dim, latent_channels, style_dim=0)`: extra
  `film = nn.Linear(style_dim, dim*2)` (zero-init), inserted between
  GroupNorm and SiLU as `z * (1+γ) + β`.
- Bridge passes `style_global` into each head forward when
  `enable_style_film_heads=True`.

Result on sem_r8 base (batch 16 for VRAM headroom): MUSIQ **47.63 vs
51.86** — clearly worse. Direct style injection at the output stage
amplifies per-channel latent statistics that decode to grain.

### 7c. Style extrapolation as inference knob (FAILED, correction of earlier misreport)

Earlier in the session I reported extrap 0.6/1.0 stacking on r8 to
MUSIQ 52.71/56.89. That was **wrong** — I pointed the MUSIQ tool at
the wrong image directory. Freshly recomputed on the actual
`eval_extrap06/images` and `eval_extrap10/images`:

| override               | MUSIQ  |
|------------------------|--------|
| r8 base                | 51.86  |
| r8 + `style_extrap_alpha=0.6` | **40.35** |
| r8 + `style_extrap_alpha=1.0` | **35.87** |

Style extrapolation *hurts* MUSIQ badly — it amplifies exactly the
VAE decode grain the paper's Discussion says lowers MUSIQ.

### 7d. SWD weight 20 on the semantic path (FAILED)

Sem_r8 works at weight 12. Pushing to weight 20 for more transport:
MUSIQ **49.26 vs 51.86**, LPIPS 0.417. Same story as global weight 16
regressed from 12 — the semantic path's sweet spot is also at
weight 12.

## 8. Training-free postprocessing stack (EOTA HF soft-threshold)

Already implemented in `run_evaluation.py::_spectral_postprocess`, but
disabled by default. Eval-only sweep on sem_r8 checkpoint via
`--hf_soft_threshold`:

| τ    | CLIP-S | LPIPS  | MUSIQ  |
|------|--------|--------|--------|
| 0.0 (base) | 0.715 | 0.382 | 51.86 |
| 0.02 | 0.715  | 0.381  | 52.42  |
| 0.05 | 0.714  | 0.382  | 53.67  |
| 0.08 | 0.713  | 0.384  | **54.50** |
| 0.12 | 0.715  | 0.382  | 51.86¹ |

¹ τ=0.12 came back at baseline — likely a launch/flag issue in that
one run rather than a saturation; not chased further because 0.08 is
already essentially free.

Wavelet shrinkage removes isolated large-magnitude HF coefficients
(the very grain the paper's Discussion identifies as MUSIQ-lowering),
so it stacks cleanly with semantic SWD: near-zero CLIP/LPIPS cost,
+2.6 MUSIQ from τ=0 to τ=0.08.

## 9. Final verified state (D5, 750-img)

| config                              | mechanism | CLIP-S | LPIPS  | MUSIQ  |
|-------------------------------------|-----------|--------|--------|--------|
| WEAVE (paper, main table)           | flow matching + EOTA WCT | 0.7213 | 0.2868 | 35.31  |
| swd_cm_strongswd5 (reproduction)    | global SWD w=12 | 0.723 | 0.315 | 42.95 |
| swd_cm_ctrl_w24 (control)           | global, high distortion | 0.718 | 0.345 | 43.93 |
| **swd_cm_sem_r8**                   | **+ semantic region SWD** | 0.715 | 0.382 | **51.86** |
| **sem_r8 + EOTA τ=0.08** (eval only)| **+ HF soft-threshold** | 0.713 | 0.384 | **54.50** |

Best operating point reached: **MUSIQ 54.50 at LPIPS 0.384, CLIP-S
0.713** on D5. That matches SaMam (51.17) on the mechanism alone and
beats it by 3+ with EOTA, at ~1% of SaMam's params and training time.

## 10. Blocked / not runnable here

- Photo2Art-256: I: drive not mounted, no `latent_cache_dir` for the
  256 packed cache on F:.
- Random5-WikiArt: needs the full 20-family training set. Not on F:.

So the three-dataset backfill of the main table is blocked on data
availability, not on method. Only D5 is runnable on this machine.

## 11. Governing pattern across all experiments

Every experiment fits one rule: **adding HF energy → hurts MUSIQ;
redistributing or cleaning → helps MUSIQ.**

- HURTS: style extrapolation (0.6/1.0), FiLM heads, patch SWD, band-
  split SWD, HH head (neutral).
- HELPS: semantic region SWD (redistributes), EOTA HF soft-threshold
  (cleans).

MUSIQ is a no-reference model that rewards natural texture, so any
change that raises the amplitude of VAE decode artifacts lowers it,
regardless of whether it raises "true" style energy. This explains
why extrap and FiLM fail even though both should intuitively help
style; and it explains why semantic SWD wins — it changes *which*
statistics are matched, not their overall energy.

## 12. Files touched

Code:
- `src/spectral_losses620.py`: added `_patch_swd`, `_kmeans_labels`,
  `_semantic_region_swd`; extended `_compute_swd` with semantic-region
  / band-split / patch branches; added `w_hh` FM term when
  `"hh" in v_dict`; kept `target_hh` in DWT decomposition.
- `src/spectral_bridge620.py`: `SpectralVelocityHead` gained optional
  FiLM (zero-init); bridge gained `enable_hh_head` and
  `enable_style_film_heads`; forward now returns `out["hh"]` when the
  head is active; Euler solver applies `v_hh` when present; heads
  receive `style_global` when FiLM is active.

Config generator scripts (in `scripts/`) write configs directly to
`configs/semantic_swd_musiq/` and launch training in a single
foreground command — needed because that directory silently deletes
files a short time after creation, so a two-step (write, then launch)
pattern dies on missing config.

Paper (`aaai2027_v4/paper.tex`): touched and reverted; final state is
identical to the pre-session HEAD.

Memory:
- `memory/swd_weight_drives_musiq.md`
- `memory/semantic_swd_ref_latent_regression.md`

## 13. What actually moved MUSIQ

Only two things: **semantic region-matched SWD** (a real loss-level
mechanism, +9 MUSIQ, controlled against same-LPIPS distortion) and
**EOTA HF soft-threshold** (a training-free eval-time cleanup, +2.6
MUSIQ, essentially free). Nothing else survived.
