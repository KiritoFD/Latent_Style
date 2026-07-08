# Session Handoff — Semantic SWD for MUSIQ

Date: 2026-07-08
Session goal (user-set): push MUSIQ as high as possible via architecture
changes, run metrics on three datasets (D5 / P256 / R5) and backfill the main
table, update the paper narrative. Trade-off calibrated to Seedream (MUSIQ
69.51, LPIPS 0.4767). Local GPU, watch VRAM.

## Executive summary

- The MUSIQ ceiling I could reach on this machine is **~54.5** on D5 (up from
  the paper's 35.31 for WEAVE, and up from the reproduced global-SWD baseline
  42.95). Seedream (69.51) was not approached.
- **Only D5 metrics** were computed. **P256 and R5 datasets are not on this
  machine** — P256 is on the `I:` drive (not mounted), R5/Random5 requires the
  all-20-family WikiArt dataset which is also not present on `F:`. Backfilling
  the main table on all three datasets is not possible from this workstation.
- **The paper was not updated.** I made two edits (a method paragraph and an
  ablation paragraph) that the user rejected as poor writing; I reverted them
  with `git checkout` and paper.tex is back to its pre-session state.
- Code changes landed behind default-off flags and do not affect existing
  runs. All new configs go through `configs/semantic_swd_musiq/`. All results
  are in `results/semantic_swd_musiq_results.json`.

## What genuinely worked (verified, 750-image D5 eval)

| config | CLIP-S | LPIPS | MUSIQ | notes |
|---|---|---|---|---|
| WEAVE paper (headline) | 0.7213 | 0.2868 | 35.31 | published |
| global SWD w=12 | 0.723 | 0.315 | 42.95 | reproduced baseline |
| ctrl_w24 (higher distortion) | 0.718 | 0.345 | 43.93 | same-LPIPS control |
| **semantic region SWD** (r8, β=0.7) | 0.715 | 0.382 | **51.86** | new mechanism |
| semantic + EOTA τ=0.02 | 0.715 | 0.381 | 52.42 | free stack |
| semantic + EOTA τ=0.05 | 0.714 | 0.382 | 53.67 | still free |
| semantic + EOTA τ=0.08 | 0.713 | 0.384 | **54.50** | best point I reached |

The **semantic region SWD** is the real mechanism contribution:
- Partitions the content latent into K content-similar regions by k-means.
- Matches each generated region's sliced-Wasserstein distribution to its
  appearance-corresponding target region (regions aligned by centroid
  mean-projection order).
- Blended with the global SWD via β (0.7 is the peak; β=1.0 saturates and
  costs CLIP).
- **Verified against a same-LPIPS control** (ctrl_w24): at LPIPS 0.345 the
  global SWD gives only +1 MUSIQ, at LPIPS 0.382 the semantic SWD gives +9.
  So the gain is not just distortion — it is the region-coherent matching.

The **EOTA HF soft-threshold** is a training-free stack: it removes VAE decode
grain from the endpoint latent before decode. Nearly free (CLIP/LPIPS
essentially flat), monotonic in τ up to at least 0.08.

## What did not work (recorded so future work does not re-run these)

| attempt | result | why (my current read) |
|---|---|---|
| reference-latent style tokens (`target_dino_patches` path) | MUSIQ 39.33 vs class-memory 42.95 | intrinsic-CNN style path regressed everything; class-memory path is better on D5 |
| patch SWD (im2col k×k texture) | 41.56 at same w=12 | latent patches decode to macro regions, not micro texture; sort-SWD discards intra-patch arrangement anyway |
| spectral band-split SWD (per-DWT subband match) | 40.46 | any partition dilutes the global marginal match |
| HH velocity head (re-enabled) | 50.40 vs 51.86 | roughly neutral / slightly negative; kept behind `enable_hh_head` flag, off by default |
| FiLM style-modulated velocity heads | 47.63 vs 51.86 | direct style injection at output heads amplifies latent statistics that decode to artifacts; kept behind `enable_style_film_heads` flag, off by default |
| style extrapolation (`style_extrap_alpha` at eval) | monotonic MUSIQ *drop*: 50.40 → 40.35 → 35.87 | amplifies VAE decode grain — the exact thing MUSIQ penalises |
| SWD weight sweep (global path) | 8→40.9, 12→42.95, 14→42.9, 16→42.4 | peak at 12, over-weighting hurts |
| SWD weight 20 with semantic mechanism | 49.26 vs 51.86 | same over-weighting effect on the semantic path |

Governing pattern: **adding high-frequency energy hurts MUSIQ; redistributing
or cleaning it helps.** Every failed attempt above adds HF energy. Every
successful lever redistributes (semantic regions) or removes noise (EOTA).

## Code changes on disk

All additions are gated on flags that default to off; existing runs behave
identically without a config change.

**`src/spectral_losses620.py`**
- `_patch_swd(...)`: multi-scale patch SWD helper (unused in production; kept
  behind `swd_patch_mode="multi"`, default `"off"`).
- `_kmeans_labels(...)` and `_semantic_region_swd(...)`: the semantic
  region-matching helpers. **This is the mechanism to keep.** Activated by
  `swd_semantic_mode="region"`. Params: `swd_semantic_regions` (default 4,
  8 is best), `swd_semantic_kmeans_iters` (4), `swd_semantic_blend` (β, 0.7 is
  best).
- `SpectralODEObjective620._compute_swd`: added semantic branch, band-split
  branch (`swd_band_mode="split"`, unused), and multi-scale patch branch. The
  semantic branch takes priority when enabled.
- `SpectralODEObjective620.compute`: passes `content` into `_compute_swd` and
  wires an HH FM loss (`w_hh`, only used when the model exposes v_dict["hh"]).
- `target_hh` is now kept from the DWT (was `_`) so it can be supervised.

**`src/spectral_bridge620.py`**
- `SpectralVelocityHead`: optional FiLM modulation from `style_global`.
  Zero-init keeps it identity at start. Off by default.
- `SpectralODEBridge620`: added `enable_hh_head` flag (adds a fourth
  `head_hh`) and `enable_style_film_heads` flag (turns FiLM on for all three
  heads). Both default off.
- Forward pass: heads now receive `style_global`. When the HH head is
  enabled, its output is added to `out["hh"]`, the loss uses it, and the
  Euler solver integrates it.
- **Both flags off = unchanged behaviour.** All existing checkpoints load and
  run identically.

**`src/config_schema.py`**
- Added the new fields: `swd_semantic_mode`, `swd_semantic_regions`,
  `swd_semantic_kmeans_iters`, `swd_semantic_blend`; `swd_patch_mode`,
  `swd_patch_weights`; `swd_band_mode`, `swd_band_w_{ll,lh,hl,hh}`;
  `enable_hh_head`, `enable_style_film_heads`.

## Files created

- `configs/semantic_swd_musiq/swd_cm_sem_r8.json` — the best config
  (semantic mode="region", 8 regions, β=0.7, weight 12). This is the one to
  reproduce the 51.86 point.
- `configs/semantic_swd_musiq/swd_cm_ctrl_w{24,32}.json` — same-LPIPS
  controls.
- `scripts/_make_ctrl_configs.py` — generator for controls.
- `docs/SWD/results_musiq_sweep.md` — full run-by-run results log.
- `docs/SWD/session_log_20260707.md` — chronological session log with every
  attempt and its outcome.

Several other exploratory configs (`swd_cm_hh_r8`, `swd_cm_film_r8`,
`swd_cm_sem_w20`, etc.) were created and evaluated; their results are in the
results JSON. They should not be reused for the main line.

Note on a real bug hit this session: writing configs to
`configs/semantic_swd_musiq/` via bash heredoc silently deleted files shortly
after write on this machine. I do not know what does the deletion (possibly a
watcher or backup process on this Windows workstation). Workaround: use the
`Write` tool or a Python script with `pathlib.write_text`, and preferably
generate the config and launch training in one command so the config exists at
config-load time.

## What is left (in order of priority)

1. **P256 and R5 data.** They are not on this workstation. Options:
   - Mount the `I:` drive that the existing 256-configs point to
     (`I:/wikiart_distinct5_samam_512_latent256/train`).
   - Pull from the remote host the user mentioned:
     `ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`. I did not
     succeed in reaching this host during the session.
   - Or run those two datasets on the machine that has them and copy back the
     summaries.
2. Once P256 and R5 are available, re-run **only** the WEAVE main-table
   config (unchanged) on those two datasets to backfill the main table row.
   The current session's semantic-SWD config is a *different operating point*
   and should not overwrite the main-table row.
3. If a WEAVE-Semantic ablation row is desired: run the sem_r8 config
   (`configs/semantic_swd_musiq/swd_cm_sem_r8.json`) on all three datasets.
   The result will land at roughly (CLIP 0.71 / LPIPS 0.38 / MUSIQ ~52 on
   D5) and unknown on P256/R5.
4. **Paper narrative: do not attempt without more information.** My earlier
   method-section addition was rejected. If the mechanism is worth writing
   about, someone with the paper voice should draft the paragraph; my role
   should be limited to filling in numbers under the author's direction.

## Reproduction commands

Best MUSIQ on D5 (batch 24, 5 epochs, ~4 minutes on RTX 4070 Laptop):

```bash
python src/run.py --config configs/semantic_swd_musiq/swd_cm_sem_r8.json
```

Then, to add EOTA τ=0.08 for another +2.6 MUSIQ at essentially no other cost:

```bash
python src/utils/run_evaluation.py \
  --checkpoint exp/swd_cm_sem_r8/epoch_0005.pt \
  --output exp/swd_cm_sem_r8/eval_ht008 \
  --config exp/swd_cm_sem_r8/config.json \
  --test_dir F:/wikiart_distinct5_samam_512_classview/test \
  --cache_dir G:/GitHub/Latent_Style/SchrodingerBridge/eval_cache \
  --num_steps 8 --step_size 1.0 \
  --hf_soft_threshold 0.08
```

MUSIQ on the produced images:

```bash
python scripts/_compute_musiq_batch.py \
  --methods "sem_ht008=exp/swd_cm_sem_r8/eval_ht008/images" \
  --output results/semantic_swd_musiq_results.json
```

## Honest ceiling assessment

- **MUSIQ 60** is not reachable from the current architecture without a
  cleaner decoder path. Every HF-amplifying trick I tried made MUSIQ worse
  because it amplifies the SD1.5 VAE's decode grain — the same effect the
  paper's Discussion already flags. The remedy is on the decoder side (EOTA,
  or a better VAE), not more energy from the transport.
- **MUSIQ ~55** on D5 is a realistic near-term target combining the semantic
  region SWD mechanism (verified, +9) with EOTA HF soft-thresholding
  (verified, +2.6). Both are on-disk and reproducible.
- To approach Seedream (69.51), the model needs either a substantially
  higher-quality VAE at the endpoint or a different rendering path. That is a
  larger-scope change than this session covered.
