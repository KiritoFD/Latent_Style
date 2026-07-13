# Repository and Remote Audit

Date: 2026-07-13
Scope: local `G:\GitHub\Latent_Style\SchrodingerBridge`, remote `administrator@100.115.18.62`, drive `I:`.

## Executive State

The project currently has two distinct states that should not be conflated:

1. **Submitted paper bundle state**: `aaai2027_v4/` was updated and committed in `0867d43d7` with the metric-block radar, current `paper.pdf`, and the first formal `SUPPLEMENTARY_MATERIAL.md`.
2. **Active research worktree state**: the repo contains many uncommitted cleanups, probe outputs, generated supplement drafts, and historical experiment deletions. The worktree is not clean.

The previous documentation statement "all failed experiment code has been deleted and the tree is clean" is no longer accurate as a current-state statement. It should be read as historical cleanup intent, not as present evidence.

## Local Git State

Command:

```powershell
git status --porcelain
```

Observed summary:

| Status | Count | Interpretation |
|---|---:|---|
| Deleted | 473 | Large historical config/tool cleanup is staged as working-tree deletion, not yet committed in the current local tree. |
| Modified | 44 | Includes source code, docs, evaluation utilities, and generated paper artifacts. |
| Untracked | 299 | Includes new configs, probe reports, logs, supplement build artifacts, and helper scripts. |
| Total | 816 | The repository must be treated as dirty. |

Top-level changed areas:

| Area | Role |
|---|---|
| `aaai2027_v4/` | Paper bundle; contains both committed formal files and untracked build/supplement scratch files. |
| `configs/` | Many obsolete configs deleted; many new probe/sweep configs untracked. |
| `docs/713/`, `docs/model_probe/`, `docs/delivery/` | Current diagnostic evidence and summary docs. |
| `src/`, `tests/`, `scripts/`, `tools/` | Active implementation and helper changes, not all paper-facing. |
| `tools/ablation256/` | Large historical ablation subtree marked deleted. |

## Paper Bundle State

The committed AAAI v4 paper state is:

| Artifact | Status |
|---|---|
| `aaai2027_v4/paper.tex` / `paper.pdf` | Committed in `0867d43d7`; compiles locally. |
| `aaai2027_v4/make_radar_metric_blocks.py` | Committed source of the current radar figure. |
| `aaai2027_v4/radar_metric_blocks_A_clip_dinos_robustbreak.png` | Committed current radar. |
| `aaai2027_v4/SUPPLEMENTARY_MATERIAL.md` | Committed first formal supplement draft; now needs expansion. |

Untracked scratch in the same folder includes `supplement.tex`, `make_supplement_figures.py`, `supplement_figures/`, build logs, and alternate radar normalization PNGs. These are not yet authoritative.

## Remote State

Remote host:

```text
administrator@100.115.18.62 -p 2222
hostname: USER-20250629DN
```

Remote filesystem observations:

| Path | Exists | Git repo | Role |
|---|---:|---:|---|
| `I:\Github\Latent_Style` | yes | no | Remote project root / working copy container. |
| `I:\Github\Latent_Style\SchrodingerBridge` | yes | no | Remote synchronized code and experiment tree. |
| `I:\checkpoints` | no | n/a | Old checkpoint path in some docs is stale for this remote. |
| `I:\latent_style_remote_curated` | yes | no | Historical remote archaeology index from earlier cleanup. |

Important implication: remote state cannot be audited with `git status`. It must be audited by file paths, timestamps, logs, summaries, and local commits.

## Remote Experiment Evidence

Key remote directory:

```text
I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe
```

Observed experiment folders, newest first:

| Folder | Last seen role |
|---|---|
| `inf_timing` | Generation-only timing benchmark. |
| `target_hf_content_anchor_ft6` | Content-anchor HF placement; safe but below subband-only. |
| `target_hf_subband_texture_ft6` | Conservative alternate; best off-DINO-S / DINO-C balance. |
| `target_hf_texture_ft6` | Stationary texture-stat route. |
| `target_hf_spatial_energy_ft6` | Energy-bounded spatial route; too much content cost. |
| `target_hf_subband_head_ft6` | Subband code into HF heads; not better than subband residual. |
| `target_hf_hybrid_ft6` | Shared pooled + per-band residual; no gain. |
| `target_hf_subband_ft6` | Best current usable architecture point. |
| `target_hf_spatial_ft6` | Raw spatial route; high style but content collapse. |
| `target_hf_delta_strong_ft6` | First usable HF route. |
| `target_hf_delta_ft15` | Early HF-only route curve. |
| `target_latent_fusion_ft15`, `target_latent_fusion_ft3` | Global target-token fusion; path exists but over-controls LL. |

Remote summaries mirror local files under:

```text
docs/model_probe/*.json
docs/model_probe/*.md
```

The most important aggregate file is:

```text
docs/model_probe/target_hf_delta_eval_summary.json
```

## Current Evidence Hierarchy

Use this priority order when writing the method or supplement:

1. **Paper raw metrics**: `aaai2027_v4/paper.tex`, Table 1.
2. **Radar generation**: `aaai2027_v4/make_radar_metric_blocks.py` and committed radar PNG.
3. **HF probe aggregate**: `docs/model_probe/target_hf_delta_eval_summary.json`.
4. **HF diagnosis narrative**: `docs/713/HF_ARCHITECTURE_PROBE_2026-07-13.md` and `docs/model_probe/HF_DELTA_DIAGNOSIS_2026-07-13.md`.
5. **Timing**: `docs/model_probe/generation_only_timing_summary.json`.
6. **Remote raw artifacts**: `I:\Github\Latent_Style\SchrodingerBridge\exp\model_probe\...`.
7. **Historical archive**: `docs/archive/` and `I:\latent_style_remote_curated/`; use only for traceability, not for current claims without re-verification.

## Cleanup Recommendations

Near-term cleanup should be conservative:

1. Do not delete untracked probe evidence until the expanded supplement and method notes cite the stable aggregate files.
2. Keep `aaai2027_v4/` formal source files separate from generated scratch (`supplement.aux`, logs, alternate radar PNGs).
3. Update `docs/README.md` and `docs/delivery/DELIVERY_SUMMARY.md` to reflect the dirty worktree and remote non-git status.
4. Commit documentation updates separately from large source/config cleanup.
5. Only after the paper-facing documents stabilize, decide whether to commit the large config/tool deletions or move them into an archive branch.

## One-line Current Truth

The paper bundle is coherent and committed, but the broader repo is an active dirty research workspace; remote `I:` is a synchronized experiment filesystem, not a git-controlled source of truth.
