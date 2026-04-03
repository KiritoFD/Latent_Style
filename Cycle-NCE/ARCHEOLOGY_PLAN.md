# Archaeology Plan - What's Done and What's Left
Updated: 2026-04-03 03:00 CST

## REPORT FILES
- PART1: ARCHAEOLOGY_REPORT.md (138,695 bytes ≈ 135KB)
- PART2: ARCHAEOLOGY_PART2.md (96,930 bytes ≈ 95KB)
- TOTAL: ~230KB of project knowledge

## DONE (covered across PART1+PART2)

### Core Code (100% scanned):
- model.py, losses.py, trainer.py, run.py (4 core)
- opt.py (640 lines), ablate.py (403 lines), exp.py (161 lines), res.py (165 lines)
- batch_distill_full_eval.py (251 lines), eval_final_works.py (311 lines)
- orthogonal_8plus6.py (170 lines), prob.py, prob-swd.py, probe_odd_patch_snr.py
- analyze_brightness_color_alignment.py (429 lines), build_style_color_stats.py
- calibrate_half_epoch_vram.py, fewshot_ukiyoe_pipeline.py (444 lines)
- rerun_reuse_full_eval_and_export_csv.py (488 lines)
- report_exp_brightness_color_vs_train.py (224 lines)
- 20 .bat launcher files (full read)

### history_configs/ (100% scanned):
- 109 configs fully read and categorized into 16 families

### Experiment Results (comprehensive):
- 109 history_configs configs mapped
- 31 Optuna HPO trials (configs + results)
- Ablate43 (17 experiments, full config matrix)
- ZeroConstraint Z01-Z03 (6 eval entries from exp.csv)
- 42_A01 Macro_Only (epoch 30, 60 + distill variants)
- All scan series (scan01-scan06)
- All nce series (11 dirs)
- All master_sweep series (21 dirs)
- All decoder series (A/B/C, D0-D7, D-160, D-sweetspot)
- All micro01-05 series (full 4-epoch trajectories)
- All spatial-adagn series (5 dirs)
- G0/G1 series, dict series
- All ablate A/M/E series
- FinalMicro_2 F01/F02 CSV data
- Exp series (exp_1-6, G1, G2, S1-S10)
- Clocor1 E1-E6 series
- style_oa (9 subdirs + multi-epoch data)
- Inject I0-I7 (configs found in history_configs)
- Early experiments: 1swd-dit-2style, strong-1swd-dit-5style, style8-, full-v2, etc.
- exp.csv (274 rows, all 17 groups parsed)
- All supplemental CSVs (benchmark, ablation, color, style_oa)
- summary/ (12 JSONs read)
- grid/ (41 PNG files, 105MB indexed)
- fewshot_runs/ (ukiyo-e pipeline details)

## NOT DONE / LOW PRIORITY
[ ] experiments-swd8/ (27 subdirs - most lack eval data, explored at overview level)
[ ] experiments-cycle/ subdirs (ablation50_repro, sweep_swd_reborn - mostly configs)
[ ] RAR archives (optuna_hpo.zip 9GB, Exp.rar 753MB, abl.rar 540MB, color.rar 135MB)
[ ] prob.py full source (partially read)

## ACTIVE EXPERIMENTS (no new eval data at 03:00)
- 42_A02-A10, micro_E*, T0*, Z* - still training or queued
- Only A01-Macro has eval results so far (ep30, ep60)


## DONE 2026-04-03 03:03:
- [x] Decoder-H-MSCTM series (4 experiments)
- [x] Micro01-05 full 4-epoch trajectories WITH ALL METRICS (style, dir, lpips, class)
- [x] G2_High_TV_Test (no evals, has training log)
- [x] Confirmed: NO GIT HISTORY in this repo (no commits)
- [x] Code version timeline reconstructed from snapshots
- [x] Tokenizer distillation coverage documented (5×4×2=40 checkpoints)

PART2 current size: 104999 bytes

---

## NEW PRIORITY: Feb 13-17 Signal Separation & Diff-Gram Experiments

**Data Source**: `C:\Users\xy\full_history` (full repository history)

**Context**: 
- Around **Feb 13**, a major experiment integration happened (`e3358c2`).
- 52 runs were analyzed and categorized into Tier A (33), B, and C.
- `runs_detailed.json` (4214 lines) contains detailed metrics for all runs.
- Around **Feb 13-17**, there were critical experiments involving `diff-gram`, signal separation, and SWD vs Gram comparisons.

### Key Commits (Signal Separation Era)
- `e3358c2` (Feb 13 14:28): "实验整合进experiments目录" - Integrated 52 runs, created `docs/experiments_cycle/` with full metrics.
- `1599a2f` (Feb 13 14:08): "风格消融实验" - Style ablation experiments.
- `8097ef7` (Jan 28 16:47): "MSE完全爆炸" - MSE exploded with cross-attn.
- `8c3edfd` (Feb 22 00:10): "在style-8注入最好；做了一点算子融合" - Fusion experiments.
- `d3526ef` (Feb 17 22:52): "diff-gram继续消融，换一下NORM方式，到0.07了" - Diff-gram ablation to 0.07.
- `84b525f` (Feb 17 01:49): "SWD某些情况下有微弱的作用，GRAM完全没用" - SWD slightly useful, GRAM useless.
- `025b77e` (Feb 21 16:16): "diff-gram在sdxl-fp32上表现极差" - Diff-gram very poor on SDXL-FP32.
- `0a0c55f` (Feb 09 13:35): "Document style-first discipline and add proto-separation E7/E8 experiments" - Proto-separation experiments.

### Tasks
- [ ] Extract and analyze `docs/experiments_cycle/data/runs_detailed.json` (4214 lines) from `e3358c2`.
- [ ] Analyze `docs/experiments_cycle/data/runs_metrics.csv` (53 lines) for the full 52-run comparison.
- [ ] Compare Diff-Gram vs SWD performance (Feb 17 era commits: `84b525f`, `d3526ef`, etc.).
- [ ] Investigate "proto-separation" experiments E7/E8 (`0a0c55f` commit).
- [ ] Trace code changes in `Thermal/src/losses.py` and `Thermal/src/model.py` during this signal separation era.

---

## Pre-Feb 8 Detailed History Summary

**Status**: ✅ Completed and written to `ARCHAEOLOGY_PRE_SCRATCH.md`.
**Data Source**: `C:\Users\xy\full_history` (54 commits from Jan 13 to Feb 7).
**Key Phases**:
1. Jan 13-15: Data encoding (WikiArt, VAE).
2. Jan 16-21: DiT experiments (Failed) → Thermal project starts.
3. Jan 22-27: `LGTUNet` (605 lines) built with `AdaGN`, `ResidualBlock`, `SelfAttention`, `StyleGate`.
4. Jan 28-31: Cross-Attention attempt (1033 lines) → "MSE exploded" → Rollback to AdaGN (217 lines).
5. Feb 1-7: Infra optimization, Loss weight normalization, Structure vs Style loss no longer opposing.
6. Feb 8: `Cycle-NCE/src/` born with 251-line `model.py`.



- **[DONE]** Mar 6 - Mar 13 (Phase 2.5): Tokenizer Distillation, HybridStyleBank failure, Revert to Decoder-D configs.
- **[TODO]** Mar 13 - Mar 22: Cross-Attn Return (Brightness Constraint), Color Loss refinement.
- **[TODO]** Mar 22 - Mar 30: The Global/Window Attention Injection (The 1200+ line era).
- **[TODO]** Mar 30 - Apr 02: The 'Micro-Batch' Great Simplification (Code cut by 65%).


## 📅 Updated Progress (2026-04-03 07:12)
- **Phase 2.6 (Mar 19 - Apr 2)**: ✅ DONE (Attention Renaissance & Micro-batch revolution).
- **Phase 2.5 (Mar 08 - Mar 18)**: ✅ DONE (NCE + Adaptive Mixing).
- **Phase 2.4 (Feb 22 - Mar 06)**: ✅ DONE (Color Anchors & SWD Domain breakthrough).
- **Phase 2.1-2.3 (Feb 08 - Feb 21)**: ⚠️ **ALIGNMENT REQUIRED**. (Gram Whitening, Style Maps, Signal Separation).

## 🔍 Next Mission: Feb 08 - Mar 13 Alignment (Code vs. Experiment)
**Objective**: Map `C:\Users\xy\full_history` commits to `Y:\experiments` results.
**Key Nodes**:
1. **Feb 13 (Gram Whitening)**: Commit `c505d3d6`. Did it improve Style Ratio in `overfit50` runs?
2. **Feb 15 (No-Edge Pruning)**: Commit `ff580af`. Did removing Edge loss kill Structure score?
3. **Feb 17 (SWD vs Gram)**: Commit `84b525f`. The "SWD slight advantage" moment.
4. **Feb 22 (Self-Similarity/StyleMaps)**: Commit `14763d0`. Did 5-style runs benefit from this?
5. **Mar 08 (NCE + Gating)**: The birth of `MSContextualAdaGN`.

**Action**: Scan `history_configs` for files dated in this range -> find corresponding experiment in `Y:\experiments` -> check `full_eval` stats.


## 🔍 Mission: Code-Experiment Alignment (Feb 08 - Mar 13)
**Goal**: Map `C:\Users\xy\full_history` commits to `Y:\experiments` results.
**Status**: 🔴 Active. Aligning Phase 2.1-2.3 (Gram Whitening, SWD, Gating).

### 📅 Key Alignment Nodes:
1. **Feb 13 (Gram Whitening & Signal Sep)**: 
   - *Code*: `c505d3d` (Feb 14) - Introduced SVD-based channel whitening.
   - *Hypothesis*: Did this reduce 'fog' or improve style ratio?
   - *Target*: Look for `style_ratio` spikes in `overfit50` or `full_300` runs around this date.
2. **Feb 17 (SWD vs Gram)**: 
   - *Code*: `84b525f` (Feb 17) - 'SWD slight advantage, Gram useless'.
   - *Target*: Identify the run where SWD score surpassed Gram score.
3. **Feb 22 (5-Style / Style Maps)**: 
   - *Code*: `14763d0` - Domain SWD 5.77x breakthrough.
   - *Target*: `5style` experiments, check style classifier score.
4. **Mar 08 (NCE + Gating)**: 
   - *Code*: `4992e06` - 'NCE loss is effective'.
   - *Target*: Experiments named `nce-*` or `gate-*`.

## ✅ Code-Experiment Alignment Status (2026-04-03 07:15)
- **Phase 2.3 (Feb 17-21)**: ✅ ALIGNED. swd-256 experiment confirms SWD emergence (clip:0.52, lpips:0.31).
- **Phase 2.4 (Feb 22-Mar 6)**: ✅ ALIGNED. nstyle-proj (200ep, 5-style) confirms StyleMaps success (clip_style=0.68, acc=42.3%, Domain SWD 5.77x).
- **Phase 2.5 (Mar 8-18)**: ✅ ALIGNED. NCE+Gate experiments prove massive gains (clip:0.52→0.69, acc:0.0→0.40).
- **Phase 2.6 (Mar 19-Apr 2)**: ✅ ALIGNED. style_oa_5 confirms Cross-Attn success (clip_style=0.698@120ep). Trainer purge Apr 2 verified by micro_E experiments (0.693/0.707).

### Key Finding: NCE Revolution
NCE Loss + StyleAdaptiveSkip Gating (Mar 8) was the SINGLE BIGGEST performance jump:
- Style: +32% (0.52→0.69) | Content: degraded (0.31→0.52) | Acc: 0.0→0.40
- Gate provides marginal gain over pure NCE (+0.8% clip, +1.8% acc)
- Pure Decoder architecture (no_norm) underperforms NCE by 3-11%

## 2026-04-03 12:15 - DATA DISCOVERY: master_sweep epoch 100 summaries found

Located in Y:\experiments\summary\:
- master_sweep_05_patch_micro__epoch_0100: ST=0.635, P2A=0.672, FID_ratio=0.96%
- master_sweep_14_narrow_micro__epoch_0100: ST=0.663, P2A=0.630, FID_ratio=2.95%

Key: Sweep 14 has better FID quality, Sweep 05 has better P2A style score.
Both are multi-style joint training — expected to be lower than single-style peaks.
Analysis written to ARCHAEOLGY_PART3.md.
