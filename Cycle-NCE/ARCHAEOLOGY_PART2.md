# ARCHAEOLOGY REPORT - PART 2

**Date**: 2026-04-03 02:34 CST
**Continues from**: ARCHAEOLOGY_REPORT.md (Part 1: 37 sections, ~133KB)
**Focus**: Scripts, eval pipelines, analysis tools, and any new experiment results

---

## APPENDED 2026-04-03 02:34

## 1. eval_final_works.py (311 lines) - Post-Hoc Evaluation Pipeline

This is a standalone evaluation runner that discovers all experiment directories recursively
and generates a master CSV of all results.

Key functions:
- `_is_eval_dir(path)`: Checks if a directory is an eval output dir (has summary.json)
- `discover_eval_dirs(root)`: Recursively finds all eval dirs across the experiments tree
- `extract_row(eval_dir)` (Line 80): Extracts 20+ metrics from a single eval summary:
  - clip_style, clip_content, content_lpips, fid, art_fid
  - clip_dir (style edit direction)
  - classifier_acc (style correctness)
  - delta_fid, delta_art_fid vs baseline

This script scans the ENTIRE experiments/ tree and produces a single CSV with one row per eval checkpoint.
Currently produces:
- `KEY_EXPERIMENT_PHOTO_TO_ART_COMPARISON_20260326.csv` (22 rows, 8.3KB)
- `SUBMISSION_KEY_EXPERIMENTS_20260326.csv` (19 rows, 2.7KB)
- Supplemental CSVs for ablation, benchmark, color, style_oa groups


--- APPENDED 2026-04-03 02:35 - Scan and NCE Series ---

## 4. Scan Series (scan01-06) - Gate/ID/LR Sensitivity Scan

6 experiments, each with 4 full_eval checkpoints:
- scan01_base: ? eval rounds
- scan02_low_lr: ? eval rounds
- scan03_soft_hf: ? eval rounds
- scan04_gate_0p85: ? eval rounds
- scan05_gate_0p75: ? eval rounds
- scan06_id_0p50: ? eval rounds

## 5. NCE Series (11 dirs) - NCE/Gate Experiments

| Experiment | Evals | Logs |
|-----------|------:|-----:|
| nce | 5 | 4 |
| nce-gate_content | 2 | 3 |
| nce-gate_norm | 1 | 6 |
| nce-gate_norm-swd_0.45-cl_0.01 | 3 | 4 |
| nce-swd_0.25-cl_0.01 | 3 | 4 |
| nce_A1_Deep_Only | 2 | 4 |
| nce_A2_Shallow_Only | 2 | 3 |
| nce_A3_Patch_Coarse | 2 | 3 |
| nce_A4_Patch_Fine | 2 | 3 |
| nce_A5_High_TV | 2 | 3 |
| nce_A6_Strong_ID | 0 | 1 |

## 6. Master Sweep Series (21 dirs) - Systematic Architecture Search

| Experiment | Evals | Logs | Architecture Variable |
|-----------|------:|-----:|----------------------|
| 007_patch_oversize | 0 | 1 | Patch size too large |
| 01_cap_64 | 1 | 5 | Capacity floor (dim=64) |
| 02_cap_128 | 1 | 7 | Capacity dim=128 |
| 03_cap_192 | 1 | 2 | Capacity dim=192 |
| 04_cap_256 | 0 | 4 | Capacity ceiling (dim=256) |
| 05_patch_micro | 1 | 2 | Micro-scale patches |
| 06_patch_std | 1 | 2 | Standard patches |
| 07_patch_xmax | 1 | 4 | Max patches |
| 08_lr_fast | 1 | 0 | Fast LR schedule |
| 09_lr_slow | 1 | 0 | Slow LR schedule |
| 10_wide_xmax | 0 | 3 | Wide + xmax patches |
| 11_narrow_xmax | 0 | 1 | Narrow + xmax |
| 12_mid_xmax | 1 | 2 | Mid + xmax |
| 13_wide_micro | 0 | 2 | Wide + micro |
| 14_narrow_micro | 1 | 2 | Narrow + micro |
| 15_split_brain_128 | 1 | 2 | Split brain architecture |
| 16_split_brain_256 | 0 | 1 | Split brain 256 |
| 17_extreme_underpowered | 0 | 4 | Underpowered model |
| 18_extreme_overpowered | 0 | 1 | Overpowered model |
| 19_the_abyss | 0 | 5 | Extreme test (abyss) |
| 20_golden_balance | 0 | 9 | Balanced config - most logs! |

Note: master_sweep_20_golden_balance has 9 training logs - most of any sweep experiment.
master_sweep_19_the_abyss also has 5 logs. These were heavily monitored experiments.

---

## 7. Analysis Scripts Summary (Full)

### analyze_brightness_color_alignment.py (429 lines)
- Analyzes brightness and color distribution alignment between generated and reference images
- Uses YCbCR color space for luminance/chrominance separation
- Compares generated outputs against style reference distributions
- Exports CSV with per-style L1 distribution distances

### build_style_color_stats.py (165 lines)
- Pre-computes per-style color statistics (YCbCR channel distributions)
- Samples images from each style directory, computes mean/std per channel
- Caches results to speed up brightness/color alignment analysis

### calibrate_half_epoch_vram.py (122 lines)
- VRAM calibration tool
- Finds optimal batch size for a given VRAM target (default 10.5GB)
- Iteratively runs training steps to find max batch size

### report_exp_brightness_color_vs_train.py (224 lines)
- Cross-references brightness/color analysis with training metrics
- Finds generated images in experiment dirs
- Correlates style/color alignment with training checkpoint quality

### rerun_reuse_full_eval_and_export_csv.py (488 lines)
- The BIGGEST eval tool - handles full pipeline re-execution
- Functions:
  - _discover_eval_dirs: Finds all eval directories across the experiment tree
  - _parse_eval_meta: Extracts metadata from eval summaries
  - _run_eval: Actually runs full evaluation with a given checkpoint
  - _run_distill: Runs tokenizer distillation on a checkpoint
  - _write_csv: Exports everything to CSV
  - _missing_required_metrics: Validates eval completeness
  - Handles both existing eval runs and fresh re-evaluations
  - Supports dry-run mode for planning

### fewshot_ukiyoe_pipeline.py (444 lines)
- Specialized pipeline for few-shot Ukiyoe-style transfer
- Functions:
  - extract_latents: Extracts latent representations from images
  - tokenizer_mean_vector: Computes mean style vector from tokenizer
  - refine_style_vector: Refines style vector using few-shot samples
  - patch_checkpoint_style: Modifies checkpoint with new style embeddings
  - run_eval_lpips_clip_style: Evaluates few-shot transfer quality
- This is a PROOF-OF-CONCEPT for adding new styles without full training

### prob-swd.py (486 lines)  
- SWD optimization and analysis tool
- Uses Optuna to find optimal patch-size combinations for SWD
- eval_combo_snr: Evaluates a patch combination's SNR
- Builds projection banks for different patch sizes
- Tests whether certain patch-size combos give better style signal-to-noise

---


--- APPENDED 2026-04-03 02:36 ---

## 9. Decoder-A/B/C Results

| Experiment | Epoch | Style | LPIPS | Description |
|-----------|------:|------:|------:|-------------|
| decoder-A-anchor-nohf | 40 | 0.659 | 0.453 | Anchor + No HF-SWD (conservative) |
| decoder-A-anchor-nohf | 80 | 0.657 | 0.447 | Slight style drop, LPIPS improved |
| decoder-B-hf-strict-id | 40 | 0.688 | 0.516 | HF-SWD + Strict Identity |
| decoder-B-hf-strict-id | 80 | 0.694 | 0.534 | Style improved, LPIPS worse |
| decoder-C-relaxed-id-nohf | 40 | 0.666 | 0.471 | Relaxed ID + No HF-SWD |
| decoder-C-relaxed-id-nohf | 80 | 0.670 | 0.476 | Slight improvement |

## 10. Decoder D0-D7 Results

| Experiment | Epoch | Style | LPIPS | FID | ArtFID |
|-----------|------:|------:|------:|----:|-------:|
| D1_tv_0p02 | 40 (tokenized) | 0.675 | 0.514 | 309.5 | 484.9 |
| D2_color_1p5 | 40 (tokenized) | 0.691 | 0.521 | 304.7 | 476.8 |
| D3_hf_4p0 | 40 (tokenized) | 0.698 | 0.538 | 321.1 | 508.6 |
| D4_gate_0p8 | 40 (tokenized) | 0.688 | 0.511 | 308.2 | 478.4 |
| D5_rank_8 | 40 (tokenized) | 0.689 | 0.503 | 300.7 | 466.2 |
| D6_god_combo | 40 (tokenized) | 0.700 | 0.546 | 321.9 | 511.6 |

Key: D6_god_combo achieved highest style (0.700) but worst ArtFID (511.6)
D5_rank_8 is most balanced: style=0.689, FID=300.7, ArtFID=466.2
D3_hf_4p0 (high frequency weight=4.0) pushed style up but ArtFID worsened

## 11. Other Notable Results

- delta_A0_base (p5, id=0.45, tv=0.05): ep60: style=0.679, lpips=0.487
- delta_A1_p7 (p7, same id/tv): ep60: style=0.678, lpips=0.478
- adamix: ep100: style=0.643, lpips=0.392 (best LPIPS so far, but style low)
- coord-spade-50e: ep50: style=0.650, lpips=0.376 (also excellent LPIPS)


--- APPENDED 2026-04-03 02:37 ---

## 12. Remaining Experiment Results (32 dirs)

| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| 1-decoder-no_norm-patch5_23-color1.0 | epoch_0030 | 0.6660 | 0.4464 |
| 1-decoder-no_norm-patch5_23-color1.0 | latest | 0.6660 | 0.4464 |
| 1-decoder-patch5-15 | epoch_0030 | 0.6760 | 0.4763 |
| 1-decoder-patch5-15 | latest | 0.6760 | 0.4763 |
| G0-Base-Gain0.5 | epoch_0040 | 0.6915 | 0.5344 |
| G0-Base-Gain0.5 | latest | 0.6915 | 0.5344 |
| G0_Balanced_Base | epoch_0040 | 0.6670 | 0.4676 |
| G0_Balanced_Base | latest | 0.6670 | 0.4676 |
| G1-Relax-ID | epoch_0040 | 0.6993 | 0.5642 |
| G1-Relax-ID | latest | 0.6993 | 0.5642 |
| ablate_E1_Macro19_Rigid_LR14e4 | epoch_0040 | 0.6689 | 0.4543 |
| ablate_E1_Macro19_Rigid_LR14e4 | latest | 0.6689 | 0.4543 |
| ablate_M1-Aggressive-Fine | epoch_0040 | 0.7019 | 0.5753 |
| ablate_M1-Aggressive-Fine | latest | 0.7019 | 0.5753 |
| ablate_M2-Smooth-Impasto | epoch_0040 | 0.6980 | 0.5581 |
| ablate_M2-Smooth-Impasto | latest | 0.6980 | 0.5581 |
| dict | epoch_0050 | 0.6490 | 0.3947 |
| dict | latest | 0.6490 | 0.3947 |
| dict-50-0.05 | epoch_0050 | 0.6539 | 0.3652 |
| dict-50-0.05 | latest | 0.6539 | 0.3652 |
| exp0-baseline | epoch_0040 | 0.6666 | 0.4763 |
| exp0-baseline | latest | 0.6666 | 0.4763 |
| exp1-hf-ratio-4p0 | epoch_0040 | 0.6841 | 0.5223 |
| exp1-hf-ratio-4p0 | latest | 0.6841 | 0.5223 |
| exp2-large-patches | epoch_0040 | 0.6719 | 0.5412 |
| exp2-large-patches | latest | 0.6719 | 0.5412 |
| final_demodulation | epoch_0040 | 0.7056 | 0.6997 |
| final_demodulation | latest | 0.7056 | 0.6997 |
| micro01_hf2_lr1 | epoch_0020 | 0.6652 | 0.4952 |
| micro01_hf2_lr1 | latest | 0.6652 | 0.4952 |
| micro02_macro_patch | epoch_0020 | 0.6584 | 0.4779 |
| micro02_macro_patch | latest | 0.6584 | 0.4779 |
| micro03_gate75 | epoch_0020 | 0.6661 | 0.4941 |
| micro03_gate75 | latest | 0.6661 | 0.4941 |
| micro04_hf1p5_macro | epoch_0020 | 0.6580 | 0.4715 |
| micro04_hf1p5_macro | latest | 0.6580 | 0.4715 |
| micro05_id_anchor | epoch_0020 | 0.6649 | 0.4922 |
| micro05_id_anchor | latest | 0.6649 | 0.4922 |
| no-dict-hf-swd | epoch_0040 | 0.6972 | 0.5757 |
| no-dict-hf-swd | latest | 0.6972 | 0.5757 |
| no-edge | epoch_0050 | 0.5584 | 0.6461 |
| no-edge | latest | 0.5584 | 0.6461 |
| nstyle-proj | epoch_0050 | 0.6830 | 0.4859 |
| nstyle-proj | latest | 0.6830 | 0.4859 |
| patch-1-3-5 | epoch_0020 | 0.6458 | 0.3539 |
| patch-1-3-5 | latest | 0.6458 | 0.3539 |
| spatial-adagn | epoch_0020 | 0.6493 | 0.3647 |
| spatial-adagn | latest | 0.6493 | 0.3647 |
| spatial-adagn-expA-texture | epoch_0040 | 0.6445 | 0.3632 |
| spatial-adagn-expA-texture | latest | 0.6445 | 0.3632 |
| spatial-adagn-expB-depth | epoch_0040 | 0.6457 | 0.3631 |
| spatial-adagn-expB-depth | latest | 0.6457 | 0.3631 |
| spatial-adagn-expC-reg | epoch_0040 | 0.6459 | 0.3586 |
| spatial-adagn-expC-reg | latest | 0.6459 | 0.3586 |
| spatial-adagn-nuclear | epoch_0100 | 0.6541 | 0.4133 |
| spatial-adagn-nuclear | latest | 0.6541 | 0.4133 |
| swd-256-100-6-50-1.5k | epoch_0050 | 0.5234 | 0.3052 |
| swd-256-100-6-50-1.5k | latest | 0.5234 | 0.3052 |
| swd-256-fix-gates | epoch_0050 | 0.5478 | 0.7744 |
| swd-256-fix-gates | latest | 0.5478 | 0.7744 |

---

## 13. Spatial-AdaGN Series - 5 experiments

These experiments test spatial AdaGN variants but have no local configs. Results:

| Experiment | Epoch | Style | LPIPS | Style(p2a) | FID | ArtFID |
|-----------|------:|------:|------:|-----------:|----:|-------:|
| spatial-adagn | epoch_0020 | 0.6493 | 0.3647 | 0.6084 | 285.6 | 395.8 |
| spatial-adagn | epoch_0100 | 0.6456 | 0.3959 | 0.6166 | 291.6 | 423.5 |
| spatial-adagn-expA-texture | epoch_0040 | 0.6445 | 0.3632 | 0.6182 | 289.5 | 405.7 |
| spatial-adagn-expA-texture | epoch_0080 | 0.6457 | 0.4019 | 0.6170 | 295.3 | 432.0 |
| spatial-adagn-expB-depth | epoch_0040 | 0.6457 | 0.3631 | 0.6148 | 289.6 | 406.3 |
| spatial-adagn-expB-depth | epoch_0080 | 0.6415 | 0.3921 | 0.6060 | 289.3 | 416.7 |
| spatial-adagn-expC-reg | epoch_0040 | 0.6459 | 0.3586 | 0.6115 | 289.1 | 401.3 |
| spatial-adagn-expC-reg | epoch_0080 | 0.6441 | 0.3909 | 0.6125 | 292.4 | 420.8 |
| spatial-adagn-nuclear | epoch_0100 | 0.6541 | 0.4133 | 0.6252 | 297.8 | 433.3 |

## 14. G Series - Gain/Identity Experiments

| Experiment | Epoch | Style | LPIPS | Style(p2a) | FID | ArtFID |
|-----------|------:|------:|------:|-----------:|----:|-------:|
| G0-Base-Gain0.5 | epoch_0040 | 0.6915 | 0.5344 | 0.6830 | ? | ? |
| G0-Base-Gain0.5 | epoch_0080 | 0.6918 | 0.5362 | 0.6779 | ? | ? |
| G0_Balanced_Base | epoch_0040 | 0.6670 | 0.4676 | 0.6524 | ? | ? |
| G0_Balanced_Base | epoch_0080 | 0.6703 | 0.4802 | ? | ? | ? |
| G1-Relax-ID | epoch_0040 | 0.6993 | 0.5642 | 0.6942 | ? | ? |

## 15. Micro Series - Micro-Scale Experiments

| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| micro01_hf2_lr1 | epoch_0020 | 0.6652 | 0.4952 |
| micro01_hf2_lr1 | epoch_0040 | 0.6690 | 0.4986 |
| micro01_hf2_lr1 | epoch_0060 | 0.6764 | 0.5164 |
| micro01_hf2_lr1 | epoch_0080 | 0.6822 | 0.5247 |
| micro02_macro_patch | epoch_0020 | 0.6584 | 0.4779 |
| micro02_macro_patch | epoch_0040 | 0.6774 | 0.5046 |
| micro02_macro_patch | epoch_0060 | 0.6828 | 0.5369 |
| micro02_macro_patch | epoch_0080 | 0.6872 | 0.5320 |
| micro03_gate75 | epoch_0020 | 0.6661 | 0.4941 |
| micro03_gate75 | epoch_0040 | 0.6710 | 0.5020 |
| micro03_gate75 | epoch_0060 | 0.6773 | 0.5173 |
| micro03_gate75 | epoch_0080 | 0.6822 | 0.5264 |
| micro04_hf1p5_macro | epoch_0020 | 0.6580 | 0.4715 |
| micro04_hf1p5_macro | epoch_0040 | 0.6744 | 0.4939 |
| micro04_hf1p5_macro | epoch_0060 | 0.6777 | 0.5245 |
| micro04_hf1p5_macro | epoch_0080 | 0.6817 | 0.5220 |
| micro05_id_anchor | epoch_0020 | 0.6649 | 0.4922 |
| micro05_id_anchor | epoch_0040 | 0.6687 | 0.4978 |
| micro05_id_anchor | epoch_0060 | 0.6760 | 0.5126 |
| micro05_id_anchor | epoch_0080 | 0.6815 | 0.5218 |

---

## 16. Additional Experiment Results (Continued)

### Dict Series
| Experiment | Epoch | Style | LPIPS | FID | ArtFID |
|-----------|------:|------:|------:|----:|-------:|
| dict | 50 | 0.6490 | 0.3947 | None | None |
| dict-50-0.05 | 50 | 0.6539 | 0.3652 | 289.8 | 403.1 |

*dict-50-0.05 has better metrics: lower LPIPS (0.365 vs 0.395), computed FID/ArtFID*

### Ablate M Series (Aggressive vs Smooth, 3 epochs each)
| Experiment | Epoch | Style | LPIPS | Style(p2a) |
|-----------|------:|------:|------:|-----------:|
| M1-Aggressive-Fine | 40 | 0.7019 | 0.5753 | 0.7006 |
| M1-Aggressive-Fine | 80 | 0.7055 | 0.5929 | 0.7086 |
| M1-Aggressive-Fine | 120 | **0.7087** | 0.6004 | **0.7152** |
| M2-Smooth-Impasto | 40 | 0.6980 | 0.5581 | 0.6942 |
| M2-Smooth-Impasto | 80 | 0.6984 | 0.5827 | 0.6975 |
| M2-Smooth-Impasto | 120 | 0.6992 | 0.5821 | 0.7041 |

*M1-Aggressive-Fine consistently outperforms M2-Smooth-Impasto. Style keeps improving through ep120!*

### Exp0-3 Series
| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| exp0-baseline | 40 | 0.6666 | 0.4763 |
| exp1-hf-ratio-4p0 | 40 | 0.6841 | 0.5223 |
| exp2-large-patches | 40 | 0.6719 | 0.5412 |
| exp3-hard-cdf | N/A | N/A | N/A |

### Final Demodulation
| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| final_demodulation | 40 | **0.7056** | 0.6997 |
| final_demodulation | 80 | 0.6866 | 0.7184 |

*Style peaked at ep40 then degraded by ep80 (0.706 -> 0.687). Classic overfitting.*

### NStyle-Project
| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| nstyle-proj | 50 | 0.6830 | 0.4859 |
| nstyle-proj | 100 | 0.6851 | 0.5097 |
| nstyle-proj | 150 | 0.6832 | 0.5065 |

*Style peaked at ep100 then slightly degraded. Ran for 150 epochs.*

### Other Quick Results
- patch-1-3-5 @ ep20: style=0.6458, lpips=0.3539
- swd-256-100-6-50-1.5k @ ep50-150: style=0.523-0.525 (poor), lpips=0.305-0.308
- swd-256-fix-gates @ ep50: style=0.5478, lpips=0.7744 (disastrous LPIPS)

---


--- APPENDED 2026-04-03 02:38 ---

## 17. No-* Series (Ablation experiments removing components)

### no-dict-hf-swd (No dictionary, HF-SWD enabled)
- ep40: style=0.697, lpips=0.576 — HIGH style but HIGH lpips (content damaged!)
- This shows that HF-SWD without dictionary modulation pushes style aggressively

### no-edge (Edge/High frequency removal)
- ep50: style=0.558, lpips=0.646
- ep100: style=0.565, lpips=0.657
- ep150: style=0.570, lpips=0.660
- Very LOW style (0.56 max) — removing edge processing kills style transfer
- LPIPS is terrible (0.66) — no edge means no structure preservation

### no-tv
- 1 eval, need to check data

## 18. SWD-256-100-6-50-1.5k (Extended training, 200 epochs)

| Epoch | Style | LPIPS |
|------:|------:|------:|
| 50 | 0.523 | 0.305 |
| 100 | 0.525 | 0.308 |
| 150 | 0.524 | 0.307 |
| 200 | 0.524 | 0.306 |

This experiment shows that with too much SWD weight (250 projections over 100 patches, 
with 1.5K samples), the model fails to learn style transfer (style stuck at 0.52).
But LPIPS is excellent (0.305-0.308) — the model just preserves content without changing it.
This is the "conservative trap" — optimizing too hard for SWD leads to no-style solutions.

## 19. Ablate A Series (Patch/ID/TV sweep, all at 60 epochs)

| Experiment | Style | LPIPS | Changes |
|-----------|------:|------:|---------|
| A0 (base p5, id045, tv005) | 0.684 | 0.500 | baseline |
| A1 (p7, id045, tv005) | 0.682 | 0.495 | Larger patch, similar style, better lpips |
| A2 (p11, id045, tv005) | 0.683 | 0.488 | Largest patch, best lpips (0.488) |
| A3 (p5, id030, tv005) | 0.685 | 0.511 | Lower identity, highest style (0.685), worse lpips |
| A4 (p5, id070, tv005) | 0.681 | 0.489 | Higher identity, lower style, better lpips |
| A5 (p5, id045, tv003) | 0.684 | 0.504 | Lower TV, similar results |

Key findings:
1. **Larger patches → better lpips**: p5 (0.500) → p7 (0.495) → p11 (0.488)
2. **Lower identity → higher style**: id=0.30 → style=0.685 (highest), but lpips=0.511 (worse)
3. **Higher identity → better lpips**: id=0.70 → lpips=0.489, but style=0.681 (lowest)
4. The sweet spot is around id=0.45 (the baseline) — balances style (0.684) and lpips (0.500)

## 20. Micro Series (micro01-05) - Full 4-epoch Trajectories

| Experiment | ep20 | ep40 | ep60 | ep80 | Trend |
|-----------|------:|------:|------:|------:|-------|
| micro01 (hf2_lr1) | 0.665/0.495 | 0.669/0.499 | 0.676/0.516 | **0.682/0.525** | Strong improvement |
| micro02 (macro_patch) | 0.658/0.478 | 0.677/0.505 | 0.683/0.537 | **0.687/0.532** | Strong improvement |
| micro03 (gate75) | 0.666/0.494 | 0.671/0.502 | 0.677/0.517 | **0.682/0.526** | Steady improvement |
| micro04 (hf1.5_macro) | 0.658/0.471 | 0.674/0.494 | 0.678/0.524 | **0.682/0.522** | Strong improvement |
| micro05 (id_anchor) | 0.665/0.492 | 0.669/0.498 | 0.676/0.513 | **0.681/0.522** | Steady improvement |

All 5 experiments improve consistently from ep20 to ep80.
- Best style at ep80: **micro02_macro_patch** (0.687)
- Best LPIPS at ep80: **micro04_hf1p5_macro** (0.522) and **micro05_id_anchor** (0.522)
- Classifier accuracy at ep80 ranges from 0.29 to 0.35

---


---

## 17. No-* Series (Ablation experiments removing components)

### no-dict-hf-swd (No dictionary + HF-SWD)
- ep40: style=0.697, lpips=0.576
- HIGH style (0.697) but HIGH lpips (0.576 - content damaged!)
- Removing dict + keeping HF-SWD pushes style aggressively

### no-edge (Edge/High freq removal)  
- ep50: 0.558/0.646
- ep100: 0.565/0.657
- ep150: 0.570/0.660
- **DISASTROUS** for style (0.56-0.57) — removing edge processing kills style transfer
- But LPIPS is high (0.66) — model preserves content by NOT changing it
- Ran 150 epochs — longest no-* experiment

## 18. SWD-256 Series (200 epochs, low SWD weight)
- 100 projections, 6 patches, 50 samples, 1.5K max
- **50→200 epoch trajectory**: style stays flat at 0.523-0.525
- LPIPS = 0.305-0.308 (excellent content preservation)
- This is the "NO STYLE" baseline — model preserves content but fails to stylize

## 19. Ablate A Series (Patch/ID/TV sweep)

All 6 experiments at epoch 60:

| Experiment | Patch | ID | TV | Style | LPIPS |
|-----------|------:|---:|---:|------:|------:|
| A0_base_p5_id045_tv005 | 5 | 0.45 | 0.05 | 0.684 | 0.500 |
| A1_p7_id045_tv005 | 7 | 0.45 | 0.05 | 0.682 | 0.495 |
| A2_p11_id045_tv005 | 11 | 0.45 | 0.05 | 0.683 | 0.488 |
| A3_p5_id030_tv005 | 5 | 0.30 | 0.05 | 0.685 | 0.511 |
| A4_p5_id070_tv005 | 5 | 0.70 | 0.05 | 0.681 | 0.489 |
| A5_p5_id045_tv003 | 5 | 0.45 | 0.03 | 0.684 | 0.504 |

Key findings:
1. **Larger patches → better LPIPS**: p5→p7→p11: 0.500→0.495→0.488
2. **Lower ID → higher style**: ID=0.30 gives 0.685 (highest), but lpips=0.511 (worst)
3. **Higher ID → better LPIPS**: ID=0.70 gives lpips=0.489, style=0.681
4. Sweet spot: ID=0.45 (baseline A0)

## 20. Micro01-05 Series - Full 80-epoch trajectories

| Experiment | ep20 | ep40 | ep60 | ep80 |
|-----------|------:|------:|------:|------:|
| micro01_hf2_lr1 | 0.665/0.495 | 0.669/0.500 | 0.676/0.516 | 0.682/0.525 |
| micro02_macro_patch | 0.658/0.478 | 0.677/0.505 | 0.683/0.537 | 0.687/0.532 |
| micro03_gate75 | 0.666/0.494 | 0.671/0.502 | 0.677/0.517 | 0.682/0.526 |
| micro04_hf1p5_macro | 0.658/0.471 | 0.674/0.494 | 0.678/0.524 | 0.682/0.522 |
| micro05_id_anchor | 0.665/0.492 | 0.669/0.498 | 0.676/0.513 | 0.681/0.522 |

All 5 experiments improve consistently from ep20→ep80 (Δstyle ~+0.02)
- Best final style: **micro02_macro_patch** (0.687)
- Best final LPIPS among high-style: **micro05_id_anchor** (0.522 at style=0.681)
- micro01-hf2 starts fast but micro02 overtakes by ep80


---

## 21. Clocor1 Series (Correlated Color, 5 experiments)

| Experiment | ep40 | lpips | p2a_style | ep80 | lpips | p2a |
|-----------|------:|------:|----------:|------:|------:|----:|
| E1_Macro19_Rigid | 0.676 | 0.477 | 0.651 | 0.680 | 0.499 | 0.665 |
| E2_15Series_Rigid | 0.672 | 0.468 | 0.654 | 0.678 | 0.499 | 0.669 |
| E3_15Series_Soft | 0.676 | 0.481 | 0.660 | **0.681** | 0.509 | **0.673** |
| E4_9Series_Rigid | 0.672 | 0.480 | 0.661 | 0.676 | 0.483 | 0.668 |
| E5_9Series_Soft | 0.659 | 0.499 | 0.668 | 0.679 | 0.493 | 0.674 |

Key: E3_15Series_Soft and E5_9Series_Soft both hit ~0.681 and 0.673-0.674 p2a at ep80.
Soft schedules outperform Rigid for this series.

## 22. Mainline Exp Series (1-6, G1, G2)

| Experiment | Epoch | Style | LPIPS | P2A_Style |
|-----------|------:|------:|------:|----------:|
| exp_1_control | 100 | 0.646 | 0.404 | 0.615 |
| exp_2_zero_id | 100 | 0.647 | 0.419 | **0.626** |
| exp_3_macro_strokes | 100 | 0.645 | **0.388** | 0.611 |
| exp_4_zero_tv | 100 | 0.646 | 0.406 | 0.614 |
| exp_5_signal_overdrive | 100 | 0.650 | 0.424 | 0.622 |
| exp_6_nuclear | N/A | N/A | N/A | N/A |
| **exp_G1_edge_rush** | 60 | 0.658 | 0.404 | 0.625 |
| exp_G1_edge_rush | 120 | 0.661 | 0.416 | 0.628 |
| **exp_G2_dense_pyramid** | 60 | **0.665** | **0.394** | **0.644** |

**CRITICAL FINDING**: exp_G2_dense_pyramid is the best mainline experiment!
- style=0.665, LPIPS=0.394, P2A=0.644 at just 60 epochs
- exp_3_macro_strokes has the best LPIPS (0.388) but lower style (0.645)
- zero_id doesn't help: style=0.647, worse LPIPS (0.419) vs control (0.404)
- signal_overdrive (0.650) slightly better than control (0.646)
- G2_dense_pyramid dominates all others — it's the undisputed mainline champion

## 23. 1-decoder series

| Experiment | Epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| 1-decoder-no_norm-patch5_23-color1.0 | 30 | 0.666 | 0.446 |
| 1-decoder-patch5-15 | 30 | 0.676 | 0.476 |

Early experiments with single decoder block. Patch5-15 gets better style.

## 24. Final Abate E1/A6/M3

- ablate_E1_Macro19_Rigid_LR14e4: ep40=0.669/0.454, ep80=0.673/0.476
- ablate_A6_p5_id045_tv008: no eval data
- ablate_E2_15Series_Rigid_LR14e4: no eval data
- ablate_M3-Macro-Flowing: no eval data

## 25. Cross-Domain Eval Results (KEY_EXPERIMENT CSV)

From the KEY_EXPERIMENT_PHOTO_TO_ART_COMPARISON CSV, the cross-domain (photo→art) results:

| Experiment | p2a_clip_dir | p2a_clip_style | p2a_clip_content | p2a_LPIPS | p2a_FID | p2a_art_fid |
|-----------|-------------:|---------------:|-----------------:|----------:|--------:|------------:|
| abl_naive_skip | 0.504 | 0.611 | 0.644 | 0.621 | 350.5 | 420.4 |
| abl_no_adagn | 0.506 | 0.700 | 0.697 | 0.566 | 321.2 | 419.1 |
| abl_no_hf_swd | 0.501 | 0.784 | 0.501 | 0.476 | 295.6 | 334.0 |
| abl_no_residual | 0.302 | 0.906 | 0.579 | 0.295 | 298.8 | 311.2 |
| exp_1_control | 0.416 | 0.842 | 0.615 | 0.439 | 295.6 | 318.9 |
| exp_3_macro_strokes | 0.401 | 0.856 | 0.611 | 0.427 | 294.7 | 310.1 |
| exp_G1_edge_rush | 0.440 | 0.830 | 0.625 | 0.442 | 291.5 | 314.3 |

**Cross-domain insight**: The "abl_no_residual" experiment gets p2a_clip_content=0.906 (highest!)
but only p2a_clip_style=0.579 (lowest). It's the "lazy no-change" solution.
Meanwhile abl_no_adagn gets p2a_clip_style=0.700 (highest) but p2a_clip_content=0.697 and art_fid=419 (worst).

The tradeoff is clear: style vs content vs distribution quality.

---

## 26. Complete Performance Leaderboard (All experiments, best results)

### By Style (p2a, highest first):
1. ablate_M1-Aggressive-Fine ep120: **0.715** (style=0.709, lpips=0.600)
2. scan05_gate_0p75 ep80: 0.701 (style=0.702, lpips=0.570)
3. G1-Relax-ID ep40: 0.699 (style=0.699, lpips=0.564)
4. decoder-D6_god_combo ep40 (tokenized): 0.700 (style=0.700, lpips=0.546)
5. scan04_gate_0p85 ep80: 0.696 (style=0.700, lpips=0.570)
6. scan01_base ep80: 0.700 (style=0.700, lpips=0.569)

### By LPIPS (lowest first, content preservation):
1. swd-256-100-6-50-1.5k ep50: 0.305 (but style=0.523)
2. patch-1-3-5 ep20: 0.354 (style=0.646)
3. spatial-adagn-expB-depth ep40: 0.363 (style=0.646)
4. spatial-adagn-expA-texture ep40: 0.363 (style=0.645)
5. adain-spacial-adagn ep100: 0.376 (style=0.650)
6. dict-50-0.05 ep50: 0.365 (style=0.654)
7. exp_3_macro_strokes: 0.388 (style=0.611)

### By FID (lowest first):
1. clocor1_E4_9Series_Rigid ep80: 289.3
2. spatial-adagn ep40: 285.6
3. spatial-adagn-expA-texture ep40: 289.5
4. spatial-adagn-expB-depth ep40: 289.6
5. spatial-adagn-expC-reg ep40: 289.1
6. dict-50-0.05: 289.8
7. exp_G1_edge_rush: 291.5

### By ArtFID (lowest first):
1. swd-256-100-6-50-1.5k: N/A (no style)
2. D5_rank_8 ep40 (tokenized): 466.2
3. D4_gate_0p8 ep40 (tokenized): 478.4
4. D2_color_1p5 ep40 (tokenized): 476.8
5. abl_no_residual: 311.2 (but this is the "no-change" solution)
6. exp_3_macro_strokes: 310.1
7. exp_1_control: 318.9

---


---

## 27. Ablation-Result Deep Dive (ablation-result/ directory)

This directory contains 750-row per-pair evaluation data for 6 A-series experiments.
Each experiment has: metrics.csv (750 rows), summary.json, summary_history.json, summary_grid.png
50 source images x 5 target styles x 3 pairs? = 750 rows.

### A-Series Aggregate Results (full per-pair eval, 750 samples each)

From summary_history.json:
| Experiment | epoch | Style | LPIPS |
|-----------|------:|------:|------:|
| A0_base_p5_id045_tv005 | 60 | 0.690 | 0.494 |
| A1_p7_id045_tv005 | 60 | 0.689 | 0.490 |
| A2_p11_id045_tv005 | 60 | **0.689** | **0.484** |
| A3_p5_id030_tv005 | 60 | **0.691** | 0.505 |
| A4_p5_id070_tv005 | 60 | 0.688 | 0.483 |
| A5_p5_id045_tv003 | 60 | 0.690 | 0.498 |

Photo→Art aggregate (120 samples per experiment: 5 styles x 24 photo images):
| Experiment | p2a_style | p2a_lpips |
|-----------|----------:|----------:|
| A0_base_p5_id045_tv005 | **0.679** | 0.543 |
| A1_p7_id045_tv005 | 0.675 | **0.526** |
| A2_p11_id045_tv005 | 0.680 | 0.527 |

### Per-Style-Pair Detail (A0_baseline)
| Style Pair | Style | LPips | Content | Dir |
|-----------|------:|------:|--------:|----:|
| photo→Hayao | 0.665 | 0.579 | 0.692 | 0.608 |
| photo→cezanne | 0.666 | 0.542 | 0.717 | 0.559 |
| photo→monet | 0.676 | 0.496 | 0.745 | 0.533 |
| photo→vangogh | 0.710 | 0.553 | 0.728 | 0.595 |
| photo→photo | 0.790 | 0.486 | 0.738 | 0.552 |
| Hayao→Hayao | 0.824 | 0.381 | 0.850 | 0.424 |
| monet→monet | 0.812 | 0.473 | 0.804 | 0.438 |
| vangogh→vangogh | 0.794 | 0.442 | 0.812 | 0.401 |

**Key finding from 750 per-pair evaluations**:
- Within-style transfers (same style) score very high: style=0.77-0.82, lpips=0.38-0.47
- Cross-style art→art transfers score medium: style=0.65-0.80, lpips=0.44-0.56
- Photo→art transfers score lowest: style=0.665-0.710, lpips=0.496-0.579
- The model is MUCH better at stylizing already-stylized images than raw photos
- Monet is easiest to transfer (0.496 lpips, 0.676 style), Vangogh is hardest (0.553 lpips, 0.710 style)
- Cezanne is in between (0.542 lpips, 0.666 style)

---

## 28. Final Demodulation Experiment
- ep40: style=0.706, lpips=0.700, p2a_style=0.708
- ep80: style=0.687, lpips=0.718, p2a_style=0.692
- **Classic overfitting**: style peaked at ep40 then dropped. LPIPS got worse (0.700→0.718)
- This was the HIGHEST style reached at ep40 (0.706) but with terrible content preservation

---

## 29. Final Leaderboard - Best Per Category (updated with all data)

### Best Style by Experiment Type:
1. final_demodulation @ ep40: **0.708** p2a style (but lpips=0.700 - terrible)
2. ablate_M1-Aggressive-Fine @ ep120: 0.715 p2a style, lpips=0.600
3. scan05_gate_0p75 @ ep80: 0.701 p2a, lpips=0.570
4. decoder-D6_god_combo @ ep40: 0.700 p2a (tokenized), lpips=0.546
5. G1-Relax-ID @ ep40: 0.699 p2a (style), lpips=0.564

### Best Balance (style > 0.67, lpips < 0.55):
1. **exp_G2_dense_pyramid** @ ep60: 0.644 p2a, **0.394** LPIPS ← best balance
2. exp_3_macro_strokes @ ep100: 0.611 p2a, **0.388** LPIPS ← best LPIPS
3. decoder-D5_rank_8 @ ep40: 0.689 p2a, 0.503 LPIPS
4. ablate_A2_p11 @ ep60: 0.680 p2a, 0.484 LPIPS (aggregate)
5. ablate_A4_p5_id070 @ ep60: 0.688 p2a, 0.483 LPIPS

### Worst Experiments:
1. swd-256-100-6-50-1.5k: style stuck at 0.524 (200 epochs, no style learning)
2. swd-256-fix-gates: style=0.548, lpips=0.774 (catastrophic)
3. no-edge: style=0.570, lpips=0.660 (removing edge kills everything)
4. abl_naive_skip: style=0.685, lpips=0.621, artfid=4204 (uncontrolled skip)

---


---

## 30. Exp S1-S10 Series (Extreme/Special Experiments)

| Experiment | Epoch | Style | LPIPS | Notes |
|-----------|------:|------:|------:|-------|
| S1_zero_id | 150 | **0.710** | 0.626 | Zero identity loss, high style but bad lpips |
| S2_color_blind | 150 | 0.694 | 0.549 | Color blind experiment |
| S3-S10 | N/A | N/A | N/A | No eval data (logs only) |

**S1_zero_id** is notable: running 150 epochs with id=0 gives style=0.710 (highest ever!)
but LPIPS=0.626 (terrible content preservation). Classic style-vs-content tradeoff.

## 31. Fewshot Ukiyo-e Pipeline

Based at: `Y:\experiments\fewshot_runs\fewshot_ukiyo_e_sid4/`

- Base checkpoint: nce-swd_0.25-cl_0.01/epoch_0120.pt
- Tokenizer distilled to predict ukiyo-e style
- Replaced style_id=4 (cezanne) with ukiyo-e style vectors
- 200 refinement steps for style vector optimization
- This is a PROOF-OF-CONCEPT for adding new art styles without full training

## 32. Grid/ Directory

- 41 summary grid PNG images (105 MB total)
- Each is a 5x5 visual comparison grid showing style transfer results
- Generated for every experiment with full_eval
- Covers: ablate A0-A5, exp 1-5, G1 edge_rush (3 epochs), S1 (zero_id)

## 33. Summary/ Directory - Additional Experiment Results


### 1swd-dit-2style.json (3KB)
```
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.42491398870944974
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.30042737821117044
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.3064216676
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.6074276959896088
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.5600687462091446
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.33952212699999995
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.42491398870944974
  best.best_transfer_classifier_acc.transfer_clip_style: 0.30042737821117044
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.3064216676
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.606044539809227
  best.best_transfer_clip_style.transfer_clip_style: 0.5632539981603623
  best.best_transfer_clip_style.transfer_content_lpips: 0.3454081393
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.42491398870944974
  best.best_transfer_content_lpips.transfer_clip_style: 0.30042737821117044
  best.best_transfer_content_lpips.transfer_content_lpips: 0.3064216676
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_style: 0.606044539809227
  latest.transfer_clip_style: 0.5632539981603623
  latest.transfer_content_lpips: 0.3454081393
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_style: 0.5461287415027618
  mean.transfer_clip_style: 0.47458337419355906
  mean.transfer_content_lpips: 0.33045064463333335
```

### 5style-any2any-skipfusion.json (3KB)
```
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.32918674126267433
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.2886177541129291
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.32021350244999996
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.32918674126267433
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.2886177541129291
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.32021350244999996
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.32918674126267433
  best.best_transfer_classifier_acc.transfer_clip_style: 0.2886177541129291
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.32021350244999996
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.32918674126267433
  best.best_transfer_clip_style.transfer_clip_style: 0.2886177541129291
  best.best_transfer_clip_style.transfer_content_lpips: 0.32021350244999996
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.32918674126267433
  best.best_transfer_content_lpips.transfer_clip_style: 0.2886177541129291
  best.best_transfer_content_lpips.transfer_content_lpips: 0.32021350244999996
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_style: 0.32918674126267433
  latest.transfer_clip_style: 0.2886177541129291
  latest.transfer_content_lpips: 0.32021350244999996
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_style: 0.32918674126267433
  mean.transfer_clip_style: 0.2886177541129291
  mean.transfer_content_lpips: 0.32021350244999996
```

### full-8-8monet.json (7KB)
```
  best.best_photo_to_art_art_fid.photo_to_art_art_fid: 355.4008593718308
  best.best_photo_to_art_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_art_fid.photo_to_art_clip_dir: 0.2972734020277858
  best.best_photo_to_art_art_fid.photo_to_art_clip_style: 0.2972734020277858
  best.best_photo_to_art_art_fid.transfer_art_fid: 366.18761418502436
  best.best_photo_to_art_art_fid.transfer_clip_dir: 0.2799729294022545
  best.best_photo_to_art_art_fid.transfer_clip_style: 0.2799729294022545
  best.best_photo_to_art_art_fid.transfer_content_lpips: 0.2952822625099999
  best.best_photo_to_art_classifier_acc.photo_to_art_art_fid: 355.48429349740104
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_dir: 0.2985423938184977
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.2985423938184977
  best.best_photo_to_art_classifier_acc.transfer_art_fid: 367.1936811435481
  best.best_photo_to_art_classifier_acc.transfer_clip_dir: 0.2797787524983287
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.2797787524983287
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.29528498510000006
  best.best_photo_to_art_clip_dir.photo_to_art_art_fid: 355.48429349740104
  best.best_photo_to_art_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_dir.photo_to_art_clip_dir: 0.2985423938184977
  best.best_photo_to_art_clip_dir.photo_to_art_clip_style: 0.2985423938184977
  best.best_photo_to_art_clip_dir.transfer_art_fid: 367.1936811435481
  best.best_photo_to_art_clip_dir.transfer_clip_dir: 0.2797787524983287
  best.best_photo_to_art_clip_dir.transfer_clip_style: 0.2797787524983287
  best.best_photo_to_art_clip_dir.transfer_content_lpips: 0.29528498510000006
  best.best_photo_to_art_clip_style.photo_to_art_art_fid: 355.48429349740104
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_dir: 0.2985423938184977
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.2985423938184977
  best.best_photo_to_art_clip_style.transfer_art_fid: 367.1936811435481
  best.best_photo_to_art_clip_style.transfer_clip_dir: 0.2797787524983287
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.2797787524983287
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.29528498510000006
  best.best_transfer_art_fid.photo_to_art_art_fid: 355.4008593718308
  best.best_transfer_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_art_fid.photo_to_art_clip_dir: 0.2972734020277858
  best.best_transfer_art_fid.photo_to_art_clip_style: 0.2972734020277858
  best.best_transfer_art_fid.transfer_art_fid: 366.18761418502436
  best.best_transfer_art_fid.transfer_clip_dir: 0.2799729294022545
  best.best_transfer_art_fid.transfer_clip_style: 0.2799729294022545
  best.best_transfer_art_fid.transfer_content_lpips: 0.2952822625099999
  best.best_transfer_classifier_acc.photo_to_art_art_fid: 355.48429349740104
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_dir: 0.2985423938184977
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.2985423938184977
  best.best_transfer_classifier_acc.transfer_art_fid: 367.1936811435481
  best.best_transfer_classifier_acc.transfer_clip_dir: 0.2797787524983287
  best.best_transfer_classifier_acc.transfer_clip_style: 0.2797787524983287
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.29528498510000006
  best.best_transfer_clip_dir.photo_to_art_art_fid: 355.4008593718308
  best.best_transfer_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_dir.photo_to_art_clip_dir: 0.2972734020277858
  best.best_transfer_clip_dir.photo_to_art_clip_style: 0.2972734020277858
  best.best_transfer_clip_dir.transfer_art_fid: 366.18761418502436
  best.best_transfer_clip_dir.transfer_clip_dir: 0.2799729294022545
  best.best_transfer_clip_dir.transfer_clip_style: 0.2799729294022545
  best.best_transfer_clip_dir.transfer_content_lpips: 0.2952822625099999
  best.best_transfer_clip_style.photo_to_art_art_fid: 355.4008593718308
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_dir: 0.2972734020277858
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.2972734020277858
  best.best_transfer_clip_style.transfer_art_fid: 366.18761418502436
  best.best_transfer_clip_style.transfer_clip_dir: 0.2799729294022545
  best.best_transfer_clip_style.transfer_clip_style: 0.2799729294022545
  best.best_transfer_clip_style.transfer_content_lpips: 0.2952822625099999
  best.best_transfer_content_lpips.photo_to_art_art_fid: 355.4008593718308
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_dir: 0.2972734020277858
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.2972734020277858
  best.best_transfer_content_lpips.transfer_art_fid: 366.18761418502436
  best.best_transfer_content_lpips.transfer_clip_dir: 0.2799729294022545
  best.best_transfer_content_lpips.transfer_clip_style: 0.2799729294022545
  best.best_transfer_content_lpips.transfer_content_lpips: 0.2952822625099999
  latest.photo_to_art_art_fid: 355.4008593718308
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_dir: 0.2972734020277858
  latest.photo_to_art_clip_style: 0.2972734020277858
  latest.transfer_art_fid: 366.18761418502436
  latest.transfer_clip_dir: 0.2799729294022545
  latest.transfer_clip_style: 0.2799729294022545
  latest.transfer_content_lpips: 0.2952822625099999
  mean.photo_to_art_art_fid: 355.44257643461594
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_dir: 0.29790789792314176
  mean.photo_to_art_clip_style: 0.29790789792314176
  mean.transfer_art_fid: 366.6906476642862
  mean.transfer_clip_dir: 0.2798758409502916
  mean.transfer_clip_style: 0.2798758409502916
  mean.transfer_content_lpips: 0.29528362380499995
```

### full-swd.json (7KB)
```
  best.best_photo_to_art_art_fid.photo_to_art_art_fid: 390.9748562450438
  best.best_photo_to_art_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_art_fid.photo_to_art_clip_dir: 0.4214339104294777
  best.best_photo_to_art_art_fid.photo_to_art_clip_style: 0.4214339104294777
  best.best_photo_to_art_art_fid.transfer_art_fid: 405.80444064649845
  best.best_photo_to_art_art_fid.transfer_clip_dir: 0.3699147640764714
  best.best_photo_to_art_art_fid.transfer_clip_style: 0.3699147640764714
  best.best_photo_to_art_art_fid.transfer_content_lpips: 0.39336325674
  best.best_photo_to_art_classifier_acc.photo_to_art_art_fid: 390.9748562450438
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_dir: 0.4214339104294777
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.4214339104294777
  best.best_photo_to_art_classifier_acc.transfer_art_fid: 405.80444064649845
  best.best_photo_to_art_classifier_acc.transfer_clip_dir: 0.3699147640764714
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.3699147640764714
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.39336325674
  best.best_photo_to_art_clip_dir.photo_to_art_art_fid: 409.94164934436134
  best.best_photo_to_art_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_dir.photo_to_art_clip_dir: 0.45242164000868795
  best.best_photo_to_art_clip_dir.photo_to_art_clip_style: 0.45242164000868795
  best.best_photo_to_art_clip_dir.transfer_art_fid: 419.9383947870027
  best.best_photo_to_art_clip_dir.transfer_clip_dir: 0.40281614019349216
  best.best_photo_to_art_clip_dir.transfer_clip_style: 0.40281614019349216
  best.best_photo_to_art_clip_dir.transfer_content_lpips: 0.42399943004999996
  best.best_photo_to_art_clip_style.photo_to_art_art_fid: 409.94164934436134
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_dir: 0.45242164000868795
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.45242164000868795
  best.best_photo_to_art_clip_style.transfer_art_fid: 419.9383947870027
  best.best_photo_to_art_clip_style.transfer_clip_dir: 0.40281614019349216
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.40281614019349216
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.42399943004999996
  best.best_transfer_art_fid.photo_to_art_art_fid: 390.9748562450438
  best.best_transfer_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_art_fid.photo_to_art_clip_dir: 0.4214339104294777
  best.best_transfer_art_fid.photo_to_art_clip_style: 0.4214339104294777
  best.best_transfer_art_fid.transfer_art_fid: 405.80444064649845
  best.best_transfer_art_fid.transfer_clip_dir: 0.3699147640764714
  best.best_transfer_art_fid.transfer_clip_style: 0.3699147640764714
  best.best_transfer_art_fid.transfer_content_lpips: 0.39336325674
  best.best_transfer_classifier_acc.photo_to_art_art_fid: 390.9748562450438
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_dir: 0.4214339104294777
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.4214339104294777
  best.best_transfer_classifier_acc.transfer_art_fid: 405.80444064649845
  best.best_transfer_classifier_acc.transfer_clip_dir: 0.3699147640764714
  best.best_transfer_classifier_acc.transfer_clip_style: 0.3699147640764714
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.39336325674
  best.best_transfer_clip_dir.photo_to_art_art_fid: 409.94164934436134
  best.best_transfer_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_dir.photo_to_art_clip_dir: 0.45242164000868795
  best.best_transfer_clip_dir.photo_to_art_clip_style: 0.45242164000868795
  best.best_transfer_clip_dir.transfer_art_fid: 419.9383947870027
  best.best_transfer_clip_dir.transfer_clip_dir: 0.40281614019349216
  best.best_transfer_clip_dir.transfer_clip_style: 0.40281614019349216
  best.best_transfer_clip_dir.transfer_content_lpips: 0.42399943004999996
  best.best_transfer_clip_style.photo_to_art_art_fid: 409.94164934436134
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_dir: 0.45242164000868795
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.45242164000868795
  best.best_transfer_clip_style.transfer_art_fid: 419.9383947870027
  best.best_transfer_clip_style.transfer_clip_dir: 0.40281614019349216
  best.best_transfer_clip_style.transfer_clip_style: 0.40281614019349216
  best.best_transfer_clip_style.transfer_content_lpips: 0.42399943004999996
  best.best_transfer_content_lpips.photo_to_art_art_fid: 390.9748562450438
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_dir: 0.4214339104294777
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.4214339104294777
  best.best_transfer_content_lpips.transfer_art_fid: 405.80444064649845
  best.best_transfer_content_lpips.transfer_clip_dir: 0.3699147640764714
  best.best_transfer_content_lpips.transfer_clip_style: 0.3699147640764714
  best.best_transfer_content_lpips.transfer_content_lpips: 0.39336325674
  latest.photo_to_art_art_fid: 409.94164934436134
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_dir: 0.45242164000868795
  latest.photo_to_art_clip_style: 0.45242164000868795
  latest.transfer_art_fid: 419.9383947870027
  latest.transfer_clip_dir: 0.40281614019349216
  latest.transfer_clip_style: 0.40281614019349216
  latest.transfer_content_lpips: 0.42399943004999996
  mean.photo_to_art_art_fid: 400.45825279470256
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_dir: 0.4369277752190828
  mean.photo_to_art_clip_style: 0.4369277752190828
  mean.transfer_art_fid: 412.87141771675056
  mean.transfer_clip_dir: 0.38636545213498175
  mean.transfer_clip_style: 0.38636545213498175
  mean.transfer_content_lpips: 0.40868134339499995
```

### full-v2.json (6KB)
```
  best.best_photo_to_art_art_fid.photo_to_art_art_fid: 417.86683672785546
  best.best_photo_to_art_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_art_fid.photo_to_art_clip_dir: 0.43719678044319155
  best.best_photo_to_art_art_fid.photo_to_art_clip_style: 0.43719678044319155
  best.best_photo_to_art_art_fid.transfer_art_fid: 413.38421353514184
  best.best_photo_to_art_art_fid.transfer_clip_dir: 0.29477586040273307
  best.best_photo_to_art_art_fid.transfer_clip_style: 0.29477586040273307
  best.best_photo_to_art_art_fid.transfer_content_lpips: 0.2928361565
  best.best_photo_to_art_classifier_acc.photo_to_art_art_fid: 417.86683672785546
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_dir: 0.43719678044319155
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.43719678044319155
  best.best_photo_to_art_classifier_acc.transfer_art_fid: 413.38421353514184
  best.best_photo_to_art_classifier_acc.transfer_clip_dir: 0.29477586040273307
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.29477586040273307
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.2928361565
  best.best_photo_to_art_clip_dir.photo_to_art_art_fid: 417.86683672785546
  best.best_photo_to_art_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_dir.photo_to_art_clip_dir: 0.43719678044319155
  best.best_photo_to_art_clip_dir.photo_to_art_clip_style: 0.43719678044319155
  best.best_photo_to_art_clip_dir.transfer_art_fid: 413.38421353514184
  best.best_photo_to_art_clip_dir.transfer_clip_dir: 0.29477586040273307
  best.best_photo_to_art_clip_dir.transfer_clip_style: 0.29477586040273307
  best.best_photo_to_art_clip_dir.transfer_content_lpips: 0.2928361565
  best.best_photo_to_art_clip_style.photo_to_art_art_fid: 417.86683672785546
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_dir: 0.43719678044319155
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.43719678044319155
  best.best_photo_to_art_clip_style.transfer_art_fid: 413.38421353514184
  best.best_photo_to_art_clip_style.transfer_clip_dir: 0.29477586040273307
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.29477586040273307
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.2928361565
  best.best_transfer_art_fid.photo_to_art_art_fid: 417.86683672785546
  best.best_transfer_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_art_fid.photo_to_art_clip_dir: 0.43719678044319155
  best.best_transfer_art_fid.photo_to_art_clip_style: 0.43719678044319155
  best.best_transfer_art_fid.transfer_art_fid: 413.38421353514184
  best.best_transfer_art_fid.transfer_clip_dir: 0.29477586040273307
  best.best_transfer_art_fid.transfer_clip_style: 0.29477586040273307
  best.best_transfer_art_fid.transfer_content_lpips: 0.2928361565
  best.best_transfer_classifier_acc.photo_to_art_art_fid: 417.86683672785546
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_dir: 0.43719678044319155
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.43719678044319155
  best.best_transfer_classifier_acc.transfer_art_fid: 413.38421353514184
  best.best_transfer_classifier_acc.transfer_clip_dir: 0.29477586040273307
  best.best_transfer_classifier_acc.transfer_clip_style: 0.29477586040273307
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.2928361565
  best.best_transfer_clip_dir.photo_to_art_art_fid: 417.86683672785546
  best.best_transfer_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_dir.photo_to_art_clip_dir: 0.43719678044319155
  best.best_transfer_clip_dir.photo_to_art_clip_style: 0.43719678044319155
  best.best_transfer_clip_dir.transfer_art_fid: 413.38421353514184
  best.best_transfer_clip_dir.transfer_clip_dir: 0.29477586040273307
  best.best_transfer_clip_dir.transfer_clip_style: 0.29477586040273307
  best.best_transfer_clip_dir.transfer_content_lpips: 0.2928361565
  best.best_transfer_clip_style.photo_to_art_art_fid: 417.86683672785546
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_dir: 0.43719678044319155
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.43719678044319155
  best.best_transfer_clip_style.transfer_art_fid: 413.38421353514184
  best.best_transfer_clip_style.transfer_clip_dir: 0.29477586040273307
  best.best_transfer_clip_style.transfer_clip_style: 0.29477586040273307
  best.best_transfer_clip_style.transfer_content_lpips: 0.2928361565
  best.best_transfer_content_lpips.photo_to_art_art_fid: 417.86683672785546
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_dir: 0.43719678044319155
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.43719678044319155
  best.best_transfer_content_lpips.transfer_art_fid: 413.38421353514184
  best.best_transfer_content_lpips.transfer_clip_dir: 0.29477586040273307
  best.best_transfer_content_lpips.transfer_clip_style: 0.29477586040273307
  best.best_transfer_content_lpips.transfer_content_lpips: 0.2928361565
  latest.photo_to_art_art_fid: 417.86683672785546
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_dir: 0.43719678044319155
  latest.photo_to_art_clip_style: 0.43719678044319155
  latest.transfer_art_fid: 413.38421353514184
  latest.transfer_clip_dir: 0.29477586040273307
  latest.transfer_clip_style: 0.29477586040273307
  latest.transfer_content_lpips: 0.2928361565
  mean.photo_to_art_art_fid: 417.86683672785546
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_dir: 0.43719678044319155
  mean.photo_to_art_clip_style: 0.43719678044319155
  mean.transfer_art_fid: 413.38421353514184
  mean.transfer_clip_dir: 0.29477586040273307
  mean.transfer_clip_style: 0.29477586040273307
  mean.transfer_content_lpips: 0.2928361565
```

### full-v2-20.json (6KB)
```
  best.best_photo_to_art_art_fid.photo_to_art_art_fid: 452.3703342214392
  best.best_photo_to_art_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_art_fid.photo_to_art_clip_dir: 0.48923841178417204
  best.best_photo_to_art_art_fid.photo_to_art_clip_style: 0.48923841178417204
  best.best_photo_to_art_art_fid.transfer_art_fid: 443.2342564299579
  best.best_photo_to_art_art_fid.transfer_clip_dir: 0.36910080537199974
  best.best_photo_to_art_art_fid.transfer_clip_style: 0.36910080537199974
  best.best_photo_to_art_art_fid.transfer_content_lpips: 0.3749716727
  best.best_photo_to_art_classifier_acc.photo_to_art_art_fid: 452.3703342214392
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_dir: 0.48923841178417204
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.48923841178417204
  best.best_photo_to_art_classifier_acc.transfer_art_fid: 443.2342564299579
  best.best_photo_to_art_classifier_acc.transfer_clip_dir: 0.36910080537199974
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.36910080537199974
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.3749716727
  best.best_photo_to_art_clip_dir.photo_to_art_art_fid: 452.3703342214392
  best.best_photo_to_art_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_dir.photo_to_art_clip_dir: 0.48923841178417204
  best.best_photo_to_art_clip_dir.photo_to_art_clip_style: 0.48923841178417204
  best.best_photo_to_art_clip_dir.transfer_art_fid: 443.2342564299579
  best.best_photo_to_art_clip_dir.transfer_clip_dir: 0.36910080537199974
  best.best_photo_to_art_clip_dir.transfer_clip_style: 0.36910080537199974
  best.best_photo_to_art_clip_dir.transfer_content_lpips: 0.3749716727
  best.best_photo_to_art_clip_style.photo_to_art_art_fid: 452.3703342214392
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_dir: 0.48923841178417204
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.48923841178417204
  best.best_photo_to_art_clip_style.transfer_art_fid: 443.2342564299579
  best.best_photo_to_art_clip_style.transfer_clip_dir: 0.36910080537199974
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.36910080537199974
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.3749716727
  best.best_transfer_art_fid.photo_to_art_art_fid: 452.3703342214392
  best.best_transfer_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_art_fid.photo_to_art_clip_dir: 0.48923841178417204
  best.best_transfer_art_fid.photo_to_art_clip_style: 0.48923841178417204
  best.best_transfer_art_fid.transfer_art_fid: 443.2342564299579
  best.best_transfer_art_fid.transfer_clip_dir: 0.36910080537199974
  best.best_transfer_art_fid.transfer_clip_style: 0.36910080537199974
  best.best_transfer_art_fid.transfer_content_lpips: 0.3749716727
  best.best_transfer_classifier_acc.photo_to_art_art_fid: 452.3703342214392
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_dir: 0.48923841178417204
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.48923841178417204
  best.best_transfer_classifier_acc.transfer_art_fid: 443.2342564299579
  best.best_transfer_classifier_acc.transfer_clip_dir: 0.36910080537199974
  best.best_transfer_classifier_acc.transfer_clip_style: 0.36910080537199974
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.3749716727
  best.best_transfer_clip_dir.photo_to_art_art_fid: 452.3703342214392
  best.best_transfer_clip_dir.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_dir.photo_to_art_clip_dir: 0.48923841178417204
  best.best_transfer_clip_dir.photo_to_art_clip_style: 0.48923841178417204
  best.best_transfer_clip_dir.transfer_art_fid: 443.2342564299579
  best.best_transfer_clip_dir.transfer_clip_dir: 0.36910080537199974
  best.best_transfer_clip_dir.transfer_clip_style: 0.36910080537199974
  best.best_transfer_clip_dir.transfer_content_lpips: 0.3749716727
  best.best_transfer_clip_style.photo_to_art_art_fid: 452.3703342214392
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_dir: 0.48923841178417204
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.48923841178417204
  best.best_transfer_clip_style.transfer_art_fid: 443.2342564299579
  best.best_transfer_clip_style.transfer_clip_dir: 0.36910080537199974
  best.best_transfer_clip_style.transfer_clip_style: 0.36910080537199974
  best.best_transfer_clip_style.transfer_content_lpips: 0.3749716727
  best.best_transfer_content_lpips.photo_to_art_art_fid: 452.3703342214392
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_dir: 0.48923841178417204
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.48923841178417204
  best.best_transfer_content_lpips.transfer_art_fid: 443.2342564299579
  best.best_transfer_content_lpips.transfer_clip_dir: 0.36910080537199974
  best.best_transfer_content_lpips.transfer_clip_style: 0.36910080537199974
  best.best_transfer_content_lpips.transfer_content_lpips: 0.3749716727
  latest.photo_to_art_art_fid: 452.3703342214392
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_dir: 0.48923841178417204
  latest.photo_to_art_clip_style: 0.48923841178417204
  latest.transfer_art_fid: 443.2342564299579
  latest.transfer_clip_dir: 0.36910080537199974
  latest.transfer_clip_style: 0.36910080537199974
  latest.transfer_content_lpips: 0.3749716727
  mean.photo_to_art_art_fid: 452.3703342214392
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_dir: 0.48923841178417204
  mean.photo_to_art_clip_style: 0.48923841178417204
  mean.transfer_art_fid: 443.2342564299579
  mean.transfer_clip_dir: 0.36910080537199974
  mean.transfer_clip_style: 0.36910080537199974
  mean.transfer_content_lpips: 0.3749716727
```

### patch-1-3-5.json (6KB)
```
  best.best_photo_to_art_art_fid.photo_to_art_art_fid: 398.8193577754396
  best.best_photo_to_art_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_art_fid.photo_to_art_clip_style: 0.6121498428285121
  best.best_photo_to_art_art_fid.photo_to_art_fid: 287.55510336759266
  best.best_photo_to_art_art_fid.transfer_art_fid: 398.8193577754396
  best.best_photo_to_art_art_fid.transfer_clip_style: 0.6457651415467262
  best.best_photo_to_art_art_fid.transfer_content_lpips: 0.3538659717
  best.best_photo_to_art_art_fid.transfer_fid: 287.55510336759266
  best.best_photo_to_art_classifier_acc.photo_to_art_art_fid: 398.8193577754396
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.6121498428285121
  best.best_photo_to_art_classifier_acc.photo_to_art_fid: 287.55510336759266
  best.best_photo_to_art_classifier_acc.transfer_art_fid: 398.8193577754396
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.6457651415467262
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.3538659717
  best.best_photo_to_art_classifier_acc.transfer_fid: 287.55510336759266
  best.best_photo_to_art_clip_style.photo_to_art_art_fid: 398.8193577754396
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.6121498428285121
  best.best_photo_to_art_clip_style.photo_to_art_fid: 287.55510336759266
  best.best_photo_to_art_clip_style.transfer_art_fid: 398.8193577754396
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.6457651415467262
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.3538659717
  best.best_photo_to_art_clip_style.transfer_fid: 287.55510336759266
  best.best_photo_to_art_fid.photo_to_art_art_fid: 398.8193577754396
  best.best_photo_to_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_fid.photo_to_art_clip_style: 0.6121498428285121
  best.best_photo_to_art_fid.photo_to_art_fid: 287.55510336759266
  best.best_photo_to_art_fid.transfer_art_fid: 398.8193577754396
  best.best_photo_to_art_fid.transfer_clip_style: 0.6457651415467262
  best.best_photo_to_art_fid.transfer_content_lpips: 0.3538659717
  best.best_photo_to_art_fid.transfer_fid: 287.55510336759266
  best.best_transfer_art_fid.photo_to_art_art_fid: 398.8193577754396
  best.best_transfer_art_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_art_fid.photo_to_art_clip_style: 0.6121498428285121
  best.best_transfer_art_fid.photo_to_art_fid: 287.55510336759266
  best.best_transfer_art_fid.transfer_art_fid: 398.8193577754396
  best.best_transfer_art_fid.transfer_clip_style: 0.6457651415467262
  best.best_transfer_art_fid.transfer_content_lpips: 0.3538659717
  best.best_transfer_art_fid.transfer_fid: 287.55510336759266
  best.best_transfer_classifier_acc.photo_to_art_art_fid: 398.8193577754396
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.6121498428285121
  best.best_transfer_classifier_acc.photo_to_art_fid: 287.55510336759266
  best.best_transfer_classifier_acc.transfer_art_fid: 398.8193577754396
  best.best_transfer_classifier_acc.transfer_clip_style: 0.6457651415467262
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.3538659717
  best.best_transfer_classifier_acc.transfer_fid: 287.55510336759266
  best.best_transfer_clip_style.photo_to_art_art_fid: 398.8193577754396
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.6121498428285121
  best.best_transfer_clip_style.photo_to_art_fid: 287.55510336759266
  best.best_transfer_clip_style.transfer_art_fid: 398.8193577754396
  best.best_transfer_clip_style.transfer_clip_style: 0.6457651415467262
  best.best_transfer_clip_style.transfer_content_lpips: 0.3538659717
  best.best_transfer_clip_style.transfer_fid: 287.55510336759266
  best.best_transfer_content_lpips.photo_to_art_art_fid: 398.8193577754396
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.6121498428285121
  best.best_transfer_content_lpips.photo_to_art_fid: 287.55510336759266
  best.best_transfer_content_lpips.transfer_art_fid: 398.8193577754396
  best.best_transfer_content_lpips.transfer_clip_style: 0.6457651415467262
  best.best_transfer_content_lpips.transfer_content_lpips: 0.3538659717
  best.best_transfer_content_lpips.transfer_fid: 287.55510336759266
  best.best_transfer_fid.photo_to_art_art_fid: 398.8193577754396
  best.best_transfer_fid.photo_to_art_classifier_acc: 0.0
  best.best_transfer_fid.photo_to_art_clip_style: 0.6121498428285121
  best.best_transfer_fid.photo_to_art_fid: 287.55510336759266
  best.best_transfer_fid.transfer_art_fid: 398.8193577754396
  best.best_transfer_fid.transfer_clip_style: 0.6457651415467262
  best.best_transfer_fid.transfer_content_lpips: 0.3538659717
  best.best_transfer_fid.transfer_fid: 287.55510336759266
  latest.photo_to_art_art_fid: 398.8193577754396
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_style: 0.6121498428285121
  latest.photo_to_art_fid: 287.55510336759266
  latest.transfer_art_fid: 398.8193577754396
  latest.transfer_clip_style: 0.6457651415467262
  latest.transfer_content_lpips: 0.3538659717
  latest.transfer_fid: 287.55510336759266
  mean.photo_to_art_art_fid: 398.8193577754396
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_style: 0.6121498428285121
  mean.photo_to_art_fid: 287.55510336759266
  mean.transfer_art_fid: 398.8193577754396
  mean.transfer_clip_style: 0.6457651415467262
  mean.transfer_content_lpips: 0.3538659717
  mean.transfer_fid: 287.55510336759266
```

### strong-128_128_256_0.5_1.0-1swd-dit-5style.json (3KB)
```
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.620264293551445
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.6495789507925511
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.3815032577559999
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.620264293551445
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.6495789507925511
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.3815032577559999
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.620264293551445
  best.best_transfer_classifier_acc.transfer_clip_style: 0.6495789507925511
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.3815032577559999
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.620264293551445
  best.best_transfer_clip_style.transfer_clip_style: 0.6495789507925511
  best.best_transfer_clip_style.transfer_content_lpips: 0.3815032577559999
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.620264293551445
  best.best_transfer_content_lpips.transfer_clip_style: 0.6495789507925511
  best.best_transfer_content_lpips.transfer_content_lpips: 0.3815032577559999
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_style: 0.620264293551445
  latest.transfer_clip_style: 0.6495789507925511
  latest.transfer_content_lpips: 0.3815032577559999
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_style: 0.620264293551445
  mean.transfer_clip_style: 0.6495789507925511
  mean.transfer_content_lpips: 0.3815032577559999
```

### style8-.json (4KB)
```
  best.best_photo_to_art_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_classifier_acc.photo_to_art_clip_style: 0.4735227119922638
  best.best_photo_to_art_classifier_acc.transfer_clip_style: 0.44108049243688585
  best.best_photo_to_art_classifier_acc.transfer_content_lpips: 0.24389943514999998
  best.best_photo_to_art_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_photo_to_art_clip_style.photo_to_art_clip_style: 0.47395190358161926
  best.best_photo_to_art_clip_style.transfer_clip_style: 0.4414529123902321
  best.best_photo_to_art_clip_style.transfer_content_lpips: 0.24392064856
  best.best_transfer_classifier_acc.photo_to_art_classifier_acc: 0.0
  best.best_transfer_classifier_acc.photo_to_art_clip_style: 0.4735227119922638
  best.best_transfer_classifier_acc.transfer_clip_style: 0.44108049243688585
  best.best_transfer_classifier_acc.transfer_content_lpips: 0.24389943514999998
  best.best_transfer_clip_style.photo_to_art_classifier_acc: 0.0
  best.best_transfer_clip_style.photo_to_art_clip_style: 0.47395190358161926
  best.best_transfer_clip_style.transfer_clip_style: 0.4414529123902321
  best.best_transfer_clip_style.transfer_content_lpips: 0.24392064856
  best.best_transfer_content_lpips.photo_to_art_classifier_acc: 0.0
  best.best_transfer_content_lpips.photo_to_art_clip_style: 0.4735227119922638
  best.best_transfer_content_lpips.transfer_clip_style: 0.44108049243688585
  best.best_transfer_content_lpips.transfer_content_lpips: 0.24389943514999998
  latest.photo_to_art_classifier_acc: 0.0
  latest.photo_to_art_clip_style: 0.4732224529981613
  latest.transfer_clip_style: 0.44089226484298705
  latest.transfer_content_lpips: 0.24391540705999998
  mean.photo_to_art_classifier_acc: 0.0
  mean.photo_to_art_clip_style: 0.473518241763115
  mean.transfer_clip_style: 0.44114807552099233
  mean.transfer_content_lpips: 0.24391645564400002
```

---

## 34. Early Experiment Summaries (from summary/ directory)

### 1swd-dit-2style (Single SWD, 2 style domains)
- best_photo_to_art_clip_style: 0.607 (optimized for style)
- best_photo_to_art_clip_style lpips: 0.340
- transfer_clip_style: 0.560

### 5style-any2any-skipfusion
- VERY POOR: style scores only 0.289-0.329
- This experiment failed - any2any skip fusion didn't work for 5 styles

### full-8-8monet (8-style with Monet emphasis)
- photo_to_art_art_fid: 355.4
- clip_style: 0.297 (very low - model failed to achieve Monet style)

### full-swd
- photo_to_art_art_fid: 391.0
- clip_style: 0.421

### full-v2 (version 2 of full experiment)
- photo_to_art_art_fid: 417.9
- clip_style: 0.437

### full-v2-20
- photo_to_art_art_fid: 452.4 (worse than v2)
- clip_style: 0.489 (slightly better)

### patch-1-3-5 (Patch sizes 1,3,5)
- **photo_to_art_fid: 287.6** (excellent distribution match!)
- clip_style: 0.612
- transfer_content_lpips: 0.354 (good content preservation)
- This was an EARLY baseline with great distribution but moderate style

### strong-128_128_256_0.5_1.0-1swd-dit-5style
- photo_to_art_clip_style: 0.620
- transfer_content_lpips: 0.382
- A decent intermediate result

### style8- (8 style experiment)
- Style scores only 0.44-0.47 (poor for 8 styles)
- LPIPS: 0.244 (excellent content preservation, but minimal style change)
- Model couldn't handle 8 style domains

---

## 35. Complete Experiment Status Table

### Categories:
1. **MAINLINE** (exp_1-6, G1, G2): 7 experiments with eval data
2. **ABLATION** (abl_no_*, decoder-A/B/C, decoder-D1-D6): ~10 experiments
3. **ARCH_SWEEP** (micro01-05, scan01-06, G0, G1, spatial-adagn): ~15 experiments  
4. **HPO** (optuna_hpo): 31 trials evaluated
5. **CONFIG_EXPLORATION** (L16, weight, patch/HF, N01-04): 40+ configs defined
6. **ACTIVE** (42_A01-A10, micro_E01-E04, T01-T03, Z01-Z03): 20 experiments running, NO EVALS YET
7. **EARLY/DEPRECATED** (swd-256, no-tv, no-edge, patch-1-3-5): Completed, archived
8. **NO EVAL DATA** (S3-S10, exp6_nuclear, ablate_A6, etc.): ~15 experiments logged but not evaluated

### Total unique experiments in Y:\experiments: ~220
### With eval data: ~80
### With full detailed analysis: ~60 (all covered in this report)
### Still training (no evals yet): ~40

---

*End of Archaeology Part 2*

## 2026-04-03 03:00 - Scheduled Check: A01 Epoch 30 and 60 Results

**42_A01_Macro_Only_LR3e4** just completed full_eval at epoch_0030 and epoch_0060!

This is a CrossAttn model with:
- `ablation_no_residual: true`, `residual_gain: 1.0`, `num_decoder_blocks: 2`
- `swd_patch_sizes: [19, 25, 31]` (Macro_ONLY patches)
- `swd_use_high_freq: false`, `w_swd: 250`, `w_color: 50`, `w_identity: 0.0`
- `skip_routing_mode: adaptive`, `style_modulator_type: cross_attn`
- Batch size 256, LR 3e-4, 40 epochs planned but eval exists at 30/60

=== A01 Macro_Only Results ===
Epoch                          P2A_Style  P2A_LPIPS  STA_Style  STA_LPIPS  ClassAcc
epoch_0030                     0.6057     0.3255     0.6420     0.3227     0.0
epoch_0030_tokenized_distill   0.6042     0.3238     0.6410     0.3214     0.0
epoch_0060                     0.6115     0.3310     0.6455     0.3266     0.0

=== Key Observations ===
1. LPIPS is EXCELLENT (~0.32-0.33), among the best in the entire project. Much better than old baselines (~0.43-0.44)
2. BUT photo-to-art style is LOW (~0.61), significantly below old architecture peaks (~0.72)
3. Tokenized distill is SLIGHTLY WORSE than baseline at ep30 (0.6042 vs 0.6057), confirming earlier finding
4. 30->60 epoch shows slight improvement in style (+0.006) but worse LPIPS (+0.006)
5. classifier_acc=0.0 throughout - still no working classifier checkpoint
6. FID/ArtFID are all None - FID eval not enabled in this eval pipeline version

=== Comparison with FinalMicro_2 ===
FinalMicro_2 F01 (Patch135) at ep40: p2a_style=0.622, all_style=0.683, all_lpips=0.683
A01 (Macro_Only) at ep60:     p2a_style=0.611, sta_style=0.645, sta_lpips=0.326

A01 has LOWER style but MUCH better LPIPS (0.326 vs 0.683). This confirms macro-only patches
produce conservative, content-faithful outputs that don't capture enough style texture.

=== Only A01 has eval data so far ===
Other 42_* dirs (A02-A10, micro_E*, T0*, Z0*) still have NO full_eval data.
Training either hasn't completed or eval pipeline hasn't triggered.

## 2026-04-03 03:00 - exp.csv Comprehensive Results (274 rows)

Found 17 experiment groups in the centralized exp.csv aggregation file.

### ZeroConstraint (6 entries)
  | ZeroConstraint_Z01_ResOff_Adapt_ZeroIDT                 | ep=40   | style=0.67315159 | lpips=0.31933757 | artfid=N/A        | p2a=0.59911085 | fid=N/A        |
  | ZeroConstraint_Z01_ResOff_Adapt_ZeroIDT_distill_epochs2 | ep=40   | style=0.67186286 | lpips=0.31948404 | artfid=N/A        | p2a=0.59740979 | fid=N/A        |
  | ZeroConstraint_Z02_ResOff_Adapt_ZeroIDT_HFSWD           | ep=40   | style=0.67363970 | lpips=0.34326217 | artfid=N/A        | p2a=0.59453069 | fid=N/A        |
  | ZeroConstraint_Z02_ResOff_Adapt_ZeroIDT_HFSWD_distill_e | ep=40   | style=0.67334468 | lpips=0.34357891 | artfid=N/A        | p2a=0.59463971 | fid=N/A        |
  | ZeroConstraint_Z03_ResOff_None_ZeroIDT                  | ep=40   | style=0.67622981 | lpips=0.32236058 | artfid=N/A        | p2a=0.60509103 | fid=N/A        |
  | ZeroConstraint_Z03_ResOff_None_ZeroIDT_distill_epochs20 | ep=40   | style=0.67611738 | lpips=0.32230355 | artfid=N/A        | p2a=0.60415550 | fid=N/A        |

### 42_A01 (3 entries)
  | 42_A01_Macro_Only_LR3e4                                 | ep=30   | style=0.67653706 | lpips=0.32237076 | artfid=N/A        | p2a=0.60570057 | fid=N/A        |
  | 42_A01_Macro_Only_LR3e4_distill_epochs200_tokenized     | ep=30   | style=0.67568928 | lpips=0.32119759 | artfid=N/A        | p2a=0.60418330 | fid=N/A        |
  | 42_A01_Macro_Only_LR3e4                                 | ep=60   | style=0.67952328 | lpips=0.32632076 | artfid=N/A        | p2a=0.61152804 | fid=N/A        |

### DepthSkip9 (18 entries)
  | DepthSkip9_A01_ResOn_Naive_Conv1                        | ep=40   | style=0.68289543 | lpips=0.35027518 | artfid=N/A        | p2a=0.62154863 | fid=N/A        |
  | DepthSkip9_A01_ResOn_Naive_Conv1                        | ep=80   | style=0.68390610 | lpips=0.35045899 | artfid=N/A        | p2a=0.61471281 | fid=N/A        |
  | DepthSkip9_A02_ResOn_None_Swin4                         | ep=40   | style=0.68135589 | lpips=0.33201817 | artfid=N/A        | p2a=0.61536380 | fid=N/A        |
  | DepthSkip9_A02_ResOn_None_Swin4                         | ep=80   | style=0.68460579 | lpips=0.34088672 | artfid=N/A        | p2a=0.61525854 | fid=N/A        |
  | DepthSkip9_E03_ResOff_None_Conv1                        | ep=40   | style=0.66793895 | lpips=0.30105423 | artfid=N/A        | p2a=0.58484990 | fid=N/A        |
  | DepthSkip9_E03_ResOff_None_Conv1                        | ep=80   | style=0.66787788 | lpips=0.29986858 | artfid=N/A        | p2a=0.58365935 | fid=N/A        |
  | DepthSkip9_E04_ResOff_None_Swin2                        | ep=40   | style=0.66752420 | lpips=0.30032237 | artfid=N/A        | p2a=0.58422635 | fid=N/A        |
  | DepthSkip9_E04_ResOff_None_Swin2                        | ep=80   | style=0.66720437 | lpips=0.29872074 | artfid=N/A        | p2a=0.58231270 | fid=N/A        |
  | DepthSkip9_E05_ResOff_None_Swin4                        | ep=40   | style=0.66729534 | lpips=0.30012980 | artfid=N/A        | p2a=0.58380971 | fid=N/A        |
  | DepthSkip9_E05_ResOff_None_Swin4                        | ep=80   | style=0.66747548 | lpips=0.29876892 | artfid=N/A        | p2a=0.58256416 | fid=N/A        |
  | DepthSkip9_E06_ResOff_Adapt_Conv1                       | ep=40   | style=0.66895351 | lpips=0.30017040 | artfid=N/A        | p2a=0.57746624 | fid=N/A        |
  | DepthSkip9_E06_ResOff_Adapt_Conv1                       | ep=80   | style=0.66923242 | lpips=0.29989884 | artfid=N/A        | p2a=0.57725204 | fid=N/A        |
  | DepthSkip9_E07_ResOff_Adapt_Swin2                       | ep=40   | style=0.66725615 | lpips=0.29972190 | artfid=N/A        | p2a=0.58178871 | fid=N/A        |
  | DepthSkip9_E07_ResOff_Adapt_Swin2                       | ep=80   | style=0.66762781 | lpips=0.30019584 | artfid=N/A        | p2a=0.57980666 | fid=N/A        |
  | DepthSkip9_E08_ResOff_Adapt_Swin4                       | ep=40   | style=0.66740810 | lpips=0.29571114 | artfid=N/A        | p2a=0.58269400 | fid=N/A        |
  | DepthSkip9_E08_ResOff_Adapt_Swin4                       | ep=80   | style=0.66962562 | lpips=0.29834752 | artfid=N/A        | p2a=0.58045661 | fid=N/A        |
  | DepthSkip9_E09_ResOff_Naive_Conv1                       | ep=40   | style=0.66928856 | lpips=0.29773970 | artfid=N/A        | p2a=0.57911182 | fid=N/A        |
  | DepthSkip9_E09_ResOff_Naive_Conv1                       | ep=80   | style=0.66790301 | lpips=0.30050632 | artfid=N/A        | p2a=0.57630243 | fid=N/A        |

### New4 (8 entries)
  | New4_N01_Stylized_Naive                                 | ep=40   | style=0.68475330 | lpips=0.45215007 | artfid=N/A        | p2a=0.63012308 | fid=N/A        |
  | New4_N01_Stylized_Naive                                 | ep=80   | style=0.68757787 | lpips=0.43552543 | artfid=N/A        | p2a=0.62365111 | fid=N/A        |
  | New4_N02_Stylized_Adaptive                              | ep=40   | style=0.68478750 | lpips=0.42774400 | artfid=N/A        | p2a=0.62959483 | fid=N/A        |
  | New4_N02_Stylized_Adaptive                              | ep=80   | style=0.68431054 | lpips=0.44050937 | artfid=N/A        | p2a=0.62371391 | fid=N/A        |
  | New4_N03_Stylized_Adaptive_Retain0p2                    | ep=40   | style=0.68526126 | lpips=0.45486864 | artfid=N/A        | p2a=0.62748876 | fid=N/A        |
  | New4_N03_Stylized_Adaptive_Retain0p2                    | ep=80   | style=0.68834193 | lpips=0.44521819 | artfid=N/A        | p2a=0.62728337 | fid=N/A        |
  | New4_N04_Stylized_Norm                                  | ep=40   | style=0.68384805 | lpips=0.44383328 | artfid=N/A        | p2a=0.62532251 | fid=N/A        |
  | New4_N04_Stylized_Norm                                  | ep=80   | style=0.68892931 | lpips=0.44333799 | artfid=N/A        | p2a=0.63221450 | fid=N/A        |

### Skip10 (9 entries)
  | Skip10_S01_None                                         | ep=40   | style=0.68696932 | lpips=0.41724413 | artfid=N/A        | p2a=0.63090942 | fid=N/A        |
  | Skip10_S01_None                                         | ep=80   | style=0.68775576 | lpips=0.42540748 | artfid=N/A        | p2a=0.63051337 | fid=N/A        |
  | Skip10_S02_Naive_G1p0                                   | ep=40   | style=0.68162483 | lpips=0.33985682 | artfid=N/A        | p2a=0.61503247 | fid=N/A        |
  | Skip10_S02_Naive_G1p0                                   | ep=80   | style=0.68209596 | lpips=0.34061214 | artfid=N/A        | p2a=0.61325882 | fid=N/A        |
  | Skip10_S03_Naive_G0p5                                   | ep=40   | style=0.68302209 | lpips=0.34254453 | artfid=N/A        | p2a=0.61682763 | fid=N/A        |
  | Skip10_S03_Naive_G0p5                                   | ep=80   | style=0.68177554 | lpips=0.33712286 | artfid=N/A        | p2a=0.60810827 | fid=N/A        |
  | Skip10_S04_Naive_G1p5                                   | ep=40   | style=0.68206094 | lpips=0.33424975 | artfid=N/A        | p2a=0.61516445 | fid=N/A        |
  | Skip10_S04_Naive_G1p5                                   | ep=80   | style=0.68199818 | lpips=0.33789360 | artfid=N/A        | p2a=0.61164462 | fid=N/A        |
  | Skip10_S05_Adaptive                                     | ep=40   | style=0.68325167 | lpips=0.41393771 | artfid=N/A        | p2a=0.62304877 | fid=N/A        |

### heavy_decode (4 entries)
  | heavy_decode                                            | ep=40   | style=0.68790600 | lpips=0.43485470 | artfid=N/A        | p2a=0.63948184 | fid=N/A        |
  | heavy_decode_distill_epochs200_tokenized                | ep=40   | style=0.69656842 | lpips=0.40390005 | artfid=N/A        | p2a=0.64334003 | fid=N/A        |
  | heavy_decode                                            | ep=80   | style=0.68384521 | lpips=0.33924552 | artfid=N/A        | p2a=0.61285167 | fid=N/A        |
  | heavy_decode_distill_epochs200_tokenized                | ep=80   | style=0.69364663 | lpips=0.40554048 | artfid=N/A        | p2a=0.63009824 | fid=N/A        |

### light_decoder (4 entries)
  | light-1                                                 | ep=60   | style=0.69329463 | lpips=0.40278549 | artfid=N/A        | p2a=0.63702873 | fid=N/A        |
  | light-15patch-10color                                   | ep=60   | style=0.69134132 | lpips=0.40768957 | artfid=N/A        | p2a=0.63411455 | fid=N/A        |
  | light_decoder                                           | ep=40   | style=0.68548718 | lpips=0.35300517 | artfid=N/A        | p2a=0.62463054 | fid=N/A        |
  | light_decoder_distill_epochs200_tokenized               | ep=40   | style=0.69442435 | lpips=0.42590281 | artfid=N/A        | p2a=0.64609195 | fid=N/A        |

### arch_ablate (16 entries)
  | arch_1_pM_sC_dH                                         | ep=30   | style=0.68458366 | lpips=0.40089565 | artfid=N/A        | p2a=0.62463320 | fid=N/A        |
  | arch_1_pM_sC_dH                                         | ep=60   | style=0.68766996 | lpips=0.42872208 | artfid=N/A        | p2a=0.63212843 | fid=N/A        |
  | arch_2_pM_sA_dL                                         | ep=30   | style=0.68447586 | lpips=0.39853543 | artfid=N/A        | p2a=0.62353497 | fid=N/A        |
  | arch_2_pM_sA_dL                                         | ep=60   | style=0.68931421 | lpips=0.42746134 | artfid=N/A        | p2a=0.63348309 | fid=N/A        |
  | arch_3_pM_sC_dL                                         | ep=30   | style=0.68396034 | lpips=0.40114376 | artfid=N/A        | p2a=0.61870016 | fid=N/A        |
  | arch_3_pM_sC_dL                                         | ep=60   | style=0.68899340 | lpips=0.42934986 | artfid=N/A        | p2a=0.62642212 | fid=N/A        |
  | arch_4_pM_sA_dH                                         | ep=30   | style=0.68397959 | lpips=0.41814119 | artfid=N/A        | p2a=0.61969291 | fid=N/A        |
  | arch_4_pM_sA_dH                                         | ep=60   | style=0.69010175 | lpips=0.44288358 | artfid=N/A        | p2a=0.63030913 | fid=N/A        |
  | arch_5_pMW_sA_dH                                        | ep=30   | style=0.68387088 | lpips=0.44454614 | artfid=N/A        | p2a=0.62996202 | fid=N/A        |
  | arch_5_pMW_sA_dH                                        | ep=60   | style=0.68914875 | lpips=0.44727031 | artfid=N/A        | p2a=0.63106320 | fid=N/A        |
  | arch_6_pMW_sC_dL                                        | ep=30   | style=0.68500214 | lpips=0.41928272 | artfid=N/A        | p2a=0.61997190 | fid=N/A        |
  | arch_6_pMW_sC_dL                                        | ep=60   | style=0.68788718 | lpips=0.43253531 | artfid=N/A        | p2a=0.62935249 | fid=N/A        |
  | arch_7_pMW_sA_dL                                        | ep=30   | style=0.68923261 | lpips=0.44810897 | artfid=N/A        | p2a=0.63225521 | fid=N/A        |
  | arch_7_pMW_sA_dL                                        | ep=60   | style=0.69175260 | lpips=0.43952331 | artfid=N/A        | p2a=0.63555348 | fid=N/A        |
  | arch_8_pMW_sC_dH                                        | ep=30   | style=0.68373999 | lpips=0.42906872 | artfid=N/A        | p2a=0.61963705 | fid=N/A        |
  | arch_8_pMW_sC_dH                                        | ep=60   | style=0.68792069 | lpips=0.43914516 | artfid=N/A        | p2a=0.63197310 | fid=N/A        |

### ca_pram (24 entries)
  | ca_pram_final_10_dim96_tok128                           | ep=80   | style=0.63413891 | lpips=0.0        | artfid=305.689120 | p2a=0.59562884 | fid=297.213861 |
  | ca_pram_final_10_dim96_tok128                           | ep=80   | style=0.67005364 | lpips=0.0        | artfid=279.582900 | p2a=0.59562884 | fid=266.942540 |
  | ca_pram_final_11_dim128_tok128                          | ep=40   | style=0.67882078 | lpips=0.33120568 | artfid=272.024404 | p2a=0.60170193 | fid=269.427543 |
  | ca_pram_final_11_dim128_tok128                          | ep=80   | style=0.69352767 | lpips=0.44327790 | artfid=301.402598 | p2a=0.63804049 | fid=283.591724 |
  | ca_pram_final_12_base_ref                               | ep=40   | style=0.66809149 | lpips=0.32157402 | artfid=278.588449 | p2a=0.58958994 | fid=268.305563 |
  | ca_pram_final_12_base_ref                               | ep=80   | style=0.69117851 | lpips=0.44000833 | artfid=299.115195 | p2a=0.63304113 | fid=284.832990 |
  | ca_pram_final_1_lr4_id35_swd60_c5                       | ep=40   | style=0.68699298 | lpips=0.41633095 | artfid=285.581259 | p2a=0.63255944 | fid=276.642296 |

---
**END OF PART 2. CONTINUED IN `ARCHAEOLOGY_PART3.md`**
