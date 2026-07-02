# Checkpoint Deletion Log — M26

- **Date**: 2026-07-03
- **Task**: M26 cleanup — delete meaningless checkpoint (.pt) files from `exp/exp_ours/`
- **Reference**: [experiment_audit.md](./experiment_audit.md)
- **Scope**: Local deletions only (`exp/` is git-ignored). Only `.pt` files were deleted; all non-`.pt` files (config.json, src/, logs/, full_eval/, train.log) were preserved.

## Summary

- **Files deleted**: 41
- **Files kept (SOTA ckpts)**: 6
- **Keep-files verified intact post-deletion**: Yes (all 6 confirmed present)

## Deleted Files

### Batch 1 — Early (wikiarts_5 smoke + full) — 10 files

| # | Original Path |
|---|---|
| 1 | `exp/exp_ours/early/clean_base_v2_local/epoch_0005.pt` |
| 2 | `exp/exp_ours/early/clean_base_v2_local/epoch_0010.pt` |
| 3 | `exp/exp_ours/early/clean_base_v2_relu2/epoch_0003.pt` |
| 4 | `exp/exp_ours/early/task3_baseline_1ep/epoch_0001.pt` |
| 5 | `exp/exp_ours/early/task3_combo_a_1ep/epoch_0001.pt` |
| 6 | `exp/exp_ours/early/task3_combo_b_3ep/epoch_0001.pt` |
| 7 | `exp/exp_ours/early/task3_combo_b_3ep/epoch_0002.pt` |
| 8 | `exp/exp_ours/early/task3_combo_b_3ep/epoch_0003.pt` |
| 9 | `exp/exp_ours/early/task4_iter/r2a_with_film/epoch_0001.pt` |
| 10 | `exp/exp_ours/early/task4_iter/r2a_with_film/epoch_0002.pt` |

### Batch 2 — Local T (except T11, T10) — 11 files

| # | Original Path |
|---|---|
| 1 | `exp/exp_ours/local_t/630_local_r1_depth2/epoch_0005.pt` |
| 2 | `exp/exp_ours/local_t/630_local_r2_dim32/epoch_0005.pt` |
| 3 | `exp/exp_ours/local_t/630_local_r3_gate_init0/epoch_0005.pt` |
| 4 | `exp/exp_ours/local_t/630_local_t14_casi/epoch_0005.pt` |
| 5 | `exp/exp_ours/local_t/630_local_t15_llgqca/epoch_0005.pt` |
| 6 | `exp/exp_ours/local_t/630_local_t18a_wll05/epoch_0005.pt` |
| 7 | `exp/exp_ours/local_t/630_local_t18b_wll10/epoch_0005.pt` |
| 8 | `exp/exp_ours/local_t/630_local_t19a_depth6/epoch_0005.pt` |
| 9 | `exp/exp_ours/local_t/630_local_t19b_dim96/epoch_0005.pt` |
| 10 | `exp/exp_ours/local_t/630_local_t22_tone_bias/epoch_0005.pt` |
| 11 | `exp/exp_ours/local_t/630_local_t26_ll_ycbcr/epoch_0005.pt` |

### Batch 3 — Phase 4 (except 4F.1, 4I.2b, 4I.7b, 4J.1) — 20 files

| # | Original Path |
|---|---|
| 1 | `exp/exp_ours/phase4/630_phase1d_verify/epoch_0002.pt` |
| 2 | `exp/exp_ours/phase4/630_phase1d_verify_v2/epoch_0003.pt` |
| 3 | `exp/exp_ours/phase4/630_phase4a2_adain_0/epoch_0003.pt` |
| 4 | `exp/exp_ours/phase4/630_phase4a2_extrap_0/epoch_0003.pt` |
| 5 | `exp/exp_ours/phase4/630_phase4a2_w_ll_0/epoch_0003.pt` |
| 6 | `exp/exp_ours/phase4/630_phase4b1_freq_a05/epoch_0003.pt` |
| 7 | `exp/exp_ours/phase4/630_phase4b1_freq_a1/epoch_0003.pt` |
| 8 | `exp/exp_ours/phase4/630_phase4b3_dwt_a1/epoch_0003.pt` |
| 9 | `exp/exp_ours/phase4/630_phase4c_dino_clean_lvl2/epoch_0003.pt` |
| 10 | `exp/exp_ours/phase4/630_phase4d_lvl2/epoch_0003.pt` |
| 11 | `exp/exp_ours/phase4/630_phase4e_db2_lvl1/epoch_0003.pt` |
| 12 | `exp/exp_ours/phase4/630_phase4e_db2_lvl2/epoch_0003.pt` |
| 13 | `exp/exp_ours/phase4/630_phase4f_lvl4/epoch_0003.pt` |
| 14 | `exp/exp_ours/phase4/630_phase4g1a_lock_ll/epoch_0003.pt` |
| 15 | `exp/exp_ours/phase4/630_phase4g2_per_subband/epoch_0003.pt` |
| 16 | `exp/exp_ours/phase4/630_phase4h4f_sota_dim96/epoch_0003.pt` |
| 17 | `exp/exp_ours/phase4/630_phase4i10b_ept_t01/epoch_0005.pt` |
| 18 | `exp/exp_ours/phase4/630_phase4i2a_sota_heun/epoch_0003.pt` |
| 19 | `exp/exp_ours/phase4/630_phase4j2_wct_aligned/epoch_0005.pt` |
| 20 | `exp/exp_ours/phase4/630_phase4j6_fewshot_popart/epoch_0005.pt` |

## Kept Files (SOTA Checkpoints — DO NOT DELETE)

| # | Path | Role |
|---|---|---|
| 1 | `exp/exp_ours/phase4/630_phase4f_lvl3/epoch_0003.pt` | 4F.1 remote SOTA |
| 2 | `exp/exp_ours/phase4/630_phase4i2b_sota_heun_5ep/epoch_0005.pt` | 4I.2b Heun SOTA |
| 3 | `exp/exp_ours/phase4/630_phase4i7b_cosine_heun_a085_5ep/epoch_0005.pt` | 4I.7b remote final SOTA |
| 4 | `exp/exp_ours/phase4/630_phase4j1_dwt_route/epoch_0005.pt` | 4J.1 DWT route starting point |
| 5 | `exp/exp_ours/local_t/630_local_t11_stochastic_dwt_p08/epoch_0005.pt` | T11 local SOTA |
| 6 | `exp/exp_ours/local_t/630_local_t10_stochastic_dwt/epoch_0005.pt` | T10 lpips BEST |

## Notes

- All 41 candidate files in the DELETE list were verified to exist on disk before deletion.
- All 6 keep-files were verified present both before and after deletion.
- No non-`.pt` files were touched.
- No git operations were performed (`exp/` is git-ignored; deletions are local only).
