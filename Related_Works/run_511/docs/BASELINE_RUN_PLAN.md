# Baseline Run Plan

Updated: 2026-05-12

This plan is for baseline reproduction only. Ablation runs are handled separately under `SchrodingerBridge/ablation_destructive_7epoch`.

## Main-Table Scope

Main paper baseline target:

```text
AdaIN
StyTR2
AesPA-Net
AesFA
CAST
StyleID
SaMST
Ours
```

## Current Status

| Method | Current usable row | Status | Next action |
| --- | --- | --- | --- |
| AdaIN | `adain_v32k` strict 750 | complete | Use existing row; no retrain needed unless reviewer asks for another protocol. |
| SaMST | `samst_strict` strict 750 | complete | Use existing row; keep artifact diagnostics because visual grain is visible. |
| StyleID | `styleid_strict` strict 750 | complete enough | Training-free; current row is complete but semantically weak. Re-run only if we tune prompts/settings. |
| StyTR2 | `stytr2_smoke6`, 5 images | smoke only | Need full or pretrained run. Scratch training is slow; prefer official/pretrained weights if available. |
| CAST | `cast_smoke3` train ok, infer failed | adapter incomplete | Fix inference/export path first, then smoke, then full 750. |
| AesFA | `aesfa_timing_probe` failed | adapter incomplete | Fix repo/dependency/weight issue, then smoke, then full 750. |
| AesPA-Net | launcher exists; no successful run | adapter incomplete | Run preflight, resolve checkpoint/dependency gaps, then smoke/full. |
| S2WAT | 2000-epoch per-style checkpoints; run_511 smoke passed | adapter ready | Launch strict 750 with `Related_Works/run_511/launchers/run_s2wat_750.bat`; default runs inference + base/guard eval. |
| Ours | `S-add__K-1_C-0_W-20_Col-0`, epoch 7 | complete | Use epoch 7 as requested. |

## Supplement / Reusable Rows

| Method | Existing data | Suggested handling |
| --- | --- | --- |
| CUT/FastCUT | `Related_Works/runs/cut_5x5/infer_5x5/images`, 1250 images | Reuse for supplement/time-to-quality, not main AST table. |
| S2WAT legacy | `Related_Works/runs/s2wat_bs1_safe_e2000_full_eval/images`, 1495 images | Keep as legacy reference only; strict run_511 adapter is now preferred. |
| SDedit/SDTurbo | legacy 5x5 folders exist | Supplement only unless paper story shifts toward diffusion comparisons. |
| SANet/AdaAttN/OTFormer/ACID-Style | no local adapted runner yet | Defer until main-table baselines are stable. |

## Execution Order

1. Keep completed rows fixed: AdaIN, SaMST, StyleID, Ours.
2. Run S2WAT strict 750 if supplement comparison is needed now.
3. Repair and smoke test missing main baselines in this order: CAST, AesFA, AesPA-Net, StyTR2.
4. Only after smoke passes, launch full 750 for the repaired baselines.
5. Run standard metrics first: `eval_750.py` and `eval_guard_750.py`.
6. Run advanced artifact metrics only for visually suspicious or paper-critical rows.
7. Refresh inventories with `python Related_Works\scripts\collect_repro_inventory.py`.

## Launch Scripts

Prepared but not started:

- `Related_Works/run_511/launchers/run_baseline_preflight_queue.bat`
- `Related_Works/run_511/launchers/run_baseline_full_queue.bat`
- `Related_Works/run_511/launchers/run_s2wat_750.bat`

All individual `.bat` launchers now call their colocated `.py` files and write to `Related_Works/run_511/outputs`.
