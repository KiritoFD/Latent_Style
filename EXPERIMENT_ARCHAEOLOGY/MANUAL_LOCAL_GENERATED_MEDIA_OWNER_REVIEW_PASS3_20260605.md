# Local Generated Media Owner Review Pass 3 - 2026-06-05

This pass continues the local generated-media owner review. The media-count
scan was used only to find the next candidate cluster; every row in the CSV was
then checked by opening the exact directory plus nearby summary, metrics, log,
time, README, or ledger evidence.

No cleanup was performed in this pass.

## Reviewed And Retained

### Docs Control Outputs

- `SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\legacy256_no_op_identity_5x5`
- `SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\legacy512_no_op_identity_5x5`
- `SchrodingerBridge\docs\experiments\idt_eval_20260602\legacy256_overfit50\idt_5x5`
- `SchrodingerBridge\docs\experiments\idt_eval_20260602\wikiart512_5style\idt_5x5`
- `SchrodingerBridge\docs\experiments\idt_eval_20260602\distinct5_512\idt_5x5`

These are no-op or IDT control outputs, not disposable generated media. They
have `summary.json`, `metrics.csv`, and direct docs/timing references. The
no-op directories are explicitly listed in
`metric_hacking_noop_20260602\README.md`; the IDT directories are backed by
`idt_build_summary.json` and timing master rows.

### Timing Outputs

- `SchrodingerBridge\exp\timing_20260602\lancet_generate750_current_b2_tchunk5_vaebs2`
- `SchrodingerBridge\exp\timing_20260601\lancet_fulleval750_b2_tchunk5_vaebs2`
- `SchrodingerBridge\exp\timing_20260602\run_eval_png750_b12_v2_w8_grid`
- `SchrodingerBridge\exp\timing_20260602\run_eval_png750_b12_v2_w8_nogrid`
- `SchrodingerBridge\exp\timing_20260602\lancet_from_scratch_e8_generate750`
- `SchrodingerBridge\exp\timing_20260602\lancet_from_scratch_e8_full_eval_direct750`

These dirs are timing evidence. The opened evidence includes internal
`timings_sec.wall_total` values, external `*_time.txt` records where present,
and docs/timing references. The from-scratch pair preserves the difference
between generation-only wall time and direct full-eval wall time.

### Distinct5 Compact / Calibration Outputs

- `SchrodingerBridge\exp\distinct5_512_ema_variant_f_annealed_prototype_ot_queue_e3_b44_remote\full_eval`
- `SchrodingerBridge\exp\distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3_b44_remote\full_eval`
- `SchrodingerBridge\exp\local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync\generation_only_step_000350_timed`

The F/K directories are current compact-anchor eval outputs, with docs same-cost
inventory and timing source-open support. The local ckptsync directory is
calibration-only, but its parent config, training CSV, generation-only summary,
and cleanup ledger show that non-mainline weights were already deleted while
the data needed for indexing was retained.

## Delete Whitelist

None.

## Remaining Work

Continue below this cluster in the local generated-media candidate list. Dataset
mirrors and formal evidence bundles should remain separate from generated-output
cleanup decisions. Deletion remains whitelist-only: open exact path, verify
summary/log/config/reference context, write policy, then delete only if the row
is explicitly whitelisted.
