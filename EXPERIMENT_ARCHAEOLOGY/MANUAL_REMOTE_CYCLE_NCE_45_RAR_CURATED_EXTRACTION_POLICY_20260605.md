# Remote Cycle-NCE 45.rar Curated Extraction Policy - 2026-06-05

This pass updates the decision for:

```text
I:\Github\Latent_Style\Cycle-NCE\45.rar
```

No deletion was performed. Remote current-state check confirms the archive
still exists and is `507.452 MB`, last written `2026-04-06 00:11:44`.

## Source Evidence

- `manual_remote_cycle_nce_45_rar_policy_20260605.csv`
- `manual_remote_cycle_nce_45_rar_entry_classes_20260605.csv`
- `manual_remote_cycle_nce_45_rar_run_ledger_20260605.csv`
- `manual_remote_cycle_nce_45_rar_summary_overview_20260605.csv`
- `manual_remote_cycle_nce_45_rar_summary_metrics_20260605.csv`
- `manual_remote_cycle_nce_45_rar_text_evidence_20260605.csv`

The archive has no expanded `Cycle-NCE\45` directory in the current remote
tree. Therefore the archive is currently the only container for these four
historical runs:

- `45_01_golden_funnel`
- `45_02_naked_fusion`
- `45_03_macro_dictator`
- `45_04_micro_rebel`

## Entry Class Breakdown

| class | entries | MB | decision |
| --- | ---: | ---: | --- |
| `config` | 4 | 0.014 | extract all |
| `summary_json` | 8 | 0.156 | extract all |
| `metrics_csv` | 8 | 0.745 | extract all |
| `training_csv` | 5 | 0.040 | extract all |
| `source_or_structured` | 42 | 26.704 | extract all |
| `other` | 9 | 0.117 | extract all |
| `generated_or_eval_image` | 6008 | 116.955 | extract all or owner-selected representative set plus manifest |
| `weight` | 12 | 423.787 | do not extract by default for nonweight package |

Nonweight evidence is not trivial: it is about `144.731 MB` by uncompressed
entry sizes and includes
configs, summaries, metrics, training CSVs, source/structured evidence,
ma-probe artifacts, and 6008 generated/eval images.

## Policy Decision

Current decision: `keep_unique_archive_payload`.

Reason: deleting `45.rar` now would delete unique nonweight evidence, not only
old checkpoints.

Allowed future cleanup path:

1. Extract a curated nonweight evidence package from `45.rar`.
2. Preserve all configs, summaries, metrics CSVs, training CSVs, source files,
   structured ma-probe CSV/JSON, and archive-root metadata.
3. Decide whether to extract all 6008 images or an owner-approved representative
   image set with a manifest that records omitted images.
4. Verify extracted files by entry count and byte size against
   `manual_remote_cycle_nce_45_rar_entry_classes_20260605.csv`.
5. Only after that, create a new delete whitelist for the original RAR or its
   weight portion.

Blocked action:

- Do not delete `45.rar` by archive size.
- Do not classify it as weight-only.
- Do not delete it before extracting or otherwise preserving the unique
  nonweight evidence.

## Cleanup Impact

If owner approves full nonweight extraction and original archive deletion, the
net space release depends on whether 6008 images are extracted:

- Extracting all nonweight evidence may re-materialize about `144.731 MB` by
  uncompressed entry sizes.
- The old weights are about `423.787 MB`.
- The archive itself is `507.452 MB`.

The safe cleanup value is therefore not the full archive size unless the
nonweight evidence is moved to a different storage location or represented by
an owner-approved subset.
