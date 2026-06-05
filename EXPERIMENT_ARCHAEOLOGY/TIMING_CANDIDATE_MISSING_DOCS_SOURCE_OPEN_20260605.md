# Timing Candidate Missing Docs Source-Open Pass - 2026-06-05

This pass source-opened the 26 timing candidate rows from
`timing_candidate_claim_reconciliation_20260605.csv` where
`in_docs_timing_master=False`.

It did not edit `SchrodingerBridge/docs/timing/training_inference_timing_master.csv`,
paper TeX, paper PDFs, source code, or Related_Works files. The output is:

- `timing_candidate_missing_docs_source_open_20260605.csv`

## Method

Directory scans were used only for navigation and for resolving exact path
ambiguities. The conclusions in the CSV come from opened source files:

- local docs/inventory files
- local `summary.json` files
- local training CSV files
- local generation logs
- local cleanup ledgers
- exact remote I-drive training CSV files
- exact remote I-drive `full_eval/.../summary.json` files

Remote checks used the SSH target:

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

For remote I-drive evidence, each row was opened by exact run path. No cleanup
was performed in this pass.

## Source-Open Counts

| bucket | rows | status |
| --- | ---: | --- |
| Distinct5 local compact anchors and SaMST e5 | 5 | opened |
| WikiArt512 timing rows | 3 | opened |
| local Distinct5 ckptsync calibration | 1 | opened |
| TokenizerClean audit packets | 5 | opened |
| remote phase1 ablations | 6 | opened |
| remote compact ablations | 4 | opened |
| remote path kinetic packets | 2 | opened |
| total | 26 | opened |

## Promotion Decisions

| decision | rows | meaning |
| --- | --- | --- |
| `promote_candidate_with_caveat` | 1, 4, 8 | usable claim support after owner approval; caveats stay visible |
| `promote_only_if_owner_accepts_missing_artfid_packet` | 2, 3 | H rows have closed metrics but no retained indexed targetwise ArtFID packet |
| `promote_timing_note_with_caveat` | 24, 44, 45 | timing is backed by opened docs note; some values are external wall rather than summary-internal wall |
| `owner_review_before_docs_promotion` | 50-54 | current TokenizerClean audit evidence, but not auto-promoted into docs timing |
| `retain_archaeology_only_nonmainline_calibration` | 43 | local calibration with weights deleted; logs/images/summary retained |
| `retain_archaeology_only_trajectory_ablation` | 56-65, 67-68 | phase/compact/path trajectory evidence, not current paper anchor |

## Local Distinct5 Findings

Rows 1-4 use
`SchrodingerBridge/docs/experiments/2026-06-04-distinct5_same_cost_inventory.csv`
for train wall minutes and each run's local `full_eval/.../summary.json` for
`timings_sec.wall_total`.

Important caveat: the local run directories for F/H/K did not provide a
separate retained in-dir training log in this evidence surface. The train wall
is inventory-backed, while the inference/eval wall is summary-backed.

Row 8, SaMST e5, was opened through both the generation log and packet status:

- `generate.log` ends with `total=750` and `elapsed_sec=323.071`.
- `packet_status.json` records `train_wall_seconds=6958.502907`,
  `inference_wall_seconds=323.071`, and `inference_ms_per_img=430.76133333333337`.
- The CSV keeps the existing original unit for train time, `115.9750 min`, and
  records the source-open packet seconds in the verification column.

## WikiArt512 Findings

Rows 24, 44, and 45 are not interchangeable:

- row 24 is full-eval wall timing, not pure generation. The opened timing note
  records external wall `210.67s`; the opened summary records internal
  `timings_sec.wall_total=206.792325715`.
- row 44 is pure generation-only PNG timing. The opened timing note records
  external wall `54.80s`; the opened summary records `mode=generation_only`,
  `generated_count=750`, and internal `timings_sec.wall_total=46.791428548`.
- row 45 is from-scratch external wall timing. The opened training CSV has
  eight epoch rows but no wall-time column, so `66.56s` train wall is
  note-backed. The generation-only external wall is `55.16s`. The same timing
  note separately records direct full eval as `106.62s`.

For rows 43, 44, and 45, the retained generated image directories were opened
and counted at 750 PNG outputs where applicable.

## Local Ckptsync Calibration Finding

Row 43 is a local WSL calibration surface:

- `training_20260605_061543.csv` has two epoch rows with
  `90.04354214668274s` and `10.457379579544067s`, totaling `100.500922s`.
- `generation_only_step_000350_timed/summary.json` has
  `mode=generation_only`, `generated_count=750`, and
  `timings_sec.wall_total=78.31742824899999`.
- `cleanup/manual_deleted_checkpoints_20260605.csv` confirms
  `epoch_0001.pt`, `epoch_0002.pt`, and `step_000350.pt` were deleted.

This row is audit-only because it is non-mainline calibration evidence.

## Remote TokenizerClean Findings

Rows 50-54 were opened on remote I drive by exact path. Each training CSV was
opened and summed by `epoch_time_sec`; each summary was opened for
`timings_sec.wall_total` and `analysis.all_pairs_overview`.

| row | train rows | train sum sec | eval wall sec | clip_style | content_lpips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 3 | 291.682382106781 | 97.27645479800049 | 0.6928812634547552 | 0.35569425093333334 |
| 51 | 3 | 290.10184431076 | 100.00101462700059 | 0.6910941313505173 | 0.35521524370666663 |
| 52 | 3 | 160.100493192673 | 151.64121619984508 | 0.6905754188696543 | 0.3195148776666667 |
| 53 | 3 | 179.849926233292 | 87.60610380000435 | 0.6957186478773753 | 0.3353660955466667 |
| 54 | 3 | 198.769279003143 | 96.89431783699911 | 0.6963431750933329 | 0.3313274828266667 |

Row 54 has a path-quality caveat. The successful evidence directory is a
top-level directory whose separator character renders as `?` in console output
but is not ASCII question mark. Its character code is `61532`; the CSV records
it as `exp[U+F03C]saswd_axis_h_base_seed42_b44_saswd_semantic`.

These TokenizerClean rows are current audit evidence, but this pass does not
promote them into docs timing without owner review.

## Remote Phase/Compact/Path Findings

Rows 56-65 and 67-68 were opened on remote I drive by exact path. They are
valid archaeology timing rows, but not current compact manuscript anchors.

| row | train rows | train sum sec | eval wall sec | clip_style | content_lpips |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 56 | 8 | 506.223088264465 | 94.8003261089998 | 0.687648866891861 | 0.45274286515999995 |
| 57 | 8 | 507.093955278397 | 95.04837335200136 | 0.6849461014270782 | 0.46229617677333334 |
| 58 | 8 | 504.975627660751 | 95.20408842699908 | 0.687321089029312 | 0.44460011107999997 |
| 59 | 8 | 512.740225315094 | 95.09122094299892 | 0.6890299293200175 | 0.4450634233866667 |
| 60 | 8 | 511.389257669449 | 95.94725668200044 | 0.6891492090622584 | 0.44326236296 |
| 61 | 8 | 525.351942062378 | 95.38143512199895 | 0.6970923771063486 | 0.35925581901333337 |
| 62 | 3 | 200.378971099854 | 95.3742954019981 | 0.6948112870057425 | 0.34021655054666666 |
| 63 | 3 | 198.939151287079 | 95.61086336899825 | 0.6985142537355423 | 0.36818666126666666 |
| 64 | 3 | 198.993928909302 | 95.69641234200026 | 0.6966254591941832 | 0.36294960470666665 |
| 65 | 3 | 199.461369514465 | 95.98097363600027 | 0.6957354170878729 | 0.3458943149066666 |
| 67 | 3 | 199.039002180099 | 88.40818769996986 | 0.6817093665599825 | 0.46676262926666673 |
| 68 | 3 | 199.110300779343 | 88.4397754999809 | 0.6790328614314397 | 0.5073434180933333 |

Row 61 spans two training logs: epochs 1-7 in
`training_20260602_073021.csv` plus epoch 8 in
`training_20260602_074357.csv`.

## Remaining Gaps

These are the gaps after this pass:

- The docs timing master has not been edited. Promotion still needs owner
  decision and a separate docs-table update.
- H rows 2-3 need owner acceptance if used without a retained indexed
  targetwise ArtFID packet.
- WikiArt from-scratch wall values are note-backed external timings; the
  opened training CSV and summaries do not contain those wall values.
- TokenizerClean rows 50-54 are current audit evidence, not automatically
  paper-facing.
- Phase/compact/path rows 56-65 and 67-68 are retained as trajectory evidence.
- The 370 docs timing rows not covered by the overlay still need their own
  source-open review before prose use.
