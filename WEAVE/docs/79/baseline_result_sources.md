# Baseline Result Sources And Current Consolidation State

This note connects three things that were previously mixed together:

1. the AAAI v4 main-table metric CSV,
2. local `results/` image packets,
3. remote `I:` image/table sources.

## 1. Main-table metric source

The v4 paper table in `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027_v4\paper.tex` is mirrored by the remote CSV:

`I:\results\tables\main_table.csv`

A local snapshot is saved as:

`G:\GitHub\Latent_Style\SchrodingerBridge\docs\79\main_table_v4_remote.csv`

This CSV contains 39 rows: 13 methods x 3 dataset columns. It should be treated as the current authoritative table-value snapshot, but not as proof that every row has a clean local image packet. Image-packet proof is tracked separately in `results_manifest.csv`.

## 2. Local image-packet registry

The current local image-packet registry is:

`G:\GitHub\Latent_Style\SchrodingerBridge\docs\79\results_manifest.csv`

Summary:

| Canonical dataset | Rows | Clean 750 rows | Main issue |
|---|---:|---:|---|
| `D5-512` | 13 | 4 | Several methods have a clean direct 750 plus nested duplicate copies; `styleshot` has 745. |
| `P2A-256` | 13 | 11 | `cut` and `seedream` are missing locally in unified `results\P256`. |
| `R5-512` | 3 | 2 | True Random5 currently only has `stylealigned`, `styleshot`, `zstar`; `styleshot` has 740. |
| `R5-WikiArt-legacy` | 5 | 0 | Directory name conflicts with detected D5 style set. |
| `random20-or-mixed` | 1 | 0 | `sdturbo` has 1123 images and a mixed 20-style style set. |

## 3. Remote image sources checked this pass

Remote host:

`ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62`

Confirmed remote paths:

| Dataset/protocol | Remote path | Count observed | Interpretation |
|---|---|---:|---|
| P2A-256 clean baselines | `I:\exp_256_photo2art` | 7 methods x 750 | Clean remote packet for `adain`, `identity`, `samam`, `samst`, `sdturbo`, `styleid`, `wct`. |
| P2A-256 duplicate subset | `I:\Github\Latent_Style\exp_baseline_256` | 3 methods x 750 | Duplicate/alternate clean packets for `adain`, `samst`, `wct`. |
| P2A-256 Seedream | `I:\Github\Latent_Style\seedream45_api\protocol_a_800\images` | 721 | Incomplete; cannot be a clean 750 row without repair. |
| P2A-256 CUT candidate | `I:\Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\infer_5x5\images` | 2427 | Needs protocol filtering to 5 x 5 x 30 = 750 before use. |
| D5 source test split | `I:\results\eval_protocol_750` | 5 x 30 = 150 | Source/test input split, not generated results. |
| v4 table metrics | `I:\results\tables\main_table.csv` | 39 rows | Exact flattened main-table value snapshot. |
| v4 auxiliary tables | `I:\results\aaai2027_v4_tables` | CSV-only | Artifact/probe/ledger tables, no image packets. |
| Curated remote experiment DB | `I:\latent_style_remote_curated\by_dataset` | CSV-only | Historical experiment metrics by dataset, useful for archaeology but not an image-packet root. |

Not found on remote in this pass:

| Requested/expected path | Result |
|---|---|
| `I:\Github\Latent_Style\SchrodingerBridge\results` | Missing |
| `I:\Github\Latent_Style\SchrodingerBridge\exp\baseline_wikiarts20` | Missing at this exact path |
| `I:\Github\Latent_Style\SchrodingerBridge\exp\wikiarts20_eval` | Missing at this exact path |
| `I:\Github\Latent_Style\SchrodingerBridge\results\R5-512` | Missing |

## 4. Immediate consolidation decisions

| Area | Decision |
|---|---|
| D5 local results | Keep `G:\GitHub\Latent_Style\SchrodingerBridge\results\D5-512` as the canonical local image root. Do not recursively count nested duplicates for table rows; use direct 750 files where present. |
| P2A local results | Keep `results\P256` for now but label it canonically as `P2A-256` in docs. Missing `cut` and `seedream` should not be silently filled from non-750 packets. |
| R5 local results | Use `results\R5-512` as canonical Random5. Do not use `results\R5-WikiArt` as Random5 until a manifest proves it contains the Random5 styles. |
| `R5-WikiArt` table label | The v4 table uses `R5-WikiArt`, but current local evidence suggests this label conflates Random5, D5-style legacy packets, and possibly random20/wikiarts20. Treat every R5 table row as needing source-level audit. |
| Large image moves | Do not bulk-move `R5-WikiArt` images yet. First derive a manifest by parsing filenames and, where needed, matching to source/target protocol manifests. |

## 5. Next work items

1. Add metric-source columns to `results_manifest.csv`: `clip_s_source`, `lpips_source`, `musiq_source`, and `main_table_value_status`.
2. For P2A-256, decide whether to derive `cut` from the 2427 remote images or rerun/export a clean 750 packet.
3. For P2A-256 Seedream, either repair the 721 packet or mark the v4 row as table-only without clean local images.
4. For R5, separate true `R5-512` from `R5-WikiArt-legacy` and `random20-or-mixed`; do not merge these into one table claim.
5. For D5, prune or ignore nested duplicate copies and repair `styleshot` missing 5 images if the row is to remain in the table.
