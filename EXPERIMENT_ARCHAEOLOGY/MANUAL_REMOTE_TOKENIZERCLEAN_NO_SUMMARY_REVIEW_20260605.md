# Remote TokenizerClean no-summary checkpoint review - 2026-06-05

## Scope

Remote root:

`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

This pass reviewed the 28 checkpoint directories that were left after the first TokenizerClean cleanup because they had no `summary.json`. The rule was stricter than the previous pass: no summary means the checkpoint may be the only payload, so deletion is allowed only for short probe/calibration dirs with config/training evidence retained.

## Inputs

- `manual_remote_tokenizerclean_no_summary_review_20260605.csv`
- `manual_remote_tokenizerclean_no_summary_cleanup_policy_20260605.csv`
- `manual_remote_tokenizerclean_exp_internal_evidence_after_no_summary_cleanup_20260605.csv`

The review opened:

- top-level file lists;
- weight names;
- `config.json` when present;
- `training_*.csv` tail rows when present;
- `remote_train.log` when present;
- summary existence checks.

## Review result before deletion

| class | dirs | checkpoint size | decision |
|---|---:|---:|---|
| `uncited_probe_or_calibration_no_summary` | 18 | 362.391 MB | delete checkpoint only |
| `manual_review_required` | 7 | 379.322 MB | keep until owner review or summary generation |
| `orphan_weight_no_config_no_summary` | 3 | 170.017 MB | keep until owner review |

## Deleted in this pass

Deleted:

- 18 checkpoint files.
- `362.391 MB`.

Deletion ledger:

- `cleanup/manual_remote_tokenizerclean_no_summary_probe_checkpoint_cleanup_20260605.csv`
- `cleanup/manual_remote_tokenizerclean_no_summary_probe_checkpoint_cleanup_by_dir_20260605.csv`

Deleted directories were all uncited, no-summary, short probe/calibration forms such as:

- `wikiart512_ema_direct_atom_residual_calib_b16/b64/b80`
- `local_patch_org_*_120b_*`
- `pair_content_spatial_head_*_120b_*`
- `pairrel_signed_spatial_head_*_120b_*`
- `split_axis_*_120b_*`

The deletion script only removed `.pt/.ckpt/.pth`; it retained config/log/training CSV evidence.

## Kept after review

Still retained:

| retained class | dirs | files | size | reason |
|---|---:|---:|---:|---|
| `keep_payload_until_summary_or_owner_review` | 7 | 10 | 379.322 MB | no summary but training CSV suggests a real payload; checkpoint may be needed to produce missing eval |
| `keep_orphan_until_owner_review` | 3 | 11 | 170.017 MB | no config and no summary, so deletion needs owner-level provenance review |

Representative retained dirs:

- `tokenizer_t01_carrier_base_b160`
- `wikiart_distinct5_ema_lancet_spectralstat_e2_b80`
- `wikiart512_ema_spectral_stat_full_e2_from_tok_b48`
- `axis_scale_probe`
- `field_budget_release_probe`
- `pair_relative_geometry_release_probe`

## TokenizerClean checkpoint state after this pass

| class | dirs | remaining checkpoint files | remaining size |
|---|---:|---:|---:|
| cited/docs/master/paper | 34 | 122 | 3813.414 MB |
| current `aaai2027_*` packets | 9 | 24 | 1451.217 MB |
| kept no-summary payload/review | 10 | 21 | 549.339 MB |
| already cleaned uncited summary-backed dirs | 44 | 0 | 0 MB |
| cleaned no-summary probe/calibration dirs | 18 | 0 | 0 MB |
| no checkpoint dirs | 30 | 0 | 0 MB |

Total TokenizerClean `exp` checkpoint state after this pass:

- 167 checkpoint files.
- `5813.970 MB`.

## Remaining gap

The remaining TokenizerClean checkpoint cleanup is now narrow:

1. Produce or recover summaries for 7 retained no-summary payload dirs, or confirm they are obsolete.
2. Owner-review 3 orphan dirs without config/summary.
3. Apply packet-specific thinning to current/cited dirs only after docs/master citation migration.
4. Handle generated-image evidence separately; this pass did not touch PNG/CSV/summary surfaces.
