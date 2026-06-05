# Local Generated Media Owner Decision Manifest - 2026-06-05

Scope:

```text
G:\GitHub\Latent_Style\SchrodingerBridge\exp\diagnostics\seedream_gap
G:\GitHub\Latent_Style\SchrodingerBridge\exp\inference\inference_param_sweep_t01e8_quick
G:\GitHub\Latent_Style\SchrodingerBridge\exp\inference\inference_param_sweep_t01e8_fine
```

This manifest follows pass5 and keeps the same cleanup boundary: no deletion was
performed. It turns the prior `retain_pending_owner` rows into an owner decision
packet with exact child/point rows, counts, sizes, and representative samples.

## Live Manual Checks

- `seedream_gap`: 7 child input sets, each 750 JPG files, total 5250 JPG files.
- `inference_param_sweep_t01e8_quick`: 14 parameter points, each 250 JPG files,
  total 3500 JPG files.
- `inference_param_sweep_t01e8_fine`: 8 parameter points, each 250 JPG files,
  total 2000 JPG files.
- Representative images were visually opened from seedream_gap, quick, and fine
  and were valid generated/style-transfer images.
- All paths remain ignored/untracked under the `SchrodingerBridge/exp/**`
  generated-output rule.

Row-level manifest:

- `manual_local_generated_media_owner_decision_manifest_20260605.csv`

## Owner Options

For each row, the owner can choose:

- Keep: retain as visual/diagnostic lineage.
- Migrate: preserve the listed sample path plus a compact parameter/sample
  manifest, then archive or move bulk images with path mapping.
- Delete later: only after an explicit owner-approved exact whitelist and a
  post-delete verification CSV.

## Current Decision

All rows remain `delete_whitelist=no`.

No directory in this manifest is a cleanup target yet because none has duplicate
proof, closed metric replacement proof, or owner authorization.
