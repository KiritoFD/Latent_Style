# Remote TokenizerClean Cited/Current Media Manifest - 2026-06-05

Scope:

```text
I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp
```

This manifest follows the earlier keep/no-delete policy for the 26 retained
cited/current media directories. It reopens the exact same 26 remote paths and
maps each directory to representative artifacts: summaries, CSV/metrics, grids,
generated media buckets, and checkpoints where present.

No deletion was performed.

## Live Recheck Method

- A temporary remote helper was used only to avoid Windows command-line length
  limits.
- The helper contained a fixed list of the 26 exact directory names. It did not
  discover directories by scanning `exp`.
- For each exact path it counted files and sampled artifact paths for JSON
  summaries, CSV files, grid-like media, generated media buckets, and checkpoint
  weights.

Row-level manifest:

- `manual_remote_tokenizerclean_cited_current_media_manifest_20260605.csv`

## Manual Conclusions

- The 8 `aaai2027_*` packet directories are current/formal full-eval packets.
  Each retains three `full_eval\epoch_*` summaries, metrics CSVs, grids,
  750-image generated buckets per epoch, and `epoch_0001.pt` through
  `epoch_0003.pt`.
- The 18 cited media directories are heterogeneous evidence surfaces. Large
  examples include `diagnostics`, `configs`, and
  `wikiart512_ema_spectral_stat_full_adapt_e2_b48`; each has summary/CSV/grid
  evidence and generated media that must stay path-addressable until a separate
  archive migration manifest exists.
- `configs` remains a path-ambiguity case: root `SchrodingerBridge\configs`
  is not the media-bearing cited surface; this manifest records
  `SchrodingerBridge\exp\configs`.
- `wikiart512_ema_spectral_stat_full_adapt_e2_b48` remains the largest retained
  cited/current model packet in this manifest, with 3608 media files, 8
  summary-like JSON files, 9 CSV files, and 8 weight-like files.

## Cleanup Boundary

All 26 rows remain `delete_whitelist=no`.

Future archive/migration can only happen after an owner-approved manifest maps
the old cited paths to new archive paths and preserves at least:

- summary JSON files
- metrics/training CSV files
- summary grids or representative grids
- generated image buckets selected by owner
- checkpoints for current/formal or cited model packets

Do not delete or move these directories by media count, size, or extension.
