# Results Snapshot 2026-06-11

- Generated at: `2026-06-10T18:47:06.522100+00:00`
- Source root: `G:\GitHub\Latent_Style\SchrodingerBridge\aaai2027`
- Total moved files: `265`

## Counts

- `baselines/json`: `6`
- `baselines/logs`: `14`
- `introstyle_dino/json`: `2`
- `misc/csv`: `1`
- `misc/logs`: `3`
- `paper_audit/json`: `8`
- `paper_audit/text`: `4`
- `round1/csv`: `14`
- `round1/json`: `16`
- `round1/jsonl`: `8`
- `round1/logs`: `60`
- `vlm/csv`: `77`
- `vlm/json`: `1`
- `vlm/jsonl`: `33`
- `vlm/logs`: `18`

## Contract

- Root-level loose result files are moved under this snapshot by topic and file type.
- `index.csv` is the machine-readable manifest.
- Active live `SaMST` watcher logs are intentionally left outside this snapshot while training is running.

## Current Summary

- `SaMST` is no longer stuck at the first public eval point; the segmented auto-resume controller is active and is currently advancing the common frontier from `epoch_0005` to `epoch_0010`.
- The wikiarts5 new-data variant board has been stabilized onto a fixed point CSV, so plotting and annotation edits now come from one source of truth.
- The main cleanup target in this pass is aaai2027 root clutter: small CSV / log / JSON / JSONL / TXT files are being consolidated into this timestamped snapshot instead of remaining loose at root.

## Current Conclusion

- Keep experiment evidence, but stop leaving it flat at aaai2027 root.
- Preserve live runs and formal result packets; archive or delete only temporary scripts and disposable scratch artifacts after they are indexed.

