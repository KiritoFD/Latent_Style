# Cross-Cache Dedup Hash Audit - 2026-06-05

This pass audits local and remote cache duplicates. It performs no deletion.

## Scope

Local roots opened:

- `eval_cache`
- `SchrodingerBridge\eval_cache`
- `Cycle-NCE\eval_cache`

Remote roots opened:

- `I:\Github\Latent_Style\eval_cache`
- `I:\Github\Latent_Style\SchrodingerBridge\eval_cache`
- `I:\Github\Latent_Style\Cycle-NCE\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\eval_cache`
- `I:\Github\Latent_Style_TokenizerClean\Cycle-NCE\eval_cache`

## Outputs

- `cross_cache_local_root_inventory_20260605.csv`
- `cross_cache_local_duplicate_hashes_20260605.csv`
- `cross_cache_local_duplicate_groups_20260605.csv`
- `cross_cache_remote_root_inventory_20260605.csv`
- `cross_cache_remote_duplicate_hashes_20260605.csv`
- `cross_cache_remote_duplicate_groups_20260605.csv`
- `cross_cache_dedup_summary_20260605.csv`
- `cross_cache_dedup_policy_20260605.csv`

## Findings

| scope | duplicate groups | SHA256 match groups | hash mismatch groups | potential reclaim if one copy retained |
| --- | ---: | ---: | ---: | ---: |
| local cache roots | 5 | 5 | 0 | 105.292363 MB |
| remote cache roots | 26 | 24 | 2 | 583.884105 MB |

Local exact duplicates include `art_inception.pth` and several `ref_feats_*.pt` files shared across root, `SchrodingerBridge`, and `Cycle-NCE` eval caches.

Remote exact duplicates include several `ref_feats_*.pt` files and manual CLIP cache files shared across `I:\Github\Latent_Style\eval_cache` and `I:\Github\Latent_Style\Cycle-NCE\eval_cache`. Remote mismatch groups prove why name/size-only deletion is unsafe: some same-name/same-size files have different SHA256 hashes, including `refs\main` and a mixed VAE safetensors group.

## Policy

No cache files were deleted in this pass.

Even exact SHA256 duplicates are retained for now because code and historical reproduction may expect cache-local paths, especially:

- `Cycle-NCE\eval_cache`
- `SchrodingerBridge\eval_cache`
- remote `I:\Github\Latent_Style\Cycle-NCE\eval_cache`

The next cleanup step is a loader/path-reference audit. A deletion whitelist is only valid after proving either that all consumers use the canonical root cache or that the duplicate root can be replaced by a documented symlink/redirect without breaking reproduction.

## Current Decision

- `all_hash_match=True`: retain pending loader/path audit.
- `all_hash_match=False`: retain, not a duplicate.
- `deletion_whitelist`: `no` for every row in `cross_cache_dedup_policy_20260605.csv`.
