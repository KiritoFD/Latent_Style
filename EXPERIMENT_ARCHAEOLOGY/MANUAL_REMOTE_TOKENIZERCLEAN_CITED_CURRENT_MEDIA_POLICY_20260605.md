# Remote TokenizerClean Cited/Current Media Archive Policy - 2026-06-05

This pass manually source-opened the retained generated-media surfaces in
`I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge` after the prior
uncited-media cleanup.

No deletion was performed. The policy output is:

- `manual_remote_tokenizerclean_cited_current_media_archive_policy_20260605.csv`

## Inputs Opened

- `manual_remote_tokenizerclean_remaining_media_classes_after_cleanup_20260605.csv`
- `manual_remote_tokenizerclean_generated_media_inventory_after_cleanup_20260605.csv`
- `manual_remote_tokenizerclean_exp_citation_graph_all_20260605.csv`
- exact remote directories under `I:\Github\Latent_Style_TokenizerClean\SchrodingerBridge\exp`

Remote checks used:

```text
ssh -p 2222 -o LogLevel=ERROR administrator@100.115.18.62
```

## Scope

The retained media surface has 26 keep rows:

| class | dirs | media files | media MB |
| --- | ---: | ---: | ---: |
| current AAAI2027 packets | 8 | 18024 | 1777.997 |
| cited media dirs | 18 | 28459 | 5723.521 |
| total | 26 | 46483 | 7501.518 |

The 50 prior `delete_uncited_summary_backed_media` rows remain at zero media
after cleanup and are not part of this keep-policy pass.

## Manual Open Findings

All 8 current AAAI2027 packet directories were opened under remote `exp`.
Each has retained full-eval summaries, CSV evidence, 3 checkpoint weights, and
2253 media files. These are current/formal packet surfaces, not deletion
candidates.

The 18 cited media dirs were opened under remote `exp`. Important details:

- `diagnostics` is the largest cited surface: 7723 media files, 2868.443 MB,
  53 summary-like JSON files, 52 CSV files, and 3 paper hits.
- `wikiart512_ema_spectral_stat_full_adapt_e2_b48` is the next largest:
  3608 media files, 1759.842 MB, 8 summary-like JSON files, 9 CSV files, and
  8 weights.
- `configs` has a path ambiguity: root `SchrodingerBridge\configs` exists but
  has zero media; the media-bearing cited surface is `SchrodingerBridge\exp\configs`
  with 152 media files, 169 summary-like JSON files, and 224 CSV files.
- `runs`, `legacy`, and `diagnostics` have paper hits and must not be moved or
  thinned without a citation-preserving manifest.
- tokenizer_t01 dirs contain historical summaries, CSV files, retained weights,
  and generated media. They are cited evidence, not zero-hit media.

## Policy

No generated media in these 26 dirs is deleted in this pass.

Allowed future action:

- Build a citation-to-artifact manifest for each cited/current directory.
- For current AAAI2027 packets, archive generated images per epoch only after
  owner chooses which packet stays paper-facing.
- Keep summaries, metrics, CSV files, summary grids, configs, and checkpoints
  in place unless a separate owner-approved migration policy maps the original
  paths to new archive paths.
- For large cited surfaces such as `diagnostics` and
  `wikiart512_ema_spectral_stat_full_adapt_e2_b48`, create a curated
  representative image set before moving bulk generated images.

Blocked action:

- Do not delete these directories by media count or size.
- Do not treat root `configs` as proof that cited `exp/configs` media is absent.
- Do not archive or move paper-hit directories without a manifest that preserves
  the exact cited path relationship.

## Remaining Gap

This pass resolves the cited/current media delete question as `keep_no_delete`.
It does not create the actual archive tarballs or migration manifests. That is
a separate owner-approved migration task.
