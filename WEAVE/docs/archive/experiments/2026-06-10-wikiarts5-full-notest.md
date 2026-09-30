# WikiArts-5 Full-Notest Dataset

Date: 2026-06-10

Purpose:

- replace the earlier `1000-per-style` Distinct5 train subset with a larger five-style train pool
- keep the current Distinct5 test split fixed
- materialize a reusable train root for later 512 resize / latent caching / tokenizer-family relaunches

Source and exclusion:

- full source root:
  - `F:\wikiart\wikiart`
- excluded test split:
  - `F:\wikiart_distinct5_samam_512_classview\test`
- output root:
  - `F:\wikiarts_5_full_notest`
- materialization mode:
  - `hardlink`

Per-style counts:

- `Early_Renaissance`
  - source: `1391`
  - excluded test: `30`
  - kept train: `1361`
- `Impressionism`
  - source: `13060`
  - excluded test: `30`
  - kept train: `13030`
- `Minimalism`
  - source: `1337`
  - excluded test: `30`
  - kept train: `1307`
- `Rococo`
  - source: `2089`
  - excluded test: `30`
  - kept train: `2059`
- `Ukiyo_e`
  - source: `1167`
  - excluded test: `30`
  - kept train: `1137`

Totals:

- source images:
  - `19044`
- excluded test images:
  - `150`
- kept train images:
  - `18894`

Artifacts:

- train root:
  - `F:\wikiarts_5_full_notest\train`
- summary json:
  - `F:\wikiarts_5_full_notest\summary.json`
- summary csv:
  - `F:\wikiarts_5_full_notest\summary.csv`

Implementation:

- built with:
  - [build_wikiarts5_full_dataset.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/build_wikiarts5_full_dataset.py)
- exclusion rule:
  - test stems are normalized by removing the optional `Style__` prefix before matching against the full source stems
- verification:
  - `missing_test_stems_in_source = 0` for all five styles
