# QEdge vs DualPath VLM Direct Interim

Date: 2026-06-09

This note tracks the direct blind local comparison:

- `QEdgePattn e01`
- `DualPath e01`
- `Seedream`

Unlike the earlier family-level local review, this packet places the two current successor families into the same comparison panel, so the read is no longer only indirect through:

- `vs Knee`
- or `vs Seedream` separately

## Live artifacts

- manifest:
  - [vlm_manifest_qedgee01_vs_dualpathe01_vs_seedream_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_qedgee01_vs_dualpathe01_vs_seedream_20260609.csv)
- compact combined board:
  - [qedge_vs_dualpath_interim_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedge_vs_dualpath_interim_board_20260609.csv)
  - [qedge_vs_dualpath_interim_board_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/qedge_vs_dualpath_interim_board_20260609.md)
- current method summary:
  - [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.method_summary.csv)
- current interim summary:
  - [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.interim_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.interim_summary.csv)
- live stdout:
  - [vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.stdout.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_qedgee01_vs_dualpathe01_vs_seedream_20260609.stdout.log)

## Current interim read

Completed cases:

- `557`

Current overall wins:

- `Seedream = 547 / 557`
- `DualPath e01 = 10 / 557`
- `QEdgePattn e01 = 0 / 557`

Current mean local scores:

- `DualPath e01`
  - style `2.162`
  - structure `3.512`
  - artifact `2.621`
- `QEdgePattn e01`
  - style `1.968`
  - structure `3.345`
  - artifact `2.379`
- `Seedream`
  - style `4.847`
  - structure `4.618`
  - artifact `4.814`

## Current interpretation

This batch is still far too early to close the family race.

But it already supports three provisional reads:

1. `Seedream` is still dominating both current successor families head-to-head.
   - It is no longer literally `all cases won`, but it remains overwhelmingly dominant.

2. Within the internal successor pair:
   - `DualPath e01` is currently reading slightly cleaner than `QEdgePattn e01` on local perceptual means.
   - This is directionally consistent with the earlier family-level local `VLM` read.

3. The direct blind panel is not fully one-sided:
   - `QEdgePattn e01` already has more discrete `best_structure` subwins.
   - `DualPath e01` now keeps a clearer mean structure/artifact edge as the blind panel grows.
   - `DualPath e01` has now picked up `10` direct overall blind wins.

This does not overturn the corrected first `DINO` structure read:

- `QEdgePattn e01 DINO = 0.02635`
- `DualPath e01 corrected DINO = 0.02635`
- `DualPath e09 DINO = 0.02742`

So the current combined interpretation remains:

- `QEdgePattn`
  - tied first structure-side geometry at `e01`
  - but already ahead on discrete structure-side blind subwins
- `DualPath`
  - slightly better current perceptual mean cleanliness
  - and slightly better mean structure/artifact scores
  - with `10` direct overall blind wins now on the board
- `Seedream`
  - still overwhelmingly ahead on the actual blind compare
