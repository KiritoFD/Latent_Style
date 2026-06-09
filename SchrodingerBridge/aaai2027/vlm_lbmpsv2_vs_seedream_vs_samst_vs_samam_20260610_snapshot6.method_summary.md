# Four-Way VLM Snapshot 6

Source snapshot:

- [vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_vs_samam_20260610_snapshot6.jsonl)

Method summary:

| Method | Cases | Wins | WinRate | StyleWins | StructWins | ArtifactWins | MeanStyle | MeanStruct | MeanArtifact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Seedream_repaired750` | 205 | 116 | 0.566 | 113 | 108 | 96 | 4.185 | 4.444 | 4.151 |
| `SaMAM_2250` | 205 | 89 | 0.434 | 88 | 92 | 108 | 3.912 | 4.371 | 4.302 |
| `SaMST_e15` | 205 | 0 | 0.000 | 4 | 0 | 0 | 3.444 | 3.312 | 2.980 |
| `LBM-PS-v2_e13` | 205 | 0 | 0.000 | 0 | 5 | 1 | 1.683 | 1.941 | 1.610 |

Current read:

- `Seedream` still leads overall.
- `SaMAM_2250` remains the strongest non-Seedream external baseline.
- `SaMAM_2250` remains ahead on the cleaner-image axes:
  - artifact wins
  - mean artifact control
- `Seedream` remains ahead on style identity and total overall wins.
- `SaMST_e15` remains style-active but non-winning.
- `LBM-PS-v2_e13` remains substantially behind all external baselines.
