# Protocol A Engineering Table

Source CSV: `Related_Works/baseline_pipeline/results/protocol_eval_table_protocol_a_800.csv`

Reference manifest: `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`

Current protocol: `5 source styles x 5 target styles x 30 source images = 750 outputs`; styles are `photo / monet / vangogh / cezanne / Hayao`. `ukiyoe` is intentionally excluded.

| Baseline | Images | CLIP-style up | CLIP-content up | LPIPS-content down | Eval time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ours_pareto_probe_4_epoch_0001` | 750 | 0.6908 | 0.8394 | 0.4184 | 21.7s |
| `cut` | 750 | 0.7588 | 0.7794 | 0.4906 | 23.0s |
| `samst` | 750 | 0.7253 | 0.7752 | 0.5390 | 21.9s |
| `s2wat` | 750 | 0.7138 | 0.7464 | 0.5263 | 21.6s |
| `styleid` | 750 | 0.7777 | 0.6402 | 0.5928 | 27.7s |
| `sdturbo` | 750 | 0.7769 | 0.6505 | 0.6265 | 21.6s |
| `sdedit_str_0p10` | 750 | 0.7023 | 0.8759 | 0.3236 | 21.7s |
| `sdedit_str_0p20` | 750 | 0.7063 | 0.7772 | 0.4087 | 21.7s |
| `sdedit_str_0p35` | 750 | 0.6966 | 0.6899 | 0.4904 | 23.1s |
| `sdedit_str_0p40` | 750 | 0.6968 | 0.6727 | 0.5155 | 21.8s |

Notes:

- This is a live engineering table, not the final AAAI claim table.
- ArtFID/FID/CFSD and paper-exact `20 x 40 = 800` protocol refresh are still pending.
- StyleID inference wall time is recorded in `runtime_summary_protocol_a_800.csv`: about `614-622s` per target, about `51.4min` total for five targets.
