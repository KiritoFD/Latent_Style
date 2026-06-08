# Content Edge Purity Metrics

These diagnostics target SaMST-like failures where semantic layout remains recognizable but output edges are dominated by texture/grain rather than content structure.

| method | run | images | content_edge_purity_up | content_edge_energy_share_up | flat_edge_energy_share_down | strong_edge_extra_rate_down | orientation_consistency_up | lowpass_grad_corr_up | flat_chroma_energy_share_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LBM | LBM-K_e1 | 450 | 0.54015 | 0.67093 | 0.20724 | 0.11572 | 0.76773 | 0.84628 | 0.34152 |
| LBM | LBM-Knee_e13 | 450 | 0.28772 | 0.58010 | 0.28290 | 0.29945 | 0.70014 | 0.49848 | 0.33431 |
| LBM | LBM-PS-v2_e13 | 450 | 0.11520 | 0.47788 | 0.35434 | 0.38732 | 0.66271 | 0.35907 | 0.41273 |
| SaMST | SaMST_e15 | 450 | 0.10152 | 0.48623 | 0.33735 | 0.40418 | 0.59622 | 0.35199 | 0.38909 |
| Seedream | Seedream_repaired750 | 450 | 0.38853 | 0.62217 | 0.23867 | 0.25977 | 0.75997 | 0.60275 | 0.32137 |
