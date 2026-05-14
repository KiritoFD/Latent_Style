# Content Edge Purity Metrics

These diagnostics target SaMST-like failures where semantic layout remains recognizable but output edges are dominated by texture/grain rather than content structure.

| method | run | images | content_edge_purity_up | content_edge_energy_share_up | flat_edge_energy_share_down | strong_edge_extra_rate_down | orientation_consistency_up | lowpass_grad_corr_up | flat_chroma_energy_share_down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ours | epoch_0007 | 750 | 0.42008 | 0.62313 | 0.25657 | 0.15355 | 0.73079 | 0.70059 | 0.38414 |
| Ours | epoch_0008 | 750 | 0.39922 | 0.61659 | 0.26303 | 0.16539 | 0.72763 | 0.66685 | 0.38500 |
| Ours | residual_1p25 | 750 | 0.35725 | 0.59937 | 0.27803 | 0.19459 | 0.71474 | 0.61357 | 0.38330 |
| SaMST | samst_strict | 750 | 0.60655 | 0.63995 | 0.19886 | 0.06902 | 0.92843 | 0.89792 | 0.37874 |
