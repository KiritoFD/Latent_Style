# Probe 713 Style Path Summary

Config: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\config.json`
Checkpoint: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\epoch_0005.pt`
Samples: 16

## Mode Ranking By Latent Style Transfer Ratio

Latent ratios are not DINO-S. They are used only to select candidates for DINO-S evaluation.

| mode | LH ratio | HL ratio | HH ratio | global L2 content | time s |
|---|---:|---:|---:|---:|---:|
| configured | 0.2334 | 0.2722 | 0.0889 | 0.301337 | 5.085 |
| per_subband_wct | 0.2334 | 0.2722 | 0.0889 | 0.301337 | 4.892 |
| per_subband_adain | 0.2270 | 0.2633 | 0.0819 | 0.299682 | 4.912 |
| configured_hh_off | 0.2334 | 0.2722 | 0.0000 | 0.293751 | 4.926 |
| configured_lhhl_strong_hh_base | 0.1568 | 0.2063 | 0.0889 | 0.309671 | 5.011 |
| configured_strong | 0.1568 | 0.2063 | 0.0000 | 0.318653 | 4.940 |
| no_endpoint | 0.0000 | 0.0146 | 0.0000 | 0.287521 | 5.174 |
| spatial_fiber_adain | 0.0000 | 0.0000 | 0.0000 | 0.308067 | 4.864 |
| spatial_fiber_wct | 0.0000 | 0.0000 | 0.0000 | 0.310216 | 4.978 |

## Endpoint Delta Over Flow

| band | flow delta | endpoint delta | endpoint/flow |
|---|---:|---:|---:|
| ll | 0.566322 | 0.000000 | 0.000 |
| lh | 0.069069 | 0.092325 | 1.337 |
| hl | 0.071994 | 0.094749 | 1.316 |
| hh | 0.000000 | 0.134380 | 1840713.014 |

## Learned Velocity Style-Swap Sensitivity

| band | across-style std | mean rms |
|---|---:|---:|
| ll | 0.662215 | 0.960637 |
| lh | 0.051795 | 0.114643 |
| hl | 0.052889 | 0.118889 |

## Block Cross-Attention Snapshot

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.054926 | 0.063849 | 0.338538 | 3.718563 |
| 1 | 0.042748 | 0.027478 | 0.426785 | 1.985792 |
| 2 | 0.044891 | 0.058538 | 0.482270 | 5.250878 |
| 3 | 0.043530 | 0.075964 | 0.589825 | 8.371243 |
