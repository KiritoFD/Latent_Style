# Probe 713 Style Path Summary

Config: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\config.json`
Checkpoint: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\epoch_0005.pt`
Samples: 4

## Mode Ranking By Latent Style Transfer Ratio

Latent ratios are not DINO-S. They are used only to select candidates for DINO-S evaluation.

| mode | LH ratio | HL ratio | HH ratio | global L2 content | time s |
|---|---:|---:|---:|---:|---:|
| configured | 0.2652 | 0.3469 | 0.1526 | 0.243923 | 0.877 |
| per_subband_wct | 0.2652 | 0.3469 | 0.1526 | 0.243923 | 0.867 |
| configured_strong | 0.2652 | 0.3469 | 0.1526 | 0.243923 | 0.879 |
| per_subband_adain | 0.2579 | 0.3389 | 0.1455 | 0.242084 | 0.860 |
| spatial_fiber_wct | 0.0066 | 0.1980 | 0.0000 | 0.253654 | 0.887 |
| spatial_fiber_adain | 0.0000 | 0.1961 | 0.0000 | 0.251391 | 0.901 |
| no_endpoint | 0.0000 | 0.0072 | 0.0000 | 0.227970 | 0.861 |

## Endpoint Delta Over Flow

| band | flow delta | endpoint delta | endpoint/flow |
|---|---:|---:|---:|
| ll | 0.444474 | 0.000000 | 0.000 |
| lh | 0.068494 | 0.082472 | 1.204 |
| hl | 0.075047 | 0.098767 | 1.316 |
| hh | 0.000000 | 0.130405 | 1862734.332 |

## Learned Velocity Style-Swap Sensitivity

| band | across-style std | mean rms |
|---|---:|---:|
| ll | 0.662215 | 0.960637 |
| lh | 0.051795 | 0.114643 |
| hl | 0.052889 | 0.118889 |

## Block Cross-Attention Snapshot

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.054926 | 0.080336 | 0.338325 | 4.638953 |
| 1 | 0.042748 | 0.026605 | 0.457955 | 1.819208 |
| 2 | 0.044891 | 0.062471 | 0.522413 | 5.255525 |
| 3 | 0.043530 | 0.098727 | 0.628046 | 9.735327 |
