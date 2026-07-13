# Probe 713 Style Path Summary

Config: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\config.json`
Checkpoint: `SchrodingerBridge\exp\710_infra_t11_distinct5_5ep\epoch_0005.pt`
Samples: 32

## Mode Ranking By Latent Style Transfer Ratio

Latent ratios are not DINO-S. They are used only to select candidates for DINO-S evaluation.

| mode | LH ratio | HL ratio | HH ratio | global L2 content | time s |
|---|---:|---:|---:|---:|---:|
| configured | 0.1949 | 0.2675 | 0.0866 | 0.274951 | 1.203 |
| per_subband_wct | 0.1949 | 0.2675 | 0.0866 | 0.274951 | 1.211 |
| per_subband_adain | 0.1889 | 0.2615 | 0.0791 | 0.273228 | 1.216 |
| configured_lhhl_strong_hh_base | 0.1607 | 0.2460 | 0.0866 | 0.284075 | 1.208 |
| configured_hh_off | 0.1949 | 0.2675 | 0.0000 | 0.266659 | 1.226 |
| configured_strong | 0.1607 | 0.2460 | 0.0000 | 0.293789 | 1.220 |
| spatial_fiber_wct | 0.0000 | 0.0459 | 0.0000 | 0.285042 | 1.201 |
| spatial_fiber_adain | 0.0000 | 0.0324 | 0.0000 | 0.282620 | 1.211 |
| no_endpoint | 0.0000 | 0.0104 | 0.0000 | 0.259653 | 1.203 |

## Endpoint Delta Over Flow

| band | flow delta | endpoint delta | endpoint/flow |
|---|---:|---:|---:|
| ll | 0.509675 | 0.000000 | 0.000 |
| lh | 0.069224 | 0.089231 | 1.289 |
| hl | 0.071548 | 0.096982 | 1.355 |
| hh | 0.000000 | 0.134030 | 1813636.060 |

## Learned Velocity Style-Swap Sensitivity

| band | across-style std | mean rms |
|---|---:|---:|
| ll | 0.662210 | 0.960653 |
| lh | 0.051794 | 0.114641 |
| hl | 0.052887 | 0.118890 |

## Learned Style-Swap Time Sweep

| t | LL std | LH std | HL std |
|---:|---:|---:|---:|
| 0.1 | 0.503028 | 0.047973 | 0.048927 |
| 0.5 | 0.662210 | 0.051794 | 0.052887 |
| 0.9 | 0.198149 | 0.036808 | 0.031234 |

## Path Separation

Same target style latent; only the learned `style_id` path or cross-attention route changes.

| scenario | LH ratio | HL ratio | HH ratio | content L2 | L2 to configured |
|---|---:|---:|---:|---:|---:|
| learned_target_no_endpoint | 0.0000 | 0.0104 | 0.0000 | 0.259653 | 0.093984 |
| learned_source_no_endpoint | 0.0000 | 0.0238 | 0.0000 | 0.182252 | 0.309786 |
| learned_shift_no_endpoint | 0.0000 | 0.0162 | 0.0000 | 0.263870 | 0.329534 |
| configured_target_endpoint_target_id | 0.1949 | 0.2675 | 0.0866 | 0.274951 | 0.000000 |
| configured_target_endpoint_source_id | 0.2238 | 0.2786 | 0.0866 | 0.203539 | 0.295414 |
| configured_target_endpoint_shift_id | 0.1991 | 0.2750 | 0.0866 | 0.279049 | 0.316155 |
| no_cross_attn_no_endpoint | 0.0000 | 0.0293 | 0.0000 | 0.199393 | 0.221538 |
| full_cross_attn_no_endpoint | 0.0000 | 0.0153 | 0.0000 | 0.399168 | 0.232512 |

## Block Cross-Attention Snapshot

| block | style gate | delta abs | ca in std | ca out std |
|---:|---:|---:|---:|---:|
| 0 | 0.054926 | 0.069902 | 0.343324 | 4.162324 |
| 1 | 0.042748 | 0.028658 | 0.441331 | 2.105918 |
| 2 | 0.044891 | 0.057700 | 0.498422 | 5.152232 |
| 3 | 0.043530 | 0.085879 | 0.602769 | 9.599442 |
