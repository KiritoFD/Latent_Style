# Deep Target/Condition Path Probe

Config: `configs\exp_probe_target_latent_fusion_ft15.json`
Checkpoint: `exp\model_probe\target_latent_fusion_ft15\epoch_0012.pt`
Data root: `I:\datasets\wikiart_distinct5_samam_512_latents_ema\train`
Load info: `{'missing': 0, 'unexpected': 0}`

## Training Target Bands

| band | target-content RMS | target-style RMS | target/content | target/style | target delta RMS |
|---|---:|---:|---:|---:|---:|
| ll | 4.434661e-01 | 1.761756e+00 | 3.301107e-01 | 1.168133e+00 | 4.434661e-01 |
| lh | 7.744394e-01 | 1.677997e-08 | 1.384662e+00 | 3.081957e-08 | 7.744394e-01 |
| hl | 8.221098e-01 | 1.214474e-08 | 1.383380e+00 | 2.138209e-08 | 8.221098e-01 |
| hh | 7.348568e-01 | 1.121391e-08 | 1.397395e+00 | 2.175436e-08 | 7.348568e-01 |

## Condition Sensitivity

| band | base RMS | style-id delta/base | target-latent delta/base |
|---|---:|---:|---:|
| ll | 2.865427e-01 | 9.172430e-02 | 1.070553e+00 |
| lh | 2.285213e-01 | 2.467504e-01 | 2.829781e-01 |
| hl | 3.182508e-01 | 1.714922e-01 | 2.366293e-01 |

## Model Debug

- `v_ll_abs`: 2.915109e-01
- `v_lh_abs`: 1.680171e-01
- `v_hl_abs`: 1.908585e-01
- `style_latent_conditioning_active`: 1.000000e+00
- `target_latent_token_fusion_active`: 1.000000e+00
