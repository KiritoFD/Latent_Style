# Deep Target/Condition Path Probe

Config: `configs\exp_probe_target_latent_fusion_ft15.json`
Checkpoint: `exp\model_probe\target_latent_fusion_ft15\epoch_0015.pt`
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
| ll | 2.934506e-01 | 9.572907e-02 | 1.063186e+00 |
| lh | 2.211324e-01 | 2.874823e-01 | 3.077617e-01 |
| hl | 3.063448e-01 | 2.017439e-01 | 2.586555e-01 |

## Model Debug

- `v_ll_abs`: 2.947118e-01
- `v_lh_abs`: 1.645456e-01
- `v_hl_abs`: 1.851318e-01
- `style_latent_conditioning_active`: 1.000000e+00
- `target_latent_token_fusion_active`: 1.000000e+00
