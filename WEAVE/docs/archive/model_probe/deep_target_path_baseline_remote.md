# Deep Target/Condition Path Probe

Config: `I:\Github\Latent_Style\SchrodingerBridge\configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
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
| ll | 2.476530e-01 | 5.244086e-01 | 0.000000e+00 |
| lh | 1.820686e-01 | 3.511469e-01 | 0.000000e+00 |
| hl | 2.229519e-01 | 3.145343e-01 | 0.000000e+00 |

## Model Debug

- `v_ll_abs`: 1.840233e-01
- `v_lh_abs`: 1.362381e-01
- `v_hl_abs`: 1.445515e-01
- `style_latent_conditioning_active`: 0.000000e+00
- `target_latent_token_fusion_active`: 0.000000e+00
