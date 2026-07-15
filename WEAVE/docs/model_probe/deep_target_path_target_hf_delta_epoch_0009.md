# Deep Target/Condition Path Probe

Config: `configs\exp_probe_target_hf_delta_ft15.json`
Checkpoint: `exp\model_probe\target_hf_delta_ft15\epoch_0009.pt`
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
| ll | 3.390794e-01 | 3.660954e-01 | 0.000000e+00 |
| lh | 2.207019e-01 | 4.702380e-01 | 4.229822e-02 |
| hl | 2.759756e-01 | 4.061183e-01 | 1.440104e-01 |
| hh | 1.886720e-01 | 5.792471e-01 | 4.357769e-02 |

## Model Debug

- `v_ll_abs`: 2.557144e-01
- `v_lh_abs`: 1.598107e-01
- `v_hl_abs`: 1.759055e-01
- `style_latent_conditioning_active`: 1.000000e+00
- `target_latent_token_fusion_active`: 0.000000e+00
- `target_latent_hf_head_fusion_active`: 1.000000e+00
- `target_latent_hf_head_gate`: 7.982977e-02
- `target_latent_hf_delta_gate_mean`: 9.797674e-02
- `v_hh_abs`: 1.360502e-01
