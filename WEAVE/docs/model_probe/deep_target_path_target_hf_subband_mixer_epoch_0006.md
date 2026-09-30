# Deep Target/Condition Path Probe

Config: `configs\exp_probe_target_hf_subband_mixer_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_mixer_ft6\epoch_0006.pt`
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
| ll | 2.789186e-01 | 4.628740e-01 | 0.000000e+00 |
| lh | 2.001834e-01 | 4.806342e-01 | 3.036061e-02 |
| hl | 2.406332e-01 | 4.459361e-01 | 9.657972e-02 |
| hh | 1.666844e-01 | 6.001784e-01 | 1.249495e-01 |

## Model Debug

- `v_ll_abs`: 2.073379e-01
- `v_lh_abs`: 1.489604e-01
- `v_hl_abs`: 1.603417e-01
- `style_latent_conditioning_active`: 1.000000e+00
- `target_latent_token_fusion_active`: 0.000000e+00
- `target_latent_hf_head_fusion_active`: 0.000000e+00
- `target_latent_hf_spatial_fusion_active`: 0.000000e+00
- `target_latent_hf_subband_fusion_active`: 1.000000e+00
- `target_latent_hf_subband_mixer_active`: 1.000000e+00
- `target_latent_hf_texture_fusion_active`: 0.000000e+00
- `target_latent_hf_spatial_energy_fusion_active`: 0.000000e+00
- `target_latent_hf_subband_delta_gate_mean`: 1.727611e-01
- `target_latent_hf_subband_mixer_gate`: 2.038604e-01
- `target_latent_hf_subband_mixer_offdiag_abs`: 3.592532e-03
- `v_hh_abs`: 1.259142e-01
