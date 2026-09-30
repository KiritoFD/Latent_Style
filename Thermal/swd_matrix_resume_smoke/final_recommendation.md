# SWD Matrix Recommendation

- Final epoch: 1
- Selected best patch (from E1): 2

## Baseline (E0)

- margin_mean: -0.125840
- identity_mse_latent_mean: 0.368916
- hf_ratio_mean: 0.000557

## Final-Epoch Deltas vs E0

| Experiment | dMargin | dIdentity% | dHF% | dDiversity | Effective |
|---|---:|---:|---:|---:|---|
| E1_p2 | 0.0000 | 0.00% | 0.00% | 0.0000 | False |

## Notes

- `noise_shortcut=true` means margin improved but high-frequency ratio overshot threshold.
- `structure_break=true` means identity structure degraded by more than 10%.