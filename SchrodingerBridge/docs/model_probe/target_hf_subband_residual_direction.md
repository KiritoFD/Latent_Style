# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.285418 | 0.098730 | 0.111191 | 0.012611 | 0.993232 | 0.012381 |
| hl | 0.248043 | 0.089052 | 0.095598 | 0.009870 | 0.994926 | 0.008637 |
| hh | 1.194715 | 0.275021 | 0.265744 | 0.084656 | 0.957125 | 0.074540 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
