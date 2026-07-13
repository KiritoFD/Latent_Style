# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_current_delta_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_current_delta_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.285986 | 0.098894 | 0.111223 | 0.012634 | 0.993228 | 0.012402 |
| hl | 0.248166 | 0.089101 | 0.095835 | 0.009902 | 0.994901 | 0.008692 |
| hh | 1.193613 | 0.275104 | 0.266101 | 0.084791 | 0.957042 | 0.074713 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
