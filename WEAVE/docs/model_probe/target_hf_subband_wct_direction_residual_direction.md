# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_wct_direction_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_wct_direction_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.284923 | 0.098685 | 0.111331 | 0.012631 | 0.993223 | 0.012413 |
| hl | 0.247603 | 0.088948 | 0.095890 | 0.009885 | 0.994905 | 0.008695 |
| hh | 1.190456 | 0.274769 | 0.265848 | 0.084640 | 0.957178 | 0.074541 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
