# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_memdrop_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_memdrop_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.292131 | 0.099937 | 0.110203 | 0.012618 | 0.993349 | 0.012231 |
| hl | 0.254422 | 0.091123 | 0.097479 | 0.010285 | 0.994745 | 0.009017 |
| hh | 1.214170 | 0.283832 | 0.274070 | 0.089695 | 0.954743 | 0.078982 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
