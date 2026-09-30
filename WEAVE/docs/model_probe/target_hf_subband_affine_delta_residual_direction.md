# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_affine_delta_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_affine_delta_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.284910 | 0.098967 | 0.106091 | 0.012208 | 0.993786 | 0.011214 |
| hl | 0.270778 | 0.096741 | 0.099738 | 0.011451 | 0.994266 | 0.009567 |
| hh | 1.182564 | 0.286488 | 0.278927 | 0.089059 | 0.954862 | 0.079656 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
