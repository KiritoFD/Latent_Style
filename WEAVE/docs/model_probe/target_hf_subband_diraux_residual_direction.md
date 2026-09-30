# Target-HF subband diraux residual direction probe

Historical config: `configs/exp_probe_target_hf_subband_diraux_ft6.json`
Historical checkpoint: `exp/model_probe/target_hf_subband_diraux_ft6/epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.581918 | 0.242326 | 0.256700 | 0.067984 | 0.964839 | 0.066603 |
| hl | 0.520730 | 0.205745 | 0.221228 | 0.050775 | 0.974216 | 0.046955 |
| hh | 1.119453 | 0.534817 | 0.488628 | 0.248351 | 0.868121 | 0.236473 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
