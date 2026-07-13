# Target-HF residual direction probe

Config: `configs\exp_probe_target_hf_subband_mixer_ft6.json`
Checkpoint: `exp\model_probe\target_hf_subband_mixer_ft6\epoch_0006.pt`
Mode: `eval`, t-values: `[0.25, 0.5, 0.75]`

## Per-band summary

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| lh | 0.285285 | 0.098707 | 0.110965 | 0.012582 | 0.993258 | 0.012335 |
| hl | 0.247758 | 0.088968 | 0.095676 | 0.009868 | 0.994916 | 0.008654 |
| hh | 1.194434 | 0.275148 | 0.265882 | 0.084733 | 0.957087 | 0.074614 |

## Reading

The residual is directionally useful under the training velocity target, but the orthogonal fraction indicates how much of it is not target-aligned.
