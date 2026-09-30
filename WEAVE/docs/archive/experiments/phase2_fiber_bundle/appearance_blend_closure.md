# Appearance-Blend Eval-Only Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003`.
- Mechanism: eval-only `model.output_appearance_blend`.
- Matched control: parent config behavior, recorded as transfer `0.671820 / 0.314618` and all-pairs `0.703234 / 0.312550`.
- Changed knobs: only `output_appearance_blend = 0.0, 0.5, 1.0`.
- Unchanged: tokenizer, solver, topology gate, losses, checkpoint, dataset, seed, eval surface.
- Remote execution: sequential eval-only runs on the RTX 3060; no checkpoint pullback and no generated image grids.

## Results

| blend | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | transfer delta | all-pairs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.671748 | 0.314596 | 0.703189 | 0.312540 | -0.000072 / -0.000022 | -0.000044 / -0.000010 |
| 0.5 | 0.671748 | 0.314596 | 0.703189 | 0.312540 | -0.000072 / -0.000022 | -0.000044 / -0.000010 |
| 1.0 | 0.671744 | 0.314595 | 0.703187 | 0.312539 | -0.000076 / -0.000023 | -0.000047 / -0.000011 |

Runtime per point was about `151s`, with eval-only GPU memory around `2.4 GiB` during the health window and idle after the sweep.

## Decision

Decision: `flat_no_training_value`.

The output appearance affine path is not the current style bottleneck under this parent. Varying the blend across the full range creates only noise-level changes and slightly lowers CLIP-style. Do not allocate a long training lane to output-appearance-blend tuning unless a separate mechanism first shows material style response.

The cheap-first policy remains active: training-side changes now require a short probe or eval-only prior showing at least a clear `+0.005` transfer/all-pairs style gain without reopening LPIPS.
