# PC-Lowpass Eval-Only Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003`.
- Mechanism: eval-only `solver_family=solver_pc` with `solver_corrector_mode=latent_lowpass`.
- Matched control: parent config behavior, transfer `0.671820 / 0.314618` and all-pairs `0.703234 / 0.312550`.
- Changed knobs: only `solver_corrector_step_size = 0.03, 0.06, 0.10`; `solver_corrector_steps=2` and lowpass kernel stayed fixed.
- Unchanged: tokenizer, topogate, losses, appearance path, checkpoint, dataset, seed, eval surface.
- Remote execution: sequential eval-only runs on the RTX 3060; no checkpoint pullback and no generated image grids.

## Results

| step size | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | transfer delta | all-pairs delta |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.671606 | 0.313594 | 0.703035 | 0.311723 | -0.000214 / -0.001025 | -0.000199 / -0.000827 |
| 0.06 | 0.671214 | 0.312733 | 0.702729 | 0.311048 | -0.000606 / -0.001885 | -0.000504 / -0.001501 |
| 0.10 | 0.671096 | 0.311748 | 0.702628 | 0.310271 | -0.000725 / -0.002870 | -0.000606 / -0.002279 |

Runtime per point was about `149-152s`, with eval-only GPU memory around `2.4 GiB` during the health window and idle after the sweep.

## Decision

Decision: `structure_repair_not_style_path`.

PC lowpass correction does exactly what the implementation implies: it pulls structure closer to the source and lowers LPIPS, but it also lowers CLIP-style at every tested step size. Under the current style-priority target, this is not a promotable path and should not receive a training lane by itself.

Keep PC as a possible final safety correction if a future style-strong mechanism overshoots LPIPS, but do not use it as the primary route to reach `0.74 / 0.30`.
