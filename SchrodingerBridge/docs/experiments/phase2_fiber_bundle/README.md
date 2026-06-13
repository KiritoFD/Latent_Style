# Phase2 Fiber Bundle Experiment Ledger

This folder stores the controlled-variable Fiber Bundle sweep artifacts.

## Live Files

- `plot_points.csv`: fixed input table for the homepage CLIP-style / LPIPS progress plots. Every closed experiment must append or update its full/all-pairs and transfer rows here before the closure note is final.
- `curves/`: raw per-run all-checkpoint CLIP-S / LPIPS curves copied from remote eval artifacts before they are normalized into `plot_points.csv`.
- `smoe_training_manifest.csv`: Round-2 SMoE-only launch and closure status.

## Current Homepage Overlay

- `k070` epoch `1-5`, `pattn_enhanced_tok` epoch `1-10`, Fiber-SDE `sigma=0.01/0.02/0.03/0.05`, and SMoE epoch `1-3` are plotted on the AAAI2027 page-1 IDT/SaMAM/Seedream CLIP-S / LPIPS panel.
- The trace uses transfer `CLIP-S - IDT` on the y-axis and `1 - LPIPS` on the x-axis.
- All retained checkpoints are drawn and connected.
- Labels are sparse by design:
  - `k070`: `e1` and `e3 best LPIPS`
  - `pattn_enhanced_tok`: `e2 best style` and `e8 low LPIPS`
  - `smoe_translator_k070_e3`: `SMoE e1` and `SMoE e2`; e3 is plotted but unlabeled to avoid collisions.
- Source curve:
  - [k070_epoch1_5_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/k070_epoch1_5_remote_clip_lpips_curve.csv)
  - [pattn_enhanced_tok_epoch1_10_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/pattn_enhanced_tok_epoch1_10_remote_clip_lpips_curve.csv)
  - [eval/fiber_sde_sigma0p01/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p01/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p01/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p01/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p02/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p02/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p02/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p02/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p03/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p03/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p03/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p03/fiber_aligned/summary.json)
  - [eval/fiber_sde_sigma0p05/isotropic/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p05/isotropic/summary.json)
  - [eval/fiber_sde_sigma0p05/fiber_aligned/summary.json](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/eval/fiber_sde_sigma0p05/fiber_aligned/summary.json)
  - [smoe_translator_k070_e3_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/smoe_translator_k070_e3_remote_clip_lpips_curve.csv)
- Rendered page-1 figure:
  - [fig_distinct5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary.png)

## Current Fiber-SDE Matched Delta

- `sigma=0.01` isotropic: transfer `0.671501 / 0.313795`, all-pairs `0.703024 / 0.311868`.
- `sigma=0.01` fiber-aligned: transfer `0.671581 / 0.313762`, all-pairs `0.702954 / 0.311888`.
- Decision: `inconclusive_tie`; transfer improves by `+0.000080` style and `-0.000033` LPIPS, while all-pairs is slightly worse by `-0.000070` style and `+0.000020` LPIPS.
- `sigma=0.02` isotropic: transfer `0.672031 / 0.314990`, all-pairs `0.703432 / 0.313025`.
- `sigma=0.02` fiber-aligned: transfer `0.671818 / 0.314936`, all-pairs `0.703320 / 0.313015`.
- Decision: `conservative_not_promoted`; fiber-aligned preserves slightly more structure but loses style against the matched isotropic control.
- `sigma=0.03` isotropic: transfer `0.673391 / 0.316894`, all-pairs `0.704514 / 0.314930`.
- `sigma=0.03` fiber-aligned: transfer `0.673405 / 0.316883`, all-pairs `0.704633 / 0.314862`.
- Decision: `marginal_positive_continue`; fiber-aligned is better than isotropic on both style and LPIPS, but the delta is still small.
- `sigma=0.05` isotropic: transfer `0.675927 / 0.322953`, all-pairs `0.706639 / 0.320868`.
- `sigma=0.05` fiber-aligned: transfer `0.675948 / 0.323189`, all-pairs `0.706763 / 0.321093`.
- Decision: `style_upper_not_promoted`; this is the style-first upper point of the scan, but it pays clear LPIPS cost and does not reach the `0.74 / 0.30` target.

## Active Queue

- Round 2 is now `smoe_translator_k070_e3`.
- This run starts from the same `k070 epoch_0003` parent used by Fiber-SDE.
- Only `tokenizer_family` changes from `pure_latent_spatial` to `smoe_translator`; solver/loss/topogate/appearance/schedule stay inherited from the parent line.
- Remote full eval must run every epoch and update the homepage page-1 CLIP-style / LPIPS figure before closure.
- Current read through epoch 3: e1 is best for style, e2/e3 recover LPIPS while losing style, and all three remain dominated by `k070 epoch_0003`; continue only under the formal curve rule before closing SMoE-only.

## Plot Update Contract

Use `tools/experiments/update_phase2_plot_points.py` after each completed eval:

```bash
python SchrodingerBridge/tools/experiments/update_phase2_plot_points.py \
  --curve-csv <clip_lpips_curve.csv> \
  --family "FiberBundle" \
  --variant "<experiment_id>" \
  --trace-id "<experiment_id>" \
  --label-prefix "<short label>"
```

Then regenerate the homepage figures:

```bash
python SchrodingerBridge/aaai2027/scripts_gen_distinct5_full_transfer_pareto.py
python SchrodingerBridge/aaai2027/scripts_gen_distinct5_all_points_big.py
python SchrodingerBridge/aaai2027/scripts_gen_distinct5_page1_summary.py
```

Or pass `--render` to `update_phase2_plot_points.py`; it refreshes the phase2 CSV consumers including the AAAI2027 page-1 summary figure.
