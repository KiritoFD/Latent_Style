# Phase2 Fiber Bundle Experiment Ledger

This folder stores the controlled-variable Fiber Bundle sweep artifacts.

## Live Files

- `plot_points.csv`: fixed input table for the homepage CLIP-style / LPIPS progress plots. Every closed experiment must append or update its full/all-pairs and transfer rows here before the closure note is final.
- `curves/`: raw per-run all-checkpoint CLIP-S / LPIPS curves copied from remote eval artifacts before they are normalized into `plot_points.csv`.

## Current Homepage Overlay

- `k070` epoch `1-5` and `pattn_enhanced_tok` epoch `1-9` are plotted on the AAAI2027 page-1 IDT/SaMAM/Seedream CLIP-S / LPIPS panel.
- The trace uses transfer `CLIP-S - IDT` on the y-axis and `1 - LPIPS` on the x-axis.
- All retained checkpoints are drawn and connected.
- Labels are sparse by design:
  - `k070`: `e1` and `e3 best LPIPS`
  - `pattn_enhanced_tok`: `e2 best style` and `e8 low LPIPS`
- Source curve:
  - [k070_epoch1_5_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/k070_epoch1_5_remote_clip_lpips_curve.csv)
  - [pattn_enhanced_tok_epoch1_9_remote_clip_lpips_curve.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/curves/pattn_enhanced_tok_epoch1_9_remote_clip_lpips_curve.csv)
- Rendered page-1 figure:
  - [fig_distinct5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_distinct5_page1_summary.png)

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
