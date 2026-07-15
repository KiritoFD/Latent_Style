# AAAI 2027 Figure Audit - R20260603N

Date: 2026-06-03

Checkpoint label:

- `current_paper_after_agent_cleanup_before_next_path_stability_integration`

1. overall_status: `weak_reject`

2. claim_safety_band: `narrow_only`

3. evidence_closure_band: `partial`

4. blocking_issue: The main figure surface is still broader than the landed evidence spine. The Distinct5 frontier figure is strong and should remain central, but the paper still spends two separate figures on one SaMST artifact diagnosis while also carrying a visibly selective ablation scatter in the main body. That layout makes the paper look visually overcommitted on a qualitative side claim and undercommitted on the core bounded-evidence story. In addition, the current figure set does not directly visualize the no-op / ArtFID pathology that the text now treats as one of the sharpest takeaways.

5. next_action_1: Merge `fig_qual_grid_ours_vs_samst.png` and `fig_zoom_ours_vs_samst.png` into one artifact-diagnosis figure with a clear hierarchy: full 5x5 comparison first, then 2-3 linked zoom callouts that isolate the grain-like failure mode. This should consume one figure slot, not two.

6. next_action_2: Demote `fig_ablation_pareto.png` from the main paper unless it is rebuilt from a complete, non-selective packet. Its current title explicitly advertises selectivity ("6 of 12 points"), which weakens trust. If a new main-figure slot opens later, spend it on a landed same-family mechanism figure or a compact no-op / ArtFID pathology visual, not on the current selective scatter.

7. support_score (0/1/2): `1` - The framework figure and Distinct5 Pareto figure materially support the current paper, but the rest of the figure surface does not yet carry its weight cleanly.

8. fairness_score (0/1/2): `1` - The Distinct5 frontier is fair and no-op-aware, but the split qualitative presentation and selective ablation figure create avoidable cherry-pick optics.

9. artifact_path_score (0/1/2): `1` - The figure inventory and working index are usable, but the active paper still depends on a mixed top-level / `figures/` asset layout that the README itself marks as pending normalization.

10. closure_value_score (0/1/2): `1` - The current figure set helps the bounded Distinct5 story, but it does not close the live mechanism-side blocker and it still leaves the metric-pathology claim more textual than visual.

## Main-Figure Surface

- keep:
  - `SchrodingerBridge/aaai_submission/framework_lbm_main_v5.png`
  - `SchrodingerBridge/aaai_submission/figures/fig_distinct5_pareto.pdf`
- merge:
  - `SchrodingerBridge/aaai_submission/fig_qual_grid_ours_vs_samst.png`
  - `SchrodingerBridge/aaai_submission/fig_zoom_ours_vs_samst.png`
- demote:
  - `SchrodingerBridge/aaai_submission/fig_ablation_pareto.png`
- redraw:
  - one unified artifact-diagnosis figure that combines the grid and zoom evidence into a single visual argument
  - if an extra figure is later justified, use it to visualize the no-op / ArtFID contradiction or a landed same-family path-stability packet, not another selective ablation panel
