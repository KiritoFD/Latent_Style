# AAAI 2027 Figure Audit - R20260603M

- `review_cycle_id`: `R20260603M`
- `lane`: `figure_audit`
- `checkpoint_label`: `current_paper_after_agent_cleanup_and_partial_path_stability_launch`
- `scope`: `Re-audit the current main-paper figure surface against the landed evidence boundary after cleanup and the still-unlanded path-stability packet.`

- `overall_status`: `weak_reject`
- `claim_safety_band`: `narrow_only`
- `evidence_closure_band`: `partial`
- `blocking_issue`: `The current figure surface is still fragmented around one artifact claim and still carries a selective mechanism-facing ablation figure, while no landed path-stability figure exists to justify a broader kinetic/path-energy visual story. The safe reviewer-facing surface is therefore still only the Distinct5 no-op-aware frontier plus bounded qualitative and ablation support.`
- `next_action_1`: `Merge fig_qual_grid_ours_vs_samst.png and fig_zoom_ours_vs_samst.png into one composite artifact-diagnosis figure with linked inset markers so one figure, not two, carries the muddy/grain-like texture claim.`
- `next_action_2`: `Demote fig_ablation_pareto.png from the main-paper set unless it is rebuilt as a plain-language full-set mechanism figure; keep framework_lbm_main_v5.png and figures/fig_distinct5_pareto.pdf, and do not add a path-stability figure until the remote packet lands cleanly.`
- `support_score`: `1`
- `fairness_score`: `1`
- `artifact_path_score`: `1`
- `closure_value_score`: `1`

## Lane answers

- `Is the current main-paper figure set still fragmented or misleading for a reviewer?`
  - `Yes: fragmented more than outright false. The strongest safe spine is still framework + Distinct5 no-op-aware frontier + one merged qualitative artifact figure. Keeping separate grid/zoom figures and a selective ablation plot makes the surface feel broader and more settled than the current closure really is.`

- `Which exact figure should be merged, demoted, or rebuilt next?`
  - `Merge`: `fig_qual_grid_ours_vs_samst.png` + `fig_zoom_ours_vs_samst.png`
  - `Demote or rebuild`: `fig_ablation_pareto.png`
  - `Keep as the main paper spine`: `framework_lbm_main_v5.png`, `figures/fig_distinct5_pareto.pdf`

- `Does the absence of a landed path-stability figure imply the current figure surface must stay narrower?`
  - `Yes. Without a landed path-stability packet, the paper should not visually imply closed kinetic/path-energy mechanism support. The figure surface should stay centered on the Distinct5 no-op-aware frontier and the current qualitative/ablation stack, with mechanism visuals treated as bounded support rather than new closure.`
