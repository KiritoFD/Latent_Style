# AAAI 2027 writing iteration round 8

Date: 2026-06-04

Scope:
- Main draft: `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- Current PDF: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`

## Purpose

This pass treats the paper as an evaluation-contract paper:

- IDT is the primary claim.
- Distinct5-512 is the CLIP-S stress split.
- LBM is the compact proof point for low-displacement positive target movement.
- Dalton/Faraday experiment packets are not integrated until they close full_eval plus targetwise ArtFID.

## Writing changes

- Rewrote the abstract from dense metric listing into a problem-contract-evidence-method structure.
- Reframed the first page around the null hypothesis: in art-to-art Style-ID transfer, the unchanged artwork must be scored before raw target-style scores are interpreted.
- Tightened the introduction so the progression is:
  - raw target-style scores miss the unchanged-source counterfactual;
  - IDT turns that counterfactual into signed movement;
  - Distinct5 exposes three regimes;
  - LBM targets low-damage positive movement.
- Tightened contribution bullets:
  - IDT as falsification contract;
  - LBM as executable style-control proof point;
  - IDT-calibrated WikiArt evidence with cost.
- Tightened related-work wording around Style-ID systems: the point is not that compact systems are impractical, but that their target-direction evidence needs IDT.
- Rewrote method overview and representation prose to make "executable control" the central representation claim.
- Reduced defensive wording in experiments and discussion while preserving evidence boundaries:
  - SaMAM remains a point-estimate claim.
  - ArtFID remains a diagnostic, not a target-direction metric.
  - Retained-checkpoint timing remains separated from full search/evaluation cost.

## Layout fixes

- Added `\raggedbottom` to avoid stretched final-page columns.
- Wrapped the bibliography in `\small` so the final reference does not spill onto an otherwise empty page.
- Rebuilt PDF after edits.

## Verification

- Build command: `cmd /c build_paper.bat`
- Output: `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`
- Page count: 11.
- Log scan: no unresolved citations/references, no fatal errors, no overfull boxes.
- Remaining warnings: XeLaTeX font substitution warnings and underfull boxes only.
- Rendered pages inspected: page 1, page 10, page 11.

## Incremental reviewer pass

Four existing reviewer agents reviewed the round-8 state after the structural
writing pass:

- Area-chair / writing: 8.1/10.
- Statistics / experimental validity: 7.2/10.
- Method / math: 7.4/10.
- Figures / layout: 7.4/10.

Consensus:

- The paper is now coherent as an evaluation-contract paper.
- The remaining acceptance ceiling is mostly evidence closure and visual
  persuasiveness, not abstract/intro coherence.
- The next major writing change should be triggered by closed Dalton/Faraday
  packets or audited replacement qualitative examples.

Applied after the reviewer pass:

- Scoped the abstract and conclusion final contract to CLIP-S-based art-to-art
  Style-ID evaluation.
- Changed SaMAM abstract wording to point-estimate / pending paired-packet
  closure.
- Changed SaMAM/figure phrasing from generic ArtFID improvement to lowering an
  ArtFID-style aggregate while failing target movement.
- Clarified retained-checkpoint timing in the abstract as selected Distinct5
  run time excluding search and evaluation.
- Rewrote the LBM contribution bullet as a proof-point claim rather than a
  component list.
- Added method boundary text: "Matching" is endpoint projection matching, not
  bridge-law or transport-map estimation.
- Clarified Euler execution as a deterministic refinement heuristic, not a
  learned path discretization.
- Clarified the semantic projection estimator as batch-paired and axis-biased,
  used for endpoint gradients rather than population-distance estimation.
- Changed Table 1 row-sign entries to sign-only so bracketed exploratory ranges
  are not visually mistaken for formal confidence intervals.
- Weakened representation-causality wording to "coupled mechanisms associated
  with better executed movement."

## Figure 3 follow-up

- Re-audited the six candidate rows in
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/visual_metric_alignment_20260602/distinct5_visual_alignment_manifest.json`.
- Replaced the first Figure 3 row from `Impressionism -> Minimalism` with
  `Early_Renaissance -> Ukiyo-e`, because it is still an audited aligned row
  and gives clearer LBM target movement:
  - IDT CLIP-S: `0.663`
  - LBM-F CLIP-S / LPIPS: `0.724 / 0.396`
  - LBM-K CLIP-S / LPIPS: `0.712 / 0.450`
- Kept `Rococo -> Ukiyo-e` as the low-displacement contrast row.
- Regenerated
  `SchrodingerBridge/aaai_submission/figures/fig_distinct5_visual_alignment_grid_panel.jpg`
  using `scripts_gen_distinct5_visual_panel.py`.
- Updated Figure 3 caption to mark the examples as audited aligned transfers.

## Dalton sidecar status

Dalton checked the Distinct5 SaMAM/SaMST packet state after round 8.

Current conclusion:

- No fully closed Distinct5 SaMAM/SaMST packet exists yet.
- No SaMAM/SaMST run was active on the remote 3060 during the audit.
- Remote GPU appeared idle for this sidecar line.

Closest SaMST artifacts:

- e5 summary/metrics/targetwise ArtFID under
  `Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b1_e5_20260603/eval_bundle/eval_epoch5/epoch_0005/`.
- e15 summary/metrics/targetwise ArtFID under
  `Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b2_e15_20260602/eval_epoch15/epoch_0015/`.
- e5-vs-e15 comparison under
  `Related_Works/baseline_pipeline/results/samst_distinct5_512_real_b1_e5_20260603/eval_bundle/compare_e5_vs_e15/`.

Closest SaMAM artifacts:

- remote curve root:
  `I:/Github/Latent_Style/Related_Works/baseline_pipeline/results/samam_distinct5_512_mamba_b6_seg250_remote_wsl_20260601_2130_diag/eval_curve`
- includes evaluated steps `2250`, `2500`, `2750`, and `3000`.
- local later artifacts exist for `3250`, but the paper should not integrate
  them until row alignment and timing are proven.

Still missing for a closed baseline packet:

- same-scope timing bound into the packet;
- IDT-aligned per-image row report or explicit missing-row report.

Dalton has been asked to close or document the packet from existing artifacts
only, without starting long training and without editing the main paper.

## Current risks

- SaMAM Distinct5 remains point estimates until Dalton closes the paired packet.
- Distinct5 remains one fixed CLIP-separated stress split until additional splits close.
- Figure 3 remains visually conservative for LBM and should be improved only with audited examples, not handpicked unaudited images.
- The current score ceiling is likely limited more by evidence closure and visuals than by abstract/intro coherence.
