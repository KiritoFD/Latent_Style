# AAAI 2027 Writing Gate R20260603O

Date: 2026-06-03

Updated: 2026-06-04

## Scope

This gate records the current writing-only pass after the Distinct5/IDT narrative was tightened. It does not integrate new experiment claims.

Authoritative manuscript:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`

## Completed writing changes

- Abstract now leads with the IDT-floor failure mode and frames LBM as transport-first stylization.
- Distinct5 prose now treats SaMAM rows as measured checkpoints, not convergence endpoints.
- SaMAM is now described by the factual target-style test: transfer CLIP-S remains below IDT at the reported checkpoints. LPIPS is not used to reject SaMAM.
- Follow-up correction: SaMAM is not described as simply unchanged or as failing because of high LPIPS. The manuscript now states that reproduced SaMAM checkpoints can visibly alter color, contrast, and local texture, but those edits are not stably target-directed because transfer CLIP-S remains below IDT.
- SaMST is described separately as the larger-displacement / high-ArtFID branch after clearing IDT.
- Table 4 caption explicitly says SaMAM rows are not a convergence endpoint.
- Method wording replaces informal "semantic beacon" phrasing with tokenizer-as-control-signal language.
- Method follow-up resolves the active-objective mismatch: the reported Distinct5 algorithm is now written as vector-field execution with optional endpoint residuals disabled in the headline OMF rows; stochastic bridge flow matching is kept only as parent lineage / optional family context.
- Reviewer-A follow-up resolves the remaining OT overclaim: the title no longer says `OT-coupled`, the inference path no longer includes target construction, and Distinct5 headline rows are described as pairing-cache / terminal-SWD / kinetic OMF objectives with `w_flow=0.0`.
- Distinct5 headline LBM config disclosure is recorded in `SchrodingerBridge/docs/experiments/distinct5_512_20260602/resolved_headline_config.md`; it lists the resolved F/H/K config paths, shared active objective (`objective_mode=omf`, `w_flow=0.0`, `terminal_swd_weight=20.0`, `w_kinetic=1.0`), training setup, and row-specific queue/tokenizer differences. The main paper keeps only the compact active-objective statement in Method to avoid breaking layout.
- SaMAM wording was rechecked after user correction: its Distinct5 failure is target-style CLIP-S below IDT, not high LPIPS. Visible edits and ArtFID movement are treated as auxiliary observations that do not override the IDT-calibrated CLIP-S test.
- Discussion/conclusion avoid vague "retained frontier" wording and use measured positive-IDT operating-point language.
- Follow-up wording pass removed remaining verdict-style phrasing such as "decisive observation", broad "stricter test" language, theorem-like "guarantee" phrasing, and unqualified "clearest evidence" claims.
- Final-paper tone pass reduced remaining internal experiment language: "current mainline", "current bottleneck", "baseline frontier", "wins diagnostics", and "content collapse" were replaced with selected-configuration, measured-operating-point, and style/content-trade-off language where appropriate.
- Layout follow-up removed the forced `\clearpage` before the reproducibility checklist after it created a nearly blank references page; the PDF is back to 13 pages.
- Distinct5 is now framed as a CLIP-separated stress case study, with broader non-CLIP-screened and additional split validation marked as future/pending evidence rather than current proof.
- Reviewer-D follow-up tightened the split/statistics boundary: Distinct5 now states that the CLIP-prototype screen is fixed, uses only WikiArt class images, does not inspect generated outputs or checkpoints, and has an auditable split-selection note at `SchrodingerBridge/docs/experiments/wikiart_stress_splits_20260603/split_selection_audit.md`.
- The broader WikiArt512 setup is no longer dismissed because the IDT prior is "too high"; it is framed as an internal convergence reference with lower separation, while additional fixed-rule splits remain the validation path.
- The reproducibility checklist remains `Partial`: historical strict-750 has paired bootstrap intervals; Distinct5 now has paired bootstrap intervals for retained IDT-aligned LBM/SaMST transfer rows, while SaMAM rows remain point-estimate operating points until a complete paired packet is integrated.

## Hard evidence boundary

Do not write the following until new packets land:

- SaMAM has converged on Distinct5.
- SaMAM 2500/2750/3000 targetwise ArtFID trends.
- LBM longer-training improves the frontier.
- Any Faraday/Hypatia long-training result without full_eval plus targetwise ArtFID.
- Additional WikiArt stress splits support the IDT conclusion.
- That SaMAM Distinct5 IDT deltas are statistically significant. Its rows remain point estimates until paired per-image metrics land.
- Historical strict-750 table is a matched leaderboard; keep it contextual unless protocols are normalized further.
- Distinct5 is broadly validated beyond the current five-style stress split. Additional fixed-rule stress splits have been selected/materialized, but no performance claim is allowed until IDT, LBM, baseline summaries, full_eval, and targetwise ArtFID land.

Current SaMAM status from Dalton:

- 2250 has transfer CLIP-S / LPIPS / targetwise ArtFID.
- 2500/2750/3000 have transfer CLIP-S and LPIPS, but targetwise ArtFID is not complete.
- 3000 ArtFID reuse evaluation was still running at last report.
- Dalton has been retasked to deliver a paper-safe convergence packet with exact paths, full/transfer summaries, targetwise ArtFID, and paired bootstrap CIs for SaMAM-vs-IDT transfer CLIP-S if matching per-image metrics are available. Do not wait on this for writing-only cleanup, and do not integrate it until the packet is complete.

Current Faraday/Hypatia status from the latest checkpoint:

- F-longer has a complete e1-e8 full_eval packet at `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_longer_train_f_seed42_b44_e8`.
- F-longer failed the retention gate: best transfer CLIP-S was e8 `0.666298`, below the `+0.006` improvement gate over F base `0.664360`, and all LPIPS values were worse than F base `0.324528`.
- No F-longer ArtFID was computed by rule; this is not a paper-result improvement.
- K-longer training completed 8 checkpoints, but the eval packet is incomplete. Recovery eval reached e1-e4 and then stopped because `/mnt/i` was full (`448G / 448G, 0 available`). Do not write any K-longer conclusion until e5-e8 full_eval plus retention/ArtFID decisions complete.

Current extra-split status:

- The deterministic split-selection artifact is complete at `SchrodingerBridge/docs/experiments/wikiart_stress_splits_20260603/selected_splits.json`.
- Three disjoint WikiArt stress splits were selected and materialized under `Dataset/wikiart_stress_splits_512/`.
- This supports the process claim that follow-up splits are selected by a fixed rule, but it does not support any performance claim yet.
- Latent encoding, IDT baselines, LBM-F runs, full_eval, transfer/full summaries, and targetwise ArtFID are still incomplete for the new splits.

## Verification

Build command:

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission
cmd /c build_paper.bat
```

Result:

- PDF builds successfully.
- 13 pages.
- No undefined references/citations found.
- No overfull or two-column-too-tall warnings found.
- Latest build after the objective/SaMAM correction reports only a harmless `fixltx2e` package warning in the searched warning set.
- After the Reviewer-D evidence-boundary patch, the paper still builds to 13 pages with the same harmless warning class.
- After the 2026-06-04 SaMAM/IDT wording cleanup, the paper still builds successfully to 13 pages; log search finds no undefined references/citations, no overfull warnings, and no float errors.
- After the Reviewer-A objective-alignment patch, the paper still builds successfully to 13 pages (`paper_aaai2026.pdf`, 3,360,623 bytes); log search finds no undefined references/citations, no overfull warnings, and no float errors. The manuscript body no longer contains the unsafe `OT-coupled`, `path-wise flow`, or SaMAM-high-LPIPS failure wording.
- After the resolved-config disclosure patch, the paper still builds successfully to 13 pages (`paper_aaai2026.pdf`, 3,360,624 bytes); log search finds no undefined references/citations, no overfull warnings, and no float errors. Rendered checks of pages 1, 8, 9, 10, and 13 found no new layout break.
- Distinct5 paired bootstrap evidence was added for retained IDT-aligned transfer rows at `SchrodingerBridge/docs/experiments/distinct5_512_20260602/bootstrap/paired_idt_transfer_bootstrap.csv`. It covers LBM-F/H/K and SaMST e5/e15; SaMAM remains point-estimate only until a complete per-image packet is integrated.
- Table 4 was relabeled with compact column headers (`C-S`, `AFID`, `Delta_f`, `Delta_t`) to improve single-column readability without changing metrics.
- Pages 1, 7, 8, 9, and 13 were rendered for visual inspection after the latest wording/layout pass; page 9 was specifically checked after the Table 4 resize fix.

## Reviewer round

One four-reviewer adversarial round was launched after this writing gate:

- Reviewer A: method / novelty.
- Reviewer B: experimental fairness / statistics.
- Reviewer C: writing / clarity.
- Reviewer D: skeptical baseline / evaluation.

Reviewers were instructed to read only the current manuscript, not modify files, and not run experiments.

Latest Reviewer D result:

- Score `4/10`, reject-leaning.
- Main risks: Distinct5 may read as CLIP-selected around the IDT story; LBM's positive-IDT gains are small even though retained transfer rows now have paired bootstrap support; SaMAM/SaMST convergence is not fully established; cost accounting is selected-checkpoint rather than time-to-parity; historical strict-750 must remain contextual.
- Writing response already applied: SaMAM is framed as target-direction failure rather than LPIPS failure, historical strict-750 wording is contextualized, and the active training objective was aligned with the pairing-cache / terminal-SWD / kinetic OMF setting.
- Additional writing response applied: split-selection wording now states the fixed pre-output screen and auditable artifact; checklist/statistical wording now limits Distinct5 significance to rows with retained IDT-aligned paired metrics.
- Still needed for a stronger revision: full class-screening appendix/ranked list in the supplement if it can be reconstructed, SaMAM paired bootstrap if matching per-image metrics become available, complete baseline curves / best-last-AUC where available, and follow-up split evidence.

## Follow-up experiment owner

Because reviewers could perceive Distinct5 as a tailored split, a split-selection tool was added:

- `SchrodingerBridge/tools/select_wikiart_stress_splits.py`

The intended follow-up protocol is to select additional high-separation WikiArt splits with a fixed CLIP-prototype rule, then run IDT plus LBM full evaluation on remote hardware. This is pending evidence and is not integrated into the paper yet.
