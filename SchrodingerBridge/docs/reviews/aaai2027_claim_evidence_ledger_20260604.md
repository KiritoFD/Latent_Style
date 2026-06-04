# AAAI 2027 Claim-Evidence Ledger

Date: 2026-06-04

Purpose: keep the AAAI draft aggressive but evidence-bound. Each claim below
maps the current paper language to the evidence that supports it, the wording
that is currently allowed, and the gate required before strengthening it.

Primary draft:
- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.pdf`

Core result roots:
- `SchrodingerBridge/docs/experiments/distinct5_512_20260602/`
- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/`
- `SchrodingerBridge/docs/experiments/dalton_aaai2027_experiment_backlog_20260604.md`

## Claim 1: IDT Is the Null for CLIP-S-Based Art-to-Art Style-ID

Current allowed wording:
- Under CLIP-S-based art-to-art evaluation, a Style-ID method that does not
  beat IDT has not demonstrated target-style movement under that evaluator.
- IDT is the unchanged-image floor, not a weak baseline.

Current evidence:
- Distinct5-512 IDT transfer CLIP-S floor: `0.6399`.
- IDT full CLIP-S: `0.6801`.
- Stored IDT outputs and summary:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/no_op_identity_5x5_summary.json`.
- Cross-dataset no-op audit notes:
  `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/`.

Do not write yet:
- IDT is a universal stylization metric correction.
- IDT proves all prior style-transfer leaderboards are invalid.
- IDT is independent of the target-style evaluator.

Upgrade gate:
- Add at least one non-CLIP target-direction validation, or close multiple
  fixed-rule WikiArt stress splits with consistent IDT behavior.

## Claim 2: Distinct5-512 Is a Fixed-Rule Stress Split, Not a Custom Benchmark

Current allowed wording:
- Distinct5-512 is a fixed CLIP-separated WikiArt stress split.
- It is intentionally separated but not exotic.
- It is a CLIP-S stress test rather than a universal benchmark.

Current evidence:
- Selected styles: Early Renaissance, Impressionism, Minimalism, Rococo,
  Ukiyo-e.
- Split protocol: 1000 train images and 30 held-out test images per class.
- Selector: CLIP ViT-B/32 class prototype separation before model-output
  inspection.
- Dataset and audit docs under:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/`.

Do not write yet:
- Distinct5 is a new standard benchmark.
- Distinct5 alone proves broad WikiArt robustness.

Upgrade gate:
- Close at least two additional fixed-rule stress splits with IDT plus LBM-F/K
  packets, targetwise ArtFID, and timing.

## Claim 3: SaMAM Is Valid Only Through 2250 in the Current Manuscript

Current allowed wording:
- SaMAM is reported at step 2250, the last trusted checkpoint in the reproduced
  Distinct5 run.
- Through step 2250, SaMAM lowers targetwise ArtFID while remaining below IDT
  in transfer CLIP-S.
- Later SaMAM outputs are excluded from manuscript evidence as reproduction-chain
  failures unless a clean independent rerun closes a new metric packet.

Current evidence:
- SaMAM 2250: transfer CLIP-S `0.5523`, LPIPS `0.3605`, targetwise ArtFID
  `148.2`, delta `-0.0877`, train `458.6` min.
- Validity boundary:
  `SchrodingerBridge/docs/experiments/2026-06-04-samam-distinct5-valid-through-2250.md`.
- Curve source still contains later rows for audit history, but these are not
  manuscript evidence under the current decision:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/tables/clip_style_vs_1lpips_full_transfer_points.csv`.

Do not write yet:
- SaMAM clears IDT on Distinct5.
- SaMAM fails after convergence.
- SaMAM is generally worse than IDT outside this validated checkpoint range.
- SaMAM has a same-scope inference cost comparison; pure generation timing is
  missing from the current packet.

Upgrade gate:
- Clean independent SaMAM rerun beyond 2250 with aligned outputs, transfer-only
  CLIP-S, LPIPS, targetwise ArtFID, train wall, and same-scope timing.

## Claim 4: SaMST Clears IDT but Pays High Damage

Current allowed wording:
- SaMST e15 clears the IDT transfer CLIP-S floor.
- Its positive target movement comes with high LPIPS and high targetwise ArtFID
  on Distinct5-512.
- e5/e15 CLIP-S and LPIPS are close; e15 lowers targetwise ArtFID relative to
  e5 but remains in the high-damage region.

Current evidence:
- SaMST e15: transfer CLIP-S `0.6957`, LPIPS `0.6319`, targetwise ArtFID
  `444.5`, delta `+0.0558`.
- SaMST e5: transfer CLIP-S `0.6989`, LPIPS `0.6335`, targetwise ArtFID
  `465.7`.
- e5/e15 transfer CLIP-S and LPIPS differ by less than `0.004/0.002`; e15
  lowers targetwise ArtFID by `21.2` but remains high.
- Existing-artifact packet:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/baseline_packet_status_20260604/`.
  It confirms e5/e15 full and transfer metrics, targetwise ArtFID, training
  wall time, and 750/750 IDT-aligned rows for both checkpoints.

Do not write yet:
- SaMST is fully converged on all metrics.
- SaMST's high-ArtFID behavior is universal beyond the closed e5/e15
  Distinct5 packet.

Upgrade gate:
- Complete SaMST timing closure by binding same-scope inference `ms/img` into
  the packet; optional e10/e30 points may strengthen convergence but are not
  required for the current e15 row wording.

## Claim 5: LBM-F/K Occupy the Low-Displacement Positive-IDT Region

Current allowed wording:
- LBM-F/K clear IDT in transfer-only and full-scope evaluation.
- LBM-F is the low-LPIPS point; LBM-K is the higher-style point.
- Retained checkpoints were reached after 1.2 recorded training minutes within
  selected runs, excluding search and evaluation.
- Compared with SaMAM 2250, LBM-F has higher transfer CLIP-S, lower LPIPS,
  lower targetwise ArtFID, and lower retained-checkpoint training time.

Current evidence:
- LBM-F: transfer CLIP-S `0.6644`, LPIPS `0.3245`, targetwise ArtFID `126.8`,
  delta `+0.0244`.
- LBM-K: transfer CLIP-S `0.6712`, LPIPS `0.3723`, targetwise ArtFID `162.0`,
  delta `+0.0312`.
- SaMAM 2250: transfer CLIP-S `0.5523`, LPIPS `0.3605`, targetwise ArtFID
  `148.2`, train `458.6` min.
- Full-scope values are in Table 1 of the draft.
- Headline config disclosure:
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/resolved_headline_config.md`.

Do not write yet:
- LBM is universally better than SaMST.
- LBM has a normalized total time-to-result advantage over all baselines.
- LBM is visually strong in every target direction.

Upgrade gate:
- Same-scope cost packet across LBM, SaMST, and SaMAM.
- Additional fixed-rule stress split evidence.
- Stronger qualitative packet or independent target-direction validation.

## Claim 6: ArtFID Is a Diagnostic, Not a Target-Direction Metric

Current allowed wording:
- ArtFID and IDT delta answer different questions.
- Lower ArtFID can reflect target-domain realism, structure preservation, or
  both; it is not target-direction evidence by itself.

Current evidence:
- SaMAM 2.25k lowers targetwise ArtFID to `148.2` while falling below IDT.
- SaMST clears IDT but has targetwise ArtFID `444.5`.
- LBM-F has lower targetwise ArtFID `126.8` with positive IDT delta.

Do not write yet:
- ArtFID is broken.
- ArtFID should be discarded.
- ArtFID is mainly a no-op reward in all settings.

Current visual evidence:
- Figure 3 uses audited aligned rows from
  `SchrodingerBridge/docs/experiments/distinct5_512_20260602/visual_metric_alignment_20260602/distinct5_visual_alignment_manifest.json`.
- The current paper panel includes `Early_Renaissance -> Ukiyo-e` and
  `Rococo -> Ukiyo-e`; it should not be treated as a hand-picked external
  qualitative claim.

Upgrade gate:
- Additional splits and independent target-direction validation.
- If ArtFID and IDT disagree systematically, write the disagreement as a
  measured diagnostic limitation, not a metric takedown.

## Claim 7: LBM Is an Endpoint-Supervised Latent Renderer, Not a Stochastic Bridge Claim

Current allowed wording:
- LBM uses bridge/flow language operationally for deterministic latent editing.
- The active object is an endpoint-trained vector-field renderer with detached
  endpoint selection, terminal projection discrepancy, and endpoint
  velocity-magnitude penalty.

Current evidence:
- Method section defines endpoint selection as a detached non-differentiable
  supervision step over `Q_s(z_0)`.
- The headline rows use `lambda_res=0`; local endpoint residual is inactive.
- Terminal semantic projection and endpoint velocity penalty are active.

Do not write yet:
- LBM estimates a Schrodinger bridge.
- LBM learns an OT map.
- LBM learns a continuous-time stochastic path.
- Endpoint-trained fixed-step recurrence is path training.

Upgrade gate:
- Endpoint-trained vs path-trained field check.
- 1/4/8/16/256-step endpoint stability probe under matched settings.

## Claim 8: Semantic-Axis Terminal Projection Is a Retained Heuristic

Current allowed wording:
- Semantic-axis projection is an asymmetric terminal discrepancy and a retained
  implementation heuristic.
- It should not be sold as a standalone mechanism until matched axis ablations
  close.

Current evidence:
- Current method uses routing-derived projection axes.
- Current ablations identify terminal endpoint pressure, not semantic axes
  independently.

Do not write yet:
- Semantic-axis projection is the causal reason LBM works.
- Semantic-axis projection is a new SWD estimator with unbiased OT meaning.

Upgrade gate:
- Matched semantic-axis vs random-axis terminal projection ablation with
  targetwise ArtFID and LPIPS/CLIP-S.

## Claim 9: Tokenizer Quality Means Executable Control

Current allowed wording:
- Code separability alone is insufficient.
- Current coupled sweeps support executable control as the representation
  hypothesis.
- Capacity alone did not solve the Distinct5 representation frontier.

Current evidence:
- Table 6 coupled representation/routing/queue variants.
- Class-local prototypes and global atoms did not improve the operating point.
- Content-guided routing and prototype-aware queues improved the frontier.

Do not write yet:
- Tokenizer design is causally solved.
- The current tokenizer is optimal.
- Routing, queue, and tokenizer effects are fully separated.

Upgrade gate:
- Fixed-tokenizer/fixed-executor ablation with same endpoint queue, budget,
  seed, batch size, endpoint velocity penalty, and terminal projection settings.

## Citation and Bibliography Audit

Current status:
- `refs.bib` has `38` entries and the current paper has `38` unique cited keys.
- Citation coverage check: `missing=[]`, `unused=[]`.
- Recent fragile entries have been manually checked and patched:
  `AesFA`, `SaMST`, `SCSA`, `S2WAT`, `LPIPS`, and `MUSIQ`.
- Final `build_paper.bat` pass has no undefined citation/reference or BibTeX
  warning; only template font-substitution warnings remain.

Do not write yet:
- Do not add new related-work claims without verifying the BibTeX entry from
  an official source.

Upgrade gate:
- Before submission, run one more citation audit after any related-work edits.

## Next Writing Action

The post-2250 SaMAM closure path is invalid under the current manuscript
decision. The next four-reviewer pass should wait for a genuinely new stage,
such as:

1. a clean independent SaMAM rerun that supersedes the 2250 boundary;
2. a new Dalton/Faraday experiment packet with full_eval + targetwise ArtFID;
3. a major paper-structure rewrite that changes the main claims.

Until then, the active paper should keep SaMAM at `2250`, keep IDT as the
primary reporting contract, and avoid another reviewer loop merely for local
wording cleanup.
