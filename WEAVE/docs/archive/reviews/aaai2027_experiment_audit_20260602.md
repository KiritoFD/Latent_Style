# AAAI 2027 Experiment Audit Memo

Date: 2026-06-03  
Role: independent experiment-audit reviewer  
Write scope: audit only; no code or paper edits performed here

## Audit scope and counting rule

I counted only current-source evidence that is directly reachable from the active manuscript and working index:

- `G:\GitHub\Latent_Style\SchrodingerBridge\goal.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission\paper_aaai2026.tex`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\aaai2027_working_index_20260602.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\comparison_report.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\README.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\selected_style_metrics_historical_merged.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\README.zh.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\dataset_audit.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\lancet_runs.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\baselines_samam_samst.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\metric_landscape.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\tables\clip_style_vs_1lpips_full_transfer_points.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\artfid_metric_hacking\distinct5_aggregate_artfid_keypoints.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\visual_metric_alignment_20260602\README.zh.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\README.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\noop_full_transfer_summary.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\idt_eval_20260602\distinct5_512\idt_5x5\summary.json`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\distinct5_table_for_paper.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\archives\old_root_files\training_times_documentation.md`

Archive backup paper workspaces under `archives/old_paper_workspaces/` do not count as primary evidence for paper claims. If a claim depends on those files, I treat it as unsupported until promoted into `docs/experiments/` or a live result directory.

## Executive verdict

The paper has a defensible core, but not yet a defensible final story.

The strong part is already clear: the project has enough evidence to support a content-preserving frontier claim, a no-op/metric-illusion claim, and a tokenizer-execution bottleneck diagnosis on Distinct5-512.

The weak part is equally clear: the manuscript currently overreaches on the latent-metric theory story, overstates efficiency if the SaMST timing estimate is used as if it were a directly preserved same-protocol wall-clock record, and mixes two different ArtFID protocols in the Distinct5 discussion without a hard naming boundary.

If submitted now, I would expect reviewers to attack three points immediately:

1. the paper talks like it has already proven a latent-space metric correction, but the reported reproduced checkpoints still use default MSE for the flow residual;
2. the historical speedup claim is not on the same evidentiary footing as the LBM timing;
3. Distinct5 uses both targetwise official ArtFID and aggregate diagnostic ArtFID in adjacent artifacts, which is manageable only if named and separated explicitly.

## 1. Claims already sufficiently supported

### 1.1 Historical strict-750 quality claim: supported if phrased narrowly

Supported phrasing:

- LBM is a strong compact content-preserving operating point on the historical strict-750 protocol.
- LBM improves LPIPS and EC over the reproduced SaMST operating point while remaining competitive in CLIP-S.
- Artifact-sensitive diagnostics favor LBM over SaMST/S2WAT when the claim is about cleaner perceptual tradeoff, not universal best style.

Evidence:

- `G:\GitHub\Latent_Style\SchrodingerBridge\S-add__K-1_C-0_W-20_Col-0\full_eval\epoch_0008\summary.json`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\selected_style_metrics_historical_merged.csv`
- Table rows already wired into `paper_aaai2026.tex` lines 280-330 and 349-371

Why this passes:

- the metrics exist;
- the baselines are named;
- the claim can be stated as a frontier claim rather than a scalar-win claim.

What must not be claimed from this block:

- universal superiority over all baselines on all metrics;
- same-machine exact training speedup against SaMST/S2WAT unless the timing evidence is upgraded.

### 1.2 Distinct5 no-op / metric-illusion claim: strongly supported

Supported phrasing:

- raw `clip_style` is unsafe on art-to-art transfer without a no-op reference;
- transfer-only and no-op-adjusted reporting are mandatory on Distinct5-like splits;
- some baselines can look good in absolute CLIP-S while delivering very small or negative movement above the identical-image prior.

Evidence:

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\README.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\noop_full_transfer_summary.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\no_op_identity_5x5_summary.json`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\visual_metric_alignment_20260602\README.zh.md`

Why this passes:

- the no-op baselines are explicitly built and evaluated;
- the effect survives transfer-only filtering;
- the visual audit agrees with the metric audit instead of contradicting it.

### 1.3 Distinct5 LBM vs evaluated SaMAM curve: supported

Supported phrasing:

- on the evaluated Distinct5-512 curve through SaMAM-2250, LBM-F/H define a stronger content-preserving frontier;
- LBM exceeds the no-op-adjusted style floor while SaMAM remains below it on Distinct5;
- the Distinct5 bottleneck is not “no change” but incorrect or overly low-frequency style execution.

Evidence:

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\lancet_runs.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\metric_landscape.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\tables\clip_style_vs_1lpips_full_transfer_points.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\comparison_report.md`

Why this passes:

- same split;
- same 750-output all-pairs protocol;
- actual wall-time labels exist for the curve;
- the no-op reference prevents trivial overclaiming.

### 1.4 Tokenizer-capacity-alone-is-not-enough: supported within the tested family

Supported phrasing:

- in the tested Distinct5 tokenizer family, larger or more structured token sets alone did not break the frontier;
- execution-side changes such as prototype queues and content-guided routing mattered more.

Evidence:

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\lancet_runs.md`
- table and discussion in `paper_aaai2026.tex` lines 406-434

Why this passes:

- the paper has multiple controlled tokenizer variants;
- the conclusion is restricted to the tested family rather than all possible tokenizers.

## 2. Claims that currently fail

### 2.1 The “latent metric correction” story is not experimentally closed

Current paper state:

- `paper_aaai2026.tex` lines 104-109 explicitly say the implementation supports `MSE`, `Huber`, and `L1`;
- the same lines also explicitly state that the reproduced historical and Distinct5 results in the paper use the default `MSE` instantiation for the flow residual;
- `src/losses.py` confirms that `Huber` and `L1` exist as options, but current tokenizer configs are still set to the default non-Huber path.

Why this fails:

- you cannot claim that the paper has corrected the latent-space metric mistake in the flow residual if the main reported results still use default MSE there;
- the `W1` terminal-alignment claim is supported;
- the broader `Huber/L1 beats MSE in latent flow learning` claim is not.

Audit verdict: `RED`

What is still allowed:

- claim that style alignment is corrected away from endpoint latent MSE and into OT + SA-SWD terminal matching;
- claim that the consequential measured correction is the `W1`-style terminal objective.

What is not allowed yet:

- claim that the paper experimentally validates Huber/L1 as the decisive latent metric correction for the main model.

### 2.2 The historical training-speed claim is not on defensible evidence

Current paper state:

- abstract and contributions cite `310 s` for LBM and `6769 s` for SaMST;
- `training_times_documentation.md` states that the `6769 s` SaMST figure is an estimate derived from a one-epoch probe multiplied to the target epoch count, not a preserved full training wall-clock record;
- S2WAT timing is also an estimate rather than a preserved end-to-end same-protocol run.

Why this fails:

- the LBM time is a direct preserved timing record;
- the SaMST time is a probe-derived estimate;
- presenting them as equally hard experimental facts is vulnerable.

Audit verdict: `RED`

What is still allowed:

- “LBM reaches its reported operating point in 310 s on preserved logs.”
- “SaMST and S2WAT have substantially higher training cost based on preserved or probe-based timing records.”

What is not allowed yet:

- a hard same-grade speedup statement like `22x faster` in the abstract without a protocol note;
- any wording that implies a strict apples-to-apples full-run wall-clock measurement for SaMST unless that run is actually preserved.

### 2.3 Distinct5 ArtFID protocol is under-specified and currently mixed across artifacts

Current evidence split:

- targetwise official-style ArtFID is present in:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\idt_eval_20260602\distinct5_512\idt_5x5\aggregate_targetwise_artfid.json`
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\artfid_comparison_points.csv`
  - this is where paper-table values such as `idt = 216.5`, `LBM-F = 122.6`, `SaMST e15 = 395.7` are coming from
- aggregate diagnostic ArtFID is present in:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\artfid_metric_hacking\distinct5_aggregate_artfid_keypoints.csv`
  - this is where no-op is near `1.0`

Why this fails if not clarified:

- these two quantities are not the same protocol;
- both are called “ArtFID” in surrounding docs;
- a reviewer will catch the fact that no-op is `216.5` in one table and `~1.0` in another.

Audit verdict: `RED` unless renamed and separated in the paper

Required fix in writing:

- reserve `ArtFID` in the main table for one protocol only;
- call the other one `aggregate ArtFID diagnostic`;
- never compare those two values as if they came from the same question.

### 2.4 The empirical support for Theorem 1 / path stability is not in the live artifact lane

Current paper state:

- `paper_aaai2026.tex` lines 150-156 and 186-187 promise empirical path-statistics support;
- the current searchable live docs do not expose a stable artifact directory for velocity-variance or trajectory-statistics outputs;
- the only clear trace is in archived paper workspaces and planning notes, which do not count as paper evidence.

Why this fails:

- reviewers will ask where the actual probe outputs are;
- “we measured it at some point in an archive workspace” is not enough.

Audit verdict: `RED`

### 2.5 The Distinct5 SaMST story is evidence-backed but documentation-inconsistent

Current state:

- a real Distinct5 SaMST e15 result exists:
  - `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602\eval_epoch15\epoch_0015\summary.json`
- but `baselines_samam_samst.md` still says formal Distinct5 SaMST is not started;
- the working plan also still speaks as if SaMST completion is pending.

Why this is a problem:

- it makes the repo look internally inconsistent;
- a reviewer cannot tell whether the Distinct5 SaMST point is official, diagnostic, or stale.

Audit verdict: `YELLOW`

This is not a missing experiment problem. It is a ledger/index problem.

### 2.6 The tokenizer bottleneck conclusion is not yet broad enough for a global thesis

Current support:

- the tested tokenizer family supports “capacity alone is insufficient.”

Why this is still incomplete:

- the conclusion “the main remaining bottleneck is not tokenizer size” is broader than the tested family;
- without the latent-metric ablation and a stronger execution-budget study, the paper should keep this as a bounded within-family diagnosis.

Audit verdict: `YELLOW`

### 2.7 The historical bootstrap confidence-interval sentence is not currently backed by a durable artifact

Current paper state:

- `paper_aaai2026.tex` line 371 gives paired bootstrap intervals;
- searchable live evidence points mostly to archived backup paper workspaces, not a current `docs/experiments/...` result artifact.

Why this fails:

- if the CI remains in the paper, the corresponding script output needs a stable current path.

Audit verdict: `YELLOW`

## 3. Minimum experiment set to make the paper defensible

This is the smallest set I would require before calling the paper AAAI-defensible.

### E1. Distinct5 flow-loss metric ablation

Purpose:

- close the biggest theory-to-experiment gap;
- determine whether `MSE` vs `Huber` vs `L1` materially changes LPIPS, artifact metrics, or stability on the actual Distinct5 frontier.

Recommended base:

- start from the `H` family, not `K`;
- `H` is the closest current balanced point and therefore the right place to test whether metric choice moves the true frontier.

Minimum runs:

- `MSE`
- `Huber`
- `L1`

Protocol:

- same Distinct5 split;
- same batch policy as current formal remote runs;
- same tokenizer and queue settings;
- full 8 epochs each, with eval at least on epochs `1, 2, 4, 8`.

Acceptance rule:

- this experiment only needs to show one of two things:
  - `Huber/L1` meaningfully improves LPIPS or artifact-sensitive metrics at similar style; or
  - `Huber/L1` does not help, which forces the paper to narrow the latent-metric claim to terminal `W1` only.

Without E1, the current motivation overpromises.

### E2. Distinct5 time-to-parity evidence

Purpose:

- replace rhetorical efficiency claims with a clean same-split convergence story.

Minimum content:

- LBM Distinct5 curve from existing `F/H/K` points and any new metric-ablation runs;
- SaMAM Distinct5 curve through the current `250..2250` steps;
- SaMST Distinct5 curve with at least `e5/e10/e15`, not only a single failure point.

Deliverables:

- `wall_clock -> clip_style`
- `wall_clock -> LPIPS`
- `wall_clock -> delta_idt`

Critical rule:

- Distinct5 should be the speed headline;
- historical strict-750 timing should become a secondary table unless actual baseline full-run timings are recreated.

### E3. Path-stability probe with durable outputs

Purpose:

- give the theorem section a real empirical anchor.

Minimum content:

- one full model checkpoint;
- one weakened kinetic variant;
- sample `t ~ U(0,1)`;
- record mean and variance of `||v_theta(z_0, t, s)||`;
- record path length ratio or another explicit trajectory-statistic summary.

This does not need to be large. It needs to exist in a stable artifact path and be cited explicitly.

### E4. Distinct5 baseline ledger cleanup

Purpose:

- remove contradictions between the manuscript and the experiment docs.

Minimum content:

- one canonical Distinct5 baseline ledger that says exactly:
  - which SaMAM points are official;
  - whether SaMST e15 is official and what its protocol is;
  - which ArtFID numbers are targetwise official and which are aggregate diagnostic.

This is cheap but mandatory.

### Optional E5. Historical bootstrap export

Only required if the paper keeps the CI sentence in the main text.

If you do not want to create this artifact, delete the bootstrap sentence.

## 4. Remote-3060-first execution order

The order below is chosen to maximize paper value per hour and to keep the remote 3060 doing only evidence-grade work.

### Step 1: metric-ablation runs on Distinct5

Run on remote 3060 first:

- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_mse_b44_remote`
- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_huber_b44_remote`
- `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_l1_b44_remote`

Expected outputs:

- `epoch_0001.pt` ... `epoch_0008.pt`
- `full_eval/epoch_0001/summary.json`
- `full_eval/epoch_0002/summary.json`
- `full_eval/epoch_0004/summary.json`
- `full_eval/epoch_0008/summary.json`
- `metrics.csv` per eval directory
- if ArtFID is included, `aggregate_targetwise_artfid.json`

Decision after Step 1:

- if `Huber` or `L1` helps, the latent-metric story survives;
- if not, remove the broader latent-metric claim and keep only the terminal-`W1` claim.

### Step 2: SaMST Distinct5 intermediate evals

Do not retrain first. Reuse the existing Distinct5 SaMST run if checkpoints exist.

Base path:

- `G:\GitHub\Latent_Style\Related_Works\baseline_pipeline\results\samst_distinct5_512_real_b2_e15_20260602`

Expected new outputs to create from existing checkpoints:

- `...\\eval_epoch05\\epoch_0005\\summary.json`
- `...\\eval_epoch10\\epoch_0010\\summary.json`
- `...\\eval_epoch15\\epoch_0015\\summary.json` already exists

Why this is second:

- it upgrades a single-point anecdote into a curve without spending new training time.

### Step 3: path-stability probe

Run probe jobs on remote 3060 using existing checkpoints.

Recommended checkpoints:

- current balanced Distinct5 LBM checkpoint: `H e1` or `F e1`
- one no-kinetic or weakened-kinetic checkpoint; if no matching Distinct5 run exists, produce one short destructive Distinct5 run:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_no_kinetic_probe_b44_remote`

Expected outputs:

- velocity-stat CSV
- trajectory-stat JSON
- one compact PDF figure

### Step 4: ledger/protocol repair

After the numerical work is done, update the experiment docs and paper-facing tables.

Do not do this before Step 1 and Step 2, because the wording depends on whether the metric-ablation claim lives or dies.

## 5. Exact artifact paths that should be created or updated

Below is the minimum artifact contract I would require.

### Create

1. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\README.md`
2. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\curve_metrics.csv`
3. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\summary_table.csv`
4. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\figures\distinct5_loss_metric_pareto.pdf`
5. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\figures\distinct5_loss_metric_pareto.png`
6. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\figures\distinct5_loss_metric_qualitative.png`
7. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-latent-metric-ablation\runs_manifest.json`

8. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-time-to-parity\README.md`
9. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-time-to-parity\distinct5_time_to_parity_points.csv`
10. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-time-to-parity\figures\distinct5_time_to_clip_style.pdf`
11. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-time-to-parity\figures\distinct5_time_to_lpips.pdf`
12. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-time-to-parity\figures\distinct5_time_to_delta_idt.pdf`

13. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-path-stability\README.md`
14. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-path-stability\velocity_stats.csv`
15. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-path-stability\trajectory_stats.json`
16. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-path-stability\figures\velocity_variance_over_t.pdf`
17. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-path-stability\figures\path_length_ratio_bar.pdf`

18. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-artfid-protocol-note.md`

19. if the main text keeps bootstrap CIs:  
   `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-historical-bootstrap\README.md`  
   `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-03-historical-bootstrap\bootstrap_summary.csv`

### Update

1. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\baselines_samam_samst.md`
2. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\metric_landscape.md`
3. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\comparison_report.md`
4. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\distinct5_table_for_paper.csv`
5. `G:\GitHub\Latent_Style\SchrodingerBridge\docs\aaai2027_working_index_20260602.md`
6. `G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission\paper_aaai2026.tex`
7. `G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission\figures\fig_distinct5_pareto.pdf`
8. `G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission\figures\fig_distinct5_pareto.png`

### Remote result directories expected

1. `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_mse_b44_remote`
2. `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_huber_b44_remote`
3. `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_loss_l1_b44_remote`
4. `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/distinct5_512_ema_variant_h_no_kinetic_probe_b44_remote` if no matching existing destructive checkpoint is available for the path probe

## Stoplight summary by claim family

| Claim family | Verdict | Reason |
|---|---|---|
| Historical strict-750 quality frontier | `GREEN` | Current metrics and artifact-sensitive comparison support a narrow frontier claim. |
| Historical strict-750 efficiency / 22x faster headline | `RED` | Baseline times are not on the same evidentiary footing as LBM time; SaMST timing is probe-derived. |
| Distinct5 no-op / metric-illusion diagnosis | `GREEN` | Explicit no-op, transfer-only, visual audit, and adjusted gains all agree. |
| Distinct5 LBM vs evaluated SaMAM curve | `GREEN` | Same split, same 750 protocol, same wall-time axis, same reference floor. |
| Distinct5 LBM vs SaMST-512 | `YELLOW` | Evidence exists for e15, but the repo ledger is inconsistent and the curve is not yet indexed. |
| Latent metric correction beyond terminal W1 | `RED` | Main reported runs still use default MSE for the flow residual; no direct MSE vs Huber/L1 ablation yet. |
| Theorem/path-stability empirical backing | `RED` | Promised in paper, not yet promoted into current stable experiment artifacts. |
| Tokenizer bottleneck diagnosis | `YELLOW` | Supported within the tested family, but not broad enough yet for a universal bottleneck claim. |
| Historical bootstrap CI sentence | `YELLOW` | Likely true, but not yet backed by a stable current artifact path. |

## Bottom line

If I were reviewing this for AAAI 2027 today, I would say the paper already has a publishable empirical core, but its strongest claimed story is one experiment block short of being trustworthy.

The fastest path to “stable AAAI” is not more tokenizer exploration first. It is:

1. close the flow-loss metric ablation;
2. convert Distinct5 timing into a clean time-to-parity figure;
3. promote the path-stability probe into a durable artifact;
4. repair the Distinct5 baseline ledger and the ArtFID naming boundary.

Do those four things, and the paper becomes much harder to attack.
