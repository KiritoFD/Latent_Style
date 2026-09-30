# AAAI 2027 Adversarial Review Memo

Date: 2026-06-03  
Scope owner: adversarial reviewer memo only

## Evidence base read for this memo

Minimum required inputs were read:

- `G:\GitHub\Latent_Style\SchrodingerBridge\goal.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\aaai_submission\paper_aaai2026.tex`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\README.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\comparison_report.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\distinct5_table_for_paper.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\comparison_20260602\selected_style_metrics_historical_merged.csv`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-06-02-distinct5-512-lancet-representation-speed.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\README.zh.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\baselines_samam_samst.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\metric_landscape.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\dataset_audit.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\metric_hacking_noop_20260602\README.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\distinct5_512_20260602\visual_metric_alignment_20260602\README.zh.md`

I also checked repo-side supporting notes that materially affect credibility:

- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\aaai2027_working_index_20260602.md`
- `G:\GitHub\Latent_Style\SchrodingerBridge\docs\experiments\2026-05-31-aaai-saswd-plan.md`

## Executive judgment

The project is interesting and has two genuinely paper-worthy assets already:

1. a strong no-op/`idt`-anchored diagnosis showing that raw `CLIP-S` can be badly misleading on art-to-art transfer;  
2. a concrete Distinct5-512 tokenizer/execution study showing that queueing/routing/execution matter more than naive tokenizer capacity.

However, the current manuscript is still **ahead of the evidence** on the three claims that matter most to an AAAI reviewer:

- latent-distance correction,
- SA-SWD novelty,
- parity-aligned efficiency.

In its current form, I would attack the paper on those three fronts first.

---

## 1. Top rejection risks, ordered by severity

### 1. The paper's central latent-metric thesis is not actually proven

This is the biggest problem.

The draft now frames the work as correcting a latent-space distance mistake: Euclidean reconstruction pressure is said to be wrong in compressed VAE latent space, while robust losses and `W1`-style terminal matching are positioned as the fix. But the manuscript itself states that the reported checkpoints still use the **default MSE** local flow residual:

- `paper_aaai2026.tex:104-109`
- `paper_aaai2026.tex:148-148`

The repo's own working index explicitly says the same thing:

- `docs/aaai2027_working_index_20260602.md:162-168`

Specifically, the index says current evidence strongly supports the `W1` terminal-alignment argument, while the `MSE vs Huber/L1` flow-loss claim still requires direct ablation.

That means the current paper is trying to cash a theoretical check that the repo has not written experimentally. A reviewer can say, correctly:

> You are not showing that Huber/L1 is the reason the system works, because your own reported runs still use MSE for the flow residual.

At best, the evidence currently supports:

- OT-coupled endpoint construction matters,
- `W1`-style terminal matching matters,
- endpoint MSE is not the main style-alignment mechanism.

It does **not** yet support the stronger claim that robust local flow losses are a key reason for the reported results.

### 2. SA-SWD novelty is still a plan, not a finished ablation

The method text and contributions elevate semantic projection-axis selection as a nontrivial contribution:

- `paper_aaai2026.tex:132-132`
- `paper_aaai2026.tex:143-147`

But the repo evidence I found is still at the planning/tooling level:

- `docs/experiments/2026-05-31-aaai-saswd-plan.md:7-10`
- `docs/experiments/2026-05-31-aaai-saswd-plan.md:43-69`
- `tools/make_saswd_ablation_configs.py`

That is not the same as a completed matched `semantic` vs `random` projection-axis experiment.

As a reviewer, I would ask the obvious question:

> How do I know the gain comes from semantic projection-axis selection rather than from having any terminal SWD at all?

Right now the paper does not have the experiment that answers that question.

### 3. The efficiency argument is still vulnerable to "non-parity comparison" attacks

The paper is better than it was before, but it is not safe yet.

Historical table:

- `paper_aaai2026.tex:310-330`

Distinct5 table:

- `paper_aaai2026.tex:383-399`

The problem is not that the numbers are false. The problem is that they still mix:

- selected LBM checkpoints,
- selected SaMST operating points,
- partial or cumulative SaMAM curves,
- different hardware scopes across result families.

The Distinct5 documentation is honest that the current plot is a curve-vs-points diagnostic, not yet a strict time-to-parity benchmark:

- `docs/experiments/distinct5_512_20260602/metric_landscape.md`
- `docs/experiments/distinct5_512_20260602/baselines_samam_samst.md`

The historical table is also mixing "time to selected operating point" rather than matched target quality:

- `docs/experiments/comparison_20260602/selected_style_metrics_historical_merged.csv`

This is enough for internal diagnosis. It is **not** enough for a hard conference claim like "22x faster" or any rhetoric that reads as universal efficiency dominance.

Until you show wall-clock curves against a matched quality target, the efficiency story remains a reviewer attack surface.

### 4. The metric story is insightful, but still under-specified in the paper

The repo has actually done a strong job here. The problem is that the paper has not yet fully metabolized it.

The key repo evidence is solid:

- `docs/experiments/metric_hacking_noop_20260602/README.md`
- `docs/experiments/distinct5_512_20260602/visual_metric_alignment_20260602/README.zh.md`

This evidence shows a real and important decomposition:

- `CLIP-S`: target-style proximity,
- `LPIPS`: content preservation,
- aggregate ArtFID/FID: broad art-domain realism,
- visual panel: whether the change is recognizably target-style rather than just low-frequency washing or content-preserving no-op.

But the paper still uses a single `ArtFID` column in Distinct5:

- `paper_aaai2026.tex:392-399`

while the repo docs distinguish at least two different uses:

1. targetwise/pairwise ArtFID for target-style diagnostics;
2. aggregate ArtFID as a broad art-domain diagnostic where no-op can look nearly perfect.

That distinction matters. If it is not made extremely explicit in the paper, a hostile reviewer can accuse the paper of metric shopping or inconsistent evaluation.

### 5. The formal section is at risk of reading like decorative math

The theorem block is not obviously wrong. The issue is evidentiary weight.

The paper says the formal results are "paired with direct empirical validation":

- `paper_aaai2026.tex:150-155`

and then gives very specific empirical support numbers:

- `paper_aaai2026.tex:207-255`

But I could not find a clean, indexed experiment artifact under `docs/experiments/` that reproduces these theorem-facing numbers as first-class outputs. Search hits mostly lead back to the paper itself or to older internal notes, not to a review-ready evidence package.

That is dangerous. Weakly anchored math hurts more than no math in AAAI reviewing, especially when the paper already has a strong empirical angle.

### 6. The tokenizer conclusion is plausible, but over-generalized

The Distinct5 A-M ablation suite is useful and internally coherent:

- `docs/experiments/2026-06-02-distinct5-512-lancet-representation-summary.zh.md`
- `docs/experiments/2026-06-02-distinct5-512-lancet-representation-speed.md`

It supports a narrower claim:

> On Distinct5-512, naive tokenizer capacity increases do not help much; target queue quality, content-guided routing, and execution structure matter more.

What it does **not** support yet is the broader headline statement that the "main remaining bottleneck is not tokenizer size" in a general sense. That may be true, but the current evidence is one stress split plus a few historical diagnostics, not a field-wide closure of the tokenizer question.

### 7. Provenance is good enough for research notes, but not fully clean for a paper under attack

The comparison docs themselves acknowledge that some Distinct5 LANCET points are preserved as curated tables and pulled `csv/json` aggregates rather than local image folders:

- `docs/experiments/comparison_20260602/README.md`
- `docs/experiments/comparison_20260602/lancet_representative_points.csv`

That is acceptable for ongoing research, but if a reviewer or rebuttal asks for exact provenance of the main figure points, this is weaker than ideal. The paper should not sound more archival than the artifact reality.

---

## 2. Claims that are currently unsupported or only weakly supported

### A. "We corrected the latent-space distance mistake"

Weakly supported, not proven.

What is supported now:

- endpoint style alignment is not done with latent endpoint MSE;
- `W1`-style terminal matching plus OT endpoint construction are important;
- historical pure-MSE-like directions can collapse badly.

What is not yet supported:

- that `Huber/L1` local flow residuals are a decisive mechanism in the reported headline runs.

### B. "SA-SWD specifically beats ordinary terminal SWD"

Currently unsupported by finished ablation evidence.

There is a plan and tooling, but no completed matched result in the evidence bundle I read.

### C. "LBM has the strongest measured content-preserving frontier"

This is acceptable **only if scoped narrowly**:

- on Distinct5-512,
- among the currently evaluated methods,
- under the reported all-pairs 750-output protocol,
- with no-op explicitly anchored.

It is too strong if read as a global statement across datasets or baseline families, because on WikiArt512/3600 the current comparison report still says SaMAM owns the best non-idt LPIPS and best aggregate ArtFID:

- `docs/experiments/comparison_20260602/comparison_report.md`

### D. "The bottleneck is execution, not tokenizer size"

This is a good working conclusion, but still one notch too strong as a paper conclusion.

What you have:

- Distinct5 tokenizer variants A-M,
- wikiart512 generated-delta geometry diagnostics,
- content-adaptive routing and queue evidence.

What you do not have:

- a second dataset family showing the same conclusion under matched tokenizer experiments,
- or a direct proof that a stronger tokenizer with a stronger renderer consumer closes the gap.

### E. "The theorem section is directly empirically validated"

Weakly supported until the validation runs are surfaced as first-class experiment artifacts, not just as numbers quoted in the manuscript.

---

## 3. Minimum experiment matrix needed to make the submission defensible

If you want the paper to survive adversarial review, do **not** spend the next week on new tokenizer families first. Close the proof obligations.

### Matrix A. Flow-loss metric ablation: MSE vs Huber vs L1

Question:

> Is the latent-metric correction story actually true for the local flow residual, or is the current gain mostly from OT + `W1` terminal alignment?

Protocol:

- Dataset: `Distinct5-512`
- Base configs: `F` and `H` only
- Same init, same seed set, same batch (`b44`), same epochs (`3`)
- Change only the local flow residual: `MSE`, `Huber`, `L1`
- Evaluate full 750 and transfer-only

Required outputs:

- `CLIP-S`
- `LPIPS`
- `Delta_idt`
- targetwise ArtFID
- MUSIQ / MANIQA
- aligned visual panel on the same transfer rows

Acceptance rule:

- If `Huber` or `L1` wins at matched or near-matched style, keep the latent-metric thesis.
- If they do not, stop claiming that local robust loss is a key driver of the current headline results.

Why this is cheap enough:

- Distinct5 short runs already cost about `60 s/epoch` plus about `146 s/checkpoint` full eval at `b44` on the remote 3060, per
  `2026-06-02-distinct5-512-lancet-representation-speed.md`.

### Matrix B. SA-SWD semantic-axis ablation: semantic vs random

Question:

> Does semantic projection-axis selection matter beyond having a terminal SWD at all?

Protocol:

- Dataset: `Distinct5-512`
- Base config: `H` first, `F` second if H is ambiguous
- Use the existing paired-config path:
  `tools/make_saswd_ablation_configs.py`
- Same init, same seed set, same queue policy, same terminal weight
- Compare `terminal_swd_axis_source=semantic` vs `random`

Required outputs:

- `CLIP-S`
- `LPIPS`
- `Delta_idt`
- targetwise ArtFID
- MUSIQ / MANIQA
- one semantic-vs-random aligned visual panel

Acceptance rule:

- If semantic axes do not beat random axes, remove SA-SWD from the paper's novelty center.

### Matrix C. Time-to-parity curves with explicit accounting

Question:

> Is LBM actually better on wall-clock to useful quality, or are current speed claims just selected-point comparisons?

Protocol:

- Primary dataset: `Distinct5-512`
- Curves to include:
  - `LBM-F/H/K` by epoch
  - `SaMAM` by step (`250` to `2250`, continue if needed)
  - `SaMST` at least `e5/e10/e15`
- Plot:
  - `CLIP-S vs train wall time`
  - `LPIPS vs train wall time`
  - `Delta_idt vs train wall time`
- If hardware must differ, state it in the axis caption and stop using ratio rhetoric.

Acceptance rule:

- Either show genuine parity curves,
- or downgrade all efficiency language to "selected operating point cost" and nothing stronger.

### Matrix D. Formal-section empirical audit

Question:

> Are the theorem-facing empirical numbers real, indexed, and reproducible enough to keep in the main paper?

Protocol:

- No new training required for the first pass
- Use existing checkpoints:
  - one historical strict-750 checkpoint,
  - one Distinct5 checkpoint
- Export a dedicated appendix table for:
  - sampled `||v_theta(z_0,t,s)||` mean/std over `t`,
  - Euler `K` sweep error,
  - OT vs random assignment directional consistency

Acceptance rule:

- If this cannot be surfaced cleanly in one day, shrink the theorem section and move it to a design-rationale appendix.

### Matrix E. Distinct5 metric-grounding panel

Question:

> Can the paper explain, in one place, why no-op, SaMAM, SaMST, and LBM occupy different parts of the evaluation space?

Protocol:

- Methods: `idt`, `SaMAM-2250`, `SaMST-e15`, `LBM-F e1`, `LBM-K e1`
- Compute on the same aligned transfer samples:
  - `CLIP-S`
  - `LPIPS`
  - targetwise ArtFID
  - aggregate ArtFID
  - MUSIQ
  - MANIQA
- Build one 5-column figure:
  `source / idt / SaMAM / SaMST / LBM-F / LBM-K`

Acceptance rule:

- This becomes the evidence block that turns the "metric illusion" story into a contribution rather than an anecdote.

---

## 4. Specific paper sections, tables, and figures that should be rewritten

### Abstract

Current risk:

- It compresses three different claims into one paragraph: historical strict-750, Distinct5 stress split, and latent-metric correction.

Rewrite:

- Keep the historical result as one scoped sentence.
- Keep Distinct5 as a second scoped sentence.
- Remove or soften any broad latent-distance-correction claim until Matrix A lands.

### Contributions list

Current risk:

- The paper currently presents some hypotheses as if they are already closed contributions.

Rewrite:

- Separate:
  1. verified empirical contributions,
  2. method contributions,
  3. diagnostic/evaluation contributions.
- Do not let the theorem or SA-SWD novelty claim outrun the ablations.

### Method: tokenizer, metric-choice, and theorem subsections

Current risk:

- The text is now smarter than before, but still tries to elevate unclosed claims.

Rewrite:

- State plainly that the reported historical and Distinct5 checkpoints use MSE for the local flow residual unless Matrix A changes that.
- Present `W1` terminal alignment as the current empirically supported correction.
- If Matrix D is weak, demote theorems to design grounding and cut the strong "paired with direct empirical validation" phrasing.

### Historical main table and cost table

Current risk:

- The tables are informative, but they invite "selected operating point" criticism.

Rewrite:

- Make the table caption explicitly say "selected operating point on historical strict-750".
- Add a footnote that not all artifact metrics were recomputed for all baselines under one common rerun.
- Move aggressive efficiency language out of prose and into a clearly labeled "time to selected operating point" table unless Matrix C lands.

### Distinct5 figure and table

Current risk:

- This is the strongest part of the paper, but it still mixes too many evaluation meanings into one presentation.

Rewrite:

- Keep `idt` as a horizontal reference line.
- Keep `Delta_idt`.
- Add a table footnote explaining exactly which ArtFID is used there.
- If space allows, add transfer-only values or move them to supplement with explicit cross-reference.

### Artifact-sensitive subsection

Current risk:

- "Ours vs SaMST" alone is no longer enough.

Rewrite:

- Make this a broader evaluation-decomposition section:
  - no-op,
  - SaMAM,
  - SaMST,
  - LBM.
- Show that the paper understands not only why LBM wins somewhere, but also what each metric is actually measuring.

### Discussion and conclusion

Current risk:

- The conclusion currently sounds more closed than the repo evidence justifies.

Rewrite:

- Keep the narrow strong conclusion:
  LBM currently defines the best measured content-preserving region on Distinct5 among evaluated methods.
- Present "execution bottleneck over tokenizer size" as the current best explanation, not as a universal theorem.

---

## 5. Concrete 1-week execution order, using the remote 3060 where possible

### Day 1: freeze evidence and stop claim drift

Use the remote 3060 only for formal runs. Before new training:

1. archive/index the exact run paths already cited by the paper;
2. declare one source-of-truth CSV for Distinct5 and one for historical strict-750;
3. separate durable artifacts from temp mirrors and `_codex_tmp` clutter;
4. stop adding new tokenizer branches until Matrices A-C are closed.

Deliverable:

- a clean experiment index,
- a paper claim-to-artifact checklist,
- a cleaner git staging boundary.

### Day 2: Matrix A on the remote 3060

Run `MSE` vs `Huber` vs `L1` on `H` first.

Priority:

1. `H-MSE`
2. `H-Huber`
3. `H-L1`

If one robust loss looks promising, repeat on `F`.

Reason:

- This is the single most important missing experiment.

### Day 3: Matrix B on the remote 3060

Run `semantic` vs `random` axis source on the same base config and same seeds.

Reason:

- This decides whether SA-SWD is a real claim or just naming.

### Day 4: baseline curve completion and timing normalization

Remote 3060:

- continue the Distinct5 baseline curve work needed for parity plots;
- if SaMST Distinct5 only has `e15` in usable paper form, add `e5/e10`;
- if SaMAM still needs one more point to show plateau, finish it once.

Reason:

- One complete wall-clock story is worth more than three extra model variants.

### Day 5: theorem audit and metric-grounding panel

Mostly analysis, not training:

- export path-stability stats,
- export Euler sweep stats,
- build the aligned Distinct5 metric-grounding panel.

Reason:

- This day decides whether the formal section stays large or gets cut down.

### Day 6: rewrite the paper around what is now actually proven

Only after Days 2-5:

- rewrite abstract,
- rewrite contributions,
- rewrite Distinct5 result section,
- rewrite cost table language,
- rewrite artifact subsection.

Rule:

- delete unsupported claims instead of trying to rhetorically save them.

### Day 7: adversarial final pass

Before calling the draft AAAI-ready:

1. ask whether each headline claim is supported by a specific file and metric;
2. remove every sentence that needs explanation in rebuttal;
3. check that each figure answers one question only.

---

## Bottom-line recommendation

**Recommendation: reject.**

Why:

- the current manuscript is materially stronger than the earlier version,
- but the three claims that would drive accept votes are still not fully backed:
  latent-metric correction, SA-SWD novelty, and time-aligned efficiency.

If Matrices A-C land cleanly and the paper is rewritten to match that evidence, this can move quickly from reject to borderline or better. In its current state, it is still too easy for a strong reviewer to puncture the core story.
