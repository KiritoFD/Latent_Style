# AAAI 2027 Reviewer Scorecard

Date: 2026-06-03  
Role: independent reviewer, adversarial but fair  
Scope: score the current manuscript and evidence as they exist now, not the intended future version

## Material read

- `SchrodingerBridge/goal.md`
- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/README.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/comparison_report.md`
- `SchrodingerBridge/docs/experiments/comparison_20260602/selected_style_metrics_historical_merged.csv`
- `SchrodingerBridge/docs/experiments/metric_hacking_noop_20260602/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/archives/old_root_files/training_times_documentation.md`

## Executive assessment

This paper has a real paper-worthy core: it argues that domain-level artistic transfer in compressed VAE latents should be treated as controlled transport, not as direct latent reconstruction, and it backs that with a useful evaluation warning that raw CLIP-style can be badly misleading without an identical-image reference. That is the best part of the submission.

However, the submission is not AAAI-safe yet. The strongest conceptual claim is only partially converted into decisive evidence, the efficiency story still mixes timing scopes and hardware regimes in ways a skeptical reviewer can attack, and the paper currently relies too much on frontier rhetoric where a top venue will demand one or two decisive ablations. My read is that the work is promising, but still short of a stable accept.

## Scores

| Criterion | Score (1-10) |
|---|---:|
| Novelty | 6 |
| Technical correctness | 6 |
| Evidence quality | 5 |
| Fairness of comparison | 4 |
| Clarity of writing | 6 |
| Figure quality | 6 |
| Reproducibility | 5 |
| Acceptance likelihood | 3 |

## Criterion-by-criterion review

### 1. Novelty - 6/10

**Strongest positive evidence**

The paper does more than stack known modules. The useful novelty is the combination of:

- OT-coupled endpoint construction in latent space,
- semantic-aligned terminal SWD rather than plain random-projection SWD,
- a tokenizer-vs-renderer diagnosis that treats style representation as an executable control problem,
- and, most importantly, the explicit no-op / `idt` evaluation framing that exposes the raw CLIP-style prior on art-to-art transfer.

The `Distinct5-512` section is where this feels most original. The paper does not merely claim a better point; it shows that some baselines stay below the identical-image style prior while paying nonzero LPIPS.

**Strongest rejection risk**

The method story is still easier to read as a careful integration of existing ideas than as a clean new algorithmic principle. A skeptical reviewer can summarize the novelty as "latent U-Net + OT pairing + SWD regularization + style routing," then ask what is fundamentally new besides the benchmark framing. If the metric-illusion insight is not elevated into a sharper evaluation contribution, the novelty score will collapse.

### 2. Technical correctness - 6/10

**Strongest positive evidence**

The current manuscript is materially more coherent than a generic style-transfer paper. It now distinguishes path-energy regularization from endpoint style matching, explains why latent endpoint supervision should not be naive Euclidean reconstruction, and limits the formal section to design-grounding rather than pretending to solve the global Schr\"odinger Bridge problem. The theorem caveat around Theorem 1 is a good move.

**Strongest rejection risk**

The paper still stops short of proving its most ambitious thesis. The manuscript argues that the latent-space metric choice matters, but the reported main checkpoints still use the default MSE instantiation for the flow residual. That makes the current evidence strongest for the endpoint-side `W_1` / SA-SWD argument, but not yet for the broader "metric correction in latent space" claim. A strong reviewer will notice this gap immediately and ask for a direct latent-metric ablation instead of accepting the geometry argument on prose alone.

### 3. Evidence quality - 5/10

**Strongest positive evidence**

There is real breadth here:

- historical strict-750 comparison,
- artifact-sensitive metrics beyond CLIP/LPIPS,
- Distinct5-512 stress benchmark,
- no-op-adjusted style gain,
- tokenizer/queue ablations,
- and some uncertainty reporting via paired bootstrap.

This is stronger than the usual single-table style-transfer paper.

**Strongest rejection risk**

The evidence is still too fragmented. The submission mixes historical strict-750, WikiArt512 convergence, and Distinct5-512 stress results, each with different roles, and the paper asks the reader to trust a fairly complicated evidence hierarchy. That can work in an internal log; it is dangerous in a conference paper. The most important missing experiment is obvious: a clean same-protocol latent-metric ablation showing whether MSE vs Huber/L1 actually changes the frontier in the claimed direction.

### 4. Fairness of comparison - 4/10

**Strongest positive evidence**

The paper is better than average in explicitly naming protocol scope. It distinguishes historical strict-750 from Distinct5-512, flags transfer-only vs full-scope reporting, and makes the `idt` baseline explicit. That is the right instinct.

**Strongest rejection risk**

This is currently the biggest acceptance threat. The timing claims still invite attack:

- historical cost numbers are on RTX 4070 Laptop,
- Distinct5 timings are on remote RTX 3060 WSL,
- LBM time is sometimes "time to selected checkpoint" while baseline time is cumulative training time,
- SaMST training cost is partly estimated from a probe,
- and the strongest efficiency language is still presented near headline quality claims.

Even when the paper is technically honest, a hostile reviewer can still frame this as an apples-to-oranges efficiency comparison. Until there is one same-machine, same-dataset, same-clock-definition time-to-parity figure for LBM vs SaMAM vs SaMST, the fairness score stays low.

### 5. Clarity of writing - 6/10

**Strongest positive evidence**

The paper now has a real argument. The introduction has a coherent problem statement, the method is anchored around endpoint construction vs path learning vs terminal correction, and the tokenizer is no longer described as a vague embedding trick. The semantic-beacon / executor phrasing is effective.

**Strongest rejection risk**

The paper still tries to carry too many messages at once:

- latent metric geometry,
- compact efficiency,
- artifact-sensitive evaluation,
- no-op metric illusion,
- tokenizer execution bottleneck,
- and a formal design-grounding section.

That is too much load for the present manuscript. The prose is better than before, but still dense and over-committed. A reviewer under time pressure can come away thinking the central claim is shifting from section to section.

### 6. Figure quality - 6/10

**Strongest positive evidence**

The main framework figure is good enough for submission. It correctly separates inference path from training-side supervision, and it does not falsely depict target-style reference inputs at inference. The Distinct5 Pareto view also does something useful by making the `idt` floor visible.

**Strongest rejection risk**

The figure suite is not yet at the level of the manuscript ambition. The Pareto plot is informative but still somewhat cramped and annotation-heavy. More importantly, the figures do not yet deliver one decisive "this is the paper" visual. The current visuals support the argument; they do not close it.

### 7. Reproducibility - 5/10

**Strongest positive evidence**

The repository clearly contains substantial experiment bookkeeping: checkpoints, timing notes, dataset organization, evaluation scripts, comparison reports, and a working index. Relative to many submissions, this is unusually concrete.

**Strongest rejection risk**

The paper itself still reads less reproducible than the repo actually is. The checklist is only partial on several items, data preprocessing is not fully surfaced in the manuscript path, and some reported comparisons depend on mixed archival artifacts rather than a single clean rerun suite. A reviewer cannot be expected to reconstruct the project's internal evidence graph from multiple docs directories.

### 8. Acceptance likelihood - 3/10

**Strongest positive evidence**

There is a plausible acceptance path because the paper already contains:

- a meaningful evaluation insight (`idt` / raw-CLIP-style illusion),
- a compact model with strong cost behavior,
- a respectable artifact-sensitive comparison against the closest efficient baseline,
- and a concrete next-step ablation target.

This is not a dead paper.

**Strongest rejection risk**

If submitted now, I expect reviewers to converge on the following objection: "Interesting empirical project, but the strongest theoretical claim is not directly validated, and the efficiency comparison is not normalized enough to support the headline." That combination is usually fatal at AAAI.

## Recommendation

**reject**

## Three highest-ROI fixes

1. **Run and report one decisive latent-metric ablation under a single protocol.**  
   Same architecture, same dataset, same hardware, same seeds, same evaluation. Compare `MSE` vs `Huber` or `L1` where the paper claims metric robustness matters. Right now the paper argues this point harder than it proves it.

2. **Replace all vulnerable efficiency prose with a same-clock time-to-parity figure.**  
   One plot, one machine class, one dataset, one definition of time, one stop criterion. Put LBM, SaMAM, and SaMST on the same axes. Until this exists, the efficiency story remains attackable.

3. **Promote the no-op/metric-illusion result into a formal evaluation contribution, not just a cautionary observation.**  
   In every main comparison that involves art-to-art transfer, report at least raw `CLIP-S`, `LPIPS`, `idt` baseline, `\Delta_idt`, and one artifact-sensitive metric. This is the cleanest path to turning a vulnerability in prior evaluation practice into a genuine paper contribution.
