# AAAI 2027 Weak-Reject Pressure Memo

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: current paper state after the `idt` / no-op diagnosis and the repaired endpoint-metric ablation

## Bottom line

The `idt` baseline and the repaired endpoint trio materially improve the paper:

- `idt` makes the Distinct5 metric-stress story real rather than rhetorical;
- the repaired endpoint trio closes the old inactive-probe objection and now supports a narrow negative conclusion about endpoint-only pointwise supervision.

Even with those two fixes, the paper still sits at `weak_reject`. The reason is no longer paper/code mismatch on the endpoint ablation. The remaining pressure comes from unsupported novelty and fairness claims.

## Claims still unsupported or still too aggressive

1. **SA-SWD semantic-axis novelty is still not closed.**
   The paper still presents SA-SWD as a distinct contribution, but the evidence still does not isolate semantic axes against random axes on the same fixed base. Current destructive ablations show that terminal distribution matching matters; they do not yet show that semantic axis selection matters.

2. **Efficiency rhetoric still outruns normalized evidence.**
   The paper repeatedly compares `310s vs 6769s` and `1.2m vs 7.6h`. Even when labeled as operating-point observations, a skeptical reviewer can still read this as an unfair speed claim because stopping rules and training scopes differ.

3. **Theorem-support wording is still stronger than its current empirical closure.**
   The endpoint trio helps the metric story, but it does not close the “each theorem is paired with direct empirical validation” line. The path-level support packet is still missing, so this remains a live attack surface if the theory contribution stays prominent.

4. **The Distinct5 frontier claim is still vulnerable to “custom stress split” pushback.**
   The `idt` argument is real, but a skeptical reviewer can still say the paper chose a separated split where `idt` is unusually strong and then redefined what counts as winning. The paper can survive this only if it stays explicit that Distinct5 is a metric-stress benchmark, not a universal art benchmark.

## Most vulnerable tables and figures

1. **Table `tab:cost`**
   This is still the easiest fairness attack. It puts operating-point times from different methods side by side without the normalized time-to-parity artifact that would justify stronger efficiency interpretation.

2. **Figure `fig:distinct5` and Table `tab:distinct5`**
   These are powerful, but also fragile. If the prose around them drifts from “metric stress test with explicit `idt` floor” toward “broad superiority benchmark,” a skeptical reviewer can attack both the split construction and the timing columns.

3. **Contribution bullet for SA-SWD plus Method/ablation discussion**
   The current paper can defend “terminal SWD matters,” but not yet “semantic alignment of projection axes is a proven novelty driver.” This makes the SA-SWD contribution line and the related mechanism interpretation more vulnerable than the endpoint story now is.

4. **Theory-contribution wording around formal analysis**
   The formal results are not dead, but their empirical closure is thinner than the paper’s current front-facing wording suggests. This is a weaker attack surface than cost fairness, but still active.

## Minimum new evidence that would move the score

1. **One matched semantic-vs-random SA-SWD axis ablation on the fixed Distinct5 base**
   Minimum acceptable output:
   - matched configs
   - same tokenizer/backbone family
   - full-eval summaries
   - one direct comparison table or plot
   - a conclusion that either validates semantic axes or narrows the contribution if the gain is small

2. **One normalized time-to-parity artifact**
   Minimum acceptable output:
   - same hardware scope
   - explicit stop criterion
   - clarified inclusion/exclusion of eval time
   - figure or CSV suitable for direct paper use

3. **If theory prominence is kept: one compact path-support artifact**
   Minimum acceptable output:
   - path-statistics measurement tied to the theorem section
   - artifact path and short interpretation note

## Reviewer pressure rule

If the next packet closes SA-SWD isolation and normalized timing fairness, the score can plausibly move from `weak_reject` to `borderline`. If those two remain open, the repaired endpoint trio alone is not enough to move the paper out of weak-reject territory.
