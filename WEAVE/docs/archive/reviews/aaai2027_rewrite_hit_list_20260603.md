# AAAI 2027 Rewrite Hit List

Date: 2026-06-03  
Lane: `adversarial_review`  
Purpose: practical manuscript-pass hit list showing exactly which current sections, tables/figures, and claim types outrun the evidence.

## Priority 1 - Abstract

### Hit 1: abstract efficiency sentence

- current target:
  - abstract sentence with `310s vs 6769s` and `1.2 minutes vs 7.6 hours`
- why it outruns evidence:
  - normalized `time-to-parity` is still open
  - current evidence is only operating-point timing under different stopping rules and mixed comparison regimes
- rewrite action:
  - either delete the cross-method timing numbers from the abstract, or recast them as:
    - `measured operating-point wall-clock observations in reproduced records`
  - do **not** leave the current sentence in a form that reads like a generalized efficiency win
- hold until evidence:
  - one same-scope normalized time-to-parity artifact

### Hit 2: abstract Distinct5 frontier sentence

- current target:
  - abstract sentence claiming the best LBM checkpoints define `the strongest measured content-preserving frontier among the evaluated methods`
- why it outruns evidence:
  - this is only safe if constantly pinned to `Distinct5-512` as a metric-stress split
  - a skeptical reviewer can still read it as broad benchmark superiority
- rewrite action:
  - force the sentence to say:
    - `on the Distinct5-512 stress split among the currently reproduced points`
  - keep `idt` / no-op diagnosis in the same breath

## Priority 2 - Contributions list

### Hit 3: SA-SWD contribution bullet

- current target:
  - contribution bullet: `Semantic-Aligned Sliced Wasserstein (SA-SWD)`
- why it outruns evidence:
  - semantic-vs-random axis isolation is still open in the experiment log
  - current evidence closes `terminal SWD matters`, not `semantic axis selection is proven necessary`
- rewrite action:
  - downgrade from novelty-closure wording to:
    - `proposed semantic-aligned terminal matching mechanism`
    - or `semantic-aligned SWD design used in the current mainline`
- hold until evidence:
  - matched semantic-vs-random axis ablation

### Hit 4: efficiency contribution bullet

- current target:
  - contribution bullet: `measured operating-point efficiency-quality result against the closest efficient baseline SaMST`
- why it outruns evidence:
  - still mixes historical operating-point timing with comparison rhetoric
  - this is the cleanest remaining fairness attack
- rewrite action:
  - keep only the quality claim in main contributions
  - move timing specifics to experiment text or a cost note
  - or prepend `under reproduced operating-point records`

## Priority 3 - Method section

### Hit 5: local-loss / endpoint paragraph around Eq. (5) and the latent-metric paragraph

- current target:
  - lines around the parent bridge objective, endpoint-side metric discussion, and `present evidence clearly closes the endpoint-side OT + W1 terminal-matching story`
- why it outruns evidence:
  - the repaired endpoint trio supports a **negative** endpoint-only closure
  - it does **not** directly prove why `W1` terminal matching wins mechanistically
- rewrite action:
  - replace any `clearly closes` wording with:
    - `current evidence favors the OT + W1-style mainline over isolated endpoint-only pointwise supervision`
  - explicitly cite the repaired trio as a negative endpoint-only packet, not as a direct proof of `W1` optimality

### Hit 6: SA-SWD mechanism prose in Method

- current target:
  - the semantic-axis explanation and the complementary-roles paragraph
- why it outruns evidence:
  - these paragraphs currently read like semantic axis selection is already validated
  - the matched random-axis control is still running/open
- rewrite action:
  - change from declarative necessity language to proposal language:
    - `we use`
    - `we hypothesize`
    - `the planned matched random-axis control tests whether this matters beyond ordinary endpoint SWD`

## Priority 4 - Experiments section

### Hit 7: Table `tab:main` interpretation paragraph

- current target:
  - paragraph after Table `tab:main` ending in `cleanest balance ... and retraining cost`
- why it outruns evidence:
  - the table itself is mostly defensible on quality
  - the `retraining cost` part pulls in the unresolved fairness story from `tab:cost`
- rewrite action:
  - split the claim:
    - quality/frontier claim here
    - timing claim only in the dedicated cost discussion with explicit caveat

### Hit 8: Table `tab:cost`

- current target:
  - the whole table and the paragraph under it
- why it outruns evidence:
  - this is the most attackable table in the paper
  - operating-point times are shown side-by-side without normalized parity
- rewrite action:
  - either:
    - demote the table to a softer note with explicit `operating-point` labeling in the title, or
    - replace it in the next pass once time-to-parity is ready
  - if it stays, the paragraph must stop short of any comparative speed narrative beyond `recorded operating-point wall-clock`

### Hit 9: Distinct5 figure/table usage

- current targets:
  - Figure `fig:distinct5`
  - Table `tab:distinct5`
  - Distinct5 subsection prose
- why they outrun evidence:
  - the `idt` logic is supported
  - the broadest frontier language is still vulnerable to `custom stress split` pushback
- rewrite action:
  - keep `idt` and `delta_idt` front and center
  - rewrite the subsection opening so the first sentence says:
    - `Distinct5-512 is a metric-stress benchmark, not a universal art benchmark`
  - remove any wording that sounds like this table settles general AST superiority
  - if timing remains in `tab:distinct5`, keep it visibly subordinate to the metric-stress story

### Hit 10: mechanism ablation interpretation

- current target:
  - text after Table `tab:ablation`
- why it outruns evidence:
  - `terminal distribution matching matters` is supported
  - `SA-SWD as semantic novelty` is not yet supported
- rewrite action:
  - keep the current destructive-ablation claim at:
    - `terminal matching matters`
  - do not let this paragraph do implicit work for the missing semantic-vs-random proof

## Priority 5 - Discussion, Conclusion, Checklist

### Hit 11: Discussion / Conclusion frontier wording

- current target:
  - Discussion line `LBM gives the strongest measured content-preserving operating region`
  - Conclusion line `defines the strongest measured content-preserving frontier`
- why they outrun evidence:
  - they are close to safe, but still need repeated scope pinning
- rewrite action:
  - add scope every time:
    - `on Distinct5-512 among the currently reproduced points`
    - or `under the present reproduced protocols`

### Hit 12: reproducibility checklist theoretical-contribution sentence

- current target:
  - checklist statement that the formal results have `proofs and experimental validation`
- why it outruns evidence:
  - endpoint trio does not validate the theorem block
  - path-stability probe is still planned
- rewrite action:
  - change to:
    - `proofs and partial empirical support`
    - or `design-grounding analysis with empirical checks where available`

## Immediate manuscript-pass order

1. Strip or soften all comparative timing rhetoric in the abstract, contribution bullet, `tab:cost`, and Distinct5 timing mentions.
2. Rewrite SA-SWD contribution/method language from `validated novelty` to `proposed mechanism pending matched axis control`.
3. Reframe the repaired endpoint trio everywhere as a **negative endpoint-only closure**, not as a broad latent-metric victory.
4. Add scope pins to every Distinct5 superiority sentence.
5. Demote theorem-support wording in the checklist and any front-facing summary that implies full empirical closure.

## Practical pass/fail rule for the next manuscript pass

If the next draft still lets a reviewer read:

- `SA-SWD semantic alignment is already proven`,
- `LBM is faster in a generally fair sense`,
- `the repaired endpoint trio proves a broad latent-metric thesis`, or
- `Distinct5 establishes broad AST superiority`,

then the draft is still weak-reject vulnerable even before new experiments land.
