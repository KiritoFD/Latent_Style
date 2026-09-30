# AAAI 2027 Adversarial Review - 2026-06-03 R3

Reviewer lane: `adversarial_review`  
Reviewer: `Lorentz`  
Scope: current manuscript plus active Distinct5 / flow-loss evidence state

This memo is intentionally harsh. It records the fastest current rejection path
and the minimum closure action for each risk.

## 1. Latent-metric thesis is ahead of direct evidence

- risk:
  - the paper still implies that manifold-aware metric correction is a core
    proved thesis, while the matched `mse / huber / l1` closure is not finished
- why dangerous:
  - the manuscript already acknowledges that the reproduced historical and
    Distinct5 results still use the default `MSE` residual
  - the master experiment log still marks the matched metric-validation bundle
    as `unsafe`
  - the flow-loss ablation README explicitly says the current probe is not yet a
    direct test of the broader endpoint-side `W1` versus Euclidean thesis
- fastest closure:
  - finish the three matched remote arms
  - if `Huber` or `L1` do not win clearly, shrink the main claim to the
    endpoint-side OT + SA-SWD story instead of insisting that `MSE` is
    geometrically wrong everywhere
- needs remote 3060:
  - yes

## 2. Efficiency claim still has an apples-to-oranges attack surface

- risk:
  - the paper still contains operating-point versus cumulative-training-time
    comparisons that can be read as unfair speedup claims
- why dangerous:
  - `310s vs 6769s` and `1.2m vs 7.6h` are still visible enough in the current
    paper to trigger a fairness objection
  - the normalized `time_to_parity_curve` remains open in the experiment log
- fastest closure:
  - immediately soften all strong efficiency wording to
    `operating-point wall-clock observation under reproduced records`
  - do not restore stronger language until a normalized same-scope time curve
    exists
- needs remote 3060:
  - no

## 3. SA-SWD novelty is still under-proved

- risk:
  - the paper presents semantic-axis selection as meaningful novelty, but the
    matched `semantic axes` versus `random axes` ablation is still missing
- why dangerous:
  - without this, a reviewer can reduce SA-SWD to ordinary SWD plus semantic
    decoration
- fastest closure:
  - run a minimal matched Distinct5 table comparing semantic axes against random
    axes on one fixed base
  - if the gain is small, downgrade the claim from novelty to effective design
    choice
- needs remote 3060:
  - yes

## 4. Theory currently behaves more like decoration than closure

- risk:
  - the theorem stack still looks stronger in prose than in actual empirical
    support
- why dangerous:
  - the paper says the formal results are paired with direct empirical
    validation, but the path-stability probe is still open
  - limitations are present, yet still too far from the front-facing claims
- fastest closure:
  - demote the rhetoric now to `design-grounding analysis with partial
    empirical support`
  - only re-promote after the path-stability probe lands
- needs remote 3060:
  - no

## 5. Distinct5 plus `idt` can still be attacked as a custom stress split

- risk:
  - the current story can still be reframed as
    `you chose a split where the baseline struggles, then rewrote the ranking`
- why dangerous:
  - even after recent tightening, the paper still leans on frontier language
    that may sound broader than the actual evidence
- fastest closure:
  - add one compact regime-evolution figure or table showing
    `Legacy256 -> WikiArt512 -> Distinct5` for `idt` and `delta_idt`
  - keep all prose pinned to `metric diagnosis on separated art-to-art
    transfer`
- needs remote 3060:
  - no

## Bottom line

Current adversarial verdict: `weak_reject`

Reason:

- the paper has a real idea and a real Distinct5 diagnostic signal
- but the strongest current narrative strands are still ahead of the matched
  evidence on metric correction, SA-SWD isolation, and normalized efficiency
