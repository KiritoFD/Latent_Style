# AAAI 2027 Weak-Reject Gate Checklist

Date: 2026-06-03  
Owner lane: `standing_adversarial_reviewer`  
Purpose: this is the minimum evidence package required before the next review round is allowed to move above `weak_reject`.

## Gate A - Activated endpoint-metric closure

- [ ] Submit one repaired experiment packet in which the switched local loss term is on the **active** transport path.
- [ ] Use one of only two valid designs:
  - `objective_mode=omf` with `w_flow > 0`, or
  - a non-`omf` velocity-regression path where `loss_type` is provably active.
- [ ] Log resolved config values in the run artifact, including at least:
  - `objective_mode`
  - `w_flow`
  - `loss_type`
  - dataset
  - batch
  - seed
- [ ] Provide full-eval results for all compared arms on the same Distinct5 base.
- [ ] Update `aaai2027_master_experiment_log.csv` with explicit interpretation that distinguishes:
  - activated probe
  - invalidated near-null control

Required evidence:

- run config files
- resolved-config dump
- train log
- full-eval summaries
- one compact comparison table or CSV

Not accepted:

- rerunning the old `mse / huber / l1` trio with `w_flow=0.0`
- prose claiming the path is active without a resolved-config artifact

## Gate B - SA-SWD novelty isolation

- [ ] Run a fixed-base Distinct5 ablation comparing:
  - semantic projection axes
  - random projection axes
- [ ] Keep backbone, tokenizer family, dataset, seed policy, and evaluation scope matched.
- [ ] Report whether semantic axes improve at least one headline dimension without unacceptable regression:
  - better style at similar LPIPS, or
  - better LPIPS/artifact profile at similar style

Required evidence:

- matched configs
- per-arm full-eval summaries
- one direct comparison table/plot

Not accepted:

- comparing across different tokenizer families or different training scopes
- qualitative-only claims

## Gate C - Efficiency fairness closure

- [ ] Produce one normalized `time-to-parity` artifact with explicit:
  - hardware
  - stop criterion
  - compared methods
  - whether eval time is included or excluded
- [ ] Keep the comparison scope single-regime and same-clock.
- [ ] Make clear whether the claim is:
  - operating-point observation, or
  - matched time-to-parity result

Required evidence:

- timing CSV
- figure used by the paper
- short provenance note describing measurement protocol

Not accepted:

- mixing operating-point times with cumulative training curves and calling it a universal speedup

## Gate D - Paper-side boundary consistency

- [ ] Manuscript wording must stay aligned with the evidence above.
- [ ] If Gate A is still open, the paper must not claim local-loss geometry closure.
- [ ] If Gate B is still open, the paper must not oversell semantic-axis novelty.
- [ ] If Gate C is still open, the paper must not use broad efficiency rhetoric.

Required evidence:

- updated manuscript diff
- reviewer-facing note pointing to the exact sections changed

## Pass rule for the next review round

The next round may move above `weak_reject` only if:

1. Gate A is closed with an **activated** endpoint-metric probe;
2. Gate B is closed with a matched semantic-vs-random SA-SWD result;
3. Gate C is closed with a normalized timing artifact;
4. Gate D is satisfied so the manuscript does not outrun the evidence.

If any one of these four gates is open, the standing adversarial reviewer keeps the verdict at `weak_reject` or worse.
