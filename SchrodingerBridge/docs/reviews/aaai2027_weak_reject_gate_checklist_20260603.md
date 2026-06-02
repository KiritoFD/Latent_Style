# AAAI 2027 Weak-Reject Gate Checklist

Date: 2026-06-03  
Owner lane: `standing_adversarial_reviewer`  
Purpose: this is the minimum evidence package required before the next review round is allowed to move above `weak_reject`.

## Gate A - Activated endpoint-metric closure

- [x] Submit one repaired experiment packet in which the switched local loss term is on the **active** transport path.
- [x] Use one of only two valid designs:
  - `objective_mode=omf` with `w_flow > 0`, or
  - a non-`omf` velocity-regression path where `loss_type` is provably active.
- [x] Log resolved config values in the run artifact, including at least:
  - `objective_mode`
  - `w_flow`
  - `loss_type`
  - dataset
  - batch
  - seed
- [x] Provide full-eval results for all compared arms on the same Distinct5 base.
- [x] Update `aaai2027_master_experiment_log.csv` with explicit interpretation that distinguishes:
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

Current status:

- Gate A is now closed in the **negative** direction.
- The repaired endpoint-only trio was activated, fully logged, and reviewed in
  `R20260603C`.
- The allowed conclusion is narrow:
  - pure endpoint-only pointwise supervision does not recover the reviewed
    `H` mainline frontier.

## Gate B - SA-SWD novelty isolation

- [ ] Run a fixed-base Distinct5 ablation comparing:
  - semantic projection axes
  - random projection axes
- [x] Prepare a matched packet with fixed backbone, tokenizer family, dataset,
  seed policy, and evaluation scope.
- [ ] Keep backbone, tokenizer family, dataset, seed policy, and evaluation scope matched in the completed runs.
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

Current status:

- Gate B packet is prepared locally:
  - `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_semantic.json`
  - `configs/aaai2027/saswd_axis_h_base_seed42_b44_saswd_random.json`
- semantic arm is fully completed on the remote `3060`
- semantic summaries now exist for `epoch_0001` through `epoch_0003`
- random arm is now running on the remote `3060`, but the current run has
  crossed into a blocker-grade degraded-throughput state
- if the current random run finishes, its quality summaries may still be
  diagnostically usable, but its wall-clock behavior is not currently credible
  as normal formal-speed evidence
- Gate B remains open until both matched arms complete and are compared directly

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

Current status:

- adversarial rewrite hit list now exists:
  - `docs/reviews/aaai2027_rewrite_hit_list_20260603.md`
- first manuscript-boundary alignment pass is now recorded in:
  - `docs/reviews/aaai2027_boundary_alignment_pass_20260603.md`
- follow-up adversarial memo on residual overclaims now exists:
  - `docs/reviews/aaai2027_boundary_followup_overclaims_20260603.md`
- the follow-up memo's two residual risks were immediately narrowed further in
  the manuscript:
  - Distinct5 timing language now stays at operating-point context only
  - the SA-SWD contribution bullet now reads as a current-mainline design, not
    a closed semantic-axis novelty claim
- Gate D remains open until the paper diff is reviewed against the still-open
  Gate B and Gate C evidence gaps

## Pass rule for the next review round

The next round may move above `weak_reject` only if:

1. Gate A is closed with an **activated** endpoint-metric probe;
2. Gate B is closed with a matched semantic-vs-random SA-SWD result;
3. Gate C is closed with a normalized timing artifact;
4. Gate D is satisfied so the manuscript does not outrun the evidence.

If any one of these four gates is open, the standing adversarial reviewer keeps the verdict at `weak_reject` or worse.
