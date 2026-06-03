# AAAI 2027 Review Consensus - R20260603L

Date: 2026-06-03

Checkpoint label:

- `current_paper_after_path_stability_prelauch_and_surface_cleanup`

Lane agents used:

- `Hubble` - `adversarial_review`
- `Boyle` - `scorecard`
- `Turing` - `experiment_audit`
- `Hilbert` - `figure_audit`

Primary inputs:

- `SchrodingerBridge/aaai_submission/paper_aaai2026.tex`
- `SchrodingerBridge/docs/aaai2027_working_index_20260602.md`
- `SchrodingerBridge/docs/experiments/aaai2027_master_experiment_log.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_review_score_log.csv`
- `SchrodingerBridge/docs/reviews/aaai2027_gate_status_and_next_experiment_priority_20260603.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-time-to-parity/README.md`
- `SchrodingerBridge/docs/experiments/2026-06-03-path-stability-protocol.md`

## Consensus status

Current consensus remains:

- `weak_reject`

Current safety band remains:

- `claim_safety_band = narrow_only`
- `evidence_closure_band = partial`

## Shared conclusion

All four lanes converged on the same operational read:

1. the paper is now materially cleaner and more bounded than earlier drafts;
2. the strongest safe story is still the Distinct5 no-op-aware frontier and
   the negative endpoint-only / negative SA-SWD closures;
3. the single highest-value unblocked mechanism packet is still Distinct5
   path-stability / weakened-kinetic on the `H` family;
4. without that packet, the paper should not ask for stronger mechanism credit
   than it already has.

## What is safe right now

- Distinct5 no-op-aware evaluation remains the strongest paper-facing
  contribution.
- Endpoint-only pointwise supervision is negatively closed and should stay
  written that way.
- SA-SWD may remain as the retained current mainline design, but not as a
  proven semantic-axis superiority result.
- Tokenizer localization is usable only as `L`-family-local evidence.
- Timing is usable only as bounded operating-point / timing-context evidence.

## What is still unsafe

- broad latent-metric or theorem-level rhetoric;
- contribution-level SA-SWD novelty language;
- tokenizer theory written as if it were family-generic or `H`-family closed;
- strong fair-speed or parity claims;
- keeping two separate figures for one artifact-diagnosis claim.

## Figure-surface read

The figure lane adds one concrete paper-shape recommendation:

- keep `framework_lbm_main_v5.png`;
- keep `figures/fig_distinct5_pareto.pdf`;
- merge `fig_qual_grid_ours_vs_samst.png` and `fig_zoom_ours_vs_samst.png`
  into one linked artifact-diagnosis figure;
- demote or rebuild `fig_ablation_pareto.png` because the current selective
  `6 of 12` view reads like an internal diagnostic slice rather than a
  reviewer-facing main-paper figure.

## Ordered next actions

1. **Land path-stability**
   - run the Distinct5 `H`-family `base + k025 + k000` packet on the remote
     3060;
   - then run `tools/probe_path_stability.py`;
   - judge it strictly by the accept/reject rule in the protocol.

2. **Do one manuscript boundary pass**
   - demote theory / SA-SWD / tokenizer / timing wording to the currently
     landed narrow boundary;
   - do not reopen broad efficiency or semantic-axis rhetoric.

3. **Consolidate the main figure set**
   - replace the split grid-plus-zoom artifact story with one composite figure;
   - move the selective ablation figure out of the main set unless rebuilt.

## Borderline path

This cycle agrees on a narrow route from `weak_reject` toward `borderline`:

- land one positive same-family mechanism packet on path stability;
- absorb it without expanding unsupported theory rhetoric;
- reduce figure fragmentation so the evidence spine is faster to trust.
