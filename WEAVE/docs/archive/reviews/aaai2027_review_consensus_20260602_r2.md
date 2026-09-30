# AAAI 2027 Review Consensus

Date: 2026-06-02  
Round: `R20260602B`

Source reviewer files:

- `aaai2027_adversarial_review_20260602_r2.md`
- `aaai2027_scorecard_20260602_r2.md`
- `aaai2027_experiment_audit_20260602_r2.md`

Checkpoint label:

- `current_paper_plus_distinct5_and_idt_recheck`

## Consensus status

Current submission status: `weak_reject`

The paper has a real core, but the current manuscript is still ahead of its
closed evidence on three fronts:

1. flow-loss metric claim;
2. semantic SA-SWD isolation;
3. normalized efficiency claim.

## Shared strengths

1. The `idt` / no-op framing is real and should stay central.
2. Distinct5-512 is the strongest current stress benchmark slice.
3. The tokenizer-vs-renderer diagnosis is more defensible than a pure
   "bigger tokenizer" narrative.

## Shared blockers

1. The paper cannot yet present a broad latent-metric correction story as
   experimentally closed.
2. SA-SWD novelty still lacks a decisive `semantic-axis vs random-axis`
   matched ablation.
3. Speed claims should remain bounded to operating-point or time-to-parity
   wording until the timing figure is normalized.
4. Distinct5 benchmark construction and mixed comparison provenance must remain
   explicitly bounded in the writing, or they become easy reviewer attacks.

## Claims to narrow now

Narrow the paper to the following evidence-backed wording:

- measured evidence supports OT-coupled endpoint construction and `W1`-style
  terminal matching;
- raw `CLIP-S` is unsafe without the `idt` baseline;
- LBM currently holds the strongest content-preserving frontier on Distinct5
  within the evaluated comparison set.

Avoid writing:

- `Huber/L1` flow residual is already proven decisive;
- semantic SA-SWD superiority is already proven;
- universal speed or universal baseline dominance claims.
- broad external-validity wording for Distinct5-512 beyond "stress benchmark";
- comparison-table rhetoric that hides mixed provenance.

## Ordered next experiments

1. `flow_loss_metric_ablation`
   - `MSE` vs `Huber` vs `L1`
   - Distinct5-512
   - same remote 3060 protocol

2. `sa_swd_axis_ablation`
   - `semantic-axis` vs `random-axis`
   - fixed `F` or `H` base

3. `normalized_time_to_parity`
   - `LBM` vs `SaMAM` vs `SaMST`
   - wall-clock x-axis
   - include `idt`-aware reading where possible

Note on reviewer disagreement:

- adversarial + scorecard lanes rank `flow_loss_metric_ablation` first because
  it most directly constrains the paper's over-broad latent-metric thesis;
- experiment-audit ranks `sa_swd_axis_ablation` first because it is the cleanest
  closure for the method's claimed novelty.

Operational choice for now: keep `flow_loss_metric_ablation` first in the queue,
but treat `sa_swd_axis_ablation` as equally paper-critical and schedule it in
the same experiment block.

## Paper-safe immediate policy

Until the first two experiments land, the manuscript should not escalate:

- the latent-distance correction claim,
- the SA-SWD novelty claim,
- or the broad speedup rhetoric.
