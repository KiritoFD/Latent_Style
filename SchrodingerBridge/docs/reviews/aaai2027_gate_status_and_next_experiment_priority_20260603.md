# AAAI 2027 Gate Status And Next Experiment Priority

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: current Gate A / B / C status only

## Gate A - Activated endpoint-metric closure

- current status:
  - `closed`
  - closed in the **negative** direction by the repaired endpoint-only trio
- gap:
  - no remaining closure gap for Gate A itself
- can it be closed now:
  - `yes`
- cannot claim:
  - not that all latent-space `MSE/L2` is broadly invalid
  - not that `Huber` or `L1` is a universal winner
  - not that the repaired trio proves why the full `OT + SA-SWD + kinetic` mainline wins

## Gate B - SA-SWD semantic-vs-random isolation

- current status:
  - `open`
  - semantic arm completed; random arm is matched in design but currently in a runtime-blocker state
- gap:
  - missing completed random-axis full-eval summaries and direct semantic-vs-random comparison
- can it be closed now:
  - `not yet`
  - reviewer-safe policy allows closure on **quality-only** evidence if the current abnormal random run finishes cleanly with matched summaries
  - otherwise a rerun is required
- cannot claim:
  - not that semantic projection-axis selection is proven necessary
  - not that SA-SWD novelty is fully isolated
  - not that any runtime behavior from the abnormal random run is representative formal evidence

## Gate C - Efficiency fairness closure

- current status:
  - `open`
- gap:
  - missing populated same-scope `time-to-parity` CSV and the corresponding figure/provenance artifact
- can it be closed now:
  - `no`
- cannot claim:
  - not a fair comparative training-speed win
  - not a normalized time-to-quality or time-to-parity result
  - not any broad speedup rhetoric from operating-point bookkeeping alone

## Unique highest-priority next experiment

The single highest-priority next experiment is:

- **finish or cleanly rerun the matched Gate B random-axis control on Distinct5-512, then compare it directly against the completed semantic arm**

Reason:

- Gate A is already closed;
- Gate C is still a required artifact, but it does not matter until Gate B stops leaving the SA-SWD novelty claim under-isolated;
- closing Gate B determines whether the paper may keep SA-SWD as a differentiated mechanism or must downgrade it to a tested design choice.
