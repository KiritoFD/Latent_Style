# AAAI 2027 Claim Safety Memo

Date: 2026-06-03

This memo records the standing "paper-safe vs unsafe" boundary from the
persistent adversarial reviewer lane.

Primary interpretation rule:

- `goal.md` is useful as an ambition document;
- `docs/reviews/aaai2027_review_consensus_20260602_r2.md` is the current
  paper-safe boundary.

## Safe claims now

1. `idt` / no-op framing is valid and should remain explicit.
   - Raw `CLIP-S` is unsafe without the identity floor.

2. On historical strict-750, LBM is a compact content-preserving operating
   point or frontier point.
   - It may be written as close to the reproduced SaMST style level while
     preserving content slightly better.
   - It should not be written as universal dominance.

3. Distinct5-512 is a valid stress benchmark.
   - Use bounded wording such as:
     "within the evaluated comparison set, LBM currently defines the strongest
     content-preserving frontier."

4. Mechanism-wise, the current evidence safely supports:
   - OT-coupled endpoint construction,
   - `W1`-style terminal matching,
   - and the diagnosis that the main bottleneck is not merely tokenizer size,
     but whether target-style control survives execution through the
     content-conditioned renderer.

## Unsafe claims now

1. `Huber/L1` flow residual has already been proven decisive.
2. The broader latent-metric blind-spot correction story is already closed.
3. Semantic SA-SWD superiority is already proven.
4. Universal speedup or universal baseline dominance rhetoric.
5. Broad external-validity claims for Distinct5-512.
6. Comparison prose that hides mixed provenance.

## Single highest-ROI next experiment

- `flow_loss_metric_ablation`

Minimum closure:

- dataset: `Distinct5-512`
- base family: one fixed `F` or `H` line
- only switch: `bridge.loss_type = MSE / Huber / L1`
- hardware: remote `3060`
- output: full strict-750 paper bundle

## Repository/evidence issues most visible to reviewers

1. `comparison_20260602` still mixes fresh JSON, remote aggregate pulls,
   archive proxies, and recovered historical folders.
2. Some representative runs no longer have one uninterrupted artifact chain
   from config -> checkpoint -> generated directory -> summary -> figure input.

Writing policy until the next experiment block lands:

- do not let the prose outrun the evidence;
- where the evidence is mixed, say so directly;
- where the story is not closed, keep it in plan/review docs instead of the
  contributions list.
