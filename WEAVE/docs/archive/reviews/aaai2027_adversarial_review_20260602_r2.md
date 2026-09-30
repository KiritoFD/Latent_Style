# AAAI 2027 Adversarial Review

Date: 2026-06-02  
Round: `R20260602B`

Scope:

- `aaai_submission/paper_aaai2026.tex`
- `docs/experiments/comparison_20260602/comparison_report.md`
- `docs/reviews/aaai2027_review_consensus_20260603.md`

Overall verdict: `Reject`

## Top rejection reasons

1. The latent-metric thesis is not closed by current evidence.
   - Current headline runs still use `MSE` for the local flow residual.
   - This supports the endpoint-side `W1` story, not a full
     `MSE -> Huber/L1` correction claim.

2. SA-SWD novelty is under-isolated.
   - There is still no matched `semantic-axis vs random-axis` ablation under
     the same base configuration.

3. Efficiency rhetoric remains vulnerable.
   - The paper still mixes selected operating points and incomplete parity
     curves across LBM, SaMST, and SaMAM.

4. The paper is trying to sell too many messages at once.
   - The strongest evidence is narrower than the current claim set.

5. The formal section asks for more trust than the current evidence bundle
   justifies.

6. Distinct5-512 is still vulnerable to a benchmark-construction attack.
   - Because the split was screened to maximize style separation, the paper
     must keep stating this clearly and avoid implying broad external validity
     beyond the split.

7. Some headline comparison provenance remains heterogeneous.
   - Mixed legacy aggregates, recovered archive metrics, and fresh comparison
     points should stay explicitly labeled until the table is normalized.

## Real strengths

1. The `idt` / no-op framing is genuinely paper-worthy.
2. Distinct5-512 is the strongest current benchmark slice.
3. The tokenizer-vs-renderer diagnosis is more mature than a naive "larger
   style embedding" story.

## Claim that must be narrowed now

Replace broad wording like "we correct the latent-space distance mistake" with:

> current measured evidence supports OT-coupled endpoint construction and
> `W1`-style terminal matching as the key empirically validated corrections;
> the role of local flow-loss choice still requires direct ablation.

## Highest-ROI next experiment

Run a matched Distinct5-512 ablation:

- `MSE` vs `Huber` vs `L1`
- same stable base family (`F` and/or `H`)
- same init / seed / batch / epoch budget
- full strict-750 evaluation on the remote 3060

## Additional caution

Until the comparison provenance is cleaner, keep the current strongest paper
story centered on:

1. `idt`-anchored evaluation;
2. Distinct5 content-preserving frontier;
3. tokenizer-vs-renderer diagnosis.
