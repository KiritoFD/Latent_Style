# AAAI 2027 Scorecard Review

Date: 2026-06-02  
Round: `R20260602B`

Scope:

- `aaai_submission/paper_aaai2026.tex`
- `docs/experiments/aaai2027_master_experiment_log.csv`
- `docs/experiments/comparison_20260602/comparison_report.md`
- `docs/reviews/aaai2027_review_consensus_20260603.md`

## Scores

- `novelty: 7/10`
  - The combination of OT-coupled endpoint assignment, latent flow matching,
    SA-SWD, and the tokenizer-vs-renderer diagnosis is meaningfully more novel
    than another style-token variant.
  - The `Distinct5-512 + idt floor` framing is a real paper contribution.
  - The mechanism novelty is still not fully closed because semantic SA-SWD is
    not yet isolated against random-axis baselines.

- `technical_quality: 5/10`
  - The method is internally coherent and the endpoint / kinetic components are
    directionally supported.
  - The causal chain is still too long where the draft turns endpoint-side
    evidence into broader latent-metric claims.
  - The theorem reads more like design grounding than decisive proof.

- `experimental_rigor: 4/10`
  - There is solid protocol structure: strict-750, all-pairs evaluation, `idt`,
    multiple metrics, and artifact-sensitive diagnostics.
  - The three key paper-closing blocks remain unfinished:
    `MSE vs Huber vs L1`, `semantic vs random axis`, and normalized
    `time-to-parity`.

- `clarity: 6/10`
  - The writing is readable and the intended contributions are visible.
  - The paper still carries too many parallel messages, which dilutes the main
    story.

- `reproducibility: 4/10`
  - The evidence indexing and experiment ledger are strong positives.
  - Several checklist items remain partial or absent, and timing scope is not
    yet normalized enough for strong reproducibility confidence.

- `significance: 6/10`
  - The `idt` adjustment can matter for the field if it is formalized cleanly.
  - The current significance is closer to a strong diagnostic plus a frontier
    result than a task-defining replacement narrative.

Average score: `5.33/10`

Overall confidence: `medium`  
Overall recommendation: `Weak Reject`

## Reviewer's bottom line

The core idea is publishable in principle, but the current claim set still
extends beyond the directly closed evidence.
