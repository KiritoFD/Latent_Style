# AAAI 2027 Tokenizer + Timing Re-Audit

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: current `paper_aaai2026.tex` after the latest tokenizer-boundary and timing-context wording changes

## Short verdict

For the two targeted attack surfaces, the manuscript is now **mostly paper-safe**.

- **Tokenizer / renderer-side mechanism claims:** now paper-safe at current scope.
- **Timing / efficiency / parity wording:** materially safer than before, with one mild residual summary-risk.

## Tokenizer / renderer-side mechanism audit

Current wording is now bounded correctly:

- abstract pins the diagnosis to `the tested Distinct5 tokenizer variants` and `in this setting`;
- related work no longer reads like a general theorem about tokenizer size;
- discussion and conclusion now frame the result as a current probe-level mechanism diagnosis rather than a universal tokenizer law.

Reviewer reading:

- safe as a bounded Distinct5-family mechanism diagnosis;
- not currently vulnerable on the earlier `tokenizer size is not the bottleneck` overclaim line.

## Timing / efficiency / parity audit

The new timing-context wording is a substantial improvement and is now mostly aligned with the Gate C boundary:

- the Distinct5 figure is explicitly labeled as a `same-scope timing-context artifact`;
- the text explicitly rejects a `false normalized parity claim`;
- historical timings are framed as operating-point records rather than parity evidence.

## Remaining exact risky sentence

The only sentence I would still flag as mildly vulnerable is in the Conclusion:

> `Under the reproduced historical strict-750 protocol, LBM reaches a cleaner style-content trade-off than SaMST, S2WAT, StyleID, and AdaIN-family baselines while remaining practical to retrain and evaluate.`

Why this is still mildly risky:

- `practical to retrain and evaluate` is broader than the newly established timing-context boundary;
- it compresses two different evidence types:
  - historical operating-point bookkeeping
  - Distinct5 same-scope timing context
- a skeptical reviewer could still ask: practical in what normalized sense?

## Remaining risky claim type

Even if the wording above stays, the paper must still avoid this claim type:

- any sentence that upgrades the Distinct5 timing-context artifact into a fair normalized time-to-parity or comparative training-speed win.

## Bottom line

Tokenizer-side wording is now safe.  
Timing-side wording is almost safe, but the Conclusion still contains one mild summary sentence that is looser than the rest of the revised timing boundary.
