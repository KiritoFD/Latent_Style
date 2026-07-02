# Tokenizer Claim Evidence Matrix

Date: 2026-06-03

Scope: tokenizer / style-representation claims only. This matrix records which
claims are safe now, what exact evidence object supports each one, and what
next evidence object would be needed for a stronger version.

| safe-now claim | current evidence object | next evidence object required for a stronger claim |
| --- | --- | --- |
| The tokenizer is a style-side control map `T_phi: s -> c_s`, not a target-image encoder. | `src/style_tokenizer.py` object definition and current single-code interface. | None for the bounded object claim. |
| Tokenizer claims should be phrased at the level of executed representation, not code geometry alone. | Existing theory queue plus the landed `L`-family execution-alignment packet showing that executed geometry is the more relevant evidence object for style gain. | Same-family (`H`) execution-alignment packet if the paper wants this phrasing tied directly to the reviewed H family rather than a successor family. |
| Code-space separability alone is insufficient evidence of tokenizer success. | Landed `L`-family correlations: tokenizer-code geometry is only a partial predictor, while executed output geometry tracks `delta_idt` better. | Cross-family replication of the same result, ideally in the reviewed H family. |
| Capacity alone is not sufficient; execution through the renderer remains a live bottleneck candidate. | Current tokenizer family ablations plus the executed-geometry reading from the landed `L`-family probe. | Frozen-tokenizer / fresh-executor versus frozen-executor / fresh-tokenizer probe to localize whether execution is primary rather than merely plausible. |
| The tokenizer bottleneck is not proven to be pure code collapse. | Distinct5 `idt` / no-op boundary plus the landed `L`-family result that output-side evidence matters more than code-only evidence. | Frozen-side swap probe and code-to-output alignment probe together, so representation-side and executor-side weakness can be separated. |
| In at least one landed successor family (`L`), executed output geometry is more predictive of no-op-adjusted style gain than tokenizer-code geometry alone. | `docs/experiments/2026-06-03-tokenizer-execution-alignment-l-family/README.md` landed correlations and summary artifacts. | Same conclusion reproduced in the reviewed H family or another matched family to move from family-local to broader tokenizer claim. |
| Distinct5 no-op strength constrains tokenizer claims to end-to-end executed control, not code geometry alone. | `delta_idt` / `idt` audit artifacts plus current tokenizer theory memos. | Direct `code separability vs delta_idt` probe if the paper wants to quantitatively tie tokenizer quality to no-op-adjusted style gain. |
| It is still only a hypothesis that the next correct representation is a target-style carrier plus execution-risk gate. | Current theory queue and current lack of direct identifying evidence. | Matched carrier-vs-residual-vs-gated representation probe read through executed outputs and `delta_idt`, not just code geometry. |
| It is still only a hypothesis that identity / texture / geometry are real separable latent factors. | Current factorized tokenizer design in code, but no causal factor-identifiability evidence. | Field intervention / ablation probe measuring code-to-output alignment and `delta_idt` consequences for each factor separately. |
| It is blocked to claim that tokenizer theory is fully closed. | Current evidence is split across a landed `L` successor packet and a blocked original `H` packet; correlations are informative but not identification proofs. | Combined same-family execution-alignment plus frozen-side localization probe, or equivalent family-matched closure packet. |

## Practical reading rule

If a tokenizer claim depends on style being preserved **after** content-shaped
execution, then the primary evidence object must be one of:

- executed output geometry
- `delta_idt`-linked output behavior
- a frozen-side localization probe

Code geometry alone is sufficient only for bounded pre-execution claims.
