# Tokenizer Claim Matrix

Date: 2026-06-03

Scope: tokenizer / style-representation claims only. This matrix is operational:
each row states the current support status and the next probe needed to move the
claim.

| claim | current support status | required next probe |
| --- | --- | --- |
| The tokenizer is a style-side control map `T_phi: s -> c_s`, not a target-image encoder. | `supported` | None required for the bounded object definition. Keep future wording anchored to `src/style_tokenizer.py` and the existing single-code interface. |
| Tokenizer quality should be judged by end-to-end executed control, not code geometry alone. | `supported` | None required for the bounded claim. Preserve the code-to-output distinction in future analysis. |
| Capacity alone is not sufficient; execution through the renderer is the measured bottleneck. | `supported` | Run the code-to-output alignment probe to tighten this from bounded diagnosis to stronger mechanism evidence. |
| The current tokenizer has already learned the correct target-style representation. | `blocked` | Run code-to-output alignment plus `delta_idt`-linked output analysis; without executed style-gain evidence this claim should not be made. |
| Tokenizer collapse has been proven. | `blocked` | Run frozen-tokenizer / fresh-executor versus frozen-executor / fresh-tokenizer to separate representation weakness from executor weakness. |
| Larger tokenizer capacity does not matter. | `blocked` | Run the same frozen-side swap probe and compare capacity changes under matched execution; current evidence only rules out a simple capacity-only story. |
| The next correct representation is a target-style carrier plus execution-risk gate. | `hypothesis` | Run a matched carrier-vs-residual-vs-gated representation probe and read it through executed outputs, not code geometry alone. |
| Identity / texture / geometry are real separable latent factors in the learned style code. | `hypothesis` | Run a factor identifiability probe: intervene on each field separately and measure code-to-output alignment plus `delta_idt` consequences. |
| Distinct5 `idt` / no-op strength proves the tokenizer is bad. | `blocked` | Run target-style gain versus code-separability under `idt`; no-op failure alone is only an end-to-end execution diagnostic, not a tokenizer verdict. |
| Better tokenizer code separability should imply better no-op-adjusted style gain. | `hypothesis` | Run the `code separability vs delta_idt` probe explicitly; this is currently plausible but not yet demonstrated. |
| Representation-side failure and executor-side failure can already be localized from current results. | `blocked` | Run the frozen-tokenizer / fresh-executor versus frozen-executor / fresh-tokenizer probe; current evidence does not identify which side is primary. |
| Tokenizer claims may safely stay at the "executed representation" level. | `supported` | None required for the narrow wording boundary; keep stronger factorization claims out of manuscript-level conclusions until new probes land. |

## Working rule

If a tokenizer claim cannot survive the question

- "would this still be true if code geometry looked good but executed outputs did not?"

then it should remain either `hypothesis` or `blocked`.
