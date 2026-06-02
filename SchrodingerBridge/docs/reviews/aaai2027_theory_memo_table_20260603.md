# AAAI 2027 Theory Memo Table

Date: 2026-06-03

Scope: current paper/code-state audit for theory-owner lane only. This memo is
restricted to claim boundary, object-level consistency, and required closure.

| module_or_claim | current_paper_wording_band | evidence_status | exact reason | required_closure |
| --- | --- | --- | --- | --- |
| Tokenizer representation as executable control | `narrow` | partially supported | Current ablations support a bounded claim that code capacity alone is not the main bottleneck, and that queue/routing changes can improve the executed style-content frontier. They do not yet prove a general theory of target-style carrier factorization or risk-gated execution. | Keep wording at "measured bottleneck is execution through a content-conditioned renderer"; treat carrier-plus-gate as next-stage hypothesis, not established mechanism law. |
| OMF-vs-FM wording for active Distinct5 result family | `unsafe` | object mismatch | The paper's FM wording describes random-time bridge supervision, but the active Distinct5-family config chain resolves to `objective_mode=omf`. For this family, the main training object is closer to one-step endpoint-delta prediction with terminal matching and endpoint kinetic regularization, not clean empirical proof of the random-time FM objective. | Split paper wording by result family: describe current mainline as OMF-style endpoint-delta training unless a separate non-OMF family is the one being reported. |
| Endpoint metric story: "latent MSE is bad" | `unsafe` | overclaimed | The only robust object-level statement is that pointwise Euclidean endpoint reconstruction is high-risk for target-style alignment in compressed latents. That does not imply all latent-space MSE/L2 terms are bad. Velocity-energy L2 and same-chart distillation MSE remain natural. | Rewrite the thesis around object type: unsafe for target-style endpoint reconstruction, reasonable for velocity regression, path energy, and teacher-student alignment in a shared representation. |
| Flow-loss `MSE / Huber / L1` trio as evidence for the metric thesis | `unsafe` | invalid probe / near-null control | Resolved configs show `objective_mode=omf`, `w_flow=0.0`, and only `loss_type` switches across the three arms. In the active OMF path, this means the intended loss-kernel comparison does not hit the main training force. The near-overlap of `MSE`, `Huber`, and `L1` is therefore expected and non-dispositive. | Do not cite this trio as latent-metric evidence. Replace with a valid endpoint-objective ablation or a non-OMF / `w_flow > 0` block that actually activates the compared loss term. |
| SA-SWD as necessary semantic terminal matching | `narrow` | under-supported | Current evidence supports that terminal distribution matching matters and that SA-SWD is a coherent design. It does not yet close the stronger novelty claim that semantic projection-axis selection is necessary beyond random or ordinary SWD. | Run a matched semantic-vs-random SWD ablation and keep current wording to "semantic-aligned terminal matching is the proposed mechanism" rather than "proven necessary". |
| `idt` / no-op interpretation of raw `CLIP-S` on Distinct5 | `safe` | well-supported | Distinct5 shows a high unchanged-image prior, and `delta_idt` cleanly separates raw target-style affinity from no-op-adjusted style gain. The current note and table support a bounded protocol-level diagnosis: raw `CLIP-S` alone can mislead on separated art-to-art transfer. | Keep Distinct5 claims centered on `idt`, transfer-only filtering, and `delta_idt`. Do not generalize to "all AST evaluation is invalid" or "all prior baselines are degenerate". |
| Efficiency claims as mathematical statements | `unsafe` | overclaimed | Current evidence is operating-point wall-clock observation under fixed reproduced protocols, mixed hardware contexts, and method-specific stopping rules. It is not a universal speedup theorem, not a normalized time-to-quality law, and not a cross-hardware complexity statement. | Reduce wording to measured operating-point wall-clock comparisons only. If stronger efficiency wording is desired later, it needs a normalized same-scope time-to-parity study. |

## Minimal reading rule for current drafting

1. If a sentence claims mechanism, check that the trained object in code matches
   the mathematical object in the paper.
2. If a sentence claims metric superiority, check that the compared loss term is
   actually active on the main training path.
3. If a sentence claims speed, downgrade it to an operating-point observation
   unless it is backed by a normalized time-to-parity protocol.
