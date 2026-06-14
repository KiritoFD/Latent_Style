# SMoE Translator Closure Decision

Date: 2026-06-14

## Status

- Family: `smoe_translator_k070_e3`.
- Parent: `k070 epoch_0003`.
- Switch delta: `tokenizer_family=pure_latent_spatial -> smoe_translator`.
- Solver, loss, topogate, appearance, batch, and schedule were kept fixed.
- Closure status: `cost_stopped_not_promoted`.

## Evidence

- All retained checkpoints through `epoch_0015` have remote `CLIP-S + LPIPS` eval.
- Best style point: `epoch_0009`, transfer `0.672774 / 0.327155`, all-pairs `0.704251 / 0.322688`.
- Best late candidate-curve Pareto point: `epoch_0014`, transfer `0.672185 / 0.324834`, all-pairs `0.703218 / 0.322686`.
- Stop point: `epoch_0015`, transfer `0.671284 / 0.333647`, all-pairs `0.702173 / 0.330398`.
- e15 matched delta vs parent: transfer style `-0.000536`, transfer LPIPS `+0.019029`; all-pairs style `-0.001061`, all-pairs LPIPS `+0.017848`.
- Runtime observability at e15: `translation_delta_from_identity=0.018724`, `routing_entropy=1.541963`, `effective_experts=4.716263`, `spatial_abs=0.813047`.
- e15 training time: `1597.6s`; eval wall: `268.7s` from summary, `294.9s` from trainer log.

## Decision

Stop the SMoE-only lane after `epoch_0015`. The family is not formally converged by the automatic patience rule because `epoch_0014` reset the candidate-curve Pareto counter, but the observed return is not worth the extra remote time: the best style lift is small and consistently costs too much LPIPS against the matched parent.

Do not launch `SMoE + fiberwise_swd` from this parent. That would make the next loss-only experiment depend on a tokenizer parent that is already too costly and not promoted. Keep the implementation switch available, but park this branch until a cheaper SMoE variant or stronger parent exists.

## Artifacts

- Curve: `docs/experiments/phase2_fiber_bundle/curves/smoe_translator_k070_e3_remote_clip_lpips_curve.csv`.
- e15 eval: `docs/experiments/phase2_fiber_bundle/eval/smoe_translator_k070_e3/epoch_0015/`.
- Matched delta: `docs/experiments/phase2_fiber_bundle/control_delta.csv`.
- Homepage plot data: `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
- Rendered page-1 figure: `aaai2027/figures/fig_distinct5_page1_summary.png`.
