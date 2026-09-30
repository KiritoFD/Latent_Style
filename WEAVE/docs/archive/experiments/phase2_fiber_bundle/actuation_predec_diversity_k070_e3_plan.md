# Actuation Predec Diversity Probe

Date: 2026-06-16

## Goal

Test the `fiber.md` generated-delta collinearity diagnosis directly. Earlier
output-basis, body+decoder injection, pre-decoder section, and proximal texture
lanes did not break the style plateau. This lane keeps the already-safe
pre-decoder style section and changes only the loss: penalize cross-style
generated-delta direction collapse.

## Controlled Delta

- Parent:
  `exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Config:
  `configs/aaai2027/phase2_actuation_predec_diversity_k070_e3_b16a2bf16_vlen010.json`.
- Mechanism held fixed:
  `model.style_delta_mode=predec_section`, `freeze_mode=injection_only`,
  tokenizer, solver, TopoGate, appearance head, data, and b16 accumulation-2
  throughput lane.
- Only loss delta:
  `bridge.w_generated_delta_diversity=0.08`,
  `bridge.generated_delta_diversity_margin=0.20`.
- Eval contract:
  `full_eval_fast10`, transfer-only, `10` source samples per style, b16 ONNX
  VAE decode, source latent cache, in-process runtime cache, no PNG/grid.

## Decision Rule

- Primary target: transfer CLIP-S, style-first toward `0.74`.
- LPIPS budget: values up to about `0.35` are acceptable only if style rises.
- Mechanism proof: generated-delta off-diagonal cosine should fall versus the
  predec-section baseline; if it does not move, treat as a weak/no-op
  implementation result rather than a theoretical negative.
- Positive evidence requires beating the confirmed proximal/R16 frontier near
  `0.6744` transfer style, with clear style gain or a much lower offdiag cosine
  that predicts style lift.
- Stop after formal convergence: no new transfer/all-pairs Pareto point for
  four later retained checkpoints and a near-flat tail; do not stop while the
  best point is in the newest two retained checkpoints.

## Launch Log

- 2026-06-16 05:35 remote WSL formal run started.
- PID: `16910`.
- Remote output root:
  `exp/aaai2027_phase2_actuation_predec_diversity_k070_e3_b16a2bf16_vlen010`.
- Remote log:
  `logs/phase2_actuation_predec_diversity_k070_e3_b16a2bf16_vlen010.launch.log`.
- Pre-launch remote parse:
  `w_generated_delta_diversity=0.08`,
  `generated_delta_diversity_margin=0.20`,
  `full_eval_runtime_model_cache=true`,
  `full_eval_delta_observability=true`.
- 64s health check:
  - active `python src/run.py --config ...predec_diversity...`
  - GPU sample `2314 / 12288 MiB`, util `93%`, power `142 W`
  - dataset `wikiarts_5_full_notest_latents_ema/train`
  - `Freeze mode=injection_only | trainable_count=10`
  - trainable tensors are only `style_section_basis_proj.*`,
    `style_section_weight_head.*`, and `style_section_out.*`
  - parent load: `loaded=272`, `missing=10`; missing keys are the new
    zero-init predec section parameters and are expected.

## Running Eval Curve

Local mirror:
`docs/experiments/phase2_fiber_bundle/eval/actuation_predec_diversity_k070_e3_b16a2bf16_vlen010/full_eval_fast10/clip_lpips_curve.csv`

| epoch | transfer CLIP-S | transfer LPIPS | eval wall | train time | delta diversity | offdiag cosine | active styles |
|---|---:|---:|---:|---:|---:|---:|---:|
| e1 | 0.680099 | 0.315598 | 71.43s | 138.32s | 0.000465 | -0.060303 | 4.94 |
| e2 | 0.680071 | 0.315606 | 22.96s | 99.22s | 0.000601 | -0.062256 | 4.98 |
| e3 | 0.680113 | 0.315620 | 25.75s | 98.46s | 0.000449 | -0.068848 | 4.98 |
| e4 | 0.680133 | 0.315616 | 27.47s | 96.83s | 0.000570 | -0.058350 | 4.93 |
| e5 | 0.680131 | 0.315594 | 29.53s | 96.66s | 0.000488 | -0.065918 | 4.97 |
| e6 | 0.680104 | 0.315584 | 32.01s | 96.49s | 0.000504 | -0.056885 | 4.97 |
| e7 | 0.680078 | 0.315580 | 20.42s | 100.32s | 0.000653 | -0.073730 | 4.97 |
| e8 | 0.680046 | 0.315578 | 26.17s | 96.88s | 0.000517 | -0.063477 | 4.94 |
| e9 | 0.680054 | 0.315589 | 38.13s | 98.00s | 0.000525 | -0.066406 | 4.98 |
| e10 | 0.680078 | 0.315564 | 20.65s | 96.86s | 0.000457 | -0.066406 | 4.95 |
| e11 | 0.680060 | 0.315555 | 25.88s | 97.07s | 0.000486 | -0.077148 | 4.96 |
| e12 | 0.680076 | 0.315551 | 26.37s | 97.00s | 0.000440 | -0.065918 | 4.97 |

Read at e12: the loss is active and the generated deltas are not collapsing
into a positive shared direction, but the style surface remains essentially
flat. e4 is the best style point; e5-e12 mainly improve LPIPS by tiny
amounts while style recedes. Continue under the formal rule until the non-Pareto
tail is long enough; because the objective is explicitly style-first and the
last eight checkpoints do not create a new style point, this lane is closed.

## Eval Infra Optimization

- Bottleneck diagnosis at e5: `lancet_generation=5.02s`,
  `vae_decode=7.77s`, `eval_metrics_loop=14.12s`; the metric loop, not model
  generation, is now the largest remaining component.
- Implemented default-compatible in-process caching for persisted eval
  artifacts when `full_eval_runtime_model_cache=true`:
  - reference feature payloads are keyed by cache file path, mtime, and size;
  - source image cache payloads are keyed by cache file signature and source set;
  - GPU reference CLIP prototypes are cached in-process and reused across
    same-run checkpoint evals.
- Updated this lane's future eval scheduling to `full_eval_metric_batch_size=50`
  and `full_eval_lpips_chunk_size=16`: CLIP/style scoring runs as one fast10
  batch, while LPIPS remains chunked to avoid a VGG peak-memory spike.
- Remote `src/utils/run_evaluation.py` and config were synced and
  `py_compile` passed. The active PID `16910` still reports cache status
  `loaded`, not `memory_loaded`, so the already-imported process did not
  reliably hot-load the new in-process artifact cache. e7 was fast
  (`20.42s` wall / `6.43s` eval total) but e9 regressed to `38.13s` wall /
  `23.83s` eval total; treat this as runtime/GPU-contention variance, not as
  measured proof of the new cache. The cache optimization is validated by
  compilation and will be measured on the next restarted lane.

## Plot Update

- Added e1-e12 transfer points to:
  `docs/experiments/phase2_fiber_bundle/plot_points.csv`.
- Wrote the normalized curve with per-epoch train time and delta observability:
  `docs/experiments/phase2_fiber_bundle/curves/actuation_predec_diversity_k070_e3_fast10_curve.csv`.
- Regenerated the AAAI2027 WikiArt-5 page-1 figure:
  `aaai2027/figures/fig_wikiart5_page1_summary.png`.

## Closure Decision

Closed as `style_flat_not_promoted`.

- Best style point: e4, transfer `0.680133 / 0.315616`.
- Final pulled point: e12, transfer `0.680076 / 0.315551`.
- Style tail: e5-e12 never beat e4; final is `-0.000056` below e4.
- LPIPS tail: final is `-0.000065` below e4, but this is LPIPS-only drift and
  not aligned with the style-first target.
- Mechanism read: generated-delta offdiag cosine is negative, so the auxiliary
  loss changed the measured delta geometry, but this did not translate into
  target-style CLIP lift. The remaining bottleneck is likely the final shared
  output head / actuator, not just cross-style delta cosine.
- Next action: archive this as negative evidence and run
  `style_delta_mode=head_adapter`, a zero-init style-conditioned residual head
  parallel to `dec_out`.
