# 2026-05-30 Tokenizer Restart Design

## Baseline Choice

This restart uses documented baselines instead of the failed `high032` tokenizer run.

1. `t01_ws0p03_g6_nl0p05` is the raw-style/Pareto baseline.
   Evidence: `docs/experiments/2026-05-20-256-diffeomorphic-tangent-progress.md` reports `clip_style=0.7264`, `LPIPS=0.5170`, `clip_content=0.7570`, `DINO-SSM=0.0263`. The 2026-05-22 regression-fix note confirms the checkpoint can reproduce `clip_style=0.7264026194016138`.

2. `t00_ws0p03_g6_nl0` is the stable structure baseline.
   Evidence: the same docs report `clip_style=0.7259`, `LPIPS=0.5166`, `clip_content=0.7602`, `DINO-SSM=0.0259`.

3. EC best is a separate content-preserving reference, not the same operating point.
   Evidence: `docs/repro_report_zh/00_总览与核心结论.md` records `K2_r00_balanced_default epoch3` with `CLIP-S=0.6980`, `CLIP-C=0.8727`, `LPIPS=0.3777`, `EC=0.4343`, and `Entropy gate 5.0 epoch1` with `CLIP-S=0.6916`, `CLIP-C=0.8804`, `LPIPS=0.3684`, `EC=0.4368`.

The tokenizer line should therefore be judged against two endpoints: keep `t01/t00` style strength from collapsing, while learning a representation knob that can move toward EC-best behavior without manual per-style hacks.

Current baseline policy:

- Primary style baseline: `t01_ws0p03_g6_nl0p05`, because it is the documented paper-facing strong-style operating point.
- Stability fallback: `t00_ws0p03_g6_nl0`, because it is the adjacent stable structure point from the same sweep.
- Content-preserving reference: EC-best (`K2_r00_balanced_default epoch3` and `Entropy gate 5.0 epoch1`), because these define the low-LPIPS/high-content endpoint rather than the style endpoint.
- Invalid baseline for this restart: failed high032/set-encoder tokenizer runs. They are negative ablations only.

## Representation Hypothesis

The tokenizer is not a larger embedding. Its job is to expose a small metric space for style control.

The first implementation should represent each style as three low-dimensional fields:

- `identity`: global color/moment displacement. This should affect broad color and contrast changes.
- `texture`: local brush/roughness amplitude. This should affect high-frequency style strength.
- `geometry`: stroke transport tendency. This should affect tangent/diffeomorphic behavior only through existing LANCET consumers.

Stage 1 should keep the parameter count small, roughly embedding-scale rather than transformer-scale. A reasonable target is below 20k parameters. This prevents the tokenizer from becoming a second backbone and makes frozen-backbone diagnostics meaningful.

## Stage-1 Tokenizer

Use a factorized token table:

```text
style_id -> identity token [d_id=24]
style_id -> texture token  [d_tex=32]
style_id -> geometry token [d_geo=24]
concat -> LayerNorm -> Linear -> style_code [style_dim=160]
```

Important constraints:

- No transformer in Stage 1.
- No style latent encoder in Stage 1.
- No external teacher or Seedream path.
- Keep the LANCET consumer interface unchanged: it still receives one `style_code`.
- Store debug tensors: per-field norm, pairwise cosine, and projected code norm.

This is intentionally close to `style_emb`, but it is no longer an opaque vector. It gives us separable axes that can be frozen, reinitialized, ablated, and measured.

The first code implementation is intentionally below transformer scale: three small style tables, one learned per-field gate vector, and one linear projector. This makes the first experiment a representation probe rather than another backbone-capacity experiment.

## Diagnostics Before Full Training

A tokenizer run is invalid unless these checks pass:

1. Parameter and gradient reachability:
   `style_tokenizer.identity`, `texture`, `geometry`, and projector must receive non-zero gradients in a batch16 smoke.

2. Style separability:
   per-style `style_code` cosine matrix should not collapse to all near-1.0 after initialization or after smoke.

3. Field usage:
   field norms should be non-zero and not dominated by a single field by more than roughly 10x during early training.

Only after those pass should we run a real base on the remote 3060.

Implementation note from the first smoke:

- Replacing the table alone was insufficient: the initial forward smoke produced finite images but zero tokenizer gradients.
- Root cause: in this clean baseline, `style_code` was mostly bypassed when skip routing was disabled and the decoder modulation module was instantiated but not applied.
- Fix: apply decoder-side `NormFreeModulation` before the output delta head. After this, all tokenizer fields and the projector receive non-zero gradients in a batch2 CPU shape smoke.

## Training Plan

1. Implement Stage-1 tokenizer on clean branch `codex/tokenizer-clean-c3058eab`.
2. Derive smoke configs from documented `t00/t01` settings, not from `high032`.
3. Run batch16 smoke for correctness and gradient diagnostics.
4. Run batch80 or calibrated remote batch for 8 epochs to establish a tokenizer+LANCET base.
5. If the base is within range, alternate:
   - freeze LANCET, train tokenizer only;
   - freeze tokenizer, train LANCET consumer;
   - compare movement against `t01/t00` and EC-best endpoints.

Success is not a smoke test. The real target remains `clip_style >= 0.73` with `LPIPS` near `0.45`, verified on strict evaluation and visual grids.
