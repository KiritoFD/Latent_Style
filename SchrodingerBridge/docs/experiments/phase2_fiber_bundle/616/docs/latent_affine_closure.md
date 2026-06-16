# Latent-Affine Eval-Only Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003` from `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_vel_tok32_safe_semantic_topogate_k070_seed42_b12a1/epoch_0003.pt`.
- Mechanism: eval-only latent-space style affine after generation and before VAE decode.
- Controlled variables: no retraining, same checkpoint, same test surface, same seed `42`, same full `CLIP-S + LPIPS` eval contract.
- Switches are default-off:
  - `full_eval.latent_postprocess_mode=none`
  - `full_eval.latent_postprocess_strength=0.0`

## Results

| Variant | Transfer CLIP-S / LPIPS | All-pairs CLIP-S / LPIPS | Transfer style - IDT | All-pairs style - IDT | Decision |
|---|---:|---:|---:|---:|---|
| Parent | `0.671820 / 0.314618` | `0.703234 / 0.312550` | `+0.031900` | `+0.023111` | control |
| s0.25 | `0.674868 / 0.310584` | `0.707268 / 0.306689` | `+0.034947` | `+0.027145` | balanced positive |
| s0.50 | `0.680303 / 0.322202` | `0.712764 / 0.316212` | `+0.040382` | `+0.032641` | balanced style candidate |
| s0.75 | `0.685444 / 0.344580` | `0.717593 / 0.336945` | `+0.045523` | `+0.037470` | style ceiling candidate |

## Matched-Control Read

- `s0.25` is strictly better than the parent on style and LPIPS: transfer `+0.003047` style and `-0.004034` LPIPS; all-pairs `+0.004034` style and `-0.005861` LPIPS.
- `s0.50` is the current balanced style point: transfer `+0.008483` style for `+0.007584` LPIPS; all-pairs `+0.009530` style for `+0.003662` LPIPS.
- `s0.75` sets the current phase2 eval-only style ceiling: transfer `+0.013623` style and all-pairs `+0.014359` style, but LPIPS cost grows to `+0.029962` transfer and `+0.024396` all-pairs.

## Decision

Promote `latent_postprocess_style_affine` as the next cheap-first style injection path, not as a final model solution. It is the first eval-only mechanism in this phase that moves both transfer and all-pairs style materially, and `s0.25` also improves LPIPS. The result does not yet reach the `0.74 / 0.30` target, so the next step should be a narrower scan around `0.25-0.60` and a combined style-strong plus PC-safety screen before any long training lane.

## Artifacts

- Manifest: `latent_affine_eval_manifest.csv`.
- Matched delta: `control_delta.csv`.
- Curve: `curves/latent_affine_k070_e3_eval_only_curve.csv`.
- Eval bundles: `eval/latent_affine_k070_e3/s025`, `eval/latent_affine_k070_e3/s050`, `eval/latent_affine_k070_e3/s075`.
- Homepage plot source: `SchrodingerBridge/aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
