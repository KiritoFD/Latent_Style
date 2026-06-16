# Latent-Affine Refinement Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003`.
- Purpose: refine the positive latent-affine band and test whether `solver_pc + latent_lowpass` can repair the high-strength LPIPS cost.
- Runs: `s0.35`, `s0.45`, `s0.60`, `s0.50+PC0.10`, `s0.75+PC0.10`.
- Contract: eval-only, seed `42`, no checkpoint pullback, no generated PNG grids, same WikiArt-5 full-train / classview test surface.

## Results

| Variant | Transfer CLIP-S / LPIPS | All-pairs CLIP-S / LPIPS | Delta vs parent transfer | Delta vs parent all-pairs | Decision |
|---|---:|---:|---:|---:|---|
| Parent | `0.671820 / 0.314618` | `0.703234 / 0.312550` | control | control | control |
| s0.35 | `0.676781 / 0.313606` | `0.709329 / 0.308847` | `+0.004960 / -0.001012` | `+0.006095 / -0.003703` | structure positive |
| s0.45 | `0.679110 / 0.318818` | `0.711609 / 0.313230` | `+0.007289 / +0.004200` | `+0.008375 / +0.000680` | balanced frontier |
| s0.60 | `0.682390 / 0.330056` | `0.714810 / 0.323339` | `+0.010569 / +0.015438` | `+0.011576 / +0.010789` | style gain with structure cost |
| s0.50+PC0.10 | `0.680160 / 0.320104` | `0.712667 / 0.314519` | `+0.008340 / +0.005486` | `+0.009434 / +0.001969` | PC marginal |
| s0.75+PC0.10 | `0.685304 / 0.343517` | `0.717560 / 0.336053` | `+0.013483 / +0.028899` | `+0.014326 / +0.023503` | PC not enough |

## Decision

- `s0.45` is the current balanced frontier for this mechanism: it nearly matches the `s0.50` style gain while reducing the LPIPS cost.
- `s0.35` is the cleanest structure-positive point and is the best option if the next integration needs a safe LPIPS anchor.
- `solver_pc + latent_lowpass` repairs LPIPS only slightly: `s0.50+PC0.10` improves transfer LPIPS by about `0.0021` versus pure `s0.50`, but it does not change the conclusion or rescue `s0.75`.
- The style target `0.74` is still not reached. The next mechanism should be a style-generator change, not more postprocess strength. Use latent-affine as a cheap eval-time style amplifier and safety diagnostic.

## Artifacts

- Manifest: `latent_affine_refine_eval_manifest.csv`.
- Matched delta: `control_delta.csv`.
- Curve: `curves/latent_affine_refine_k070_e3_eval_only_curve.csv`.
- Eval bundles: `eval/latent_affine_refine_k070_e3/`.
- Homepage plot source: `SchrodingerBridge/aaai2027/page1_bundle/wikiart5_page1_clip_lpips_points.csv`.
