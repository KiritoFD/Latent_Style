# 2026-06-16 Stage Summary

## Current Retained Frontier

The active retained frontier is intentionally small and split by role rather than by mechanism count.

| role | point | transfer CLIP-S | LPIPS | status |
|---|---|---:|---:|---|
| style-first train-time retained | `I2SB low-anchor e9` | `0.701429` | `0.372203` | retained |
| balanced eval-time retained | `LatAff s0.45` | `0.679110` | `0.318818` | retained |
| structure-first eval-time retained | `LatAff s0.35` | `0.676781` | `0.313606` | retained |
| path-geometry diagnostic | `Slerp e2 peak` | `0.712038` | `0.476511` | diagnostic-only |
| style-ceiling diagnostic | `LatAff s0.75` | `0.685444` | `0.344580` | diagnostic-only |

Machine-readable copy: [retained_frontier.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/retained_frontier.csv).

## What Consistently Worked

- `I2SB / SDE` is still the only mechanism that reliably breaks the ODE-style ceiling. Whenever the model reaches the `0.70+` transfer CLIP-S band, an endpoint/bridge-style stochastic path is involved.
- Path geometry matters, but only as a first-order ingredient. `latent_slerp` produced the first clean matched gain over the same clean-I2SB parent, which is positive evidence for path shaping even though the full curve was not promotable.
- Low-anchor style preservation is better than hard lowpass replacement. The `0.50` anchor line is the first train-time point that keeps transfer style above `0.70` while bringing LPIPS down into the `0.37` band.
- Eval-time latent affine remains the cleanest cheap amplifier. `s0.35` and `s0.45` improve the parent without reopening the heavy LPIPS damage seen in the style-force diagnostics.

## What Failed And Is Now Retired

- `blend0p25`, `content_anchor`, and other scalar shrink / soft-anchor routes are retired as negative. They suppress structure drift and style actuation together, so they do not create a usable orthogonal split.
- Raw `Fiber-SDE` noise scans, mask-aware noise scans, residual-envelope noise scans, and the frozen local head follow-up are retired from the active path. They produced useful diagnostics, but no matched control delivered a target-facing style/LPIPS gain strong enough to keep spending lanes.
- `SMoE tokenizer`, `kinetic_release`, `RGB calibration`, `topology_release`, `appearance_blend`, and `PC lowpass` are retired from the active path. They either stayed flat, paid LPIPS for tiny style movement, or only repaired structure while weakening style.
- `gauge_field / fiber_flow`-style deeper rewrites are not active code paths in the current retained line. They were restored out of the main tree before closure and should only re-enter through isolated rough probes if we decide the evidence gap justifies it.

## Codebase Retention Policy

- Keep:
  - active I2SB / low-anchor / latent-slerp / latent-affine mechanisms that still support the retained frontier or the main diagnostics
  - training/eval infra, cached fast-eval path, curve CSVs, closure notes, and homepage plotting inputs
  - machine-readable retained frontier and stage-summary docs
- Retire from the main path:
  - mechanisms with explicit negative closure and no retained frontier role
  - plot prominence for failed sub-series; they stay in CSV/history but become visually secondary
  - unfinished rewrite prototypes that have not passed a matched-control closure
- For not-yet-done ideas, the default policy is now `rough_probe_before_reintegration`: do not re-open the main code path first; run a few cheap evidence points or an isolated branch before promoting the mechanism back into active configs.
