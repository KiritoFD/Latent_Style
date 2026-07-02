# `XPred + K_manifold` Candidate Packet

Date: 2026-06-07

Intent:

- keep the strongest known style-ceiling family:
  - endpoint prediction
  - barycentric target smoothing
- add the strongest currently hypothesized content-preserving kinetic family:
  - manifold-adaptive split

Why this candidate exists:

- `XPred_Barycenter b40` already reaches the `0.71+` transfer style band
- its failure mode is catastrophic LPIPS
- `K_manifold` is the most natural attempt to repair that failure without abandoning the endpoint-target geometry

Success condition:

- style remains near the `XPred_Barycenter` band
- LPIPS improves materially relative to the plain `XPred_Barycenter` line

## Evaluation correction

Important trust boundary:

- a quick rerun against the drifting mainline source was **not** paper-safe for this packet
- the checkpoint loaded non-strictly against a changed tokenizer/model schema
- trusted numbers below come only from the snapshot-matched fast rerun:
  - run-local `src`
  - current fast `run_evaluation.py` overlaid onto that run-local source
  - output root:
    - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/aaai2027_inmortal_xpred_kmanifold_seed42_b32/full_eval_fast_snapshot`

This correction materially changes the interpretation of the packet.

## Full snapshot-matched readout

| epoch | transfer CLIP-style | transfer LPIPS |
| --- | ---: | ---: |
| `e1` | `0.7008` | `0.7922` |
| `e2` | `0.7188` | `0.7496` |
| `e3` | `0.7095` | `0.7151` |
| `e4` | `0.7201` | `0.7188` |
| `e5` | `0.7211` | `0.6999` |
| `e6` | `0.7248` | `0.6881` |
| `e7` | `0.7259` | `0.6863` |
| `e8` | `0.7235` | `0.6770` |

Best retained point under the current promotion rule:

- `e7`
  - transfer `clip_style = 0.7259`
  - transfer `content_lpips = 0.6863`
  - full `clip_style = 0.7284`
  - full `content_lpips = 0.6790`

## Mechanism reading

Relative to `XPred_Barycenter` best (`e7 = 0.7161 / 0.7176` transfer):

- style improves by about `+0.0098`
- LPIPS improves by about `-0.0312`

What this means:

- `K_manifold` is a **real repair** for the endpoint-prediction family
- the improvement is not cosmetic; it moves both the style ceiling and the damage profile in the right direction
- this is now the strongest `inmortal` packet on raw transfer style

What it does **not** mean:

- it does not solve the ceiling problem by itself
- LPIPS remains far from the `0.30` target band
- the line is still too destructive to replace the paper-facing compact frontier

## Implication for the next round

This packet justifies staying inside the `x-prediction` family.

Most likely next repair:

- keep:
  - endpoint prediction
  - barycentric target smoothing
  - manifold-adaptive kinetic
- add:
  - constrained proximal high-pass refinement

Reason:

- transport-side repair is helping
- but the remaining gap is still a content/stylization decoupling problem, not a pure style-ceiling problem

## Eval timing note

Snapshot-matched fast eval is now trustworthy and materially leaner, but the main bottleneck remains VAE decode.

Observed per-epoch band on this packet:

- `wall_total`: about `84.7s` to `86.6s`
- `eval_total`: about `20.3s` to `21.2s`
- `vae_decode`: about `52.3s`

Interpretation:

- CLIP + LPIPS are no longer the suspicious part
- bigger batch helped only slightly
- the next eval-speed win has to target the decoder backend itself
