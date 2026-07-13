# Theory Map

## Current Information Flow

1. Content latent `x` enters the model as a 4-channel SD VAE latent.
2. A single-level Haar transform splits it into `LL`, `LH`, `HL`, `HH`.
3. The four subbands are concatenated along channels and projected into the compact backbone.
4. The backbone is mostly content/time driven. Learned style enters through `style_memory` tokens and cross-attention.
5. The velocity heads predict motion for LL/LH/HL and optionally HH.
6. The solver integrates the predicted subband velocities.
7. Endpoint statistical alignment injects target-style latent statistics into high-frequency bands. In the paper T11 config this is final-step per-subband WCT/AdaIN-like alignment with LL excluded.

## Working Causal Picture

The architecture succeeds because it turns content preservation and style transfer into different coordinates:

- LL carries high-energy structure and is expensive to perturb.
- LH/HL carry much of the useful mid-frequency stylistic movement.
- HH is often frozen by the learned flow and relies on endpoint alignment if style needs diagonal fine detail.
- Endpoint alignment bypasses the weak learned style path and directly imposes target high-frequency statistics.

This makes the current design simple but brittle: most style comes from a hand-coded endpoint operator, while the learned style memory/cross-attention path may be too weak to materially change the trajectory.

## Probe Hypotheses

H1. Learned cross-attention is weak relative to endpoint alignment.

Expected signal: `sum(|style_delta|)` and style-swap sensitivity inside blocks are much smaller than endpoint subband deltas.

H2. DINO-S gain is dominated by LH/HL endpoint movement, not by LL movement.

Expected signal: DINO-S-positive outputs show higher LH/HL transfer ratios while LPIPS/DINO-C degrade when LL is moved.

H3. WCT only beats AdaIN when channel covariance is meaningful at the chosen feature shape.

Expected signal: per-subband WCT and per-subband AdaIN are close for 4-channel subbands; spatial-fiber WCT may help more but risks content drift.

H4. The best model change is a small learned high-frequency style path, not replacing endpoint alignment.

Candidate changes:

- High-frequency style-conditioned velocity heads, with LL unchanged.
- Q-side style AdaLN only on DWT-routed high-frequency attention.
- A small endpoint residual/gate learned on LH/HL/HH after the ODE, initialized as no-op.

## 713 Probe Update

The first T11 probe supports H1 and H4:

- The endpoint operator is larger than learned flow in LH/HL and is the only HH mover.
- The learned style-swap velocity response is dominated by LL in absolute magnitude.
- Stronger endpoint scales are not automatically better; the latent ratio fell when LH/HL/HH scales were increased.

Therefore the next model candidate should not strengthen all style conditioning globally. It should add learned style capacity only to high-frequency velocity heads while leaving LL structurally conservative.

## 713 GPU Path-Separation Update

Run: `docs/713/probe_outputs/t11_ep5_style_path_n32_gpu_pathsep.md`.

The extra probe separates `style_id` from `style_latent`:

- With endpoint disabled, the learned path gives almost no high-frequency statistical transfer: LH `0.0000`, HL about `0.0104`, HH `0.0000`.
- With target endpoint enabled, changing only `style_id` moves the output substantially in L2, but HH ratio stays fixed and LH/HL ratios remain endpoint-dominated.
- Turning cross-attention off with endpoint disabled is not much worse than the learned target no-endpoint case for latent style ratios. This means the current learned cross-attn path affects the trajectory, but its effect is not aligned with the target high-frequency style statistics measured by the probe.
- Disabling DWT route and using full cross-attn with no endpoint increases content L2, but still does not produce useful high-frequency transfer. Full spatial style attention is therefore not the first fix.
- The time sweep shows the same leakage pattern: style-swap sensitivity is largest in LL at `t=0.1` and `t=0.5`, while LH/HL are an order of magnitude smaller.

Interpretation:

1. `style_memory -> cross-attn` is not dead, but it is not the main DINO-S-relevant style carrier.
2. The endpoint WCT/AdaIN path is the actual high-frequency appearance actuator in the current T11 model.
3. Because learned style sensitivity naturally concentrates in LL, global AdaLN/global style scaling is likely to buy style by spending structure/content.
4. A better learned injection should be both high-frequency-local and output-head-proximal: put style modulation immediately before LH/HL/HH velocity prediction, not in the shared backbone.

## Decision Rule

Use DINO-S as the primary style metric. A candidate is worth keeping only if it improves DINO-S at matched protocol without unacceptable LPIPS/DINO-C regression. CLIP-S can support interpretation but must not override DINO-S.

## Current Candidate Order

1. Train a high-frequency-only style-conditioned velocity head:
   - `style_velocity_head_enabled=true`
   - `style_vhead_hf_nonzero_init=true`
   - LL head remains zero-init/conservative.
   - Keep global style AdaLN off.
   - Keep endpoint per-subband WCT unchanged for the first A/B.
2. Evaluate an HH endpoint-off control:
   - `endpoint_adain_scale_hh=0.0`
   - no retraining; use it to estimate whether HH endpoint helps DINO-S enough to justify its content cost.
3. Only if candidate 1 fails, test a small independent HF style delta head. Avoid enabling global style AdaLN before there is evidence that DINO-S gains outweigh DINO-C/LPIPS loss.
