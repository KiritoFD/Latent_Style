# Method Exploration and Current Checkpoint Handoff

Date: 2026-07-13

This document extracts the still-useful method exploration record from older non-713 docs and reconciles it with the current HF-route probe evidence. The superseded source docs were archived to avoid confusing future method writing.

## 1. Current Adopted Checkpoint

Current paper checkpoint family:

| Field | Value |
|---|---|
| Config | `configs/exp_brk_a_ll03_10ep.json` |
| Logical checkpoint | `<EXP_ROOT>/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt` |
| Remote observed root | `I:\Github\Latent_Style\SchrodingerBridge\exp\dino_s_break\brk_a_ll03_10ep` |
| Training | 10 epochs, batch 96, seed 42 |
| Hardware | RTX 3060 12GB |
| Train time | 176.9 s |
| Inference | 8-step Euler, 750 pairs, generation-only 94.63 s |
| Main source of paper values | `aaai2027_v4/paper.tex`, Table 1 |

Current main-table row:

| Board | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|
| D5-512 | **0.4859** | 0.7075 | **0.2583** | **0.8287** |
| P2A-256 | 0.4801 | 0.6681 | 0.3116 | 0.8612 |
| R5-WikiArt | 0.5226 | 0.7747 | 0.2895 | 0.7717 |

Interpretation: this is the paper's balanced operating point, not a claim that every axis is best. DINO-S is first-tier, while DINO-C/LPIPS and cost are the main differentiators.

## 2. Method Components: Current Reading

The effective method should be described as:

```text
WEAVE = Haar wavelet coordinates
      + band-weighted rectified-flow transport
      + endpoint high-frequency style statistics
      + auxiliary style-memory conditioning
```

Current component status:

| Component | Current status | Reason |
|---|---|---|
| Haar DWT / IDWT | Core | Separates LL structure from LH/HL/HH texture bands. |
| Rectified flow matching | Core | Learns content-aware latent transport with low inference step count. |
| Band-weighted target | Core | LL is weakly stylized, HF uses target-style bands. |
| Endpoint AdaIN / HF statistics | Core style injector | Stable inference-side style-statistics lever. |
| Cross-attention / style memory | Auxiliary | Probe gates are near-closed and gradient mass is weak; do not frame as main style path. |
| SWD / contrastive SWD / edge / low-pass aux losses | Retired or non-central | Older ablations found weak or harmful contribution under the current recipe. |
| ASG | Historical / not current main story | Older 710 result was useful exploration, but current 713 diagnosis supersedes the claim that ASG is the paper mechanism. |
| Decoder AdaLN/AdaIN | Failed direction | Repeated decoder injection attempts did not improve the frontier. |

## 3. Target Objective

The current target is not source reconstruction. It already asks for style in the high-frequency bands:

```text
LL       = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)
LH/HL/HH = target_style bands
```

The old shorthand "original LL + target style H" is too crude. The correct reading is:

- LL is partially style-stat aligned but content-protected.
- LH/HL/HH are the explicit style target.
- If HH appears in the target/loss, the model needs an HH output route for that supervision to be connected.

## 4. Exploration Timeline

### Stage A: 710 minimal component audit

Useful findings retained:

| Finding | Current use |
|---|---|
| Flow matching is necessary. | Still part of the method core. |
| Haar wavelet coordinates help content/style separation. | Still the main coordinate system. |
| LL weighting matters. | Current recipe uses a weak LL style blend / low LL weight to protect content. |
| Canonical DINO protocol matters. | DINO-S is primary style; DINO-C is content; old patch-SSM DINO-C interpretation is retired. |
| VAE decode is a major inference-time fixed cost. | Timing discussion separates network generation and VAE decode. |

Findings downgraded:

| Old 710 claim | Current correction |
|---|---|
| ASG enters the main table. | Not current paper story; 713 HF-route diagnosis supersedes it. |
| DWT routing / cross-attn is the main style injector. | Cross-attention is auxiliary; target-HF route is the missing style path. |
| Code cleanup complete. | False for current worktree; repo is dirty and active. |

### Stage B: brk_a / paper checkpoint

The current paper checkpoint stabilized the low-cost operating point:

- 10 epochs on RTX 3060.
- 903K trainable parameters plus frozen VAE.
- Strong content preservation relative to high-style diffusion/editing baselines.
- DINO-S near Seedream 4.5 on D5 while LPIPS/DINO-C are much better.

Endpoint AdaIN scale variants are diagnostic operating points, but the main table must be read from `paper.tex`. Do not let older `WEAVE-m/s/q` names override the current table.

### Stage C: failed decoder/style modulation attempts

Older `docs/method.md` correctly recorded a useful negative result: simply injecting style through decoder AdaLN/AdaIN/Q/gate modulation does not improve the frontier.

Current interpretation:

- The decoder predicts velocity; generic style modulation perturbs intermediate features without giving the model the missing target-HF information path.
- Flow loss dominates the optimization; weak style-side losses cannot force arbitrary decoder perturbations to become useful style transfer.
- This failure does not prove a hard DINO-S ceiling. It proves that this family of injection points is the wrong route.

### Stage D: HF-route probes

The 713 probe series opened the missing image-specific target-style path.

| Route | Result | Interpretation |
|---|---|---|
| Global target-token fusion | Connected but over-controls LL | Too global; not the right path. |
| HF delta route | Modest style improvement | First proof that target-HF conditioning helps. |
| Raw spatial target-HF maps | Highest style, content collapse | Proves capacity but leaks target geometry. |
| Pooled per-subband HF code | Best usable route | Coordinate-free style signal, content preserved. |
| Stationary texture stats | Safe but weak alone | Useful diagnostic, not enough by itself. |
| Content-anchor placement | Safe but weaker | Placement engineering is not the next main lever. |
| Stationary-stat multi-token code | Weaker than subband-only | More coordinate-free statistics did not improve the route. |

Current best probe:

| Run | DINO-S | DINO-C | CLIP-S | LPIPS | Off DINO-S |
|---|---:|---:|---:|---:|---:|
| `target_hf_subband_ft6` | **0.488624** | 0.798123 | **0.720880** | 0.296553 | 0.403917 |

Conservative alternate:

| Run | DINO-S | DINO-C | CLIP-S | LPIPS | Off DINO-S |
|---|---:|---:|---:|---:|---:|
| `target_hf_subband_texture_ft6` | 0.488420 | **0.798815** | 0.719357 | **0.296046** | **0.404302** |

Rejected:

| Run | DINO-S | DINO-C | CLIP-S | LPIPS | Reason |
|---|---:|---:|---:|---:|---|
| `target_hf_spatial_ft6` | **0.490074** | 0.404308 | **0.748291** | 0.538240 | Content collapse from target layout leak. |
| `target_hf_multitoken_ft6` | 0.483562 | 0.794129 | 0.718699 | 0.297979 | Worse than subband-only; code removed. |
| `target_hf_subband_deep_energy_ft6` | 0.482631 | 0.794932 | 0.717588 | 0.297529 | Deep additive residual with RMS bound underperformed; code removed. |
| `target_hf_subband_film_head_ft6` | 0.482591 | 0.791672 | 0.717951 | 0.299591 | Pure HF-head FiLM conditioning underperformed; config removed. |

## 5. Correct Theory After Probing

Old theory:

```text
DINO-S 0.48 is a fundamental SAT limit.
Only LL unlock can break it.
```

Current theory:

```text
Old SAT routes saturate because target-image HF style does not have
a clean non-spatial route into HF velocity heads.
```

This is more precise. The model has capacity: raw spatial HF proves that. The problem is to expose target style without exposing target coordinates. Therefore the next lever is route topology and coordinate-free target-HF capacity, not first-order hyperparameter tuning.

## 6. Current Recommended Next Experiments

Architecture-first plan:

| Step | Experiment | Success condition |
|---|---|---|
| 1 | Orientation-specific residual depth for LH/HL/HH | Better stroke direction handling, no raw target map leak. |
| 2 | Energy-normalized HF residual | Residual energy stays bounded against existing HF heads. |
| 3 | Stronger but compact subband residual head | DINO-S improves over `target_hf_subband_ft6` without DINO-C/LPIPS collapse. |
| 4 | Full D5/P2A/R5 rerun | Required before promoting any probe to main table. |
| 5 | Matched CFG ablation only if needed | Separate CFG from DWT route, HH head, delta heads, and gate changes. |

Do not prioritize:

| Direction | Reason |
|---|---|
| More epochs alone | HF-delta 15ep did not beat subband 6ep. |
| Raw spatial target-HF maps | Already failed content metrics. |
| LL target-image shortcuts | Likely buys style by spending content. |
| More placement engineering after content-anchor | Latest placement attempt is safe but weaker. |
| Stationary-stat multi-token code | Tested and removed; it underperformed subband-only on all tracked metrics. |
| Deep additive subband residual | Tested and removed; more residual capacity did not improve the style route. |
| Pure target-HF FiLM head conditioning | Tested and removed; newly initialized HF-head FiLM weakens the learned velocity field. |
| Generic decoder AdaLN/AdaIN | Already repeatedly negative. |

## 7. Method-writing Guidance

Use this narrative:

1. WEAVE changes coordinates with Haar DWT so content-heavy LL and texture-heavy HF are separated.
2. The flow model learns content-aware latent transport under band-weighted supervision.
3. Endpoint HF statistics provide a lightweight style-statistics injector.
4. The current probe diagnosis shows that the missing route is image-specific target-HF information into HF velocity heads.
5. Cross-attention/style memory is auxiliary, not the main style mechanism.

Avoid:

- "WEAVE is best on every axis."
- "DINO-S 0.48 is a fundamental limit."
- "CFG improves content" without a matched ablation.
- "ASG is the current paper mechanism."
- "The repo is clean."

## 8. Archived Source Docs

The following non-713 docs were mined for still-valid information and then archived:

| Archived source | Valid information extracted | Superseded content |
|---|---|---|
| `docs/archive/713_external_legacy/method.md` | Effective components, decoder AdaLN/AdaIN negative results, figure-writing hints. | Hard DINO-S ceiling, old cleanup claims, old next direction. |
| `docs/archive/713_external_legacy/710_CONCLUSIONS.md` | Early component audit, DINO protocol, infra/timing observations. | ASG as main-table mechanism, old clean-state claim, old SOTA framing. |
