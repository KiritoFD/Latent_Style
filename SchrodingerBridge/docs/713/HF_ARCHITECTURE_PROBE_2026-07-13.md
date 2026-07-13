# HF Architecture Probe Notes

Date: 2026-07-13

## Current Theory

The baseline target is already style-heavy in high-frequency bands:

- `LL = 0.7 * content_LL + 0.3 * AdaIN(content_LL -> style_LL)`
- `LH/HL/HH = target_style` bands
- `HH` must have an output head, otherwise the HH target is supervised without a model path.

The main bottleneck is not the loss target. The loss asks for high-frequency target style. The bottleneck is the condition path: baseline mostly reads `style_id -> style_memory`, while image-specific `target_style` was only used to build the target. The useful fix is to connect target HF latent to HF velocity prediction while keeping LL disconnected.

## Probe Findings

Spatial target-HF injection proved the path has capacity, but it is unsafe:

- `target_hf_spatial_ft6` trained checkpoint: target-latent delta/base was about `1.03/1.04/1.04` on `LH/HL/HH`.
- DINO-S rose to `0.490074`, but DINO-C collapsed to `0.404308` and LPIPS to `0.538240`.
- Interpretation: direct target HF spatial maps leak target geometry/layout through high-frequency channels.

The first usable route was pooled target-HF:

- `target_hf_delta_strong_ft6`, eval AdaIN 1.5: DINO-S `0.487036`, DINO-C `0.799077`, LPIPS `0.295459`.
- This route sends target HF as an image-specific style vector into HF velocity prediction, without exposing target spatial coordinates. The later subband-pooled route slightly improves the main-table DINO-S while keeping similar content quality.

## Architecture Attempts After Spatial Probe

All runs used the same 6ep fine-tune recipe from `brk_a_ll03_10ep`, no hyperparameter search.

| run | target-latent route | all DINO-S | DINO-C | DINO-structure | CLIP-S | LPIPS | off DINO-S | conclusion |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `target_hf_delta_strong_ft6` | shared pooled HF -> HF heads + residual delta | 0.487036 | 0.799077 | 0.024591 | 0.717586 | 0.295459 | 0.401948 | previous best usable route |
| `target_hf_spatial_ft6` | per-band spatial HF maps -> residual delta | 0.490074 | 0.404308 | 0.046968 | 0.748291 | 0.538240 | n/a | high style, content collapse |
| `target_hf_subband_ft6` | per-band pooled HF -> residual delta only | **0.488624** | 0.798123 | **0.024536** | **0.720880** | 0.296553 | **0.403917** | best current usable architecture |
| `target_hf_hybrid_ft6` | shared pooled HF + per-band residual delta | 0.485753 | 0.797810 | 0.024605 | 0.719710 | **0.295576** | 0.400962 | extra residuals do not help |
| `target_hf_subband_head_ft6` | shared pooled HF + per-band code into main HF heads | 0.487264 | 0.798699 | 0.024614 | 0.719169 | 0.296164 | 0.402149 | slight style gain over strong, not over subband-only |
| `target_hf_spatial_energy_ft6` | shared pooled HF + energy-bounded spatial residual | 0.486100 | 0.790866 | 0.024809 | 0.720364 | 0.297755 | 0.402737 | stronger probe path, but content cost and no all DINO-S win |
| `target_hf_texture_ft6` | per-band stationary HF texture stats -> residual delta | 0.486044 | 0.798035 | 0.024473 | 0.718189 | 0.296399 | 0.401347 | safe but weaker than subband pooled |
| `target_hf_subband_texture_ft6` | per-band pooled HF + stationary texture stats -> residual delta | 0.488420 | **0.798815** | 0.024596 | 0.719357 | **0.296046** | **0.404302** | near-best, better off-style and content, but no all DINO-S win |
| `target_hf_content_anchor_ft6` | coordinate-free target HF code + content-energy placement residual | 0.484393 | 0.795462 | 0.024555 | 0.717251 | 0.298162 | 0.399538 | content-safe placement does not beat subband-only; slightly below strong/subband |

## Diagnosis

The successful path is:

`target_style -> DWT HF -> shared pooled target-HF vector -> HF velocity heads/residual delta -> LH/HL/HH`

The unsafe path is:

`target_style -> DWT HF spatial maps -> output HF residual`

The subband route is useful under the main table protocol, but the probe shows why the gain is modest rather than a large jump:

- subband-only trained delta/base: `LH 0.030`, `HL 0.097`, `HH 0.125`
- hybrid trained delta/base: `LH 0.079`, `HL 0.189`, `HH 0.092`
- subband-head trained delta/base: `LH 0.031`, `HL 0.044`, `HH 0.022`
- energy-bounded spatial trained delta/base: `LH 0.261`, `HL 0.292`, `HH 0.277`
- stationary texture-only trained delta/base: `LH 0.034`, `HL 0.133`, `HH 0.059`
- subband + texture trained delta/base: `LH 0.039`, `HL 0.211`, `HH 0.136`

By contrast, raw spatial reaches delta/base around `1.0` but destroys content. Energy-bounded spatial moves in the right probe direction, but its extra HF strength currently spends content budget without increasing all DINO-S over subband-only.

The stationary texture-stat route tested the intended safer alternative to raw spatial injection: target HF maps are reduced to per-subband mean/std/RMS/absolute-energy statistics, so target coordinates cannot pass through. The route is safe, and combining it with subband pooling increases the measured target-latent condition strength, especially on `HL/HH`. However, the main-table all DINO-S remains slightly below subband-only (`0.488420` vs `0.488624`). The useful lesson is that widening the pooled bottleneck is not enough by itself; the model benefits most from a simple per-orientation target-HF code, while additional stationary statistics mostly improve off-diagonal style and content preservation.

Important protocol note: earlier low values around `0.44` came from recomputing DINO-S with `exclude_source_from_style_refs=true`. The delivered main-table numbers use `exclude_source_from_style_refs=false`. Off-diagonal DINO-S remains the cleaner style-only sanity check and is reported above.

## Recommended Next Architecture

Do not tune gates/epochs first. The next structural change should be a constrained spatial-HF route:

- target HF spatial maps must be converted into stationary texture/energy fields, not raw target coordinates;
- output residual energy should be normalized against the existing HF head energy, so the route cannot dominate content structure;
- the content feature map should decide where texture is placed.

Current checkpoint recommendation:

- Use `target_hf_subband_ft6/epoch_0006.pt` as the new best architecture point for the main table protocol.
- Keep `target_hf_subband_texture_ft6/epoch_0006.pt` as the conservative alternate if off-diagonal DINO-S and DINO-C are prioritized over the headline all DINO-S.
- Keep `target_hf_delta_strong_ft6/epoch_0006.pt` as the older fallback because its behavior is simpler and already better than the baseline.
- Do not use raw spatial HF despite its high all DINO-S; it is content collapse.

Next structural direction: improve the subband route's learned delta/base without exposing raw target coordinates. Energy-bounded spatial gives the right probe magnitude but needs a stronger content anchor. Stationary texture stats are safe but not sufficient as the main injection path; they can be kept as a diagnostic branch, not the current primary architecture.

## Content-anchor experiment result (2026-07-13 late)

`target_hf_content_anchor_ft6` finished on remote RTX 3060 (I:):

- **Idea**: keep coordinate-free target HF codes (per-subband pooled), place residual with a **content HF energy mask** + energy bound vs current HF head velocity.
- **Train**: 6ep from `brk_a_ll03_10ep`, batch 96, final loss ≈ 1.95, ckpt `epoch_0006.pt`.
- **Eval protocol**: AdaIN 1.5, 750 pairs, canonical DINOv2-small, `exclude_source_from_style_refs=false`.

| metric | content_anchor | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.484393 | **0.488624** | −0.0042 |
| off DINO-S | 0.399538 | **0.403917** | −0.0044 |
| DINO-C | 0.795462 | 0.798123 | −0.0027 |
| CLIP-S | 0.717251 | 0.720880 | −0.0036 |
| LPIPS | 0.298162 | 0.296553 | +0.0016 |

**Verdict: FAIL as primary architecture.** Content-anchored placement is safe (no content collapse) but does not improve the main-table all DINO-S over subband-only. It is also slightly worse than `target_hf_delta_strong_ft6` (0.487036).

Current recommendation unchanged:

1. **Primary**: `target_hf_subband_ft6` (all DINO-S 0.488624)
2. **Conservative alternate**: `target_hf_subband_texture_ft6` (best off DINO-S / DINO-C balance)
3. **Fallback**: `target_hf_delta_strong_ft6`
4. Do **not** promote content-anchor or raw spatial as main path

Next structural direction (if still needed): increase subband route capacity without spatial target leak — e.g. multi-token subband codes / orientation-specific residual depth — not more placement engineering.
