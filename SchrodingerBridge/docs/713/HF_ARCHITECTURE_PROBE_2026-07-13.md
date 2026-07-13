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
| `target_hf_subband_ablate_residual` | same checkpoint, inference hooks zero subband residual deltas | 0.485770 | 0.788810 | 0.024649 | 0.720464 | 0.300980 | 0.403276 | causal ablation: residual path is contributing |
| `target_hf_subband_nomem_ft6` | subband route, generic style-memory cross-attn disabled | 0.484903 | 0.794833 | **0.024491** | 0.716728 | **0.294348** | 0.401335 | rejected: cleaner target-HF route but weaker style/content frontier |
| `target_hf_subband_memres_ft6` | subband route, target-HF code residualized against style-memory mean | 0.486561 | 0.793519 | 0.024531 | 0.719228 | 0.297730 | 0.402490 | rejected: explicit prior subtraction is below subband-only |
| `target_hf_hybrid_ft6` | shared pooled HF + per-band residual delta | 0.485753 | 0.797810 | 0.024605 | 0.719710 | **0.295576** | 0.400962 | extra residuals do not help |
| `target_hf_subband_head_ft6` | shared pooled HF residual + nominal per-band head code | 0.487264 | 0.798699 | 0.024614 | 0.719169 | 0.296164 | 0.402149 | not a clean head-conditioning test; `style_velocity_head_enabled` was off |
| `target_hf_spatial_energy_ft6` | shared pooled HF + energy-bounded spatial residual | 0.486100 | 0.790866 | 0.024809 | 0.720364 | 0.297755 | 0.402737 | stronger probe path, but content cost and no all DINO-S win |
| `target_hf_texture_ft6` | per-band stationary HF texture stats -> residual delta | 0.486044 | 0.798035 | 0.024473 | 0.718189 | 0.296399 | 0.401347 | safe but weaker than subband pooled |
| `target_hf_subband_texture_ft6` | per-band pooled HF + stationary texture stats -> residual delta | 0.488420 | **0.798815** | 0.024596 | 0.719357 | **0.296046** | **0.404302** | near-best, better off-style and content, but no all DINO-S win |
| `target_hf_content_anchor_ft6` | coordinate-free target HF code + content-energy placement residual | 0.484393 | 0.795462 | 0.024555 | 0.717251 | 0.298162 | 0.399538 | content-safe placement does not beat subband-only; slightly below strong/subband |
| `target_hf_multitoken_ft6` | per-band stationary-stat tokens -> attention residual delta | 0.483562 | 0.794129 | 0.024531 | 0.718699 | 0.297979 | 0.398793 | rejected: more statistic tokens did not improve style or content |
| `target_hf_subband_deep_energy_ft6` | deeper per-band pooled HF residual + RMS bound | 0.482631 | 0.794932 | **0.024497** | 0.717588 | 0.297529 | 0.397683 | rejected: extra residual capacity with RMS bound weakens style/content frontier |
| `target_hf_subband_film_head_ft6` | pure per-band target-HF FiLM into main HF heads | 0.482591 | 0.791672 | 0.024594 | 0.717951 | 0.299591 | 0.398305 | rejected: live head-conditioning path, but weaker than residual subband route |

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

A matched inference-time ablation on `target_hf_subband_ft6/epoch_0006.pt` gives the cleanest causal check so far. The checkpoint, solver, endpoint AdaIN, and evaluation pairs were kept fixed; forward hooks zeroed only `target_latent_hf_subband_delta_lh/hl/hh`. This drops all-pairs DINO-S from `0.488624` to `0.485770`, DINO-C from `0.798123` to `0.788810`, and LPIPS from `0.296553` to `0.300980`. Therefore the learned subband residual is not a dead branch and is not buying style by content collapse. It behaves more like a small stabilizing style transport correction: the measured style gain is modest, but removing it also harms content consistency.

The stationary texture-stat route tested the intended safer alternative to raw spatial injection: target HF maps are reduced to per-subband mean/std/RMS/absolute-energy statistics, so target coordinates cannot pass through. The route is safe, and combining it with subband pooling increases the measured target-latent condition strength, especially on `HL/HH`. However, the main-table all DINO-S remains slightly below subband-only (`0.488420` vs `0.488624`). A later stationary-stat multi-token variant also failed (`0.483562` DINO-S, `0.398793` off-DINO-S), so merely adding more coordinate-free statistic tokens is not the next lever. The useful lesson is narrower: the model benefits most from a simple per-orientation target-HF code, while additional stationary statistics mostly improve off-diagonal style/content balance only when paired with subband pooling.

Important protocol note: earlier low values around `0.44` came from recomputing DINO-S with `exclude_source_from_style_refs=true`. The delivered main-table numbers use `exclude_source_from_style_refs=false`. Off-diagonal DINO-S remains the cleaner style-only sanity check and is reported above.

The no-style-memory run separates two effects that were previously tangled. Probes showed that disabling generic style-memory cross-attention makes the target-HF path the only live style route and aligns the HF-MSE and HF-stat gradients (`cos_fm_hf_vs_stat` becomes positive). However, the full eval is worse than subband-only on DINO-S, DINO-C, CLIP-S, and off-DINO-S. Therefore `style_id -> style_memory` is not merely a harmful shortcut; it carries a useful class-level style prior. The bottleneck is that this prior can dominate the image-specific target-HF route, not that it should be removed entirely.

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

Next structural direction: improve the subband route's learned delta/base without exposing raw target coordinates. Energy-bounded spatial gives the right probe magnitude but needs a stronger content anchor. Stationary texture stats are safe but not sufficient as the main injection path; the failed multi-token statistic route should not be kept in code. Prefer simpler orientation-specific HF residual capacity and energy normalization over more statistic-token width.

After the no-memory result, avoid binary memory removal. A better structural target is a two-route style decomposition: keep style memory as a coarse category prior, but make the image-specific target-HF subband code responsible for the residual/orientation-specific component. Concretely, future probes should measure whether the target-HF branch changes the HF heads after conditioning on memory, instead of testing target-HF in isolation.

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

Next structural direction (if still needed): do not simply deepen the subband residual, add more stationary tokens, or route the pooled code through a newly initialized HF-head FiLM. All three variants underperform. The current best explanation is that the shallow additive subband residual is useful because it preserves the pretrained HF heads and adds a small direction-specific correction; larger or more invasive conditioning perturbs that learned velocity field.

## Stationary-stat multi-token result (2026-07-14)

`target_hf_multitoken_ft6` tested a wider coordinate-free code: each target HF band was encoded as four stationary statistic tokens (mean, std, RMS, absolute energy), and decoder features queried those tokens with attention to produce per-band residual velocity. The implementation was removed after the run because it did not improve the frontier.

| metric | multitoken | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.483562 | **0.488624** | -0.0051 |
| off DINO-S | 0.398793 | **0.403917** | -0.0051 |
| DINO-C | 0.794129 | 0.798123 | -0.0040 |
| CLIP-S | 0.718699 | 0.720880 | -0.0022 |
| LPIPS | 0.297979 | 0.296553 | +0.0014 |

**Verdict: FAIL; code removed.** This negative result is useful because it separates "more coordinate-free statistics" from the actual missing route. The bottleneck is not just token count; the useful path still appears to be a compact orientation-specific target-HF code with the residual energy controlled against the existing HF heads.

## Deep energy-normalized subband residual result (2026-07-14)

`target_hf_subband_deep_energy_ft6` tested the next obvious capacity hypothesis: keep the target-HF code coordinate-free and per-orientation, but replace the shallow additive residual with a two-block residual head whose output RMS is normalized against the current HF velocity. Local smoke confirmed the intended initial residual/base ratio (`~tanh(0.18)=0.178`) and nonzero gradients, so the path was live.

| metric | deep-energy | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.482631 | **0.488624** | -0.0060 |
| off DINO-S | 0.397683 | **0.403917** | -0.0062 |
| DINO-C | 0.794932 | 0.798123 | -0.0032 |
| CLIP-S | 0.717588 | 0.720880 | -0.0033 |
| LPIPS | 0.297529 | 0.296553 | +0.0010 |

**Verdict: FAIL; code/config/pipeline removed.** This falsifies the simple "more residual capacity under an RMS guardrail" story. The likely issue is that additive residual capacity still competes with the already learned HF heads instead of improving their conditioning. Future attempts should modify the HF head conditioning path itself or the training target decomposition, not bolt on a larger residual branch.

## Pure subband FiLM head-conditioning result (2026-07-14)

`target_hf_subband_film_head_ft6` was run after re-checking `target_hf_subband_head_ft6`. The older run did not isolate head conditioning: `style_velocity_head_enabled` was off, so the per-band code was not consumed by plain `VelocityHead`; the measured gain mostly came from the shared HF residual. The new run enabled style-conditioned HF heads and injected per-band target-HF codes into LH/HL/HH FiLM, with no additive target-HF residual and no raw target coordinates. Local smoke verified that the subband path was active, shared residual was inactive, target-style latent changed HF outputs, and both FiLM/head-code gradients were nonzero.

| metric | pure FiLM head | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.482591 | **0.488624** | -0.0060 |
| off DINO-S | 0.398305 | **0.403917** | -0.0056 |
| DINO-C | 0.791672 | 0.798123 | -0.0065 |
| CLIP-S | 0.717951 | 0.720880 | -0.0029 |
| LPIPS | 0.299591 | 0.296553 | +0.0030 |

**Verdict: FAIL; config/pipeline removed.** This falsifies the simple "put the target-HF code directly into the main head" hypothesis. A newly introduced FiLM path appears to disturb the pretrained HF velocity heads more than it helps style routing. The surviving method story stays simpler: use the pretrained spectral bridge, keep LL protected, and add a compact per-orientation target-HF residual only on HF bands.

## No-style-memory subband result (2026-07-14)

`target_hf_subband_nomem_ft6` tested the gradient-probe hypothesis that generic style-memory cross-attention was acting as a shortcut. The model kept the pooled per-subband target-HF residual route, but disabled `style_cross_attention_enabled`. A pretraining probe on the subband checkpoint showed the intended route change: style-id output influence fell to zero, target-latent influence stayed live, style-memory gradients vanished, and HF-MSE/HF-stat gradients became aligned. This made it a clean single-variable architecture test.

| metric | no-memory subband | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.484903 | **0.488624** | -0.0037 |
| off DINO-S | 0.401335 | **0.403917** | -0.0026 |
| DINO-C | 0.794833 | 0.798123 | -0.0033 |
| CLIP-S | 0.716728 | 0.720880 | -0.0042 |
| LPIPS | **0.294348** | 0.296553 | -0.0022 |

**Verdict: FAIL; config removed, metrics kept.** The result is important because it rejects the overly simple "remove the shortcut" theory. The style memory route is a useful coarse style prior. The remaining architectural problem is to prevent that prior from saturating the prediction while letting the target-HF route supply image-specific, orientation-aware residual style. Do not promote this checkpoint.

## Memory-residualized subband result (2026-07-14)

`target_hf_subband_memres_ft6` tested the follow-up hypothesis from the no-memory result: keep style memory as a category prior, but subtract a learned projection of the style-memory mean from each per-subband target-HF code before the HF residual heads. Local smoke confirmed nonzero gradients through the memory projection, target-HF encoder, target-HF residual head, and style memory.

| metric | memory-residualized | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.486561 | **0.488624** | -0.0021 |
| off DINO-S | 0.402490 | **0.403917** | -0.0014 |
| DINO-C | 0.793519 | 0.798123 | -0.0046 |
| CLIP-S | 0.719228 | 0.720880 | -0.0017 |
| LPIPS | 0.297730 | 0.296553 | +0.0012 |

**Verdict: FAIL; code/config removed, metrics kept.** The result partially recovers from the no-memory run but remains below the simple subband residual on all key style/content metrics. Explicitly subtracting a learned memory prior is therefore too blunt: it reduces useful class-prior signal or destabilizes content without enough image-specific gain. The next viable direction should not algebraically remove memory; it should regularize route usage or improve target-HF residual learning without perturbing the memory prior itself.

## Inference-time target-HF residual ablation (2026-07-14)

This probe tests the trained route without adding new training variables. It wraps the normal evaluator and installs three forward hooks that return zeros from `target_latent_hf_subband_delta_lh`, `target_latent_hf_subband_delta_hl`, and `target_latent_hf_subband_delta_hh`. Everything else is unchanged: same `target_hf_subband_ft6/epoch_0006.pt`, same AdaIN 1.5 eval setting, same 750 pairs, same DINOv2-small protocol.

| metric | subband-only | residual ablated | delta |
|---|---:|---:|---:|
| all DINO-S | **0.488624** | 0.485770 | -0.0029 |
| off DINO-S | **0.403917** | 0.403276 | -0.0006 |
| DINO-C | **0.798123** | 0.788810 | -0.0093 |
| CLIP-S | **0.720880** | 0.720464 | -0.0004 |
| LPIPS | **0.296553** | 0.300980 | +0.0044 |

**Verdict: PASS as a diagnostic.** The residual path has measurable causal value. The large DINO-C/LPIPS drop is especially useful: the branch is not merely injecting style texture, it also helps keep the HF transport compatible with the content latent. This supports keeping the simple subband residual in the method and suggests that future architecture work should increase its effective route strength carefully, not replace it with raw spatial maps or newly initialized invasive head conditioning.
