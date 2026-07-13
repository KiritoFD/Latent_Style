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
| `target_hf_subband_scale_1p25` | same checkpoint, residual deltas scaled to 1.25x at inference | 0.487311 | 0.788688 | 0.024654 | 0.720082 | 0.300671 | 0.404491 | no balanced gain; spends content budget |
| `target_hf_subband_scale_1p5` | same checkpoint, residual deltas scaled to 1.5x at inference | 0.487485 | 0.779830 | 0.024840 | 0.721106 | 0.305438 | 0.406744 | style-biased, content cost too high |
| `target_hf_subband_scale_hh1p5` | same checkpoint, only HH residual scaled to 1.5x | 0.487815 | 0.783560 | 0.024765 | 0.720560 | 0.303415 | 0.406092 | HH-only boost still follows style/content tradeoff |
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
| `target_hf_subband_basis_ft6` | target-HF code selects low-rank content-derived residual basis | 0.482840 | 0.793659 | 0.024564 | 0.718310 | 0.297061 | 0.398561 | rejected: safer parameterization, but too weak and below subband-only |
| `target_hf_subband_pairstats_ft6` | target-HF code plus current-vs-target HF discrepancy statistics | 0.483765 | 0.794304 | 0.024541 | 0.718318 | 0.297092 | 0.399385 | rejected: dynamic coordinate-free discrepancy signal is still weaker than target-only subband code |
| `target_hf_subband_diraux_ft6` | residual branch trained with direct direction auxiliary | 0.486150 | 0.793859 | **0.024536** | 0.718929 | 0.297425 | 0.402097 | rejected: probe direction improves, image frontier worsens |
| `target_hf_subband_timewindow_{early,late}_norm` | inference-only residual time-window causal probe | 0.48660-0.48664 | 0.79361-0.79365 | **0.024533-0.024534** | 0.71933-0.71938 | 0.297480 | 0.40254-0.40256 | rejected: temporal localization underperforms full-trajectory residual |
| `target_hf_subband_mixer_ft6` | zero-init cross-orientation mixing among pooled LH/HL/HH target-HF codes | 0.486666 | 0.793705 | 0.024535 | 0.719392 | 0.297500 | 0.402582 | rejected: learned small off-diagonal mixing but did not improve residual direction or metrics; code/config removed |
| `target_hf_subband_current_delta_ft6` | zero-init target-current pooled HF code difference | 0.486683 | 0.793621 | 0.024539 | 0.719366 | 0.297567 | 0.402626 | rejected: slightly stronger target-specific information flow but unchanged residual direction and worse image frontier; code/config removed |
| `target_hf_subband_affine_delta_ft6` | subband residual changed from scale-only to affine scale+shift | 0.482449 | 0.790343 | 0.024682 | 0.717787 | 0.298913 | 0.398861 | rejected: condition route became stronger but remained mostly off-direction; code/config removed |
| `target_hf_subband_wct_direction_ft6` | gated analytic WCT-stat direction residual | 0.486511 | 0.793320 | 0.024538 | 0.719448 | 0.297849 | 0.402438 | rejected: condition-direction probe improved, but final DINO-S/CLIP-S/content frontier worsened; code/config removed |

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

The same hook was then used as a residual-strength curve. Scaling the trained residual above its learned magnitude does not improve the balanced point: `1.25x` gives DINO-S `0.487311`, DINO-C `0.788688`, LPIPS `0.300671`; `1.5x` gives DINO-S `0.487485`, DINO-C `0.779830`, LPIPS `0.305438`. Off-diagonal DINO-S rises slightly (`0.404491` and `0.406744`), but this is paid for by content degradation. A band-specific follow-up, motivated by the direction probe below, scaled only the better-aligned HH residual to `1.5x`; it still underperformed the original balance (DINO-S `0.487815`, DINO-C `0.783560`, LPIPS `0.303415`). Thus the route is useful but not simply amplitude-limited. Future changes should improve the conditional direction of the residual, not multiply the same residual vector.

The 2026-07-14 gradient/information-flow probe separates the target image's two roles:

```text
target_style as supervision target      -> constructs LL/HF training target
target_style as condition/style_latent  -> enters the model through target-HF subband route
```

On `target_hf_subband_ft6/epoch_0006.pt`, under the actual training objective (`hf_stat_loss_enabled=false`), the target path is strong but the condition path is weak:

| band | condition-path grad/tensor | target-path grad/tensor | condition / target |
|---|---:|---:|---:|
| LL | ~0 | 5.46e-7 | ~0 |
| LH | 4.48e-6 | 1.77e-4 | 2.53% |
| HL | 2.29e-6 | 1.74e-4 | 1.32% |
| HH | 1.71e-6 | 3.46e-4 | 0.50% |

The forward intervention agrees. Replacing condition `style_latent` with content bands and then restoring one target band at a time gives a clean diagonal route:

| target condition band | output LL | output LH | output HL | output HH |
|---|---:|---:|---:|---:|
| LL | 0.000000 | 0.000002 | 0.000003 | 0.000006 |
| LH | 0.000000 | 0.075624 | 0.000003 | 0.000004 |
| HL | 0.000000 | 0.000002 | 0.097229 | 0.000007 |
| HH | 0.000000 | 0.000002 | 0.000002 | 0.119432 |

Values are output delta/base. The route is doing the intended thing: no LL leakage, almost no cross-band leakage, and per-band HF influence. The weakness is that this target-specific condition influence is small compared with the learned residual branch's generic contribution. Zeroing the target-HF residual modules changes the HF velocity by `0.474/0.396/1.081` delta/base on LH/HL/HH, but changing the condition from content bands to target bands changes it only `0.076/0.097/0.119`. The branch is live and large, but the image-specific part of it is small.

CFG is easier to interpret after this probe. In this code path, `cfg_unconditional=True` disables target-HF branches and uses the unconditional style-token route. The observed CFG/content behavior is therefore plausibly a mixture with a less target-HF-injected velocity field, not proof that CFG alone is a content-preserving style mechanism.

The stat-loss probe should not be read as the actual training objective. It was enabled only diagnostically. It shows why a naive HF-stat auxiliary is risky: stat gradients are much larger than FM-HF gradients, and the full-model gradient cosine can become negative. Group-wise decomposition is more precise: the target-HF branch itself is weakly positive (`cos≈0.29` for FM-HF vs stat), while `time_proj` is strongly conflicting (`cos≈-0.82`; HH-specific `cos≈-0.94`). This means adding a style-stat loss would mostly perturb the global transport/time conditioning rather than cleanly strengthen the target-HF route.

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

## Low-rank content-basis residual result (2026-07-14)

`target_hf_subband_basis_ft6` tested a safer residual parameterization motivated by the direction and spatial-leak probes:

```text
old: delta_hf = conv(FiLM(h, z_target_hf))
new: delta_hf = sum_r a_r(z_target_hf) * B_r(h)
```

Here the target-HF code only selects low-rank coefficients, while the spatial basis maps come from the content/backbone feature `h`. This should prevent target-coordinate leakage and force the target image to control "what style direction" rather than "where to draw it."

| metric | low-rank basis | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.482840 | **0.488624** | -0.0058 |
| off DINO-S | 0.398561 | **0.403917** | -0.0054 |
| DINO-C | 0.793659 | 0.798123 | -0.0045 |
| CLIP-S | 0.718310 | 0.720880 | -0.0026 |
| LPIPS | 0.297061 | 0.296553 | +0.0005 |

**Verdict: FAIL; code/config removed, metrics kept.** The result is useful because it rejects a tempting "target selects a content basis" story. The parameterization is safe, but it underuses target-HF style: compressing the branch into rank-4 coefficients over content-derived bases loses too much image-specific orientation/texture information. The next structural attempt should not merely constrain the residual more; it needs a better-conditioned target-HF route that preserves the simple subband residual's live corrective role.

## Pairwise current-target HF statistics result (2026-07-14)

`target_hf_subband_pairstats_ft6` tested a dynamic, coordinate-free correction signal:

```text
z'_band = z_target_band + gate * E_stats(current_HF_band, target_HF_band, target-current)
delta_band = residual(h, z'_band)
```

The motivation was the residual-direction probe: a target-only code may not know how far the current ODE state already is from target HF statistics, so adding current-vs-target discrepancy statistics could improve the residual direction without exposing target spatial maps.

| metric | pair-stats | subband-only (best) | delta |
|---|---:|---:|---:|
| all DINO-S | 0.483765 | **0.488624** | -0.0049 |
| off DINO-S | 0.399385 | **0.403917** | -0.0045 |
| DINO-C | 0.794304 | 0.798123 | -0.0038 |
| CLIP-S | 0.718318 | 0.720880 | -0.0026 |
| LPIPS | 0.297092 | 0.296553 | +0.0005 |

**Verdict: FAIL; code/config removed, metrics kept.** The result rejects the simple "current-target statistics will fix residual direction" hypothesis. The signal is safe but appears too coarse/noisy over the ODE trajectory: global discrepancy statistics do not provide a useful enough style route, and they weaken the compact target-only subband code. Future attempts should not add more global statistics to the subband code; if dynamic conditioning is revisited, it needs a stronger probe showing that the dynamic signal improves residual direction without lowering the image frontier.

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

## Target-HF residual strength curve (2026-07-14)

After the ablation result, the next hypothesis was that the branch might be correct but under-powered. The matched test multiplies only the three trained subband residual outputs at inference.

| residual scale | all DINO-S | off DINO-S | DINO-C | CLIP-S | LPIPS | reading |
|---:|---:|---:|---:|---:|---:|---|
| 0.00 | 0.485770 | 0.403276 | 0.788810 | 0.720464 | 0.300980 | removing the route hurts style and content |
| 1.00 | **0.488624** | 0.403917 | **0.798123** | 0.720880 | **0.296553** | original trained magnitude, best balance |
| 1.25 | 0.487311 | 0.404491 | 0.788688 | 0.720082 | 0.300671 | slight off-style gain, content cost |
| 1.50 | 0.487485 | **0.406744** | 0.779830 | **0.721106** | 0.305438 | style-biased, content cost too high |
| HH 1.50 only | 0.487815 | 0.406092 | 0.783560 | 0.720560 | 0.303415 | band-aware boost still trades content for off-style |

**Verdict: FAIL as an improvement, PASS as a probe.** Simple residual amplification is not the performance breakthrough. The method should keep the compact subband residual, but a better architecture must change what the residual predicts or how it is conditioned. A scalar gain only moves along the style/content tradeoff and does not improve the Pareto point.

## Target-HF residual direction decomposition (2026-07-14)

The direction probe decomposes the trained checkpoint as:

`total HF velocity = base HF head velocity + target-HF subband residual`

and compares the residual against the immediate desired correction `target_velocity - base_velocity` under the training target. It uses `t = 0.25, 0.5, 0.75`, eval mode, 4 batches of 8 latents, and no image metrics.

| band | residual/base | residual/target | cos(residual, desired) | projection onto desired | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|---:|
| LH | 0.285418 | 0.098730 | 0.111191 | 0.012611 | 0.993232 | 0.012381 |
| HL | 0.248043 | 0.089052 | 0.095598 | 0.009870 | 0.994926 | 0.008637 |
| HH | 1.194715 | 0.275021 | 0.265744 | 0.084656 | 0.957125 | 0.074540 |

**Diagnosis.** The residual branch is useful because it lowers the training-target MSE, especially on HH. But most residual energy is orthogonal to the ideal correction direction, especially on LH/HL. This explains both previous observations: removing the branch hurts, while amplifying it hurts content. The next architecture should not be a larger residual or a gain; it should make the residual direction more target-aligned without letting a new loss compete with the main transport objective.

## Residual-direction auxiliary result (2026-07-14)

`target_hf_subband_diraux_ft6` tested the direct follow-up to the direction probe: keep the exact same compact subband-residual inference graph, but add a training-side auxiliary loss that aligns each residual delta with `stopgrad(target_velocity - base_velocity)`.

The probe objective succeeded mechanically:

| run | mean MSE improvement | mean cos(residual, desired) | orthogonal fraction |
|---|---:|---:|---:|
| `target_hf_subband_ft6` | 0.031853 | 0.157511 | 0.981761 |
| `target_hf_subband_diraux_ft6` | **0.116677** | **0.322185** | **0.935725** |

But the image metrics moved backward:

| metric | diraux | subband-only | delta |
|---|---:|---:|---:|
| all DINO-S | 0.486150 | **0.488624** | -0.0025 |
| off DINO-S | 0.402097 | **0.403917** | -0.0018 |
| DINO-C | 0.793859 | **0.798123** | -0.0043 |
| CLIP-S | 0.718929 | **0.720880** | -0.0020 |
| LPIPS | 0.297425 | **0.296553** | +0.0009 |

**Verdict: FAIL; code/config removed, metrics kept.** This is a useful falsification: a residual branch can become more aligned to the immediate velocity target while still hurting the final style/content frontier. The direct auxiliary appears to spend capacity matching an instantaneous correction rather than preserving the learned ODE transport geometry. Future direction work should be less invasive: probe/gate/orthogonalize the route, or change the residual parameterization, but do not add this auxiliary loss as-is.

## Residual time-window causal probe (2026-07-14)

After the direction-auxiliary failure, the next non-invasive hypothesis was temporal: maybe the subband residual is useful only as a late HF texture correction, and applying it across the whole ODE path creates off-direction content cost. A temporary evaluator hook tested this without changing weights:

`v(t) = v_base(t) + w(t) * delta_hf(t)`

Both early and late windows used `1 / window_width` normalization, so the approximate integrated residual energy stayed comparable to the full residual.

| variant | active window | all DINO-S | off DINO-S | DINO-C | CLIP-S | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| subband-only baseline | full | **0.488624** | **0.403917** | **0.798123** | **0.720880** | **0.296553** |
| late normalized | [0.5, 1.0] | 0.486637 | 0.402562 | 0.793614 | 0.719335 | 0.297480 |
| early normalized | [0.0, 0.5] | 0.486602 | 0.402543 | 0.793654 | 0.719377 | 0.297480 |

**Verdict: FAIL as an improvement, PASS as a probe; temporary hook code removed, metrics kept.** The nearly identical early/late results and the drop from full residual indicate that the route is not merely an endpoint texture patch or an early structure term. It acts as a small continuous correction throughout the learned transport field. Future architecture should not time-gate this branch; it should change the residual basis/parameterization or improve target-HF conditioning while preserving full-path participation.

## Cross-orientation mixer result (2026-07-14)

`target_hf_subband_mixer_ft6` tested whether the bottleneck was missing coordinate-free communication among LH/HL/HH target-HF codes:

```text
z'_k = z_k + tanh(g) * sum_{j != k} A_{k,j} z_j
```

The off-diagonal matrix was initialized to zero, so the initial function exactly matched `target_hf_subband_ft6`. The path was live: the trained checkpoint had `target_latent_hf_subband_mixer_active=1`, `gate≈0.204`, and nonzero off-diagonal weights (`mean abs≈0.0036`). But the residual-direction probe was unchanged:

| run | mean MSE improvement | mean cos(residual, desired) | orthogonal fraction |
|---|---:|---:|---:|
| `target_hf_subband_ft6` | 0.031853 | 0.157511 | 0.981761 |
| `target_hf_subband_mixer_ft6` | 0.031868 | 0.157508 | 0.981754 |

Full eval also moved backward:

| metric | mixer | subband-only | delta |
|---|---:|---:|---:|
| all DINO-S | 0.486666 | **0.488624** | -0.0020 |
| off DINO-S | 0.402582 | **0.403917** | -0.0013 |
| DINO-C | 0.793705 | **0.798123** | -0.0044 |
| CLIP-S | 0.719392 | **0.720880** | -0.0015 |
| LPIPS | 0.297500 | **0.296553** | +0.0009 |

**Verdict: FAIL; code/config removed, metrics kept.** Simple cross-orientation code sharing is not the missing bottleneck.

## Target-current code delta result (2026-07-14)

The gradient/info-flow probe showed that the subband residual is large, but the target-specific part of its condition response is small. `target_hf_subband_current_delta_ft6` therefore tested a zero-init dynamic code:

```text
z'_k = z_target,k + tanh(g) * (z_target,k - z_current,k)
```

`z_current` is pooled from the current ODE state `x_t` in the same HF subband encoder/projection, with the input tensor detached. This keeps target spatial maps disconnected and starts exactly equivalent to subband-only. The gate learned a small nonzero value (`tanh(g)≈0.044`) and did slightly increase target-specific information flow:

| measure | subband-only | current-delta |
|---|---:|---:|
| LH target-latent delta/base | 0.075624 | 0.083375 |
| HL target-latent delta/base | 0.097229 | 0.105496 |
| HH target-latent delta/base | 0.119432 | 0.125972 |
| LH condition/target grad ratio | 2.53% | 2.74% |
| HL condition/target grad ratio | 1.32% | 1.44% |
| HH condition/target grad ratio | 0.50% | 0.50% |

But the residual direction stayed essentially unchanged:

| run | mean MSE improvement | mean cos(residual, desired) | orthogonal fraction |
|---|---:|---:|---:|
| `target_hf_subband_ft6` | 0.031853 | 0.157511 | 0.981761 |
| `target_hf_subband_current_delta_ft6` | 0.031936 | 0.157720 | 0.981724 |

Full eval again moved backward:

| metric | current-delta | subband-only | delta |
|---|---:|---:|---:|
| all DINO-S | 0.486683 | **0.488624** | -0.0019 |
| off DINO-S | 0.402626 | **0.403917** | -0.0013 |
| DINO-C | 0.793621 | **0.798123** | -0.0045 |
| CLIP-S | 0.719366 | **0.720880** | -0.0015 |
| LPIPS | 0.297567 | **0.296553** | +0.0010 |

**Verdict: FAIL; code/config removed, metrics kept.** The bottleneck is not simply the absence of a coordinate-free target-current code difference. Mildly increasing target-specific condition sensitivity is insufficient unless the residual direction itself changes in a way that preserves the learned transport geometry.

## Condition-direction debug and follow-up failures (2026-07-14)

`tools/probe_gradient_information_flow.py` was extended to measure not only condition strength, but also whether the target-condition perturbation points toward the training velocity target:

```text
condition_delta = v(style_latent=target) - v(style_latent=content)
desired_delta   = target_velocity - v(style_latent=content)
```

For the current best `target_hf_subband_ft6`, the target-condition route is clean but weak and mostly off-direction:

| band | condition delta/base | delta/desired | cos(delta, desired) | orthogonal fraction | MSE improvement |
|---|---:|---:|---:|---:|---:|
| LH | 0.075624 | 0.016563 | 0.053786 | 0.998340 | 0.001631 |
| HL | 0.097229 | 0.020643 | 0.044784 | 0.998798 | 0.001612 |
| HH | 0.119432 | 0.019114 | 0.031591 | 0.999377 | 0.000868 |

This explains why amplitude-only fixes have failed: the image-specific condition component is too small, but much of it is also orthogonal to the desired local correction.

Two follow-up architecture tests were run from this diagnosis:

| run | mechanism probe | full-eval result | verdict |
|---|---|---|---|
| `target_hf_subband_affine_delta_ft6` | condition/target grad ratios rose from `2.5/1.3/0.5%` to `4.1/3.0/1.2%`, and condition delta/base rose strongly on LH/HL | DINO-S `0.482449`, DINO-C `0.790343`, CLIP-S `0.717787`, LPIPS `0.298913` | failed: stronger but still mostly off-direction |
| `target_hf_subband_wct_direction_ft6` | condition-direction cosine improved from `0.054/0.045/0.032` to `0.110/0.125/0.093`; gate learned `~0.0345` | DINO-S `0.486511`, DINO-C `0.793320`, CLIP-S `0.719448`, LPIPS `0.297849` | failed: local direction improved, final transport frontier worsened |

**Updated diagnosis.** A successful architecture must satisfy three constraints simultaneously:

1. make the target-HF condition route stronger;
2. make the target-specific perturbation more target-aligned;
3. preserve the learned ODE transport geometry over the full trajectory.

The current compact subband residual remains best because it is compatible with the learned HF heads, not because its target-condition path is already strong. Do not retry affine scale+shift or analytic WCT/AdaIN direction residual as-is.
