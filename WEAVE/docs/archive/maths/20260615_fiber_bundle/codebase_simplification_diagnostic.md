# SA-Flow/LANCET Codebase Simplification & Module Diagnostic

**Date:** 2026-06-15
**Scope:** Full codebase audit + local/remote experiment evidence synthesis
**Goal:** Identify useful vs useless modules, eliminate redundancy, produce an efficient codebase

---

## 1. Experiment Evidence Summary

### 1.1 What Works (Positive Experimental Evidence)

| Module/Feature | Experiment | Result | Verdict |
|---|---|---|---|
| **PureLatentSpatialTokenizer** | Round 2 baseline (tok_baseline_global) | Default tokenizer, stable, produces best LPIPS-conditional style | **KEEP** |
| **I2SB endpoint + velocity prediction** | Round 2 sde_i2sb_* (sigma 0.25-1.0) | bridge_sigma=0.25 is the best SDE mode; endpoint mode used by all Round-2 configs | **KEEP** |
| **SWD terminal loss** | All training runs | Primary style supervision signal; essential | **KEEP** |
| **Kinetic penalty (global_l2)** | Path stability probe (k025, k000) | k000 (no kinetic) → LPIPS 0.486 (catastrophic); k025 → 0.460; base → 0.427. Kinetic is essential for content preservation | **KEEP** |
| **Sinkhorn coupling** | All recent training | GPU-native, replaced Hungarian; essential for throughput | **KEEP** |
| **Style Overdrive (t>1)** | k070 overdrive sweep | s=1.35→0.678/0.289, s=1.60→0.684/0.296; s=2.0+lataff0.6→0.722/0.342 (historical best) | **KEEP** (but needs curvature fix) |
| **Latent Affine Calibration** | k070 lataff sweep | s=0.45→0.679/0.319; best post-hoc style boost | **KEEP** |
| **XPred + Kmanifold + Pattn** | Inmortal experiments | Best training-side result: 0.735 style / 0.591 LPIPS at Stokes002 finetune e13 | **KEEP** (though LPIPS gap remains) |
| **EMA transport** | All training | Essential for stable evaluation | **KEEP** |
| **Gradient checkpointing** | All training | Essential for memory management | **KEEP** |

### 1.2 What Doesn't Work (Negative Experimental Evidence → DELETE)

| Module/Feature | Experiment | Result | Verdict |
|---|---|---|---|
| **Fiber-SDE noise (isotropic & fiber-aligned)** | sigma 0.01-0.08 sweep | Style gain 0.703→0.709 for +0.025 LPIPS; fiber-aligned NOT better than isotropic | **DELETE** |
| **SMoE Translator Tokenizer** | 15-epoch training | Best e9: 0.673/0.327; tokenizer-only does not produce enough style gain | **DELETE** (ablation complete, failed) |
| **Kinetic Release (w_kin=0.7)** | k070_kin070 probe | +0.002 style for +0.026 LPIPS; net negative | **DELETE** |
| **RGB Calibration** | s0.25-0.75 sweep | Lowers style; clipping artifacts | **DELETE** |
| **Topology Release (blend 0-0.5)** | k070 blend sweep | FLAT — no style response at all | **DELETE** |
| **Appearance Blend (0-1.0)** | k070 blend sweep | FLAT — output appearance path is not the style bottleneck | **DELETE** |
| **PC-Lowpass Corrector** | step 0.03-0.10 | Improves LPIPS but LOWERS style; structure repair, not style path | **DELETE** |
| **TopoGate / self_topology_blend** | Topology release sweep | The blend gate does nothing at default; topology release sweep was flat | **DELETE** |
| **DINO-dependent tokenizers** (DinoDict, CrossImg, ResidualAdapt, VLMPrompt) | Round 1 ablations | All retired; `validate_dino_retired_runtime()` actively blocks them | **DELETE** |
| **OMF objective** (Optimal Mass Flow) | Legacy early experiments | All Round-2 uses i2sb_endpoint; OMF is ~340 lines of dead code | **DELETE** |
| **Teacher EMA** | Never used in any promoted config | target_teacher_mode="off" everywhere | **DELETE** |
| **Ablation losses** (contrastive, residual_direction, spectral_amplitude, retinex, lowfreq_velocity, style_contrastive) | Weights = 0.0 in all configs | No evidence of value | **DELETE** |
| **Ablation attention blocks** (SpatialModulatedSelfAttn, GWOTAttention, GatedSpadeAttention, PnPSelfInject) | Round 1 backbone ablations | All retired; none is the default | **DELETE** |
| **Legacy projection_modes** in FactorizedStyleTokenizer | Various 2026-03~05 experiments | additive, carrier_residual, direct_code, concept_atoms, direct_atom_residual, class_prototypes, global_vq — all ablated, only "concat" is used | **DELETE** |
| **CDF SWD distance mode** | Never used in any promoted config | soft_cdf path is dead | **DELETE** |
| **Phighpass proximal mode** | Inmortal XPred_Phighpass | 0.680 style / 0.775 LPIPS — clear negative | **DELETE** |
| **XPred_Barycenter** | Inmortal | 0.719 style / 0.717 LPIPS — catastrophic content damage | **DELETE** |

### 1.3 Borderline (Keep But Simplify)

| Module/Feature | Evidence | Recommendation |
|---|---|---|
| **Endpoint prediction mode** | Used by Round-2; but division instability at t→1 | Keep, but fix with v-prediction or proper reweighting |
| **FactorizedStyleTokenizer** | Still the "legacy_factorized" default in some configs | Keep for baseline comparison, but strip 6 dead projection_modes |
| **Hungarian coupling** | Deprecated; CPU offload bottleneck | Keep as fallback but default to Sinkhorn |
| **Style Overdrive** | Best eval-time result, but uncontrolled extrapolation | Keep, but add curvature-aware extrapolation |
| **Latent Affine** | Best post-hoc boost, but global (non-spatial) | Keep, but document limitation |
| **Stokes/Anisotropic kinetic** | Inmortal experiments showed marginal LPIPS improvement at high style cost | Keep as optional, not default |
| **Execution budget mode** | Some Round-2 configs use it | Keep as config option |
| **Style injection (spatial_carrier_gate)** | sde_optimal_with_heuristics uses it | Keep as config option |

---

## 2. Code Audit: Dead Code Quantification

| File | Total Lines | Essential | Dead/Ablation | % Dead |
|---|---|---|---|---|
| `losses.py` | 1847 | ~1230 | ~620 | **34%** |
| `semantic_tokenizer.py` | 728 | ~502 | ~226 | **31%** |
| `style_tokenizer.py` | 458 | ~158 | ~300 | **65%** |
| `lancet_blocks.py` | 837 | ~686 | ~151 | **18%** |
| `trainer.py` | 1366 | ~1136 | ~230 | **17%** |
| `model.py` | ~1400 | ~1000 | ~400 | **29%** |
| `ot_cost.py` | 376 | ~348 | ~28 | **7%** |
| `round1_registry.py` | 139 | ~56 | ~83 | **60%** |
| `round2_registry.py` | 203 | 203 | 0 | **0%** |
| `style_families.py` | 325 | ~300 | ~25 | **8%** |
| **TOTAL** | **~7027** | **~4619** | **~2408** | **~34%** |

**Estimated code reduction: ~2400 lines (34%)** can be removed without any functional loss.

---

## 3. Config-Gated Features That Default to Off

26 config-gated branches in model.py alone default to disabled. Of these, the following are experimentally verified as useless:

| # | Feature | Config | Experiment | Verdict |
|---|---|---|---|---|
| 1 | Fiber-aligned noise | solver_fiber_aligned=False | Fiber-SDE sweep: no benefit | DELETE |
| 2 | TopoGate | semantic_self_topology_gate=False | Topology release: flat | DELETE |
| 3 | PC lowpass corrector | solver_corrector_mode="none" | PC sweep: lowers style | DELETE |
| 4 | Appearance alignment | output_appearance_alignment_mode="none" | Blend sweep: flat | DELETE |
| 5 | Style overdrive clamp-off | allow_style_overdrive=False | Overdrive WORKS; change default to True | FLIP DEFAULT |
| 6 | Proximal crossattn | proximal_mode="off" | No positive evidence | DELETE |
| 7 | Bridge sigma=0 | bridge_sigma default 0 | I2SB with sigma=0.25 WORKS | Keep as config |
| 8 | Diffeomorphic stroke | use_diffeomorphic_stroke=False | Only used in sde_optimal_with_heuristics | Keep as config |

---

## 4. Specific Deletion Plan

### 4.1 `losses.py` — Remove ~620 lines

DELETE:
- `_compute_omf_details()` and `_compute_omf()` (~340 lines) — dead OMF objective
- `_anisotropic_kinetic_loss()`, `_stokes_viscous_loss()` (~40 lines) — ablation
- `_lowfreq_velocity_loss()` (~4 lines) — zero weight everywhere
- `_style_contrastive_loss()`, `_style_signature()` (~35 lines) — zero weight
- `_residual_style_direction_loss()` (~13 lines) — zero weight
- `_spectral_amplitude_loss()` (~12 lines) — zero weight
- `_retinex_target()` (~14 lines) — zero weight
- Teacher EMA system (`_teacher_reduce`, `_update_target_teacher`, `_teacher_target`, `_teacher_alignment_loss`, state_dict extensions) (~80 lines)
- `_gradient_magnitude()`, `_diff_x()`, `_diff_y()` (~15 lines) — only used by deleted anisotropic/Stokes
- `_spectral_split_kinetic_loss()`, `_manifold_adaptive_kinetic_loss()` (~42 lines) — ablation modes
- `bridge_noise_schedule = "delayed_window"` path in `_bridge_state_and_velocity()` — dead heuristic

### 4.2 `semantic_tokenizer.py` — Remove ~226 lines

DELETE:
- `DinoDictionaryTokenizer` (~55 lines)
- `CrossImageRoutingTokenizer` (~55 lines)
- `ResidualSemanticAdapterTokenizer` (~50 lines)
- `VLMPromptStyleTokenizer` (~58 lines)

Also remove `validate_dino_retired_runtime()` from `style_families.py` (no longer needed if classes are gone).

### 4.3 `style_tokenizer.py` — Remove ~300 lines

DELETE 6 unused projection_mode paths:
- `"additive"` mode code
- `"carrier_residual"` mode code
- `"direct_code"` mode code
- `"concept_atoms"` mode code
- `"direct_atom_residual"` mode code
- `"class_prototypes"` mode code
- `"global_vq"` mode code

Keep only `"concat"` mode. This also allows removing `_atom_weights()`, `_class_prototype_weights()`, `mixture_weights()`, `_field_dropout()`, and various `_record_*_debug()` methods.

### 4.4 `lancet_blocks.py` — Remove ~151 lines

DELETE:
- `SpatialModulatedSelfAttn` (~33 lines)
- `GWOTAttention` (~41 lines)
- `GatedSpadeAttention` (~28 lines)
- `PnPSelfAttentionInject` (~28 lines)
- `_spatial_distance_bias()` (~14 lines) — only used by GWOTAttention

### 4.5 `model.py` — Remove ~400 lines

DELETE:
- `_fiber_aligned_solver_noise()` (~42 lines) — fiber-SDE failed
- `_correct_transport_state()` PC lowpass path (~37 lines) — PC corrector failed
- TopoGate-related code in integrate_transport — topology release flat
- Appearance alignment path in refine_endpoint — blend flat
- Legacy DINO lerp corrector mode
- `_runtime_content_dino_gate()` (~22 lines) — never called with valid data
- Proximal cross-attention texture path — no positive evidence
- `proximal_attn_routing_mode == "sinkhorn"` and `"gumbel_hard"` branches

CHANGE:
- `allow_style_overdrive` default → `True` (experimentally validated)
- `style_strength_max` default → `1.60` (validated sweet spot)

### 4.6 `trainer.py` — Remove ~230 lines

DELETE:
- `_maybe_initialize_tokenizer_from_latents()` and all sub-methods (~180 lines) — legacy, skipped for structured tokenizer
- `_pure_latent_uses_structured_tokenizer()` — trivial redundant wrapper
- Freeze mode alias mapping (~14 aliases → keep only canonical names)

### 4.7 `ot_cost.py` — Remove ~28 lines

DELETE:
- `_soft_cdf()`, `_cdf_grid()`, `_pairwise_from_projected_cdf()`, `_aligned_from_projected_cdf()` — dead CDF distance mode

### 4.8 `round1_registry.py` — Archive or remove ~83 lines

The DINO tokenizer and retired backbone attention specs are historical records. Either:
- (a) Move to `docs/experiments/round1_ablation_archive.py` for reproducibility
- (b) Delete entirely if historical tracking is done via git

Recommendation: **(a) archive** — keeps experiment tracking without cluttering active code.

---

## 5. Infra Fixes (From Previous Analysis)

### 5.1 OT Coupling: Hungarian → Sinkhorn Default

**Problem:** `linear_sum_assignment` offloads to CPU, O(n³), is the training throughput bottleneck.
**Fix:** Default `coupling_solver` to `"sinkhorn"` (already supported); remove Hungarian as default.

### 5.2 Hires Body Dual Execution

**Problem:** `lancet_runtime.py:714-717` runs hires body twice (once no_grad for skip, once with grad).
**Fix:** Compute skip features inside the gradient path and detach afterward:
```python
h_c = checkpoint(lambda: block(h_c, style_code, gate=0.0), h_c, style_code)
skip_32 = h_c.detach()  # No separate no_grad pass needed
```

### 5.3 Endpoint Parameterization Division Instability

**Problem:** `v = (z1 - x0) / (1-t)` with `clamp_min(1e-3)` → 1000× amplification near t=1.
**Fix:** Switch to v-prediction parameterization: predict velocity directly, no division needed.

---

## 6. Architectural Issues (From Fiber Bundle Analysis)

### 6.1 The "Pseudo-Connection" Problem

The Ehresmann connection claimed in the theory docs is NOT implemented. What exists is attention-logit blending, not tangent-space decomposition. This is the root cause of:

1. **Style Overdrive uncontrolled extrapolation** — without true vertical projection, t>1 trajectories drift horizontally
2. **Fiber-SDE failure** — noise is injected without structural guarantee of staying in the fiber
3. **Topology Release flatness** — the "connection" gate has no geometric effect

**Recommendation:** Either (a) implement a true projection operator, or (b) abandon the Ehresmann framing and rename to what it actually is (attention content-style blending).

### 6.2 The Delta Rank Collapse

Carrier codes have effective rank 3.986 (near full), but generated deltas have rank 3.324 (collapsed). Different styles produce nearly collinear residuals. This means the model is NOT learning fiber sections — it's learning a single direction with style-dependent amplitude.

**Recommendation:** Replace the `dec_out` bottleneck (single C×3×3 conv) with a multi-head style-routed delta generator.

### 6.3 Straight-Line Interpolation on Curved Manifold

Training uses ψ_t = (1-t)x_c + t·x_s, which assumes flat latent space. SD latent space is curved. Intermediates fall outside the data manifold.

**Recommendation:** Use Schrödinger bridge sampling (which adds stochasticity to the interpolation) or learn a geodesic approximation.

---

## 7. Simplified Codebase Target

### Core Pipeline (what remains after cleanup)

```
src/
├── model.py          (~1000 lines, down from 1400)
│   ├── TimeConditionedBridge
│   │   ├── _compute_delta()          [velocity mode only]
│   │   ├── forward()                 [clean: encode → backbone → delta → velocity]
│   │   ├── integrate_transport()     [Euler + I2SB solver only]
│   │   ├── _resolve_integration_horizon()  [overdrive ON by default]
│   │   └── refine_endpoint()         [simplified: no PC, no appearance align]
│   └── (no fiber noise, no PC corrector, no TopoGate, no proximal crossattn)
│
├── lancet_backbone.py (~30K, unchanged — already clean)
│
├── lancet_blocks.py   (~700 lines, down from 837)
│   ├── CrossAttnAdaGN, ResBlock, SpatialSelfAttention, AttentionBlock
│   ├── SemanticCrossAttn
│   ├── StyleMaps, _build_feature_block
│   ├── NormFreeModulation, SimpleResBlock, StyleRoutingSkip
│   └── (no GWOTAttention, no GatedSpade, no PnPSelfInject, no SpatialModulated)
│
├── lancet_runtime.py  (~46K, with hires body dual-exec fix)
│
├── losses.py          (~1200 lines, down from 1847)
│   └── OTFlowMatchingObjective
│       ├── compute()                 [i2sb_endpoint only]
│       ├── _compute_sampled_bridge_details()
│       ├── _terminal_swd_loss()      [primary style supervision]
│       ├── _kinetic_penalty_loss()   [global_l2 only]
│       ├── _cycle_consistency_loss() [if weight > 0]
│       └── (no OMF, no teacher EMA, no ablation losses)
│
├── semantic_tokenizer.py (~500 lines, down from 728)
│   ├── _BaseStructuredTokenizer
│   ├── PureLatentSpatialTokenizer    [DEFAULT]
│   └── SMoETranslatorTokenizer       [optional, keep for now]
│
├── style_tokenizer.py  (~160 lines, down from 458)
│   └── FactorizedStyleTokenizer      [concat mode only]
│
├── ot_cost.py          (~350 lines, unchanged — already lean)
│
├── trainer.py          (~1100 lines, down from 1366)
│   └── (no legacy tokenizer init, no freeze aliases)
│
├── config_schema.py    (~38K, prune retired config keys)
├── style_families.py   (~310 lines, remove DINO blocker + retired strings)
├── round2_registry.py  (unchanged — active)
│
├── run.py              (unchanged)
└── utils/
    ├── inference.py    (simplify: remove PC solver, keep Euler + I2SB)
    ├── dataset.py      (unchanged)
    ├── artfid_metric.py (unchanged)
    └── ...
```

### Archived (moved to `docs/experiments/`)

- `round1_registry.py` → `docs/experiments/round1_ablation_archive.py`
- DINO tokenizer classes → `docs/experiments/retired_tokenizers.py`

---

## 8. Performance Impact Estimate

| Metric | Before | After | Change |
|---|---|---|---|
| Total source lines | ~7027 | ~4619 | **-34%** |
| Loss computation paths | 3 (OMF, I2SB, bridge) | 1 (I2SB only) | **-67%** |
| Solver families | 5 | 2 (Euler + I2SB) | **-60%** |
| Tokenizer variants | 6 | 3 (PureLatent, SMoE, Factorized-concat) | **-50%** |
| Config-gated off-by-default features | 26 | ~10 (only the ones with positive evidence) | **-62%** |
| Dead attention blocks | 4 | 0 | **-100%** |
| Inference code paths | 8+ solver combos | 2 (Euler legacy + I2SB) | **-75%** |

---

## 9. Execution Order

1. **Phase 1 (Safe deletions):** Remove dead code from losses.py, semantic_tokenizer.py, lancet_blocks.py, ot_cost.py — no functional impact
2. **Phase 2 (Structural cleanup):** Simplify model.py (remove PC corrector, fiber noise, TopoGate, appearance alignment), simplify style_tokenizer.py
3. **Phase 3 (Config defaults):** Flip allow_style_overdrive=True, style_strength_max=1.60, coupling_solver="sinkhorn"
4. **Phase 4 (Infra fixes):** Fix hires body dual execution, fix endpoint division instability
5. **Phase 5 (Archive):** Move round1_registry + retired tokenizers to docs/experiments/

Each phase should be followed by a build/smoke test to verify no regression.
