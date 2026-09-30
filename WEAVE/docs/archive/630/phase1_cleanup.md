# Phase 1: Codebase Deep Cleanup (2026-06-30)

## Objective
删除所有已确认无效的 loss 和模块，重构到最简洁优雅的 codebase，保证性能不下降。

## Baseline Reference
- Model: SpectralODEBridge620 (clean_base_v2_local)
- Baseline metrics: clip_style=0.7293 (PASS ≥ 0.7243), content_lpips=0.3203 (PASS ≤ 0.3453)
- Model params: 903,248 (after dino_adapter removal)
- Smoke test: GPU 33.9 MB, loss ≈ 4.59

## Cleanup Summary (628/629 Ablation Conclusions)

### Direction 1: spectral_w_hh removal (628 L8: DEAD)
- **Evidence**: 628 L8 ablation, Δclip=±0.0001 (within noise)
- **Action**: Removed `spectral_w_hh` from configs/clean_base_v2*.json
- **Impact**: Zero (HH loss was never active in clean_base_v2)

### Direction 3: Placeholder metric cleanup
- **Action**: Removed 60+ dead metric keys from utils/training.py
- **Impact**: Zero (trainer.py `_avg()` guards missing keys)

### Direction 4: StyleConditioner620 cleanup (192→109 lines, -788K dead params)
- **Evidence**: `dino_adapter` (LayerNorm+Linear+SiLU+Linear, 384→1024→384) was unconditionally created but `_adapt_dino` returned original value when `adapter_enabled=False`
- **Action**: Removed dino_adapter, local_cnn, text branches (all config=false)
- **Smoke test**: PASS, 903,248 params, loss=4.592, GPU 33.9MB

### Direction 5: Block620 cleanup (636→279 lines, 56% reduction)
- **Removed**: _sparsemax function, 4 dead attn_modes (gated/gated_raw/style_select/sparsemax), FiLM modulation, style MoE, content_dino query, learnable shortcut, skip_coarse, topk truncation, style_bias_proj
- **Kept**: RMSNorm (E4+ uses it), softmax + relu2 modes, single k/v projection
- **Smoke test**: PASS, 903,248 params, loss=4.587, GPU 33.8MB

### Direction 6: integrate_transport cleanup (260→85 lines)
- **Removed**: WCT mode, multiband AdaIN, patch AdaIN, multi-level extrapolation, mean_only/std_only modes, dead lowpass modes
- **Active path preserved**: DWT→Euler→iDWT→Endpoint AdaIN (full) + Style Extrap (simple scale)
- **Smoke test**: PASS, 903,248 params, loss=4.609, GPU 33.9MB

### Dead fiber references cleanup
- Removed `solver_fiber_aligned`, `i2sb_fiber_aligned_noise`, `i2sb_fiber_project_use_gate` from lancet_runtime.py
- Removed `source_style_latent` dead passing in utils/inference.py
- Removed 4 dead fiber metric columns from utils/training.py

## Files Modified
1. `src/spectral_bridge620.py` — integrate_transport rewrite (260→85 lines)
2. `src/blocks620.py` — full cleanup (636→279 lines)
3. `src/style_encoder620.py` — rewrite (192→109 lines)
4. `src/lancet_runtime.py` — dead fiber refs removed
5. `src/utils/inference.py` — dead source_style_latent removed
6. `src/utils/training.py` — dead fiber metrics removed
7. `configs/clean_base_v2.json` — spectral_w_hh removed
8. `configs/clean_base_v2_local.json` — spectral_w_hh removed
9. `docs/theory/SpectralODE_Bridge.md` — NEW: 8-chapter theory document

## Mathematical Proof: Zero Impact on Active Path
All removed code was dead (never executed in clean_base_v2 config):
- `dino_adapter`: `adapter_enabled=False` → `_adapt_dino` returns original
- Dead attn_modes: `attn_mode="softmax"` (default) → only softmax/relu2 branches executed
- FiLM/MoE: `film_enabled=False`, `moe_enabled=False` → branches never taken
- WCT/multiband/patch: `endpoint_adain_scale=1.0` (full mode) → only full AdaIN path executed
- Multi-level DWT: `spectral_levels=1` → single-level only

## Known Issue: attn_mode not passed
`style_attn_mode: "relu2"` in config is NOT passed to SpatialBridgeBlock620 by spectral_bridge620.py. Blocks use default "softmax". This is a pre-existing bug, preserved to maintain performance parity. Documented in theory doc §7.4.

## Next Steps
- Phase 2: Deep ablation analysis for any remaining dead code
- Phase 3: Confirm absolute minimal codebase
- Phase 4: TDD masking implementation
