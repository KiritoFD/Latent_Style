# Phase 1 Cleanup Report

## Summary

Removed all experimentally-verified dead code. **4 commits, ~840 lines deleted.**

| Commit | Description |
|--------|-------------|
| `08fd47d` | Archive before cleanup |
| `2790a85` | Remove heuristic content losses (losses.py -506 lines) |
| `035851e` | Remove frequency split (ot_cost.py -337 lines) |
| `6a40a72` | Streamline training logs (trainer.py -28 lines) |
| `2556111` | Clean config params |

## What Was Removed

### 1. losses.py (942 → 436 lines)

**Removed functions:**
- `calc_latent_patch_nce_loss` — PatchNCE contrastive loss
- `calc_low_freq_structure_loss` — dead code, never called
- `_calc_local_contextual_color_loss` — local color alignment (gradient conflict with SWD)
- `_compute_omf_details` / `_compute_omf` — OMF discrete mode (253 lines)
- `_freq_split` — high/low frequency decomposition (manifold tearing)
- `_cosine_lock_loss` — cycle consistency
- `_collect_repulsive_components` — repulsive diversity loss
- `compute_debug` — debug variant

**Removed config params:**
`w_color`, `w_repulsive`, `w_nce`, `w_cycle`, `w_low_freq`, `objective_mode`, `color_patch_size`, `color_transport_mode`, `color_gumbel_tau`, `low_freq_kernel_size`, `swd_use_high_freq`, `nce_num_patches`, `nce_temperature`, `repulsive_pool_size`, `repulsive_temperature`

**Rewritten:**
`compute()` — removed OMF dispatch, now pure flow matching path

### 2. ot_cost.py (413 → 76 lines)

**Removed functions:**
- `_get_sobel_kernels` — Sobel edge detection
- `_compute_fused_hf_feature` — gradient magnitude
- `_prepare_micro_features` — high-pass feature extraction
- `_prepare_macro_features` — low-pass feature extraction
- `_select_sample_indices`, `_pairwise_from_projected`, `_aligned_from_projected` — CDF-based SWD (replaced by sort-based)
- `_branch_pairwise_cost`, `_branch_aligned_cost` — branch-level cost computation

**Removed config params:**
`swd_use_high_freq`, `swd_hf_weight_ratio`, `swd_micro_weight`, `swd_macro_weight`, `swd_micro_patch_max`, `swd_macro_patch_min`, `swd_cdf_*` (6 params), `swd_deterministic_subsample`, `swd_projection_chunk_size`

**Simplified:**
`pairwise_cost` / `aligned_cost` — now direct full-band SWD via sort projection

### 3. trainer.py (642 → 614 lines)

**Log columns:** 33 → 12
```
epoch, loss, flow, kinetic_energy, ot_cost, terminal_swd,
t_mean, velocity_abs, endpoint_abs, lr, epoch_time_sec, samples_per_sec
```

**tqdm display:** 12 → 6 items
```
loss, flow, kin, ot, tswd, t
```

### 4. config.json

Removed bridge params: `objective_mode`, `w_low_freq`, `w_cycle`, `w_color`, `w_repulsive`, `w_nce`, `low_freq_kernel_size`, `swd_use_high_freq`

Added: `w_curvature: 0.0` (for Phase 2 A/B test)

## Why These Were Safe to Remove

- `w_color=0.0`, `w_repulsive=0.0`, `w_nce=0.0` — already disabled in config
- `w_low_freq=1.0`, `w_cycle=1.0` — only used inside `_compute_omf_details` which was gated by `objective_mode="omf"`. After removing OMF mode, dead code
- `calc_low_freq_structure_loss` — defined but never called anywhere
- Frequency split — replaced by full-band SWD (faster, avoids manifold tearing)
- OMF mode — replaced by standard flow matching (theoretical superiority)

## What Remains

### Core loss pipeline (losses.py ~200 lines effective):
- OT matching (Sinkhorn/Hungarian)
- Bridge state construction (x_t, v_target)
- Flow matching loss
- Kinetic energy regularizer
- Curvature regularizer (default off)
- Terminal SWD loss
- Semantic-guided SWD
- Knowledge distillation

### Core OT cost (ot_cost.py 76 lines):
- Full-band random projection SWD
- Sort-based distance computation
- Projection bank caching
