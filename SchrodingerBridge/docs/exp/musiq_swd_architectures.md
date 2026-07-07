# MUSIQ Improvement: Semantic SWD Architecture Directions

## Background

Previous baseline: DWT Route + Attention-Weighted SWD (MUSIQ=41.11, CLIP-S=0.7275, LPIPS=0.4347)
Trade-off target: Seedream 4.5 (D5: CLIP-S=0.7198, LPIPS=0.4767, MUSIQ=69.51)

The core insight: MUSIQ rewards texture naturalness/sharpness. The baseline SWD
operates on pixel-marginal distributions globally — it cannot target local texture
quality. Semantic SWD solves this by partitioning the content into content-similar
regions and matching distributions within each region, avoiding the "muddy blend"
that global pixel-marginal SWD produces when incompatible regions (smooth sky vs.
textured foliage) share one distributional match.

## Architecture Directions (all centered on semantic SWD)

### S1: Semantic Region SWD (`swd_semantic_mode: "region"`)
- **Mechanism**: k-means on content latent partitions spatial locations into K
  content-coherent regions. Each generated region is matched to the appearance-
  corresponding target region (aligned by centroid mean-projection order).
- **Guidance signal**: content latent (k-means)
- **Key params**: `swd_semantic_regions=8`, `swd_semantic_blend=0.7`
- **Theory**: Region-coherent matching keeps per-region statistics internally
  consistent. The global SWD (blend=0.3) preserves the overall distributional
  constraint that drives MUSIQ via the reference artwork's statistics.

### S2: Semantic Patch SWD (`swd_semantic_mode: "region_patch"`) [NEW]
- **Mechanism**: Combines semantic region partitioning with multi-scale patch
  texture matching. Within each content-coherent region, local k×k patches are
  extracted and their projected distributions are matched.
- **Guidance signal**: content latent (k-means) + multi-scale patches [1,3,5]
- **Key params**: `swd_semantic_regions=6`, `swd_semantic_blend=0.7`,
  `swd_patch_sizes=[1,3,5]`, `swd_patch_weights=[0.3,0.4,0.3]`
- **Theory**: Patch-level matching lifts each sample to a C·k²-dim texture vector,
  so sliced projections carry local structure — directly targeting MUSIQ's texture
  reward. Doing this within content-coherent regions prevents cross-region texture
  contamination.
- **Implementation**: `_semantic_patch_swd()` in `spectral_losses620.py`

### S3: Semantic Band-split SWD (`swd_semantic_mode: "region_band"`) [NEW]
- **Mechanism**: Decomposes gen and target into LL/LH/HL/HH via Haar DWT, then
  applies semantic region matching within each subband independently. HF bands
  get higher weight.
- **Guidance signal**: content latent (k-means, downsampled to subband resolution)
- **Key params**: `swd_semantic_regions=6`, `swd_semantic_blend=0.8`,
  `swd_band_w_ll=0.25, swd_band_w_lh=1.0, swd_band_w_hl=1.0, swd_band_w_hh=2.0`
- **Theory**: The model lives in the wavelet domain. Full-latent SWD is dominated
  by low-frequency energy, barely constraining the HF bands MUSIQ rewards. Splitting
  into subbands and up-weighting HH routes the SWD budget to texture/detail.
  Semantic region matching within each subband maintains content coherence.
- **Implementation**: `_semantic_band_swd()` in `spectral_losses620.py`

### S4: Cross-attn Guided Semantic Region SWD
- **Mechanism**: Switches from "attention-weighted" (value multiplication) to
  "cross-attn-guided" (importance sampling) for the global SWD component, combined
  with semantic region matching.
- **Guidance signal**: cross-attention pixel entropy (importance sampling) +
  content latent (k-means regions)
- **Key params**: `swd_scale_mode="cross-attn-guided"`, `swd_semantic_mode="region"`,
  `swd_semantic_regions=8`, `swd_semantic_blend=0.7`
- **Theory**: Cross-attn entropy encodes where the model is actively routing style.
  Using it as sampling weight (not value weight) focuses SWD on regions the model
  is editing, without distorting the latent values being matched.

## Cleanup Performed

Removed non-SWD architecture directions (LL content anchor, spectral power
consistency, HH soft gate) to keep the experiment focused on semantic SWD:
- `config_schema.py`: removed `w_ll_content_anchor`, `ll_content_anchor_kernel`,
  `w_spectral_power_consistency`, `spectral_power_low_cutoff`, `dwt_hh_soft_gate`,
  `dwt_hh_gate_init`
- `spectral_losses620.py`: removed LL content anchor loss, spectral power
  consistency loss, and their init/metrics
- `blocks620.py`: removed HH soft gate parameter and forward application
- `spectral_bridge620.py`: removed HH soft gate parameter passing
- Deleted old D1-D4 config files and gen_configs.py

## Configs

All configs are based on `dwt_route_distinct5.json` (DWT route + attention-weighted
SWD, batch=48, 10 epochs, distinct5 dataset) with one SWD direction enabled each:
- `configs/musiq_s1_sem_region.json`
- `configs/musiq_s2_sem_patch.json`
- `configs/musiq_s3_sem_band.json`
- `configs/musiq_s4_sem_xattn.json`
