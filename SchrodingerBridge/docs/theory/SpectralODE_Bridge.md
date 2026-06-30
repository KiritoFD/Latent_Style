# Spectral ODE Bridge: Mathematical Theory

**Document**: FC-SB Phase 4 B2 — Native Spectral Schrödinger Bridge
**Model**: `SpectralODEBridge620` (`src/spectral_bridge620.py`)
**Loss**: `SpectralODEObjective620` (`src/spectral_losses620.py`)
**Status**: Cleaned codebase after 628/629 ablation sweeps (2026-06-30)

---

## 1. Introduction & Motivation

### 1.1 Problem Statement

Given a content latent $x_0 \in \mathbb{R}^{B \times C \times H \times W}$ and a target style latent $x_1$, learn a transport map $\phi: x_0 \mapsto x_1$ that:
- Preserves content structure (measured by LPIPS)
- Transfers style identity (measured by CLIP style similarity)
- Remains stable under Euler integration (no divergence)

### 1.2 Core Insight

Standard Flow Matching (FM) bridges operate in the **Euclidean latent space**, treating all frequency components uniformly. This creates a fundamental tension:
- **Low frequencies** (LL) carry content structure → must be preserved → should have *small* velocity
- **High/mid frequencies** (LH/HL/HH) carry style texture → must be changed → should have *large* velocity

The **Spectral ODE Bridge** resolves this by decomposing the transport into the **wavelet domain** and predicting independent velocities per subband. This allows:
1. **Frequency-disentangled training**: per-subband FM losses with independent weights
2. **Content-locked inference**: LL subband velocity ≈ 0, preserving structure
3. **Style-focused transport**: LH/HL subbands carry the style transfer signal

### 1.3 FC-SB Design Principles

| Principle | Mechanism | Ablation Evidence |
|-----------|-----------|-------------------|
| Spectral decomposition | Haar DWT → 4 subbands | 620 architecture baseline |
| Frequency-disentangled loss | Per-subband FM loss (LL/LH/HL) | 628 L8: HH loss DEAD (Δclip=±0.0001) |
| Content locking | $w_{ll} = 0$ (LL velocity not trained) | 628 D1: Δclip=-0.0167 (removing hurts) |
| Endpoint AdaIN | Fiber statistics matching at $t=1$ | 628 D2: Δclip=-0.0142 (removing hurts) |
| Style extrapolation | Fiber high-pass scaling | 628 D3: Δclip=-0.0016 (removing hurts) |
| LL weight preservation | $w_{ll} > 0$ on LL FM loss | 628 L7: Δclip=-0.0042 (removing hurts) |

---

## 2. Mathematical Foundations

### 2.1 Haar Wavelet Transform (DWT)

The 2D Haar DWT decomposes an image into 4 subbands via a $2 \times 2$ block transform. For a $2 \times 2$ block with pixels $(a, b, c, d)$:

$$
\begin{aligned}
\text{LL} &= \frac{a + b + c + d}{2} \quad &\text{(low-low: average)} \\
\text{LH} &= \frac{a + b - c - d}{2} \quad &\text{(low-high: vertical edges)} \\
\text{HL} &= \frac{a - b + c - d}{2} \quad &\text{(high-low: horizontal edges)} \\
\text{HH} &= \frac{a - b - c + d}{2} \quad &\text{(high-high: diagonal)}
\end{aligned}
$$

The coefficient $\frac{1}{2} = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}}$ ensures **orthonormality**: $\text{IDWT}(\text{DWT}(x)) = x$ exactly.

**Properties**:
- **Orthogonal**: subbands are linearly independent, no aliasing
- **Perfect reconstruction**: IDWT exactly inverts DWT
- **Energy preservation**: $\|\text{LL}\|^2 + \|\text{LH}\|^2 + \|\text{HL}\|^2 + \|\text{HH}\|^2 = \|x\|^2$
- **Single-level**: $x \in \mathbb{R}^{B \times C \times H \times W} \to$ 4 subbands in $\mathbb{R}^{B \times C \times H/2 \times W/2}$

### 2.2 Flow Matching Bridge

A flow matching bridge constructs a probability path between two distributions $p_0$ (content) and $p_1$ (style). The **linear interpolation** path is:

$$
x_t = (1 - t) \cdot x_0 + t \cdot x_1, \quad t \in [0, 1]
$$

with target velocity:

$$
v_t = \frac{dx_t}{dt} = x_1 - x_0
$$

The model learns to predict $v_\theta(x_t, t)$ by minimizing:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim \mathcal{U}[0,1]} \left[ \|v_\theta(x_t, t) - (x_1 - x_0)\|^2 \right]
$$

### 2.3 Brownian Bridge Noise (Optional)

To approximate a Schrödinger bridge (which is a diffusion process, not a deterministic ODE), we add **Brownian bridge noise** that vanishes at endpoints:

$$
x_t = (1 - t) \cdot x_0 + t \cdot x_1 + \sigma \sqrt{t(1 - t)} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

The noise scale $\sigma \sqrt{t(1-t)}$ is:
- **Zero at $t=0$ and $t=1$**: endpoints are deterministic
- **Maximum at $t=0.5$**: maximum stochasticity in the middle
- **SB-like**: mimics the variance profile of a Schrödinger bridge

---

## 3. Model Architecture

### 3.1 Overview

```
Input: x_t (B, 4C, H, W)
  │
  ├── DWT(x_t) → (LL, LH, HL, HH) each (B, C, H/2, W/2)
  │
  ├── Stack: [LL, LH, HL, HH] → (B, 4C, H/2, W/2)
  │
  ├── input_proj: Conv2d(4C → dim) → (B, dim, H/2, W/2)
  │
  ├── time_proj: sinusoidal_embedding(t) → Linear → (B, dim)
  │
  ├── style_conditioner: DINO patches → (style_tokens, style_global)
  │
  ├── N × SpatialBridgeBlock620:
  │     AdaLN(time) → Self-Attn → Cross-Attn(style) → FFN
  │
  ├── head_ll: SpectralVelocityHead(dim → C) → v_LL
  ├── head_lh: SpectralVelocityHead(dim → C) → v_LH
  ├── head_hl: SpectralVelocityHead(dim → C) → v_HL
  │   (head_hh removed: 628 L8 confirmed DEAD)
  │
  Output: {ll: v_LL, lh: v_LH, hl: v_HL}
```

### 3.2 SpectralVelocityHead

Each subband velocity head is a zero-initialized conv:

$$
v_{\text{subband}} = \text{Conv}_{3 \times 3}\left(\text{SiLU}\left(\text{GroupNorm}(h)\right)\right)
$$

**Zero initialization** ($\text{std}=10^{-3}$, $\text{bias}=0$) ensures the model starts as identity (no transport), preventing early-training instability.

### 3.3 SpatialBridgeBlock620

Each block follows the architecture:

$$
\begin{aligned}
h_1 &= \text{Norm}(x) \cdot (1 + \gamma_{\text{time}}) + \beta_{\text{time}} \\
h_{\text{sa}} &= \text{SelfAttn}(h_1) \\
x' &= x + \sigma(\text{gate}_t) \cdot h_{\text{sa}} \\
h_{\text{ca}} &= \text{CrossAttn}(Q=x', K,V=\text{style\_tokens}) \\
x'' &= \alpha \cdot x' + \tanh(g_{\text{style}}) \cdot h_{\text{ca}} \\
x_{\text{out}} &= x'' + \text{FFN}(\text{Norm}(x''))
\end{aligned}
$$

where:
- $\gamma_{\text{time}}, \beta_{\text{time}}, \text{gate}_t$ come from `time_adaln(time_emb)` (AdaLN)
- $\alpha$ is the shortcut alpha (default 1.0)
- $g_{\text{style}}$ is a learnable gate parameter (init 0.05, tanh-bounded)
- Cross-attention uses **softmax** (default) or **ReLU²** attention mode

### 3.4 StyleConditioner620

Projects cached DINO patch tokens into the bridge width:

$$
\begin{aligned}
\text{style\_tokens} &= \text{MLP}(\text{DINO\_patches}) + \text{style\_memory}[s] \\
\text{style\_global} &= \text{MLP}(\text{DINO\_cls})
\end{aligned}
$$

where `style_memory` is a learnable parameter of shape $(S, M, D_{\text{dino}})$ providing per-style context.

---

## 4. Training Objective

### 4.1 Spectral Flow Matching Loss

The target velocity is decomposed into the wavelet domain:

$$
\Delta x = x_1 - x_0
$$

$$
(\Delta_{\text{LL}}, \Delta_{\text{LH}}, \Delta_{\text{HL}}, \Delta_{\text{HH}}) = \text{DWT}(\Delta x)
$$

The per-subband FM loss is:

$$
\mathcal{L}_{\text{spectral}} = w_{\text{ll}} \cdot \text{MSE}(v_\theta^{\text{LL}}, \Delta_{\text{LL}}) + w_{\text{lh}} \cdot \text{MSE}(v_\theta^{\text{LH}}, \Delta_{\text{LH}}) + w_{\text{hl}} \cdot \text{MSE}(v_\theta^{\text{HL}}, \Delta_{\text{HL}})
$$

**Note**: The HH head and loss were removed (628 L8: DEAD, $\Delta\text{clip} = \pm 0.0001$). The HH subband is still decomposed for input (stacked with LL/LH/HL), but no velocity head predicts $v_{\text{HH}}$.

### 4.2 Weight Configuration (clean_base_v2)

| Weight | Value | Role |
|--------|-------|------|
| $w_{\text{ll}}$ | `spectral_w_ll` > 0 | LL FM loss (content structure preservation) |
| $w_{\text{lh}}$ | 1.0 | LH FM loss (vertical edge style transfer) |
| $w_{\text{hl}}$ | 1.0 | HL FM loss (horizontal edge style transfer) |
| $w_{\text{hh}}$ | ~~1.5~~ removed | HH FM loss (628 L8: DEAD) |

### 4.3 Bridge Path Construction (tri_band)

The bridge path $x_t$ is constructed using the **tri_band** mode, which applies frequency-aware interpolation:

1. **Tri-band decomposition** of content $x_0$ and target $x_1$:
   - $\text{LL}$: large-kernel lowpass (color/illumination — broad structure)
   - $\text{Mid}$: mid-kernel lowpass − LL (edges/contours — content structure)
   - $\text{HH}$: $x$ − mid-kernel lowpass (fine texture/strokes — style)

2. **Frequency-aware target projection**:
   $$x_1^{\text{proj}} = \text{LL}(x_0) + \left[\alpha_{\text{edge}} \cdot \text{Mid}(x_0) + (1 - \alpha_{\text{edge}}) \cdot \text{Mid}(x_1)\right] + \text{HH}(x_1)$$
   
   - LL: **locked to content** (preserves broad structure)
   - Mid: **α-blended** (preserves content contours with $\alpha_{\text{edge}} = 0.5$)
   - HH: **fully from target** (free style diffusion)

3. **Bridge interpolation**:
   $$x_t = (1 - t) \cdot x_0 + t \cdot x_1^{\text{proj}}$$

This ensures the transport focuses on style-relevant frequencies while protecting content structure.

---

## 5. Inference: Spectral Euler Integration

### 5.1 Integration Scheme

At inference, the model integrates the learned velocity field using **spectral Euler integration**:

$$
h_0 = x_0, \quad \Delta t = \frac{T}{N}
$$

For $i = 0, 1, \ldots, N-1$:

$$
\begin{aligned}
t_i &= \frac{i}{N} \cdot T \\
v_{\text{LL}}, v_{\text{LH}}, v_{\text{HL}} &= v_\theta(h_i, t_i, \text{style}) \\
\text{LL}_i, \text{LH}_i, \text{HL}_i, \text{HH}_i &= \text{DWT}(h_i) \\
\text{LL}_{i+1} &= \text{LL}_i + v_{\text{LL}} \cdot \Delta t \\
\text{LH}_{i+1} &= \text{LH}_i + v_{\text{LH}} \cdot \Delta t \\
\text{HL}_{i+1} &= \text{HL}_i + v_{\text{HL}} \cdot \Delta t \\
h_{i+1} &= \text{IDWT}(\text{LL}_{i+1}, \text{LH}_{i+1}, \text{HL}_{i+1}, \text{HH}_i)
\end{aligned}
$$

**Key**: The HH subband is **not integrated** ($\text{HH}_{i+1} = \text{HH}_i$), consistent with the removed HH velocity head.

### 5.2 Endpoint AdaIN (core_keep D2)

After each Euler step, an **endpoint AdaIN** correction matches fiber statistics to the target style:

1. **Decompose** into base (lowpass) and fiber (highpass):
   $$\text{base}(h) = \text{IDWT}(\text{LL}(h), 0, 0, 0)$$
   $$\text{fiber}(h) = h - \text{base}(h)$$

2. **Style fiber** (with extrapolation):
   $$\text{fiber}_{\text{style}} = \text{fiber}(x_1) \cdot (1 + \alpha_{\text{extrap}})$$

3. **First-order statistics matching**:
   $$\mu_{\text{pred}} = \text{mean}(\text{fiber}(h)), \quad \sigma_{\text{pred}} = \text{std}(\text{fiber}(h))$$
   $$\mu_{\text{target}} = \text{mean}(\text{fiber}_{\text{style}}), \quad \sigma_{\text{target}} = \text{std}(\text{fiber}_{\text{style}})$$
   $$\text{fiber}_{\text{matched}} = \frac{\text{fiber}(h) - \mu_{\text{pred}}}{\sigma_{\text{pred}}} \cdot \sigma_{\text{target}} + \mu_{\text{target}}$$

4. **α-blend** (preserve content):
   $$h_{\text{out}} = \text{base}(h) + (1 - s_{\text{adain}}) \cdot \text{fiber}(h) + s_{\text{adain}} \cdot \text{fiber}_{\text{matched}}$$

where $s_{\text{adain}} = 1.0$ (full AdaIN) and $\alpha_{\text{extrap}} = 0.1$ (style extrapolation).

### 5.3 Style Extrapolation (core_keep D3)

The style extrapolation mechanism scales the style fiber by $(1 + \alpha_{\text{extrap}})$. Since the fiber (high-pass) component has mean ≈ 0, this extrapolation effectively **amplifies the style signal magnitude** without changing its direction:

$$\text{fiber}_{\text{style}}^{\text{extrap}} = (1 + \alpha) \cdot \text{fiber}_{\text{style}}$$

This provides a mild boost to style transfer strength ($\alpha = 0.1$ → 10% amplification).

---

## 6. Ablation Conclusions (628/629)

### 6.1 Core Modules (KEEP — removing hurts performance)

| ID | Module | Δclip (remove) | Conclusion |
|----|--------|----------------|------------|
| D1 | spectral_ode architecture | -0.0167 | Core architecture, mandatory |
| D2 | endpoint_adain_scale | -0.0142 | Critical for style transfer |
| D3 | style_extrap_alpha | -0.0016 | Mild but positive effect |
| L7 | spectral_w_ll | -0.0042 | LL FM loss contributes despite low weight |

### 6.2 Dead/Harmful Modules (REMOVED)

| ID | Module | Δclip (remove) | Conclusion |
|----|--------|----------------|------------|
| L8 | spectral_w_hh (HH loss) | ±0.0001 | DEAD — no effect |
| L9 | spectral_w_lh/hl (single) | +0.0010 | HARMFUL — removing single improves (but S1+S2 combined has negative interaction, so both kept) |
| — | FiLM modulation | ~0 | DEAD — film_enabled=false, never active |
| — | Style MoE | ~0 | DEAD — moe_enabled=false, never active |
| — | content_dino query | ~0 | DEAD — query_source="concat", never active |
| — | Multi-level DWT | ~0 | DEAD — spectral_levels=1 optimal |
| — | WCT/multiband/patch AdaIN | ~0 | DEAD — endpoint_adain_scale=1.0 (full mode) |
| — | learnable shortcut | ~0 | DEAD — shortcut_alpha=1.0 (float) |
| — | skip_coarse | ~0 | DEAD — skip_coarse=false |
| — | top-k truncation | ~0 | DEAD — topk=0 |

### 6.3 Theoretical Interpretation

The ablation results reveal a **content fidelity pathway**:

$$
\text{DWT (Haar)} \xrightarrow{\text{orthogonal decomposition}} \text{AdaIN scale} \xrightarrow{\text{fiber matching}} \text{Spectral ODE}
$$

1. **Haar DWT** provides the orthogonal decomposition that enables frequency-disentangled control
2. **AdaIN scale** ($s_{\text{adain}}$) controls the trade-off between content preservation (base) and style transfer (fiber)
3. **Spectral ODE** integrates the velocity field in the wavelet domain, ensuring stable transport

The **negative interaction** between S1+S2 loss cuts (combined Δclip = -0.0018) suggests the per-subband losses have a **regularization effect** on each other — removing both simultaneously destabilizes training more than removing either alone.

---

## 7. Clean Codebase Summary

### 7.1 Active Code Path

| File | Lines | Role |
|------|-------|------|
| `src/spectral_bridge620.py` | ~260 | Model definition (SpectralODEBridge620) |
| `src/spectral_losses620.py` | ~140 | Training objective (3 per-subband FM losses) |
| `src/spectral620.py` | ~100 | Haar DWT/IDWT utilities |
| `src/blocks620.py` | ~280 | SpatialBridgeBlock620 (relu2 + softmax only) |
| `src/style_encoder620.py` | ~110 | StyleConditioner620 (DINO projection only) |
| `src/losses620.py` | — | Bridge path construction (tri_band) |

### 7.2 Removed Dead Code (628/629 cleanup)

| Category | Items Removed |
|----------|---------------|
| Attn modes | gated, gated_raw, style_select, sparsemax, _sparsemax function |
| FiLM | Pre-attn FiLM, Post-attn FiLM, style_bias_proj |
| MoE | Style MoE router, expert k/v projections |
| Query sources | content_dino, sa_out_only |
| Integration hooks | WCT, multiband AdaIN, patch AdaIN, multi-level extrapolation |
| Loss components | spectral_w_hh, 9 auxiliary losses, 60+ placeholder metrics |
| Style encoder | dino_adapter (~788K dead params), local_cnn, text branch |
| Config fields | tri_band_inference_lock, spectral_w_hh |

### 7.3 Configuration (clean_base_v2_local.json)

Active parameters:
- `contract_family: "620_spectral_ode"` → SpectralODEBridge620
- `spectral_ode_levels: 1` → single-level Haar DWT
- `bridge_path_mode: "tri_band"` → tri-band frequency-aware interpolation
- `endpoint_adain_scale: 1.0` → full endpoint AdaIN
- `style_extrap_alpha: 0.1` → 10% style fiber amplification
- `style_attn_mode: "relu2"` → (note: see §7.4)

### 7.4 Known Discrepancy: attn_mode

The config specifies `style_attn_mode: "relu2"`, but `SpectralODEBridge620.__init__` does not currently pass `attn_mode` to `SpatialBridgeBlock620`. The blocks use the default `"softmax"` mode. This is preserved as-is in the cleaned codebase to maintain performance parity with the trained baseline. Enabling `"relu2"` requires passing `attn_mode=getattr(model_cfg, 'style_attn_mode', 'softmax')` in the block constructor call.

---

## 8. Performance Reference

**Baseline** (4070 Laptop, clean_base_v2_local, 2026-06-30):
- Model params: 903,248
- allpairs clip_style: 0.7293 (pass ≥ 0.7243)
- allpairs content_lpips: 0.3203 (pass ≤ 0.3453)
- Smoke test GPU: 33.9 MB
- Training peak GPU: ~0.36 GB
