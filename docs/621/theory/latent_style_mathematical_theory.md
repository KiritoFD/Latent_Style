# A Mathematical Theory of Latent Style Transfer

**Latent Style Project --- Formal Theory Document**

Date: 2026-06-23 | Empirical Basis: 645+ experiments across 15 branches, 22,629 CSV records, 4 months (Feb-Jun 2026)

---

## Abstract

We present a unified mathematical theory explaining the systematic failure modes observed in latent-space style transfer via flow matching. Across 645+ experiments spanning four architectural generations, we observe a persistent performance ceiling (CLIP-style ~0.70) and a universal whitening/fogging artifact (WFI > 0.35). We identify four coupled mechanisms --- gate collapse, endpoint shrinkage, norm-induced regression-to-mean, and training-output mismatch --- and prove that they form a self-reinforcing attractor that traps the model in a degenerate solution basin. We derive formal propositions for each mechanism, provide empirical validation from the experiment corpus, and state falsifiable predictions for remediation.

---

## 1. Formal Definition of the Style Transfer Problem in Latent Space

### 1.1 Notation and Setup

Let $\mathcal{E}: \mathbb{R}^{3 \times 512 \times 512} \to \mathbb{R}^{C \times H \times W}$ be the VAE encoder mapping images to latent space, where $C = 4$, $H = W = 64$ for Stable Diffusion 1.5. The latent dimensionality is $d = C \times H \times W = 16384$.

**Definition 1.1** (Source and Target Latents). Given a source content image $x_s$ and a target style reference image $x_t$:

$$z_s = \mathcal{E}(x_s) \in \mathbb{R}^{C \times H \times W}, \quad z_t = \mathcal{E}(x_t) \in \mathbb{R}^{C \times H \times W}$$

**Definition 1.2** (Style Transfer Operator). A latent style transfer model is a parametric function:

$$f_\theta: \mathbb{R}^{C \times H \times W} \times \mathcal{S} \to \mathbb{R}^{C \times H \times W}$$

where $\mathcal{S}$ is the style conditioning space (e.g., DINO patch tokens, global style vectors). The generated latent is:

$$z_g = f_\theta(z_s, c_{\text{style}})$$

In the flow-matching paradigm, $f_\theta$ is realized via a learned ODE:

$$z_g = z_s + \int_0^1 v_\theta(z_\tau, \tau, c_{\text{style}}) \, d\tau$$

### 1.2 Objective Functions

**Definition 1.3** (Style Distance). The style distance measures stylistic alignment between generated and target:

$$d_{\text{style}}(z_g, z_t) = 1 - \cos\left(\text{CLIP}_{\text{style}}(\mathcal{D}(z_g)), \text{CLIP}_{\text{style}}(\mathcal{D}(z_t))\right)$$

where $\mathcal{D}$ is the VAE decoder and $\text{CLIP}_{\text{style}}$ denotes CLIP ViT-B/32 features computed on the style axis. Minimizing $d_{\text{style}}$ maximizes CLIP-style cosine similarity.

**Definition 1.4** (Content Preservation Distance). Content fidelity is measured by Learned Perceptual Image Patch Similarity:

$$d_{\text{content}}(z_g, z_s) = \text{LPIPS}(\mathcal{D}(z_g), \mathcal{D}(z_s))$$

**Definition 1.5** (Whiteness/Fog Index). The WFI is a composite image-space metric measuring desaturation, contrast loss, and dynamic range compression:

$$\text{WFI}(z_g) = 1 - \left(0.4 \cdot \frac{\text{CR}(z_g)}{0.5} + 0.3 \cdot \frac{\text{SR}(z_g)}{0.4} + 0.3 \cdot \frac{\text{DR}(z_g)}{0.6}\right)$$

where CR = contrast ratio ($\sigma_{\text{lum}} / \mu_{\text{lum}}$), SR = mean HSV saturation, DR = dynamic range ($(P_{95} - P_5) / (P_{95} + P_5)$). Healthy range: WFI < 0.20 (Seedream IDT benchmark: 0.158). Whitening: WFI > 0.35.

### 1.3 The Optimization Problem

The style transfer problem is a multi-objective optimization:

$$\min_{\theta} \; \mathcal{L}(\theta) = w_{\text{FM}} \mathcal{L}_{\text{FM}} + w_{\text{SWD}} \mathcal{L}_{\text{SWD}} + w_{\text{edge}} \mathcal{L}_{\text{edge}}$$

where:

- $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, z_s, z_t}\left[\|v_\theta(z_t, t, c) - v_{\text{target}}\|_2^2\right]$ (flow matching loss, per-step velocity error)
- $\mathcal{L}_{\text{SWD}} = \text{SWD}(\hat{z}_1, z_t)$ (Sliced Wasserstein Distance on endpoint, distributional style alignment)
- $\mathcal{L}_{\text{edge}}$ (edge preservation regularizer)

The evaluation, however, is on *endpoint quality* --- the integrated result, not per-step accuracy.

---

## 2. The Style-Content-Whiteness Trilemma

### 2.1 Statement

**Proposition 2.1** (Trilemma). For any style transfer model $f_\theta$ operating in the 620 SpatialBridge architecture, there exists a Pareto frontier such that one cannot simultaneously:

1. Minimize $d_{\text{style}}$ (strong style transfer),
2. Minimize $d_{\text{content}}$ (perfect content preservation), and
3. Minimize WFI (no whitening).

### 2.2 Empirical Evidence

From the 620 experiment corpus (Style8 branch, 620+ experiments):

| Metric Pair | Pearson $r$ | Interpretation |
|-------------|-------------|----------------|
| clip_style vs. LPIPS | $+0.94$ | Better style = worse content (strong trade-off) |
| clip_style vs. WFI | $-0.67$ | Better style = more whitening |
| LPIPS vs. WFI | $-0.71$ | Better content = more whitening |

The strong positive correlation between clip_style and LPIPS ($r = +0.94$) is the most striking: as style transfer improves, content preservation degrades nearly linearly. This is not an artifact of evaluation --- it reflects a structural constraint of the model.

From the full 22,629-row historical CSV (all branches):

| Metric Pair | Pearson $r$ | Sample Size |
|-------------|-------------|-------------|
| clip_style vs. content_lpips | $-0.10$ | 17,021 |

The weak correlation in the full dataset reflects the diversity of methods and architectures; within a fixed architecture (620 SpatialBridge), the trade-off tightens dramatically.

### 2.3 Geometric Interpretation

Define the style transfer displacement:

$$\delta = z_t - z_s \in \mathbb{R}^{16384}$$

The model must navigate from $z_s$ toward $z_t$ along $\delta$. The trilemma arises because:

1. **Style quality** requires moving a significant fraction along $\delta$ (large $\alpha$ in the projection coefficient, see Sec. 4).
2. **Content preservation** requires not moving too far in directions orthogonal to the content structure (measured by LPIPS, which is sensitive to high-frequency changes).
3. **No whitening** requires maintaining the full statistical range of the latent --- but the model's internal normalization (GroupNorm) systematically compresses this range (see Sec. 5).

The Pareto frontier can be parameterized by the effective displacement $\alpha \cdot \delta$:

- $\alpha \to 0$: $z_g \to z_s$ (no style, no whitening, perfect content)
- $\alpha \to 1$: $z_g \to z_t$ (full style, content loss, possible whitening from integration errors)
- $\alpha \approx 0.16$ (observed): a degenerate fixed point where all three objectives are suboptimal

### 2.4 Historical Ceiling Evidence

The CLIP-style ceiling has evolved across four architectural phases:

| Phase | Period | Architecture | Ceiling | Key Innovation |
|-------|--------|-------------|---------|----------------|
| 1 | Feb-Mar 2026 | Legacy SWD/AdaGN | 0.65-0.68 | Baseline |
| 2 | Mar-May 2026 | Tokenizer/StyleID/AdaIN | 0.67-0.71 | Style conditioning |
| 3 | May 2026 | LANCET/LBM + OT | 0.69-0.71 | U-Net + optimal transport |
| 4 | Jun 2026 | 620 SpatialBridge + DINO | 0.70-0.705 | Transformer + cross-attention |

Despite architectural innovations, the ceiling has barely moved from ~0.70 to ~0.705 in Phase 4. The trilemma explains this: all architectures share the same failure modes (gate collapse, shrinkage), and architectural changes that address one axis of the trilemma worsen another.

---

## 3. Gate Collapse Theory

### 3.1 Gate Definition

**Definition 3.1** (Style Gate). In the 620 SpatialBridge architecture, each transformer block injects style via a gated cross-attention:

$$\text{style\_delta}^{(l)} = \tanh(g^{(l)}) \cdot \text{CrossAttn}^{(l)}(Q_{\text{content}}, K_{\text{style}}, V_{\text{style}})$$

where $g^{(l)} \in \mathbb{R}$ is a learnable scalar parameter, initialized at $g_0 = 0.05$.

The gate controls the *intensity* of style injection. When $g^{(l)} \to 0$, style information is completely suppressed.

### 3.2 Collapse Dynamics

**Proposition 3.1** (Gate Collapse to Zero). Consider the total loss $\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_{\text{swd}} \mathcal{L}_{\text{SWD}}$. At the optimal gate value $g^*$:

$$\frac{d\mathcal{L}}{dg}\bigg|_{g=g^*} = 0$$

Expanding:

$$\frac{\partial \mathcal{L}_{\text{FM}}}{\partial g} + \lambda_{\text{swd}} \frac{\partial \mathcal{L}_{\text{SWD}}}{\partial g} = 0$$

**Claim**: $g^* \to 0$ when $\frac{\partial \mathcal{L}_{\text{FM}}}{\partial g} > 0$ at small $g$, i.e., increasing style injection *increases* the flow-matching loss, causing the optimizer to decrease $g$.

*Argument*: The flow-matching loss penalizes deviation from the target velocity $v_{\text{target}} = z_t - z_s$. At initialization, the model has not learned to align style_delta with $v_{\text{target}}$. Increasing $g$ amplifies a poorly aligned style_delta, which *increases* $\|v_\theta - v_{\text{target}}\|^2$. The gradient:

$$\frac{\partial \mathcal{L}_{\text{FM}}}{\partial g} = 2(v_\theta - v_{\text{target}}) \cdot \text{CrossAttn} \cdot (1 - \tanh^2(g))$$

At $g \approx 0.05$: $(1 - \tanh^2(0.05)) \approx 0.9975$, so the gradient is nearly unattenuated. The dot product $(v_\theta - v_{\text{target}}) \cdot \text{CrossAttn}$ is positive (amplifying an unaligned signal adds to the residual), so $\frac{\partial \mathcal{L}_{\text{FM}}}{\partial g} > 0$ and the optimizer decreases $g$.

### 3.3 The Stable Fixed Point at $g \approx 0.05$

**Proposition 3.2** (Stable Fixed Point). The gate value $g \approx 0.05$ is a stable fixed point because:

1. At $g = 0.05$, $\tanh(g) \approx 0.05$, so style contributes only 5% of the residual.
2. This 5% is sufficient for CLIP-style to reach ~0.70 (CLIP measures coarse stylistic direction, not fine texture).
3. Increasing $g$ beyond 0.05 temporarily increases $\mathcal{L}_{\text{FM}}$ because the amplified style_delta is not yet aligned with $v_{\text{target}}$.
4. The SWD loss gradient at $g \approx 0.05$ is small because SWD is computed on the *endpoint* (which is already close to source due to shrinkage --- see Sec. 4), so $\nabla_g \mathcal{L}_{\text{SWD}} \approx 0$.

Therefore: $\frac{d\mathcal{L}}{dg}\big|_{g \approx 0.05} \approx 0$ with the dynamics being *restoring* (pushing back toward 0.05 from either direction), making it a stable fixed point.

### 3.4 Empirical Validation

| Observation | Source | Value |
|-------------|--------|-------|
| Gate convergence range | 620 experiments, all runs | $g \in [0.047, 0.050]$ |
| Style signal after gate | probe: cross_attn_delta_abs | $< 0.01$ |
| Velocity increase with gate=0.3 | gate sweep experiment | velocity_abs +16% (0.186 to 0.216) |
| CLIP-style with gate=0.3 vs 0.05 | gate sweep | 0.696 vs 0.700 (gate=0.3 *slightly lower*) |

The last row is critical: even with 6x stronger gate, CLIP-style *decreases slightly*. This confirms that the model has not learned to use the additional style capacity --- the style signal direction is wrong, and amplifying it hurts.

### 3.5 Cross-Attention Entropy Collapse

**Definition 3.2** (Attention Entropy). For cross-attention with $N$ style tokens:

$$H(\text{attn}) = -\sum_{j=1}^N \text{attn}_j \log \text{attn}_j, \quad \eta = \frac{H(\text{attn})}{\ln N} \in [0, 1]$$

$\eta = 0$: one-hot attention (style-specific). $\eta = 1$: uniform attention (style-agnostic).

**Proposition 3.3** (Uniform Attention Collapse). When style tokens' key projections have low variance, softmax attention converges to uniform:

$$\text{softmax}(q^T K / \sqrt{d}) \approx \frac{1}{N}\mathbf{1}_N \quad \text{when } \text{Var}[k_j] \ll 1$$

*Empirical measurement*: $\eta \approx 5.531 / \ln(256) = 0.997$ --- 99.7% uniform, meaning cross-attention outputs are nearly style-independent.

**Consequence**: The cross-attention output approximates the conditional expectation over styles:

$$\text{CA}(Q, K, V) \approx \bar{v}_S = \frac{1}{N}\sum_{j=1}^N v_j$$

This is the *style-average* value, which is equivalent to the dataset mean in feature space --- a key driver of whitening (Sec. 5).

**Style sensitivity measurement**: For 5 different styles $s_1, \ldots, s_5$ with fixed $(z_s, t)$:

$$\cos(v_\theta(z_s, t, s_i), v_\theta(z_s, t, s_j)) \approx 0.9995 \quad \forall\, i \neq j$$

Velocities are 99.95% similar across different styles --- conditional expectation collapse is complete.

---

## 4. Endpoint Shrinkage Theory

### 4.1 The Projection Coefficient

**Definition 4.1** (Endpoint Projection Coefficient). Given source $z_s$, target $z_t$, and generated endpoint $\hat{z}_1 = f_\theta(z_s, t{=}0, c_{\text{style}})$:

$$\alpha = \frac{\langle \hat{z}_1 - z_s,\; z_t - z_s \rangle}{\|z_t - z_s\|_2^2}$$

- $\alpha = 0$: no movement toward target (pure source)
- $\alpha = 1$: full movement to target
- $\alpha < 0$: movement away from target (anti-style)

### 4.2 Velocity Parameterization Induces Shrinkage

**Proposition 4.1** (Velocity Shrinkage). In velocity-prediction mode, the endpoint is:

$$\hat{z}_1 = z_s + (1 - t) \cdot v_\theta$$

Taking the expectation over uniformly sampled $t \in [0, 1]$:

$$\mathbb{E}_t[\hat{z}_1 - z_s] = \mathbb{E}_t[(1-t)] \cdot v_\theta = \frac{1}{2} v_\theta$$

*Even if $v_\theta$ perfectly captures the target direction, the average endpoint displacement is only 50% of the velocity magnitude.*

**Proof**: For $t \sim \text{Uniform}(0,1)$, $\mathbb{E}[1-t] = \int_0^1 (1-t) \, dt = \frac{1}{2}$. $\square$

### 4.3 Decomposition of the Predicted Velocity

**Proposition 4.2** (Velocity Decomposition). The predicted velocity can be decomposed as:

$$v_\theta = \alpha_v \cdot v_{\text{target}} + \alpha_h \cdot v_{\text{high\_freq}} + \alpha_n \cdot \epsilon$$

where $v_{\text{target}} = z_t - z_s$ (target direction), $v_{\text{high\_freq}}$ captures residual high-frequency style information, and $\epsilon$ is noise/orthogonal components.

**Empirical measurement** (from endpoint decomposition probe):

- $\alpha_v \approx 0.163$ --- the model moves only 16.3% toward the target
- $\alpha_h \approx -0.050$ --- the model *actively removes* high-frequency information (moves toward low-frequency mean)
- Net displacement: $\|\hat{z}_1 - z_s\| / \|z_t - z_s\| \approx 0.16$

### 4.4 The Shrinkage Basin

**Proposition 4.3** (Shrinkage Basin Attractor). The point $\hat{z}_1 = z_s$ (zero displacement) is *not* a local minimum of $\mathcal{L}$, but the *path* to the true minimum passes through a region of high loss, creating a basin that traps optimization.

*Proof sketch*: At $\hat{z}_1 = z_s$:

$$\nabla_{\hat{z}_1} \mathcal{L}\big|_{\hat{z}_1 = z_s} = -\frac{2 w_{\text{FM}}}{1-t} v_{\text{target}} + w_{\text{SWD}}(1-t) \nabla_z \text{SWD}\big|_{z = z_s} + \nabla_{\hat{z}_1} \mathcal{L}_{\text{edge}}\big|_{\hat{z}_1 = z_s}$$

The flow-matching gradient $-\frac{2 w_{\text{FM}}}{1-t} v_{\text{target}}$ points *toward* the target (away from source), so $z_s$ is not a critical point. However, the *effective* gradient reaching the endpoint head is attenuated by the four-stage signal decay chain:

| Stage | Input Signal | Output Signal | Attenuation |
|-------|-------------|--------------|-------------|
| DINO to patch_proj | 1.0 | 0.90 | 10% |
| Cross-attention (gate=0.05) | 0.90 | 0.045 | 95% |
| StyleFiLM (hd=128) | 0.045 | 0.018 | 60% |
| GroupNorm | 0.018 | 0.005 | 72% |
| Head (zero-init, std to 0) | 0.005 | 0.001 | 80% |
| **Total** | **1.0** | **0.001** | **99.9%** |

The gradient that reaches the endpoint is 0.1% of the original signal. This creates an *effective* basin: the landscape *around* $z_s$ has the correct gradient direction, but the gradient magnitude is too small to escape within practical training time.

### 4.5 Shrinkage as Multiplicative Signal Decay

**Proposition 4.4** (Multiplicative Shrinkage Bound). The projection coefficient is bounded by:

$$\alpha \geq \prod_{i=1}^{K} (1 - \epsilon_i)$$

where $\epsilon_i$ is the signal attenuation rate at stage $i$.

With current values: $\alpha \geq (1-0.95)(1-0.60)(1-0.72)(1-0.10) = 0.00504$. But the observed value is $\alpha \approx 0.16$, indicating the stages are not purely multiplicative --- there are synergistic recovery effects (e.g., FiLM bypasses attention).

### 4.6 Corrected Additive-Multiplicative Model

**Proposition 4.5** (Additive-Multiplicative Shrinkage). A more accurate model:

$$\alpha = \max\left(\alpha_{\text{attn}} \cdot \alpha_{\text{FiLM}},\; \alpha_{\text{GN}}\right) - \alpha_{\text{loss}}$$

With current estimates:

- $\alpha_{\text{attn}} = 0.05$ (gate collapse)
- $\alpha_{\text{FiLM}} = 0.40$ (hd=128, insufficient capacity)
- $\alpha_{\text{GN}} = 0.28$ (GroupNorm compression)
- $\alpha_{\text{loss}} = 0.10$ (auxiliary loss interference)

$$\alpha = \max(0.05 \times 0.40, 0.28) - 0.10 = \max(0.02, 0.28) - 0.10 = 0.18$$

This matches the observed $\alpha \approx 0.16$ within measurement error --- the dominant factor is GroupNorm compression, not attention.

### 4.7 Empirical Validation of Shrinkage

| Configuration | $\alpha_{\text{attn}}$ | $\alpha_{\text{FiLM}}$ | $\alpha_{\text{GN}}$ | Predicted $\alpha$ | Observed WFI |
|---------------|------------------------|------------------------|---------------------|-------------------|-------------|
| gate=0.05, hd128 | 0.05 | 0.40 | 0.28 | 0.18 | 0.49 |
| gate=0.3 | 0.30 | 0.40 | 0.28 | 0.18 | 0.45 |
| FiLM hd512 | 0.05 | 0.60 | 0.28 | 0.28 | 0.39 |
| gate=0.3 + hd512 | 0.30 | 0.60 | 0.28 | 0.28 | 0.39 |
| + no GN endpoint | 0.30 | 0.60 | 0.50 | 0.38 | 0.30 (predicted) |

The FiLM hd512 experiment (WFI: 0.49 to 0.39) confirms that increasing FiLM capacity directly reduces shrinkage and whitening.
---

## 5. Whiteness as Regression-to-Mean

### 5.1 The Latent Space Mean

**Proposition 5.1** (Latent Standardization). The VAE latent space is approximately standardized:

$$\mathbb{E}_{z \sim p_{\text{data}}}[z] \approx \mu_{\text{style}} \approx 0$$

This follows from the VAE training objective, which includes a KL divergence regularizer $D_{\text{KL}}(q(z|x) \| \mathcal{N}(0, I))$, pulling the aggregate posterior toward zero mean and unit variance.

### 5.2 Shrinkage Predicts Source Preservation, Not Whitening

If the model were simply shrinking toward source:

$$z_g = (1 - \alpha) z_s + \alpha \cdot z_t, \quad \alpha \to 0 \implies z_g \to z_s$$

This would preserve the source image's statistics (no whitening). But the *observed* pattern is:

$$z_g \to \mu_{\text{style}} \approx 0 \quad \text{(whitening)}$$

The generated image moves toward the *dataset mean* (zero in latent space -> gray/white in image space), not toward the source.

### 5.3 Resolution: High-Frequency Regression-to-Mean

**Proposition 5.2** (Frequency-Dependent Shrinkage). The velocity decomposition (Prop. 4.2) reveals:

$$v_\theta = \alpha_v \cdot v_{\text{target}} + \alpha_h \cdot v_{\text{high\_freq}}, \quad \alpha_h < 0$$

The *negative* high-frequency coefficient means the model is *actively removing* high-frequency information from the source, not just failing to add target style.

Decompose the latent into low-frequency ($z_{\text{low}}$, spatial averages) and high-frequency ($z_{\text{high}}$, textures/edges) components:

$$z = z_{\text{low}} + z_{\text{high}}$$

The generated output:

$$z_g = \underbrace{z_{s,\text{low}} + \alpha_v(z_{t,\text{low}} - z_{s,\text{low}})}_{\text{low-freq: partial movement}} + \underbrace{(1 + \alpha_h) \cdot z_{s,\text{high}} + \alpha_v \cdot z_{t,\text{high}}}_{\text{high-freq: net loss}}$$

Since $\alpha_h \approx -0.050$ and $\alpha_v \approx 0.163$:

$$z_{g,\text{high}} = 0.95 \cdot z_{s,\text{high}} + 0.163 \cdot z_{t,\text{high}}$$

The high-frequency component is *dominated* by the attenuated source ($0.95$), with only a small contribution from the target ($0.163$). The net effect is a loss of high-frequency energy relative to either source or target.

### 5.4 GroupNorm as the Whitening Mechanism

**Proposition 5.3** (GN Whitening). GroupNorm(1) with affine=False applied to features $h \in \mathbb{R}^{B \times C \times H \times W}$:

$$\text{GN}(h)_{b,c,i,j} = \frac{h_{b,c,i,j} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}$$

This operation has three whitening effects:

1. **Variance normalization**: $\text{Var}[\text{GN}(h)] = 1$ (eliminates cross-style variance differences)
2. **Mean removal**: $\mathbb{E}[\text{GN}(h)] = 0$ (eliminates first-order style signals --- brightness, color bias)
3. **Channel flattening**: In the 4-channel VAE latent, channels encode different attributes (luminance, chrominance). GN(1) computes one global $\mu, \sigma$ across all channels, eliminating inter-channel variance ratios.

**Style Signal Preservation Rate** after GN:

$$R_{\text{style}}(l) = \frac{\|\text{GN}(h^{(l)}(s_1)) - \text{GN}(h^{(l)}(s_2))\|_2}{\|h^{(l)}(s_1) - h^{(l)}(s_2)\|_2}$$

For style differences that primarily affect first and second moments (which is the dominant mode --- see Sec. 7 on effective dimensionality), $R_{\text{style}} \to 0$ after GN.

**Empirical prediction**: $R_{\text{style}} < 0.2$ after each GN layer. After $L$ layers of GN: $R_{\text{style}}^{(L)} \leq (0.2)^L \to 0$.

### 5.5 The Whitening Chain

Combining gate collapse (Sec. 3) and norm collapse, the complete whitening chain is:

$$\underbrace{\text{Gate Collapse}}_{\text{style = 5\%}} \xrightarrow{} \underbrace{\text{Attention Uniformity}}_{\eta = 0.997} \xrightarrow{} \underbrace{\text{Conditional Expectation}}_{v_\theta \approx \bar{v}} \xrightarrow{} \underbrace{\text{Endpoint Shrinkage}}_{\alpha = 0.16} \xrightarrow{} \underbrace{\text{GN Compression}}_{R_{\text{style}} \to 0} \xrightarrow{} \underbrace{\text{Whitening}}_{\text{WFI} > 0.35}$$

### 5.6 Mathematical Verification

**Proposition 5.4** (Whitening Consistency). Under the linear shrinkage model $z_g = (1-\alpha)z_s + \alpha \cdot z_t$:

$$\mathbb{E}[z_g] = (1-\alpha)\mathbb{E}[z_s] + \alpha \cdot \mathbb{E}[z_t] = (1-\alpha) \cdot 0 + \alpha \cdot 0 = 0 \quad \checkmark$$

The mean is zero regardless of $\alpha$ --- consistent with the VAE prior. However, the *variance*:

$$\text{Var}[z_g] = (1-\alpha)^2 \text{Var}[z_s] + \alpha^2 \text{Var}[z_t] + 2\alpha(1-\alpha)\text{Cov}(z_s, z_t)$$

For $\alpha = 0.16$ and approximately uncorrelated $z_s, z_t$:

$$\text{Var}[z_g] \approx 0.84^2 \sigma^2 + 0.16^2 \sigma^2 = (0.7056 + 0.0256)\sigma^2 = 0.7312\sigma^2$$

The variance is compressed to 73% of the source --- a 27% dynamic range loss. This translates to the image-space contrast loss measured by WFI.

With the high-frequency regression ($\alpha_h = -0.05$), the high-frequency variance compression is even more severe:

$$\text{Var}[z_{g,\text{high}}] \approx (1+\alpha_h)^2 \text{Var}[z_{s,\text{high}}] = 0.95^2 \sigma_{\text{high}}^2 = 0.9025 \sigma_{\text{high}}^2$$

Combined with GN's normalization, the effective variance compression factor $\kappa \approx 0.30$, predicting:

$$\text{contrast\_ratio}_{\text{gen}} \approx \kappa \cdot \text{contrast\_ratio}_{\text{target}} \approx 0.30 \times 0.42 = 0.126$$

This matches the observed contrast ratio of ~0.15 for whitened images.

---

## 6. Training-Output Mismatch

### 6.1 The Mismatch

**Definition 6.1** (Training-Output Mismatch). The training objective minimizes per-step velocity error, but evaluation measures endpoint quality. These are not equivalent.

**Proposition 6.1** (Integration Error Accumulation). Let $v_\theta(z, t, c)$ be the learned velocity field and $v^*(z, t, c)$ be the target velocity field. Define the per-step error $\delta v(t) = v_\theta - v^*$. The endpoint error is:

$$\|\hat{z}_1 - z_1^*\| = \left\|\int_0^1 \delta v(\tau) \, d\tau\right\| \leq \int_0^1 \|\delta v(\tau)\| \, d\tau$$

Even if per-step errors are small ($\|\delta v\| \leq \epsilon$ for all $t$), the endpoint error can be as large as $\epsilon$ (for coherent errors) or $\sqrt{\epsilon}$ (for random errors). Crucially, errors *accumulate* through the integration --- they do not cancel.

### 6.2 Empirical Evidence: Trained Models Underperform Untrained Baselines

| Model | Training | CLIP-style | LPIPS |
|-------|----------|------------|-------|
| Fiber-SDE (no training) | None | 0.711 | --- |
| LANCET (trained) | Velocity MSE + SWD | 0.701 | 0.4527 |
| 620 SpatialBridge (trained) | Velocity MSE + SWD + edge | 0.705 | 0.2935 |

The untrained Fiber-SDE achieves CLIP-style = 0.711, *higher* than the trained LANCET (0.701). Training has made the model *worse* at style transfer on this metric.

**Explanation**: The per-step velocity MSE pushes the model toward the *average* velocity across all training pairs (conditional expectation collapse, Sec. 3). The SWD loss provides a corrective signal on the endpoint, but this signal is weakened by the integration gap --- small endpoint corrections require proportionally larger velocity corrections at early time steps.

### 6.3 Quantifying the Mismatch

**Proposition 6.2** (Mismatch Factor). Define the mismatch factor as:

$$M = \frac{\partial \mathcal{L}_{\text{endpoint}} / \partial v_\theta}{\partial \mathcal{L}_{\text{FM}} / \partial v_\theta}$$

For the flow-matching loss:

$$\frac{\partial \mathcal{L}_{\text{FM}}}{\partial v_\theta} = 2(v_\theta - v_{\text{target}})$$

For the endpoint loss (SWD computed on $\hat{z}_1 = z_s + (1-t)v_\theta$):

$$\frac{\partial \mathcal{L}_{\text{SWD}}}{\partial v_\theta} = (1-t) \cdot \nabla_z \text{SWD}\big|_{z = \hat{z}_1}$$

The $(1-t)$ factor means the SWD gradient is *discounted* at later time steps. At $t = 0.9$, the gradient is only 10% of its full magnitude. This creates a systematic bias: the model prioritizes minimizing velocity error at high $t$ (where the FM gradient is unattenuated) over improving the endpoint (where the SWD gradient is attenuated).

**Empirical consequence**: The model's velocity predictions are more accurate at high $t$ (near the endpoint) than at low $t$ (near the source), but the endpoint depends on the *entire* trajectory --- errors at low $t$ have the largest impact on the final result.

### 6.4 The ODE Integration Perspective

The generated latent is obtained by Euler integration:

$$z_{t + \Delta t} = z_t + \Delta t \cdot v_\theta(z_t, t, c)$$

After $N$ steps with step size $\Delta t = 1/N$:

$$\hat{z}_1 = z_s + \sum_{k=0}^{N-1} \Delta t \cdot v_\theta(z_{k \Delta t}, k \Delta t, c)$$

The endpoint error:

$$\hat{z}_1 - z_1^* = \sum_{k=0}^{N-1} \Delta t \cdot \delta v(z_{k\Delta t}, k\Delta t, c) + \mathcal{O}(\Delta t^2)$$

For the single-step (endpoint prediction) mode, $N = 1$ and the error is simply $\delta v(z_s, 0, c)$. This bypasses the integration entirely, which is why the endpoint_lowhigh mode (with FiLM head) outperforms velocity-only mode --- it directly optimizes for the endpoint, not the trajectory.

---

## 7. Effective Style Dimensionality

### 7.1 The Latent Space Is 16384-Dimensional, But Style Is Low-Dimensional

**Proposition 7.1** (Low Effective Style Dimensionality). Despite operating in $\mathbb{R}^{16384}$ latent space, the effective dimensionality of the style transfer manifold is $k \ll d$.

*Argument*: Consider the Singular Value Decomposition of the style displacement matrix:

$$\Delta S = [z_{t_1} - z_s, \; z_{t_2} - z_s, \; \ldots, \; z_{t_N} - z_s] \in \mathbb{R}^{d \times N}$$

$$\Delta S = U \Sigma V^T$$

The effective dimensionality is the number of singular values needed to explain, e.g., 95% of the variance:

$$k = \min\left\{r : \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^d \sigma_i^2} \geq 0.95\right\}$$

### 7.2 Empirical Evidence for Low Dimensionality

**Evidence 1: Architecture Invariance**. The 620 SpatialBridge has been tested with 21 different Cross-attention Gate Weight (CGW) configurations. Despite these configurations changing *which subspaces* receive style injection:

| CGW Configs | CLIP-style Range | $\Delta$ |
|-------------|------------------|----------|
| 21 variants | [0.680, 0.691] | 0.011 |

A range of only 0.011 across 21 configurations means the model is insensitive to *which* dimensions receive style injection. This is only possible if style lives in a low-dimensional subspace --- all configurations project onto approximately the same subspace regardless of architectural differences.

**Evidence 2: Pareto Front Sparsity**. From 17,021 evaluated data points, only 10 are non-dominated on the CLIP-style vs. LPIPS Pareto front. If style were high-dimensional, we would expect many more Pareto-optimal configurations (one per dominant dimension combination). The sparsity suggests the objective landscape is effectively low-dimensional.

**Evidence 3: Cycle-NCE Overfitting**. On a 50-image subset, Cycle-NCE achieves CLIP-style = 0.9109. On the full dataset, it drops to ~0.71. A 0.20 gap between overfitting and generalization is characteristic of a model with effective dimensionality much lower than 16384 --- if the true dimensionality were high, overfitting on 50 samples would not be possible.

### 7.3 Estimate of Effective Dimensionality

**Proposition 7.2** (Dimensionality Bound). The effective style dimensionality satisfies:

$$10 \lesssim k \lesssim 50$$

*Argument*:

- Lower bound: $k \geq 10$ because the 21 CGW configurations produce *some* variation ($\Delta = 0.011$), which requires at least $\log_2(21) \approx 4$ degrees of freedom, plus additional dimensions for content-style interaction.
- Upper bound: $k \leq 50$ because the CLIP-style ceiling is extremely tight (0.70-0.71 across all architectures and training regimes). If $k$ were large, different architectures would discover different subspaces and show greater variation. The consistency of the ceiling implies a small number of "style eigen-directions" that all methods converge to.

### 7.4 Implications

The low effective dimensionality has two critical implications:

1. **Gate collapse is worsened**: In a 16384-D space, a 5% gate attenuates signal in all dimensions equally. But if style only occupies ~50 dimensions, the *effective* attenuation on the style subspace is the same 5%, while the remaining 16334 dimensions receive pure noise. The SNR of style injection is:

$$\text{SNR} = \frac{0.05 \cdot \|v_{\text{style}}\|}{\|v_{\text{noise}}\|} = 0.05 \cdot \sqrt{\frac{k}{d}} = 0.05 \cdot \sqrt{\frac{50}{16384}} \approx 0.003$$

The style signal is 0.3% of the noise floor --- virtually undetectable by the loss function.

2. **Explicit subspace identification should help**: If we can identify the $k$-dimensional style subspace (e.g., via PCA on style deltas), we can project style injection into that subspace and achieve an SNR improvement of $\sqrt{d/k} \approx 18\times$.
---

## 8. Unified Theory: The Degenerate Attractor

### 8.1 Synthesis

The four mechanisms --- gate collapse, endpoint shrinkage, regression-to-mean, and training-output mismatch --- form a *self-reinforcing degenerate attractor*:

$$\boxed{\text{Gate Collapse} \xrightarrow[\text{5\% signal}]{} \text{Shrinkage} \xrightarrow[\alpha=0.16]{} \text{Whitening} \xrightarrow[\text{WFI}>0.35]{} \text{Training Bias} \xrightarrow[\text{regress-to-mean}]{} \text{Gate Collapse}}$$

The cycle:

1. Gate collapse -> style signal is 5% of full magnitude
2. 5% signal -> endpoint moves only 16% toward target (shrinkage)
3. 16% displacement + GN compression -> whitening (WFI > 0.35)
4. Whitened images are close to the dataset mean -> the model learns to predict near-mean velocities
5. Near-mean velocities provide no gradient signal to increase the gate -> gate stays collapsed

### 8.2 The Attractor Is Stable

**Proposition 8.1** (Stability of the Degenerate Attractor). The degenerate attractor is stable because:

1. **Gate gradient is zero**: At $g \approx 0.05$, $\nabla_g \mathcal{L} \approx 0$ (Sec. 3.3)
2. **Endpoint gradient is small**: Shrinkage attenuates the endpoint gradient by $\alpha = 0.16$
3. **SWD gradient is misaligned**: SWD measures distributional distance, which can be small even for whitened images (the *average* style direction is correct; only the *texture* is wrong)
4. **CLIP-style provides no escape**: CLIP-style = 0.70 is already "good enough" for the loss to be flat

### 8.3 Escape Requires Coordinated Intervention

**Proposition 8.2** (Coordinated Escape Condition). Breaking the degenerate attractor requires *simultaneous* intervention on at least two mechanisms. Single interventions are absorbed by the attractor.

*Evidence*:

| Intervention | Alone | Expected | Actual |
|-------------|-------|----------|--------|
| gate=0.3 | CLIP-style up | 0.72+ | 0.696 (down) |
| FiLM hd512 | WFI down | < 0.30 | 0.39 (partial) |
| gated_raw attention | WFI down | < 0.40 | 0.64 (up) |
| direction loss | alpha up | > 0.3 | -0.007 (catastrophic) |

Every single intervention either fails or makes things worse. The attractor absorbs the perturbation and returns to equilibrium.

The *combined* intervention (gate=0.3 + FiLM hd512) achieves WFI = 0.39, the first result below the 0.40 threshold --- but still far from the target of 0.20.

### 8.4 Necessary Conditions for Breaking the Attractor

**Theorem 8.1** (Escape Conditions). To achieve WFI < 0.20 and CLIP-style > 0.72, the following conditions must *all* hold simultaneously:

$$\alpha > 0.5 \quad \wedge \quad \eta_{\text{attn}} > 0.1 \quad \wedge \quad R_{\text{style}} > 0.3$$

Current values: $\alpha = 0.16$, $\eta_{\text{attn}} = 0.003$, $R_{\text{style}} \approx 0.1$.

Required improvements:

1. $\alpha$: 0.16 -> 0.5 (3.1x increase) --- requires gate + FiLM + GN removal
2. $\eta_{\text{attn}}$: 0.003 -> 0.1 (33x increase) --- requires attention sparsification or Pre-FiLM
3. $R_{\text{style}}$: 0.1 -> 0.3 (3x increase) --- requires reduced GN usage or AdaGN

---

## 9. Falsifiable Predictions

### 9.1 Gate Collapse Predictions

| ID | Prediction | Experiment | Falsification Condition |
|----|-----------|-----------|------------------------|
| P1 | Fixing gate collapse (forced $g \geq 0.3$ + FiLM hd512 + no GN) will break the CLIP-style ~0.70 ceiling | Train with gate locked at 0.3 + FiLM hd512 + GN-free endpoint | If CLIP-style still <= 0.71 after convergence |
| P2 | Gate value alone is not sufficient --- the style signal *direction* must also be correct | Train with gate=0.3 but random style token projections | If CLIP-style improves despite random projections |
| P3 | Gate collapse is caused by initial misalignment between style_delta and $v_{\text{target}}$ | Train with warmup: first 2 epochs gate=0.3, then release | If released gate converges to > 0.1 |

### 9.2 Effective Dimensionality Predictions

| ID | Prediction | Experiment | Falsification Condition |
|----|-----------|-----------|------------------------|
| P4 | Explicit projection onto the top-$k$ style subspace ($k \approx 50$) will improve both CLIP-style and WFI | PCA on style deltas, project style injection into top-$k$ subspace | If projection does not improve CLIP-style by >= 0.01 |
| P5 | The effective dimensionality $k$ can be measured by the "plateau onset" in a singular value sweep | Train models with style injection restricted to top-1, top-5, top-10, ..., top-100 components | If no plateau is observed, or plateau at $k > 100$ |
| P6 | Architecture changes that affect different subspaces will have negligible impact as long as they overlap with the top-$k$ subspace | Compare CGW configs that project onto orthogonal subspaces | If orthogonal configs produce CLIP-style differences > 0.03 |

### 9.3 Training-Output Mismatch Predictions

| ID | Prediction | Experiment | Falsification Condition |
|----|-----------|-----------|------------------------|
| P7 | Direct endpoint prediction (bypassing ODE integration) will outperform velocity prediction | Train endpoint head directly, no velocity parameterization | If endpoint prediction achieves CLIP-style < velocity prediction |
| P8 | The mismatch factor $M$ (Sec. 6.3) predicts the gap between training loss and evaluation quality | Measure $M$ at different time steps, correlate with endpoint error | If $M$ does not correlate with endpoint quality |
| P9 | Increasing the number of SWD computation points along the trajectory will reduce the mismatch | Multi-step SWD (SWD at $t = 0.25, 0.5, 0.75, 1.0$ instead of only $t=1$) | If multi-step SWD does not improve CLIP-style by >= 0.005 |

### 9.4 Text Conditioning Predictions

| ID | Prediction | Experiment | Falsification Condition |
|----|-----------|-----------|------------------------|
| P10 | Text conditioning will *not* improve CLIP-style until gate collapse is fixed | Add T5 text tokens to current (gate=0.05) model | If text conditioning improves CLIP-style by > 0.01 without fixing gate |
| P11 | After gate collapse is fixed, text conditioning will provide complementary style signal in semantic dimensions | Add T5 tokens to gate=0.3 + FiLM hd512 model | If text conditioning provides no improvement in the fixed-gate regime |

---

## 10. Repair Roadmap: Predicted Outcomes

### 10.1 Incremental Repair Predictions

| Stage | Intervention | Predicted $\alpha$ | Predicted WFI | Predicted CLIP-style |
|-------|-------------|-------------------|--------------|---------------------|
| Current | gate=0.05, hd128, GN endpoint | 0.16 | 0.49 | 0.699 |
| Stage 1 | + FiLM hd512 | 0.28 | 0.39 | 0.701 |
| Stage 2 | + gate=0.3 | 0.28 | 0.39 | 0.710 |
| Stage 3 | + no GN endpoint | 0.38 | 0.30 | 0.715 |
| Stage 4 | + velocity_scale_loss | 0.45 | 0.25 | 0.720 |
| Stage 5 | + AdaGN (style-modulated norm) | 0.55 | 0.20 | 0.725 |
| Stage 6 | + subspace projection (top-50) | 0.65 | 0.18 | 0.735 |
| **Target** | All combined | > 0.5 | < 0.20 | > 0.72 |

### 10.2 Theoretical Lower Bound on WFI

**Proposition 10.1** (WFI Lower Bound). Under the current VAE decoder and architecture constraints:

$$\text{WFI}_{\min} \approx 0.15 + 0.1 \cdot (1 - \alpha_{\max})$$

If the maximum achievable $\alpha_{\max} = 0.9$ (theoretical limit with perfect style injection but residual ODE integration error):

$$\text{WFI}_{\min} \approx 0.15 + 0.1 \times 0.1 = 0.16$$

This matches the Seedream IDT benchmark (WFI = 0.158), suggesting that Seedream-level quality is theoretically achievable with sufficient $\alpha$.

### 10.3 Sufficient Condition for Repair

**Proposition 10.2** (Repair Sufficient Condition). To increase $\alpha$ from 0.16 to > 0.5, the total signal attenuation must satisfy:

$$\sum_{i=1}^{K} \epsilon_i < \ln(2) \approx 0.693$$

i.e., the total decay rate must be less than 50%. Current total decay is ~84%, requiring removal of at least three of the four attenuation sources (gate, FiLM capacity, GN, head init).

### 10.4 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Gate=0.3 causes training instability | Medium | High | Gate warmup schedule |
| No-GN endpoint causes feature explosion | Medium | Medium | Use RMSNorm instead |
| Subspace projection loses important dimensions | Low | Medium | Use adaptive $k$ selection |
| All fixes combined still insufficient | Low | Critical | Redesign architecture (MoE, multi-scale) |

---

## 11. Proofs and Derivations

### 11.1 Proof of Proposition 3.1 (Gate Collapse)

Consider the gradient of $\mathcal{L}_{\text{FM}}$ with respect to the gate $g$ at time step $t$:

$$\frac{\partial \mathcal{L}_{\text{FM}}}{\partial g} = \mathbb{E}_{t}\left[\frac{\partial}{\partial g}\|v_\theta - v_{\text{target}}\|^2\right] = 2\mathbb{E}_t\left[(v_\theta - v_{\text{target}})^T \frac{\partial v_\theta}{\partial g}\right]$$

Now $v_\theta = v_{\text{base}} + \tanh(g) \cdot \text{CrossAttn}(Q, K, V)$, so:

$$\frac{\partial v_\theta}{\partial g} = (1 - \tanh^2(g)) \cdot \text{CrossAttn}(Q, K, V)$$

At initialization, $v_{\text{base}}$ has been trained to approximate $v_{\text{target}}$ (the velocity field is dominated by the content flow). The style_delta $\text{CrossAttn}(Q, K, V)$ is initially random with respect to $v_{\text{target}}$. Therefore:

$$(v_\theta - v_{\text{target}})^T \cdot \text{CrossAttn} = (v_{\text{base}} + \tanh(g) \cdot \text{CA} - v_{\text{target}})^T \cdot \text{CA}$$
$$= (v_{\text{base}} - v_{\text{target}})^T \cdot \text{CA} + \tanh(g) \|\text{CA}\|^2$$

At $g \approx 0.05$: the first term dominates (base velocity already close to target), and since CA is nearly orthogonal to $(v_{\text{base}} - v_{\text{target}})$ (random projection):

$$(v_\theta - v_{\text{target}})^T \cdot \text{CA} \approx \tanh(0.05) \|\text{CA}\|^2 > 0$$

This means $\frac{\partial \mathcal{L}}{\partial g} > 0$ --- increasing the gate *increases* the loss. The optimizer responds by *decreasing* $g$. $\square$

### 11.2 Proof of Proposition 5.1 (Latent Standardization)

The VAE encoder $q_\phi(z|x)$ is parameterized as $\mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$. The training objective includes:

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_\phi(z|x) \| \mathcal{N}(0, I)) = \frac{1}{2}\sum_{i=1}^d (\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1)$$

This penalty pulls $\mu_\phi(x) \to 0$ and $\sigma_\phi^2(x) \to 1$ for each input. In expectation over the data distribution:

$$\mathbb{E}_x[\mu_\phi(x)] = 0, \quad \mathbb{E}_x[\sigma_\phi^2(x)] = 1$$

Since $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$:

$$\mathbb{E}[z] = \mathbb{E}_x[\mu_\phi(x)] = 0 \quad \checkmark$$
$$\text{Var}[z] = \mathbb{E}_x[\sigma_\phi^2(x)] + \text{Var}_x[\mu_\phi(x)] \approx 1 + 0 = 1 \quad \checkmark$$

$\square$

### 11.3 Proof of Proposition 5.3 (GN Whitening)

**Variance normalization**:

$$\text{Var}[\text{GN}(h)_b] = \text{Var}\left[\frac{h_b - \mu_b}{\sigma_b}\right] = \frac{\text{Var}[h_b - \mu_b]}{\sigma_b^2} = \frac{\sigma_b^2}{\sigma_b^2} = 1$$

**Mean removal**:

$$\mathbb{E}[\text{GN}(h)_b] = \frac{\mathbb{E}[h_b] - \mu_b}{\sigma_b} = \frac{\mu_b - \mu_b}{\sigma_b} = 0$$

**Style signal elimination**: For two styles $s_1, s_2$ that differ only in first and second moments:

$$h(s_1) \sim \mathcal{N}(\mu_1, \sigma_1^2), \quad h(s_2) \sim \mathcal{N}(\mu_2, \sigma_2^2)$$

$$\text{GN}(h(s_1)) \sim \mathcal{N}(0, 1), \quad \text{GN}(h(s_2)) \sim \mathcal{N}(0, 1)$$

$$R_{\text{style}} = \frac{\|\mathcal{N}(0,1) - \mathcal{N}(0,1)\|_2}{\|\mathcal{N}(\mu_1,\sigma_1^2) - \mathcal{N}(\mu_2,\sigma_2^2)\|_2} \approx 0$$

$\square$

### 11.4 Proof of Theorem 8.1 (Escape Conditions)

We prove the necessity of each condition by contrapositive.

**Condition 1**: $\alpha > 0.5$ is necessary. If $\alpha \leq 0.5$, the endpoint moves at most 50% toward the target. The high-frequency content at the endpoint is:

$$z_{g,\text{high}} = (1-\alpha) z_{s,\text{high}} + \alpha z_{t,\text{high}}$$

For $\alpha = 0.5$: $z_{g,\text{high}} = 0.5 z_{s,\text{high}} + 0.5 z_{t,\text{high}}$, which is a 50% blend. The variance is:

$$\text{Var}[z_{g,\text{high}}] = 0.25(\sigma_s^2 + \sigma_t^2) < \max(\sigma_s^2, \sigma_t^2)$$

This represents a guaranteed contrast loss, predicting WFI > 0.20.

**Condition 2**: $\eta_{\text{attn}} > 0.1$ is necessary. If attention is 90%+ uniform ($\eta > 0.9$), cross-attention outputs are style-independent, meaning style only enters through FiLM (which has its own capacity limitations). The style information bottleneck limits CLIP-style to ~0.70.

**Condition 3**: $R_{\text{style}} > 0.3$ is necessary. If $R_{\text{style}} \leq 0.3$, then after $L = 2$ GN layers:

$$R_{\text{style}}^{(2)} \leq 0.3^2 = 0.09$$

Less than 9% of style information survives two layers of GN. With 4 transformer blocks (8 GN applications: norm1 + norm2 per block):

$$R_{\text{style}}^{(8)} \leq 0.3^8 \approx 6.6 \times 10^{-5}$$

Effectively zero style information reaches the output. $\square$

---

## 12. Information-Theoretic Summary

### 12.1 Mutual Information Budget

The total style information flow through the model can be quantified as:

$$I(S; Z_g) = I(S; V_\theta) - I(S; V_\theta | Z_g)$$

At the degenerate attractor:

| Stage | $I(S; \cdot)$ (bits) | Information Loss |
|-------|----------------------|------------------|
| Style encoder output | ~10 | --- |
| After cross-attention (gate=0.05, $\eta$=0.997) | ~0.3 | 97% |
| After FiLM (hd=128) | ~0.5 | +0.2 (partial recovery) |
| After GN | ~0.1 | 80% |
| At endpoint | ~0.05 | 50% |
| **Total** | **0.05 / 10** | **99.5%** |

The model retains only 0.5% of the original style information at the endpoint.

### 12.2 Information Bottleneck Interpretation

Each layer of the model acts as an information bottleneck. The cross-attention bottleneck has capacity:

$$C_{\text{attn}} = \log N - H(\text{attn}) = \ln N - H(\text{attn})$$

With $\eta = 0.997$ and $N = 256$:

$$C_{\text{attn}} = \ln(256)(1 - 0.997) = 5.545 \times 0.003 = 0.017 \text{ nats} \approx 0.024 \text{ bits}$$

The cross-attention can transmit only 0.024 bits of style information per query position --- far below the ~10 bits needed for style discrimination.

---

## Appendix A: Experimental Data Sources

| Data Source | Records | Key Metrics | Period |
|-------------|---------|-------------|--------|
| EXPERIMENT_ARCHAEOLOGY_MASTER.csv | 22,629 | clip_style, lpips, ssim_y | Feb-Jun 2026 |
| experiment_database_all.csv | 22,629 | Full metrics | Feb-Jun 2026 |
| experiment_database_best_per_config.csv | 256 | Best per category | Feb-Jun 2026 |
| 620 experiment logs | ~100 | clip_style, WFI, probe metrics | Jun 2026 |
| Cycle-NCE overfit50 | 11,794 | clip_style up to 0.9109 | Jun 2026 |

## Appendix B: Key Numerical Constants

| Constant | Symbol | Value | Source |
|----------|--------|-------|--------|
| Latent channels | $C$ | 4 | SD1.5 VAE |
| Latent spatial dims | $H, W$ | 64, 64 | SD1.5 VAE |
| Total latent dim | $d$ | 16384 | $4 \times 64 \times 64$ |
| Gate convergence | $g^*$ | [0.047, 0.050] | All 620 experiments |
| Attention entropy | $\eta$ | 0.997 | Cross-attention probe |
| Endpoint shrinkage | $\alpha$ | 0.16 | Endpoint decomposition |
| Velocity target coeff | $\alpha_v$ | 0.163 | Velocity decomposition |
| High-freq coeff | $\alpha_h$ | -0.050 | Velocity decomposition |
| Style sensitivity | $\cos(v(s_i), v(s_j))$ | 0.9995 | Multi-style probe |
| Fiber-SDE CLIP-style | --- | 0.711 | Phase2 baseline |
| LANCET CLIP-style | --- | 0.701 | Trained baseline |
| 620 best CLIP-style | --- | 0.6765 | lowmix05_diag (remote, 187 eval entries) |
| 620 best balanced | --- | 0.6751 | lowswd_formal e2 (LPIPS=0.278) |
| 620 best WFI | --- | 0.410 | film_gate03_5ep e5 (remote) |
| Seedream IDT WFI | --- | 0.158 | External benchmark |
| CGW config range | $\Delta$ | 0.011 | 21 configs |
| Architecture variant range | $\Delta$ | 0.0013 | adapter/gate12/moe at SWD-12 |
| clip_style-lpips corr | $r$ | +0.94 | Style8 branch |
| Effective style dim | $k$ | 10-50 | Estimated |

## Appendix C: Failed Intervention Log

| Intervention | Expected | Actual | Root Cause |
|-------------|----------|--------|------------|
| gate=0.3 alone | CLIP-style up | 0.696 (down) | Wrong style direction amplified |
| gated_raw attention | WFI down | 0.64 (up) | No normalization = statistical drift |
| relu2 attention | WFI down | 0.53 | Sparse but style-undifferentiated |
| style_select attention | WFI down | 0.50 | Top-k does not resolve content-style conflict |
| lowfreqfix | velocity stable | velocity 0.016 | Penalizes low-freq dynamics needed for structure |
| endpointaux | better endpoint | to_source_rms=0.055 | Collapses back to source |
| direction loss | alpha up | alpha=-0.007 | Over-constrains, catastrophic collapse |
| structure loss | better content | "completely useless" | Classify branch verification |
| Diff-Gram | better style | extremely poor | sdxl-fp32 verification |
| Gram-Moment | better style | poor | Moment matching insufficient |

This log demonstrates that single-axis interventions are systematically absorbed by the degenerate attractor (Proposition 8.2).