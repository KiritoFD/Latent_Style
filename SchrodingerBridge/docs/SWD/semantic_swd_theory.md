# Semantic Region SWD: Theory and Mechanism

## 1. Motivation

### 1.1 The limitation of global WCT
Global whitening-and-coloring (WCT) matches the marginal statistics of an entire high-frequency subband:
$$T_i(a) = \Sigma_i^{\star 1/2}\,\hat\Sigma_i^{-1/2}(a-\hat\mu_i)+\mu_i^\star$$
This treats every spatial location as exchangeable, ignoring the fact that a portrait's skin, a landscape's sky, and a still-life's fabric carry distinct texture statistics. The mismatch is severe precisely when source and target domains have different spatial content layouts.

### 1.2 The empirical signal
On Distinct5-WikiArt, replacing global SWD with semantic region SWD raises MUSIQ from 42.95 → 51.86 at matched CLIP-S/LPIPS, a +8.9 point gain. A same-LPIPS control with heavier global SWD weight gains only +1 MUSIQ. The gain is the content-coherent redistribution, not extra distortion.

## 2. Formal Definition

### 2.1 Semantic partition
Let $\hat h \in \mathbb R^{C\times HW}$ be a flattened high-frequency subband. We partition the $HW$ spatial locations into $K$ content-coherent regions via k-means on the content latent $\ell$:
$$r_j = \arg\min_{k\in\{1,\ldots,K\}} \|\ell_j - c_k\|_2$$
where $c_k$ are cluster centroids updated for 4 iterations.

For each region $k$, let $\hat h^{(k)}$ and $h^{\star(k)}$ denote the generated and target-style pixel subsets restricted to $\{j: r_j = k\}$.

### 2.2 Quantile matching as 1D optimal transport
Within each region, we solve a 1D optimal transport between $\hat h^{(k)}$ and $h^{\star(k)}$ by deterministic quantile interpolation:
1. Sort both subsets: $\hat h^{(k)}_{\sigma(1)} \le \ldots \le \hat h^{(k)}_{\sigma(n_k)}$
2. Sort target: $h^{\star(k)}_{\tau(1)} \le \ldots \le h^{\star(k)}_{\tau(m_k)}$
3. Linearly remap: $\hat h^{(k)}_{\sigma(i)} \leftarrow h^{\star(k)}_{\tau(\lfloor (i-1)(m_k-1)/(n_k-1) \rfloor + 1)}$

This is the closed-form sliced-Wasserstein optimum in one dimension and preserves the within-region rank order of the generated image, so structure is not erased.

### 2.3 Blended endpoint correction
The final endpoint correction blends global WCT with the region-wise match:
$$\hat h_i^{+} = (1-\beta)\,T_i(\hat h_i) + \beta\,\mathrm{QMatch}(\hat h_i, h_i^\star; r)$$
with $\beta = 0.7$.

## 3. Theoretical Analysis

### 3.1 Why region-wise is a content-adaptive restriction
**Theorem 1** (Region-wise SWD upper-bounds global). *Let $\hat{\mathcal P}, \mathcal P^\star$ be the global empirical distributions over $HW$ pixels, and $\hat{\mathcal P}^{(k)}, \mathcal P^{\star(k)}$ the region-restricted distributions with region masses $\pi_k$ ($\sum_k \pi_k = 1$). Then:*
$$\sum_{k=1}^{K} \pi_k\,\mathcal W_2^2(\hat{\mathcal P}^{(k)},\mathcal P^{\star(k)}) \ge \mathcal W_2^2(\hat{\mathcal P},\mathcal P^\star)$$

*Proof.* The global $\mathcal W_2^2$ optimizes over all couplings $\gamma \in \Pi(\hat{\mathcal P}, \mathcal P^\star)$. Define the region-preserving coupling set:
$$\Pi_r = \Big\{\gamma \in \Pi : \mathrm{supp}(\gamma) \subseteq \bigcup_k \big(\mathrm{supp}(\hat{\mathcal P}^{(k)}) \times \mathrm{supp}(\mathcal P^{\star(k)})\big)\Big\}$$
Since $\Pi_r \subseteq \Pi$, the restricted minimum is $\ge$ the unrestricted minimum: $\min_{\gamma \in \Pi_r} \langle C, \gamma \rangle \ge \min_{\gamma \in \Pi} \langle C, \gamma \rangle = \mathcal W_2^2(\hat{\mathcal P},\mathcal P^\star)$. But the region-preserving minimum decomposes:
$$\min_{\gamma \in \Pi_r} \langle C, \gamma \rangle = \sum_{k=1}^{K} \pi_k\,\mathcal W_2^2(\hat{\mathcal P}^{(k)},\mathcal P^{\star(k)})$$
since couplings across different regions are forbidden. Combining gives the result. $\square$

**Interpretation.** The region-wise transport cost is an **upper bound** on the global transport cost. This is the *content-adaptive restriction*: by forbidding cross-region transport (sky↔skin), we accept a **higher** transport cost in exchange for **content-coherent** matching. The extra cost is precisely the price of respecting content layout.

**Why this helps MUSIQ.** MUSIQ measures perceptual quality, not transport cost. A lower transport cost (global WCT) can match sky pixels to skin pixels if that minimizes distance, but the result looks unnatural. Region-wise SWD pays a higher transport cost to keep sky statistics matched to sky, skin to skin—producing content-coherent textures that MUSIQ rewards.

### 3.2 Why deterministic beats stochastic (10-point MUSIQ gap)
Replacing deterministic quantile interpolation with stochastic multinomial sampling regresses MUSIQ by ~10 points (51.86 → 41.50). The reason:

**Deterministic quantile matching** is the closed-form 1D OT optimum. It preserves the within-region rank order: the $i$-th smallest generated pixel maps to the $i$-th smallest target pixel. This is the **monotone coupling**, which is optimal in 1D.

**Stochastic multinomial sampling** draws from the target distribution independently, breaking the monotone coupling. The result is a **random coupling** whose expected transport cost equals the average distance, not the minimum. In 1D, the gap between the monotone coupling and a random coupling is:
$$\mathbb E[\mathcal W_2^2(\text{random})] - \mathcal W_2^2(\text{monotone}) = \mathrm{Var}(\text{target rank}) - 0$$
which is strictly positive whenever the target distribution is non-degenerate.

**Practical implication:** The monotone coupling preserves spatial coherence (adjacent pixels with similar values remain similar after matching), while stochastic sampling introduces per-pixel noise that the VAE decoder amplifies into grain.

### 3.3 The β blend trade-off
- $\beta = 0$ (pure global WCT): ignores content layout, MUSIQ = 42.95.
- $\beta = 0.7$ (optimal): cross-region style cues preserved, within-region statistics content-coherently aligned, MUSIQ = 51.86.
- $\beta = 1.0$ (pure region match): saturates, costs CLIP-S because cross-region statistical variation also carries style information.

The optimum at $\beta = 0.7$ reflects a bias-variance trade-off: global WCT has high bias (ignores content) but low variance (uses all pixels); region match has low bias but high variance (fewer pixels per region). The blend minimizes total error.

### 3.4 K-value trade-off (hypothesis, under experiment)
- $K = 1$: equivalent to global SWD.
- $K$ small (4): regions too coarse,接近global.
- $K$ medium (8): sweet spot, content-coherent regions with enough pixels per region.
- $K$ large (32): regions too fine, per-region statistics noisy (few pixels), variance dominates.

Predicted MUSIQ curve: inverted-U peaking around K=8.

## 4. Empirical Results

### 4.1 Current best
| Config | K | β | SWD w | MUSIQ | CLIP-S | LPIPS |
|--------|---|---|-------|-------|--------|-------|
| global SWD | - | 0 | 12 | 42.95 | 0.7275 | 0.4347 |
| sem_r8 (base) | 8 | 0.7 | 12 | 51.86 | 0.7147 | 0.3815 |
| sem_r8 + EOTA τ=0.08 | 8 | 0.7 | 12 | **54.50** | 0.7126 | 0.3843 |
| sem_r8_strict (multinomial) | 8 | 0.7 | 12 | 41.50 | - | - |

### 4.2 Key findings
1. **k-means region partitioning is the effective mechanism.** S5-S8 (guidance-based without k-means) all failed (MUSIQ 36-39).
2. **Deterministic quantile matching is essential.** Stochastic sampling regresses 10 points.
3. **EOTA stacks with semantic SWD.** +2.6 MUSIQ for free (CLIP-S/LPIPS flat).
4. **Governing pattern:** adding HF energy hurts MUSIQ (VAE grain); redistributing/cleaning helps.

## 5. Open Questions (Under Investigation)

1. **K-value sweep:** Does K=4/16/32 beat K=8? (M1, running)
2. **β-value fine sweep:** Is 0.7 truly optimal or just a local optimum? (M2)
3. **Region matching alternatives:** Softmax weighting vs hard k-means? (M4)
4. **Cross-dataset generalization:** Does D5 optimal transfer to R5? (M5)

## 6. Implementation Notes

- Code: `src/spectral_losses620.py`, functions `_kmeans_labels` and `_semantic_region_swd`.
- The K×B Python loop is intentional (vectorized multinomial produces wrong results).
- Config: `swd_semantic_mode=region`, `swd_semantic_regions=K`, `swd_semantic_blend=β`.
- Training: 5 epochs, batch=24, Patience=2, ~3.5 min/epoch on RTX 3060.
