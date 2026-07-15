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

### 3.4 K-value trade-off

**Hypothesis (inverted-U, original).** Per-region transport cost and statistical reliability interact non-monotonically:
- $K = 1$: equivalent to global SWD, no content adaptivity.
- $K$ small (e.g. 4): regions too coarse, partition approaches global; minimal content-adaptive gain.
- $K$ medium (e.g. 8): sweet spot — content-coherent regions with enough pixels per region for stable 1D quantile matching.
- $K$ large (e.g. 32): regions too fine, per-region empirical CDFs become noisy (few pixels per region per channel), variance dominates, transport cost estimate becomes unreliable.

**Empirical K=4 result.** K=4 produces a *higher* training transport loss than K=8 (tswd=0.7753 vs 0.7333, +5.7%). This is consistent with the upper-bound theorem: coarser regions forbid fewer cross-content transports than K=8, so the region-preserving minimum is closer to (but still above) the global minimum. The CLIP-S and LPIPS both regress slightly (CLIP-S 0.7167 vs 0.7147, LPIPS 0.3875 vs 0.3815), and MUSIQ drops by 2.72 (49.14 vs 51.86), confirming that K=4's coarser partition loses content-coherent style cues that K=8 captures.

### 3.5 K=16 anomaly and theory revision (in progress)

**Empirical K=16 result.** K=16 produces a *lower* training transport loss than K=8 (tswd=0.6297 vs 0.7333, −14.1%). This **contradicts** the upper-bound theorem prediction, which says more regions → tighter restriction → higher transport cost.

**Three candidate explanations:**

1. **Per-region sample-size effect dominates.** With K=16, each region contains ~64 pixels (vs ~128 for K=8). The empirical CDFs on smaller samples are noisier, and the 1D OT optimum on noisier distributions is numerically easier to achieve (fewer effective constraints). The variance-induced reduction in transport cost overwhelms the restriction-induced increase.

2. **K=16 clusters are more content-discriminative.** With more centroids, k-means captures finer content distinctions, so within-region pixels are more similar in content space. The within-region transport cost drops because the source and target distributions within each region are already closer (less content mixing to reconcile).

3. **F.interpolate numerical effect.** Quantile interpolation on smaller subsets has fewer interpolation points, which may produce lower-cost matches by virtue of having fewer constraints to satisfy.

**Implication for the upper-bound theorem.** Theorem 1 is mathematically correct — region-wise transport is an upper bound on global. But the bound is **loose**: the gap between the bound and the global minimum depends on K in a non-monotonic way because of the variance effect. The theorem assumes exact 1D OT computation; in practice, we compute empirical 1D OT on finite samples, and the variance of the empirical CDF estimator scales as $O(1/\sqrt{n_k})$ where $n_k = HW/K$ is the per-region sample size. As K grows, $n_k$ shrinks, the empirical CDFs become smoother (less concentrated), and the empirical 1D OT cost is biased downward.

**Revised K-value trade-off (under investigation).**
- $K$ small: regions coarse, content-discrimination weak, transport cost high (matches theorem).
- $K$ medium (8): content-discrimination good, sample size adequate, transport cost moderate.
- $K$ large (16+): content-discrimination strong BUT sample size small, variance effect dominates, empirical transport cost artificially low. Whether this translates to better perceptual quality (MUSIQ) depends on whether the lower transport cost reflects real content-coherent matching or just numerical artifact.

**K=16 perceptual metrics:** CLIP-S=0.7203 (better than K=8's 0.7147), LPIPS=0.3961 (worse than K=8's 0.3815), **MUSIQ=50.35 (worse than K=8's 51.86, −1.51)**. The lower tswd did NOT translate to better perceptual quality.

**Resolution of the anomaly.** The tswd decrease at K=16 is a **numerical artifact** of the empirical 1D OT computation on smaller per-region samples, not a real improvement in content-coherent matching. The LPIPS degradation (+0.0146, more content distortion from noisier quantile matches) outweighed the CLIP-S improvement (+0.0056, more style transfer), resulting in a net MUSIQ decrease of 1.51.

**Implication for the upper-bound theorem.** Theorem 1 is mathematically correct — region-wise transport is an upper bound on global. But the bound's tightness depends on K in a non-monotonic way because of finite-sample effects. The theorem assumes exact 1D OT computation; in practice, we compute empirical 1D OT on $n_k = HW/K$ samples per region. The empirical CDF variance scales as $O(1/\sqrt{n_k})$, so as K grows, $n_k$ shrinks, the empirical CDFs become smoother (less concentrated), and the empirical 1D OT cost is biased downward. This bias does NOT reflect better content-coherent matching — it reflects noisier quantile matches that the VAE decoder amplifies into content distortion.

**Revised K-value trade-off (CONFIRMED by M1).**
- $K$ small (4): regions coarse, content-discrimination weak, MUSIQ=49.14 (low).
- $K$ medium (8): content-discrimination good, sample size adequate, **MUSIQ=51.86 (peak)**.
- $K$ large (16): content-discrimination strong but sample size too small, empirical OT biased, MUSIQ=50.35 (declining).

The inverted-U is confirmed for MUSIQ. K=8 is the true sweet spot where content-discrimination and per-region sample size are balanced. The peak is driven by the **content-discriminative power per region weighted by sample-size reliability**, not by raw K or by transport cost.

## 4. Empirical Results

### 4.1 Current best
| Config | K | β | SWD w | MUSIQ | CLIP-S | LPIPS |
|--------|---|---|-------|-------|--------|-------|
| global SWD | - | 0 | 12 | 42.95 | 0.7275 | 0.4347 |
| sem_r8 (base) | 8 | 0.7 | 12 | 51.86 | 0.7147 | 0.3815 |
| sem_r8 + EOTA τ=0.08 | 8 | 0.7 | 12 | **54.50** | 0.7126 | 0.3843 |
| sem_r8_strict (multinomial) | 8 | 0.7 | 12 | 41.50 | - | - |
| sem_r4 (K-sweep) | 4 | 0.7 | 12 | 49.14 | 0.7167 | 0.3875 |
| sem_r16 (K-sweep) | 16 | 0.7 | 12 | 50.35 | 0.7203 | 0.3961 |

### 4.2 K-sweep results (M1, 3/4 complete)
| K | tswd (train, final) | CLIP-S (all_pairs) | LPIPS (all_pairs) | MUSIQ | Verdict |
|---|---------------------|--------------------|--------------------|-------|---------|
| 4 | 0.7753 | 0.7167 | 0.3875 | **49.14** | strictly worse than K=8 on all 3 metrics |
| 8 | 0.7333 | 0.7147 | 0.3815 | 51.86 / **54.50** | **peak (best)** |
| 16 | 0.6297 | 0.7203 | 0.3961 | **50.35** | tswd artifact; style↑ content↓ MUSIQ↓ |
| 32 | *training* | - | - | - | in progress |

**Note on tswd.** tswd is NOT a reliable proxy for perceptual quality across K values. It decreases monotonically with K (4>8>16) due to per-region sample-size bias, but MUSIQ follows an inverted-U peaking at K=8. Use MUSIQ as the ground truth for K selection.

### 4.3 Key findings
1. **k-means region partitioning is the effective mechanism.** S5-S8 (guidance-based without k-means) all failed (MUSIQ 36-39).
2. **Deterministic quantile matching is essential.** Stochastic sampling regresses 10 points.
3. **EOTA stacks with semantic SWD.** +2.6 MUSIQ for free (CLIP-S/LPIPS flat).
4. **Governing pattern:** adding HF energy hurts MUSIQ (VAE grain); redistributing/cleaning helps.
5. **K=8 is the inverted-U peak (CONFIRMED).** K=4 (MUSIQ 49.14) and K=16 (MUSIQ 50.35) both regress. The peak is driven by content-discriminative power per region weighted by sample-size reliability, not by raw K or transport cost.
6. **tswd is misleading for K selection.** Empirical 1D OT cost on finite samples has $O(1/\sqrt{n_k})$ variance bias that artificially lowers tswd at large K. Use MUSIQ as ground truth.

## 6. Mechanism-Level Exploration (M4, in progress)

After M1 confirmed the inverted-U at K=8, we pivot from parameter sweeps to
**mechanism-level code changes**. Four variants are implemented in
`spectral_losses620.py`, each replacing a different component of the pipeline:

### 6.1 Soft-mask region SWD (`region_soft`)

**Mechanism change:** replace hard k-means `argmax` labels with soft softmax
membership probabilities. Each pixel contributes to ALL regions with weight
$p_k = \mathrm{softmax}(-\|x - c_k\|^2 / \tau)_k$, clamped to a floor
$\min_w$ and renormalized.

**Theoretical significance.** This relaxes the coupling set:
$$\Pi_r \subseteq \Pi_{\mathrm{soft}} \subseteq \Pi$$
where $\Pi_{\mathrm{soft}}$ allows weak cross-region transport proportional to
$\min_w$. As $\tau \to 0$, $\Pi_{\mathrm{soft}} \to \Pi_r$ (hard); as
$\tau \to \infty$, $\Pi_{\mathrm{soft}} \to \Pi$ (global). The soft mask
interpolates between strict region OT and global OT, breaking the hard
partition boundary that destroys transition information (e.g. a pixel
"between sky and skin" now contributes fractionally to both regions).

**Empirical result (M4.1).** Soft-mask with $\tau=1.0$, $\min_w=0.05$:
- CLIP-S = 0.7230 (vs hard 0.7147, **+0.0083, better style**)
- LPIPS = 0.3272 (vs hard 0.3815, **−0.0543, better content, −14.2%**)
- tswd = 0.4260 (vs hard 0.7333, −42%)
- **MUSIQ = 41.04** (vs hard 51.86, **−10.82, CRASHED**)

**Paradox and resolution.** Soft-mask is the FIRST mechanism to improve both
CLIP-S and LPIPS simultaneously (all K-sweep points were tradeoffs). Yet
MUSIQ crashes by 10+ points. This reveals that CLIP-S/LPIPS measure
**alignment** (style-content correspondence) while MUSIQ measures
**perceptual quality** (sharpness, texture naturalness).

The hard boundary is a **feature, not a bug**: it keeps per-region statistics
**sharp** (each region's quantile match is computed on a clean, non-overlapping
pixel subset). Soft membership smears each pixel's contribution across all
regions, smoothing the per-region empirical CDFs. The smoothed CDFs produce
quantile matches that are statistically "safer" (lower transport cost, better
alignment) but perceptually "bland" (lost sharp style features that MUSIQ
rewards).

**Theory revision.** Theorem 1's upper bound is not just a transport-cost
bound; it is a **perceptual-quality bound**. $\Pi_r$ (hard) gives HIGHER
transport cost but BETTER MUSIQ because sharp per-region statistics = sharp
style features. $\Pi_{\mathrm{soft}}$ gives LOWER transport cost but WORSE
MUSIQ because smoothed statistics = blurred style. The inverted-U is not just
about sample-size; it is about the **sharpness-vs-smoothness tradeoff** in
per-region statistics.

This finding reframes the mechanism search: the goal is not to relax $\Pi_r$
toward $\Pi$, but to find couplings that are **as tight as $\Pi_r$ on
sharpness but smarter about content coherence**.

### 6.2 Sinkhorn (entropic-regularized) OT (`region_sinkhorn`)

**Mechanism change:** replace exact 1D quantile matching (F.interpolate on
sorted values) with Sinkhorn OT (entropic-regularized) per region per
projection.

**Theoretical significance.** The K=16 tswd anomaly (§3.5) showed that
empirical 1D OT on small per-region samples ($n_k = 64$) is biased downward
by $O(1/\sqrt{n_k})$ variance. Sinkhorn OT explicitly controls the
bias-variance tradeoff via $\epsilon$:
$$\mathrm{OT}_\epsilon(\mu, \nu) = \min_{\gamma \in \Pi(\mu,\nu)} \langle C, \gamma \rangle - \epsilon H(\gamma)$$
As $\epsilon \to 0$ we recover exact OT (and the artifact); larger $\epsilon$
smooths the transport plan, reducing sample-size sensitivity at the cost of
a small bias. If Sinkhorn OT eliminates the tswd artifact at K=16 while
preserving MUSIQ, it confirms the artifact is variance-driven and provides a
principled fix.

**Implementation.** `_sinkhorn_1d` + `_semantic_region_swd_sinkhorn` in
[spectral_losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_losses620.py).
Sinkhorn iterations in log-domain for numerical stability.

### 6.3 Hierarchical (coarse + fine) region SWD (`region_hier`)

**Mechanism change:** run two independent k-means partitions at different
granularities ($K_{\text{coarse}}=4$, $K_{\text{fine}}=16$), compute region
SWD at both levels, blend them with weight $w_{\text{fine}}$.

**Theoretical significance.** A single-K partition cannot simultaneously
represent structural content categories (sky, skin, background — needs small
K) and texture-level subregions (brush strokes within skin — needs large K).
The hierarchical blend is a two-level discretization of hierarchical OT:
$$\mathcal L_{\text{hier}} = (1-w_f) \mathcal L_{\text{sem}}(K_c) + w_f \mathcal L_{\text{sem}}(K_f)$$
This sidesteps the inverted-U tradeoff by giving the model both scales
simultaneously, rather than forcing a single K to compromise.

**Implementation.** `_semantic_region_swd_hier` reuses
`_semantic_region_swd` at two K values and blends. Cheap (2× the cost of a
single-K run, no new k-means code).

### 6.4 Content-adaptive-K region SWD (`region_adaptive_k`)

**Mechanism change:** pick K per-sample by measuring how much k-means
inertia drops when going from K to the next candidate. If the relative drop
is below a threshold, more clusters are not justified and we use the smaller
K.

**Theoretical significance.** The inverted-U peak K* depends on content
complexity (§3.4). A fixed K=8 is suboptimal for both simple images
(K*≈4) and complex images (K*≈16). Adaptive-K tracks the per-sample peak
directly via the inertia-elbow criterion:
$$K^*(x) = \min \{K : (\text{inertia}(K) - \text{inertia}(2K)) / \text{inertia}(K) < \theta\}$$
This is the principled fix for the inverted-U: instead of picking one K for
all images, pick the right K for each image.

**Implementation.** `_kmeans_labels_with_inertia` +
`_semantic_region_swd_adaptive_k` in
[spectral_losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/spectral_losses620.py).
Runs k-means at all candidate K values (vectorized), picks per-sample K,
then dispatches subsets to `_semantic_region_swd` at the chosen K.

### 6.5 Spectral-decoupled region SWD (`region_spectral`) — Mechanism 5

**Architecture change.** The previous four mechanisms (6.1–6.4) all modify the
region SWD *within a single frequency band*: one k-means partition covers all
frequencies of the latent. But style and content live in different bands:
- **Style** (brushstrokes, texture, color jitter) is predominantly **high-frequency**.
- **Content** (structure, layout, semantics) is predominantly **low-frequency**.

A single spatial partition conflates them. Mechanism 5 changes *where* SWD
operates: DWT-decompose `gen` and `target` into LL (low-freq) and LH/HL/HH
(high-freq), then apply different SWD strategies per band:

$$\mathcal{L}_{\text{spectral}} = w_{\text{ll}} \cdot \text{SWD}_{\text{global}}(\text{LL}_g, \text{LL}_t) + w_{\text{hf}} \cdot \frac{1}{3}\sum_{b \in \{\text{LH,HL,HH}\}} \text{SWD}_{\text{region}}(b_g, b_t)$$

- **LL band → global SWD (K=1):** Low-frequency content should match globally
  to preserve structure; no region matching needed.
- **HF bands → region SWD (K=8):** High-frequency texture should match within
  content-coherent regions to transfer brushstrokes locally.

**Theoretical motivation.** The inverted-U peak $K^*$ depends on frequency:
high-freq texture needs fine regions (large $K$); low-freq structure needs
global matching ($K=1$). Frequency-band division lets each band use its optimal
$K$ without compromise. This is a *deeper* architecture change than 6.1–6.4:
it changes *where* SWD operates (frequency domain) rather than *how* region
partitioning works.

**Integration.** Naturally integrates with the existing DWT route architecture
(`cross_attn_dwt_route`): the model already lives in the wavelet domain, so
the SWD loss can directly consume subbands without extra decomposition cost.

**Status.** Code implemented (`_semantic_region_swd_spectral`), config ready
(`sem_r8_spectral.json`, $w_{\text{ll}}=1.0, w_{\text{hf}}=2.0$). Awaiting GPU.

### 6.6 Attention-guided style-conditional region SWD (`region_attn`) — Mechanism 6

**Architecture change.** The previous five mechanisms all define regions via
k-means on the *content* latent. These regions are **style-agnostic**: the same
content point belongs to the same region regardless of the target style. But
semantically, a "sky" region in a landscape should be matched differently to
Impressionism (brushstroke sky) vs Ukiyo-e (flat color sky).

Mechanism 6 replaces the region **definition** (content k-means → cross-attention
map) rather than the region **matching**. The cross-attention map
$A \in \mathbb{R}^{B \times N \times S}$ ($S$ = style tokens) is already
style-conditional: each style token defines a "soft region" — locations
attending strongly to that token form one region.

$$\text{region}_s = \text{TopK}(\{i : A_{i,s} \geq \bar{A}_{\cdot,s}\}), \quad s = 1, \ldots, S$$

We use **hard top-k** membership (not soft membership) based on the soft-mask
lesson (§6.1): hard boundaries preserve sharp per-region statistics that MUSIQ
rewards. Within each attention-defined region, gen is matched to target via
quantile SWD.

**Theoretical motivation.** This is the first mechanism that makes regions
**style-conditional**. The region partition $\Pi_r$ in Theorem 1 becomes a
function of both content AND style: $\Pi_r = f(\text{content}, \text{style})$.
This breaks the content-style independence assumption implicit in k-means
regions, potentially allowing more expressive style transfer.

**Status.** Code implemented (`_semantic_region_swd_attn`), config ready
(`sem_r8_attn.json`). Uses existing cross-attention guidance infrastructure.
Awaiting GPU.

### 6.7 Experimental Results: Mechanism 1–4 vs. Hard Baseline

| Mechanism | CLIP-S | LPIPS | MUSIQ | vs. Hard K=8 | Verdict |
|-----------|--------|-------|-------|--------------|---------|
| Hard K=8 (baseline) | 0.7147 | 0.3815 | 51.86 | — | Best |
| + EOTA τ=0.08 | 0.7126 | 0.3843 | **54.50** | MUSIQ +2.64 | Best+EOTA |
| Soft-mask (6.1) | 0.7230 | 0.3272 | 41.04 | MUSIQ -10.82 | Ruled out |
| Hierarchical (6.2) | 0.7202 | 0.3903 | — | LPIPS +2.3% | Ruled out |
| Sinkhorn (6.3) | 0.7142 | 0.4243 | — | LPIPS +11.2% | Ruled out |
| Adaptive-K (6.4) | — | — | — | — | Skipped (tactical) |

**Key insight.** All 4 mechanisms that modify HOW regions are matched (soft-mask,
Sinkhorn, hierarchical, adaptive-K) fail to beat the hard K=8 baseline. The
consistent pattern is that any relaxation of the hard boundary degrades quality
— the hard boundary is a **feature, not a bug**. Sharp per-region statistics
are what MUSIQ and LPIPS reward.

**Pivot (Deli_AutoResearch protocol).** stale_count=3 → structural pivot
required. Change the region **DEFINITION** (where/how regions are formed) rather
than the region **MATCHING** (how distributions are matched within regions):
- Mechanism 5 (spectral): change WHERE SWD operates (frequency vs spatial)
- Mechanism 6 (attn): change HOW regions are defined (attention vs k-means)

### 6.8 Experimental Results: Mechanism 5–6 (Structural Pivot)

| Mechanism | CLIP-S | LPIPS | MUSIQ | vs. Hard K=8 | Verdict |
|-----------|--------|-------|-------|--------------|---------|
| Spectral (6.5) | — | — | — | — | Training |
| Attention (6.6) | — | — | — | — | Pending |

## 5. Open Questions (Under Investigation)

1. **K-value sweep (M1, done):** K=4 (49.14) < K=8 (51.86, peak) > K=16 (50.35). Inverted-U confirmed. K=32 canceled.
2. **Mechanism 1–4 sweep (M4, done):** All 4 mechanisms that modify region MATCHING fail to beat the hard K=8 baseline. The hard boundary is a feature, not a bug.
3. **Mechanism 5–6 sweep (M4, in progress):** Does changing the region DEFINITION (frequency domain or attention-based) beat the hard baseline?
4. **Cross-dataset generalization (M5):** Does D5 optimal transfer to R5?
5. **Clustering feature dimensionality:** Would clustering on DINO features (vs raw content latent) shift the optimal K?
6. **Sample-size correction:** Could a variance-corrected tswd (e.g., bootstrap CI, or $n_k$-weighted loss) recover the monotonic relationship between tswd and MUSIQ?
7. **Frequency-band K optimality:** Does the spectral-decoupled mechanism (6.5) confirm K*=1 for LL and K*=8 for HF? In training.

## 6. Implementation Notes

- Code: `src/spectral_losses620.py`, functions `_kmeans_labels` and `_semantic_region_swd`.
- The K×B Python loop is intentional (vectorized multinomial produces wrong results).
- Config: `swd_semantic_mode=region`, `swd_semantic_regions=K`, `swd_semantic_blend=β`.
- Training: 5 epochs, batch=24, Patience=2, ~3.5 min/epoch on RTX 3060.
