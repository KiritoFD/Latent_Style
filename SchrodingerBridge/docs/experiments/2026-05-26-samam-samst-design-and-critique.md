# SaMST / SaMam High-Level Reading, Design Transfer, and Critique

Date: 2026-05-26

Sources read:

- `F:\SaMST.pdf`: *Pluggable Style Representation Learning for Multi-Style Transfer*, ACCV 2024.
- `F:\SaMam.pdf`: *SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer*, CVPR paper.
- Local code:
  - `G:\GitHub\Latent_Style\Related_Works\repos\external\SaMST`
  - `G:\GitHub\Latent_Style\Related_Works\repos\SaMam`

This note is written for LANCET/SB development, not as a generic related-work summary. The key question is:

> What do these two works teach us about getting strong style without breaking structure, and where are their claims/metrics weak enough that our paper can critique them fairly?

## Executive View

SaMST and SaMam look different on the surface, but their useful design principle is the same:

> Style is injected as conditional local operators and controlled modulation, not as unconstrained spatial transport.

SaMST packages each style into a tiny learned representation and expands it into local depthwise kernels, AdaIN statistics, and channel gates. SaMam moves this pattern into a Mamba/state-space backbone: the style embedding conditions selective-scan/state updates, local convolution, instance normalization, and channel modulation.

For LANCET, the valuable part is not the exact SaMST codebook or SaMam Mamba backbone. The valuable part is the operator routing:

- local textons through style-conditioned depthwise filters;
- global color/statistics through normalized affine modulation;
- feature selection through channel gates;
- stability through identity/zero initialization;
- geometry preservation through consistency losses or scan/order constraints.

The caution is equally important. Neither paper explicitly solves semantic boundary preservation, local artifact realism, color flooding, or phase alignment. Their metrics can be strong while visual outputs still show muddy texture, noisy grain, or style leaking into the wrong semantic regions.

## SaMST: What The Paper Actually Does

SaMST targets efficient multi-style transfer. Its core claim is that style modeling and transfer can be decoupled:

1. Learn a compact style representation for each style.
2. Store these representations in a style codebook.
3. Use the selected representation to generate parameters inside a universal transfer network.

The paper describes the style representation as length `C=16`, while the local checked code uses 32-dimensional `style_representation` parameters in `networks/transfer_net.py`. This mismatch is not fatal for understanding the method; the key is still a small learned vector expanding into many operators.

Main modules:

- `SConv`: the style representation predicts depthwise convolution kernels. This is the most important local-texture module.
- `SRAdaIN`: the style representation predicts normalization statistics for global style statistics.
- `SCM`: the style representation predicts channel-wise modulation coefficients.
- `f0` reconstruction style: an auto-encoding style representation allows identity/content reconstruction.
- `Lgeo`: a geometric consistency term reduces uncontrolled geometric drift.

The training objective is conventional style-transfer supervision:

- VGG content loss;
- VGG style mean/std loss;
- reconstruction loss;
- geometric consistency loss.

Implementation details in the paper matter for comparison:

- content images: `256x256`;
- style images: `512x512`;
- batch size: `8`;
- style capacity: reported as large-scale style-code storage, with 50k style images in the main training setup;
- inference speed is low because the style knowledge has already been prepaid into the codebook and network.

### SaMST Interpretation

SaMST is not simply "fast arbitrary style transfer." It is an amortized multi-style system. The expensive part is moved into:

- pretraining the universal transfer network;
- learning/storing style representations;
- optionally training new style codes for extension.

So the fair interpretation is:

> SaMST is fast at inference because style knowledge has been compressed into learned operator parameters. It is not solving unpaired style transfer from scratch at inference time.

This is important for our paper because LANCET's claim is not just inference speed; it is an explicit latent transport objective that can expose a tunable style-content frontier with much smaller training cost.

## SaMam: What The Paper Actually Does

SaMam targets arbitrary image style transfer with a Mamba/state-space backbone. The headline motivation is global receptive field with linear-complexity sequence modeling.

The architecture has:

- a content Mamba encoder;
- a style Mamba encoder;
- a style-aware Mamba decoder;
- a Style-aware Selective Scan Structured State Space block, called `S7`;
- zigzag scanning to reduce spatial discontinuity from flattening images into sequences;
- local enhancement to compensate for SSM local-detail forgetting.

But from a LANCET design standpoint, the most important pieces are again the style-aware local/global operators:

- `SConv`: style embedding predicts depthwise convolution kernels.
- `SAIN`: style-aware instance normalization, with style-generated gamma/beta.
- `SCM`: style-aware channel modulation.
- `S7`: style embedding affects state-space update parameters, so long-range sequence propagation is also style-conditioned.
- zero-init: SAIN/SCM embedders are initialized to output zero, making the module start close to identity.

The local code confirms these details:

- `ARCHI/SAVSSM/common/SConv.py` generates per-sample depthwise kernels from the style representation and uses reflection padding.
- `ARCHI/SAVSSM/common/SAIN.py` uses instance norm plus style-generated gamma/beta, with optional zero initialization.
- `LOSS/losses.py` uses VGG content/style/identity terms; there is no explicit semantic boundary loss, CLIP loss, latent OT, topology barrier, or artifact-specific loss.

The reported loss is also conventional:

- content loss;
- style loss;
- identity image loss;
- identity feature loss;
- weights: style `10`, identity terms `1` and `50` in the paper.

### SaMam Interpretation

SaMam's Mamba backbone is not the only source of its strength. The paper's own ablations indicate that S7, SConv, SCM, SAIN-zero, local enhancement, and zigzag scan all matter.

For us, the clean takeaway is:

> The global receptive field helps, but style-conditioned operator design is the more transferable idea.

The reason is practical: LANCET's current failure is not obviously a lack of long-range context. Our failure is that stronger style pressure can become badly routed: warp fragments structure, high-frequency residuals lose phase alignment, and style texture is not sufficiently object/boundary aware.

## What We Should Learn

### 1. Style Should Be Routed Through Operators, Not Free Motion

Both papers route style through filters, norms, and gates. They do not rely on free coordinate warp as the primary style mechanism.

For LANCET this supports the current diagnosis:

- dynamic no-warp branches are useful because they add style residuals without moving content geometry;
- diffeomorphic/warp branches are dangerous because they can anti-align high-pass residuals and visually fragment images.

Design implication:

- keep no-warp style dynamics as the anchor;
- use style-conditioned high-pass local operators for textons;
- avoid free spatial resampling unless it is heavily masked by content edge/support.

### 2. SConv Is The Most Direct Borrowable Module

SaMST and SaMam both use style-conditioned depthwise convolution. This is nearly the exact missing middle layer between "stronger style loss" and "unsafe warp."

For LANCET:

- extract a high-pass band from the latent;
- use style code to generate per-channel depthwise kernels;
- apply kernels only to high-pass residuals;
- blend residually with a small/learned gate;
- use reflection padding;
- keep low-pass content/color under separate, weaker control.

Expected benefit:

- stronger local artistic texture;
- less global content movement;
- more controlled high-frequency style injection than raw SWD pressure.

Risk:

- if not edge/phase gated, the operator can still paint texture everywhere and hurt LPIPS.

### 3. Zero-Init Style Modulation Is A Real Stability Device

SaMam's zero initialization is not cosmetic. It starts style-aware blocks close to identity, then lets style operators grow as training needs.

For LANCET:

- every style-conditioned kernel/gate/norm head should start as no-op or near-no-op;
- add learnable scalar gates initialized near zero;
- monitor raw output p95, delta norm, high/low delta ratio, and high-pass cosine, not only final CLIP.

This directly addresses our observed failure pattern: stronger branches can produce large raw velocity and broken high-frequency residuals.

### 4. Geometry Consistency Is Safer Than Coordinate Warp

SaMST uses a geometric consistency loss to constrain possible translations. This is not the same as predicting an explicit warp. It is more like saying:

> Stylization should commute with simple image transforms.

For LANCET:

- add flip/rotation equivariance on the generated endpoint;
- keep it light, because too much will suppress style;
- use it as a guardrail for style operators rather than as a primary content loss.

This is especially relevant for video: if output under a small transform is unstable, temporal flicker is likely.

### 5. Style Code Should Become A Small Operator Bank

SaMST shows that a tiny representation can expand into many useful style operators. LANCET already has style IDs/embeddings, but a raw embedding is weaker than an explicit operator bank.

Candidate LANCET operator bank:

- high-pass depthwise kernel branch for textons;
- channel gate branch for palette/feature selection;
- low-pass affine branch for color/statistics;
- scalar strength gate for per-style aggressiveness;
- optional semantic/edge gate for boundary-aware routing.

The bank should be separable and constrained first. Dense unconstrained kernels are too easy to overfit or break structure.

## What We Should Not Copy

### Do Not Copy Their Training Claim Uncritically

SaMST's inference speed is real, but style knowledge is prepaid through training and style-code learning. SaMam also relies on a nontrivial full model and conventional VGG-style training.

For the paper:

- compare training time and style-extension cost explicitly;
- distinguish "inference-time cost" from "total style acquisition cost";
- avoid allowing a pretrained style-code system to pose as a cheap zero-shot method.

### Do Not Copy VGG/Gram-Only Objective As Our Main Logic

Both papers lean heavily on VGG content/style/identity objectives. These losses are useful, but they do not measure:

- semantic placement of style;
- color flooding across object boundaries;
- high-frequency grain realism;
- phase alignment of texture residuals;
- temporal stability;
- topology preservation.

For LANCET, VGG-style losses can be auxiliary only. The main thesis should remain latent transport + terminal distribution matching + content-preserving dynamics.

### Do Not Copy Mamba Just Because It Is Mamba

SaMam's Mamba backbone is interesting, but importing it wholesale would not directly solve our strongest observed failure:

- broken style routing;
- unsafe high-frequency residuals;
- weak object-boundary locking;
- warp-induced fragmentation.

If we borrow from SaMam, borrow:

- style-aware operator conditioning;
- zero-init gates;
- scan/order continuity idea if we ever build a sequence latent module.

Do not make "Mamba backbone" the next move unless diagnostics show global receptive field is the bottleneck.

## How Their Metrics Can Be Misleading

The right critique is not "SaMST/SaMam are bad." They are strong baselines. The critique is:

> Their reported metrics validate global style statistics and coarse content retention more than boundary-aware, artifact-free stylization.

Likely metric blind spots:

- ArtFID/FID-style metrics can reward global distribution match while missing local semantic leakage.
- CF/CSFD-style metrics may reward style-content correlation while not punishing muddy local texture enough.
- LPIPS can miss structured grain if object layout remains recognizable.
- VGG/Gram losses encourage palette and texture statistics, even when those statistics are spread over wrong regions.
- Edge metrics can be fooled by texture edges: extra grain can increase edge recall while degrading perceived quality.

This matches our visual experience with SaMST: it can preserve coarse structure and hit style metrics, but still look muddy/noisy or introduce large color fields in the wrong place.

## How To Hammer Them Fairly

The fair attack should be artifact- and boundary-sensitive, not rhetorical.

Recommended metric pack:

1. High-frequency patch KID
   - Run on Laplacian/high-pass images.
   - Catches dirty grain and broken local texture.

2. FFT slope / spectrum-shape error
   - Penalizes unnatural high-frequency energy distribution.
   - Useful when outputs look noisy or pointillist.

3. No-reference IQA
   - MUSIQ and MANIQA are useful for visible quality collapse.
   - Prior local evidence already suggests SaMST loses here.

4. DISTS-content
   - Better perceptual structure distance than only CLIP-content.

5. Edge-locked color drift
   - Compare low-frequency chroma changes inside content-edge or semantic masks.
   - Specifically targets color flooding / large color blocks.

6. High-pass phase consistency
   - Cosine between generated high-pass residual and content high-pass structure.
   - This is directly aligned with our LANCET diagnosis and catches warp-like fragmentation.

7. Temporal consistency for video
   - Optical-flow warped LPIPS / DISTS over adjacent frames.
   - SaMST-like noisy local texture should be exposed strongly here.

Paper framing:

> SaMST is a strong efficient baseline under coarse style-content metrics, but artifact-sensitive diagnostics reveal that its style is often achieved through muddy or grain-like local statistics. LANCET's advantage should be stated as a better efficiency-quality and artifact-sensitive trade-off, not as winning every raw style metric.

## Direct Implications For Current LANCET Work

Current internal anchor from the EMA line:

- `ema_dynamic_frontier_w32`, epoch 6: `clip_style=0.7093`, `content_lpips=0.4690`.
- This is already an excellent operating point under the current backend.
- Target priority remains: first push `clip_style > 0.72`; LPIPS can temporarily degrade to `0.49-0.50` if style gain is real.

Observed diagnosis:

- no-warp dynamic style residuals help;
- warp/diffeomorphic paths fragment images;
- current model lacks an object-aware local texton organizer.

Therefore the high-probability design path is:

1. Keep no-warp dynamic branch as the base.
2. Add style-conditioned high-pass local filtering, inspired by SConv.
3. Zero-init every new style-conditioned branch.
4. Add edge/support/phase gates before allowing stronger high-pass style.
5. Add light equivariance consistency.
6. Evaluate with CLIP-style/LPIPS plus artifact-sensitive metrics.

## Concrete Design Probes

### Probe A: High-Pass SConv

Goal:

- push CLIP-style above `0.72`;
- keep LPIPS below roughly `0.49-0.50`.

Design:

- style-generated depthwise kernels;
- applied only to latent high-pass bands;
- kernel size `3` or `5`;
- reflection padding;
- residual gate initialized at zero or very small value.

Success signature:

- CLIP-style rises;
- high-to-low delta ratio stays controlled;
- high-pass cosine does not become strongly negative;
- grids show texture organization rather than fragmentation.

### Probe B: Zero-Init SAIN/SCM-Like Gates

Goal:

- increase style controllability without raw velocity explosion.

Design:

- style-conditioned affine normalization or channel gates in latent feature blocks;
- final projections zero-initialized;
- small learnable global scalar.

Success signature:

- raw output p95 does not spike;
- less color flooding than simple style-loss increases;
- style rises without destroying LPIPS.

### Probe C: Equivariance Geometry Loss

Goal:

- reduce fragmentation and improve video stability.

Design:

- random horizontal flip first;
- optional 90-degree rotations only after stability is proven;
- transfer transformed content, invert transform, penalize endpoint mismatch.

Success signature:

- cleaner grids;
- better temporal metrics;
- little or no raw style loss.

### Probe D: Edge/Support-Gated Style Operator

Goal:

- close the qualitative gap to Seedream-like boundary-respecting stylization.

Design:

- derive an active support mask from content latent high-pass magnitude or edge proxy;
- allow high-pass style operators in texture regions;
- restrict low-pass color changes crossing strong boundaries.

Success signature:

- fewer large wrong-color regions;
- less texture in flat semantic areas;
- better Hayao/Vangogh side-by-side frames.

## How This Affects Our Paper Claim

SaMST and SaMam should be cited as evidence that efficient style transfer benefits from style-conditioned operator modulation. But our contrast should be:

- They learn representation/operator banks under VGG-style objectives.
- We learn a latent bridge/transport dynamics with explicit terminal distribution matching and kinetic/content regularization.
- Their objectives do not directly control semantic boundary leakage or artifact texture.
- Our artifact-sensitive diagnostics and video checks expose failures that headline metrics can hide.

The cleanest thesis:

> Existing efficient baselines amortize style into conditional filters and can score well on global style-content metrics, but they lack explicit transport/path control and artifact-aware structure constraints. LANCET should keep their useful operator idea while enforcing safer latent dynamics.

## Bottom Line

The best idea to steal is not Mamba. It is:

> small style representation -> constrained local operator bank -> zero-init gated residual application.

The best idea to attack is not their architecture. It is:

> headline style-transfer metrics can be satisfied by broad texture/statistic injection while still producing muddy, noisy, or semantically misplaced artifacts.

For our next model iteration, the correct direction is not "turn every style loss higher." It is stronger style pressure routed through a SaMST/SaMam-style local operator bank, locked by content edge/phase diagnostics and evaluated with artifact-sensitive metrics.

## Current VAE-Backend Implication

After the KL-f4 / SDXL / EMA replacement experiments, this reading has become more concrete.

Current verified anchors:

| backend / variant | clip_style | content_lpips | readout |
|---|---:|---:|---|
| Seedream 4.5 API, all pairs | 0.7532 | 0.3644 | golden visual target, object-aware repainting |
| Seedream 4.5 API, style-transfer subset | 0.7326 | 0.3822 | practical target for our 750/800 protocol |
| original VAE t01-like point | ~0.726 | ~0.517 | style can pass 0.72 but content is too expensive |
| KL-f4 fair f4-patch line | ~0.654 | ~0.485 | not useful as drop-in replacement |
| SDXL 256 line | <0.68 style | variable | mismatched to current 256 operators |
| EMA dynamic guard | 0.7078 | 0.4477 | best clean/content anchor |
| EMA dynamic frontier | 0.7093 | 0.4690 | good balanced anchor |
| EMA support SConv guard | 0.7110 | 0.4680 | local operator helps, still below target |
| EMA support SConv style | 0.7168 | 0.5261 | style rises by spending too much content |
| EMA routed actuator | 0.7157 | 0.5020 | structural routing helps but gate is too coarse |

Interpretation:

- KL-f4 and SDXL failures mean the immediate bottleneck is not just "VAE capacity".
- EMA is the only replacement backend worth continuing, but its clean style ceiling is currently around `0.708-0.716`.
- The SaMST/SaMam idea we tried, style-conditioned local filtering, is directionally correct: it beats scalar loss-only probes.
- But our current SConv/router is still source-edge scalar gating. It cannot decide object/region-level texton placement, so stronger style becomes flat-region drift and high-frequency mist.

This updates the next design target:

> Do not add more global style pressure. Add object/region-aware operator routing.

The next EMA probe should start from `ema_dynamic_frontier_w32` or `ema_sconv_support_w30_guard`, then add a small region-aware style bank:

- low-pass color path: bounded and optionally flat-region capped;
- high-pass texton path: style-conditioned depthwise kernels, zero-init gate;
- region gate: derived from semantic bins / low-frequency content clusters / style-pair assignment, not only edge magnitude;
- diagnostic gate: reject variants whose Seedream-gap highpass or flat-flood gap rises without real CLIP-style gain.

If this fails, the honest VAE-backend conclusion is:

> `sd-vae-ft-ema` is useful diagnostically and sometimes cleaner than the original VAE, but it does not yet outperform the original VAE under the requested `clip_style > 0.72` and `content_lpips < 0.45` target. KL-f4 and SDXL are worse drop-in backends for the current 256x256 LANCET design.
