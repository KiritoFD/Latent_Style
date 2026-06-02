# Intro / Related-Work Claim Audit

Date: 2026-06-03  
Scope: `aaai_submission/paper_aaai2026.tex` intro + related-work only  
Purpose: list unsafe or outdated claims, exact anchors, and sentence-level rewrite direction

## Audit items

| Anchor | Current claim risk | Rewrite direction |
|---|---|---|
| `Introduction`, line 38 | The sentence compresses several 2024-2025 methods into one broad trend line: "aesthetic quality, semantic consistency, or efficient global modeling through modules such as AesPA/AesFA, HSI, and SCSA." This mixes different problem settings and makes `HSI` / `SCSA` sound like the same kind of contribution as `AesFA`. | Split by function, not by date. Keep `AesPA/AesFA` as perceptual-quality arbitrary style transfer, and move `HSI` / `SCSA` into a later clause about reference-guided injection or semantic consistency. |
| `Introduction`, line 40 | "Many recent methods are still reference-image-guided" is broadly right, but the grouping currently mixes arbitrary reference-guided methods with diffusion adaptation methods without stating that the common factor is inference-time style evidence. | Rewrite the sentence so the common criterion is explicit: "methods that consume a style exemplar or style-conditioned inference signal at test time." |
| `Introduction`, line 40 | "Our setting is closer to the former compact multi-style regime" is useful, but "former" is locally ambiguous because the previous sentence names both compact multi-style and state-space lines. | Replace "former" with the exact phrase: "compact multi-style regime." |
| `Introduction`, line 42 | "indiscriminate Euclidean endpoint penalties" is currently safe, but still slightly too broad if read quickly; it can be misread as an attack on latent-space `L2` in general. | Narrow to "pointwise Euclidean endpoint reconstruction used as the style-alignment objective" and explicitly contrast that with acceptable `L2` roles such as kinetic regularization. |
| `Introduction`, line 44 | "time-conditioned latent vector field trained under bridge-style transport supervision" is safer than earlier FM-heavy wording, but still sounds more textbook-FM-clean than the active Distinct5 evidence family warrants. | Rephrase toward neutral supervision language: "trained to predict transport updates under endpoint-coupled bridge supervision." Avoid making this sentence sound like a clean empirical proof of random-time FM. |
| `Related Work / Classical and arbitrary`, line 59 | "These works mainly ask how a style exemplar should be fused at inference time" is directionally right, but too reductive for `Puff-Net` / `SCSA`, which also change feature disentanglement and semantic-region behavior. | Soften to: "These works primarily refine how style evidence is injected and aligned at inference time." |
| `Related Work / Efficient multi-style`, line 62 | "SaMST is the closest recent representative" is fine for your paper, but the paragraph currently understates that this line is about amortized fixed-style deployment rather than arbitrary-style generalization. | Add one short sentence clarifying that the line targets "a predefined style family in one deployable model" rather than unconstrained exemplar transfer. |
| `Related Work / State-space and Mamba-based`, line 65 | "arbitrary or many-style transfer" is too loose for the cited pair. It risks overclaiming what the state-space line has already established across problem settings. | Rewrite to a narrower formulation: "for efficient global mixing in recent style-transfer backbones" and avoid implying a settled state-space consensus across all transfer regimes. |
| `Related Work / State-space and Mamba-based`, line 65 | "The distinction is that LBM does not claim a new global-mixing backbone; its main contribution is objective-side" is good, but "objective-side" is a little vague. | Make the distinction concrete: "endpoint construction, transport supervision, and terminal distribution matching" rather than abstract "objective-side." |
| `Related Work / Training-free and diffusion`, line 68 | "SMS further pushes this direction by casting stylization as diffusion-side style distribution matching..." is useful, but if kept, it should be framed as another reference-guided / large-prior line, not as directly comparable compact deployment. | Add one clause making the contrast explicit: "within a large generative prior rather than a compact multi-style latent model." |
| `Related Work / Distribution matching and perceptual evaluation`, line 70 onward | This paragraph is currently the cleanest one, but "CLIP-derived scores as useful but incomplete diagnostics" still leaves open the question of what exactly the paper contributes beyond repeating prior metric caution. | Add one sentence making the paper's narrower contribution explicit: not a new universal metric, but an `idt`-anchored reporting protocol for separated art-to-art transfer. |

## Highest-priority rewrites

If only three intro/related edits are made next, they should be:

1. Narrow line 42 so it cannot be read as "latent `L2` is bad in general."
2. Tighten line 44 so the intro does not over-sell the active Distinct5 family as a clean random-time FM empirical story.
3. Split the line-38/59 literature grouping so `AesFA`, `HSI`, `SCSA`, `StyleID`, and `Z*` are not narratively collapsed into one undifferentiated "recent methods" bucket.

## Safe positioning summary

The safest intro / related-work stance remains:

- reference-guided arbitrary style transfer improves style-evidence injection and semantic alignment;
- compact multi-style transfer improves amortized deployment over a fixed style family;
- state-space methods improve backbone efficiency, not necessarily the supervision story;
- training-free diffusion methods improve raw style flexibility but usually rely on large priors and inference-time style evidence;
- this paper's supported differentiation is endpoint-side OT + SA-SWD / `W1`-style matching plus evaluation calibration, not a closed general theorem about all latent local-loss geometries.
