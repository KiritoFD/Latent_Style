# Tokenizer Representation Related-Work Refresh

Date: 2026-06-03
Scope: tokenizer / style-representation framing only; no manuscript edits

## Bottom line

The current tokenizer-related framing is still basically right. I did not find a newer 2025-2026 primary-source paper that displaces `SaMST` as the strongest direct baseline anchor for **compact multi-style style representation**. The main literature risk is not a missing stronger comparator; it is **terminology confusion** around the word `tokenizer`.

## What still holds

- `SaMST` remains the closest direct comparator for the paper's style-representation story because it is explicitly about **pluggable style representation learning for multi-style transfer** in a deployable many-style model.
- `SaMAM` / `Mamba-ST` remain useful for the **backbone-efficiency** lane, not for tokenizer semantics.
- `StyleID`, `Z*`, `SMS`, `StyleSSP`, and newer 2026 training-free diffusion papers such as `StyleGallery` still belong to the **reference-guided or large-prior stylization** lane, not to the compact style-id-conditioned representation lane.

## What needs fresher framing

The term `style tokenizer` can now attract adjacent prior art that is not actually the paper's main comparison class:

- `StyleTokenizer: Defining Image Style by a Single Instance for Controlling Diffusion Models` (ECCV 2024) is the clearest example. It is relevant as **terminology-adjacent** work because it uses tokenizer language for style control, but it is a zero-shot diffusion-control method from a single reference image, not a compact many-style transfer baseline.
- Older token-style generative papers such as `StylerDALLE` create a similar risk: they can make reviewers hear `tokenizer` as image-token or diffusion-token control rather than as a style-id-to-control map inside a feed-forward multi-style transfer system.

So the literature-side boundary should be explicit: in this paper, the tokenizer is a **style-side control map** from target style/domain identity to a compact control code, not a target-image encoder and not a VQ or diffusion image tokenizer.

## Strongest baseline framing now

If the tokenizer/representation paragraph is refreshed later, the safest baseline hierarchy is:

1. **Direct style-representation anchor**: `SaMST`
2. **Efficiency/backbone neighbors**: `SaMAM`, `Mamba-ST`
3. **Adjacent but different control regime**: `StyleTokenizer`, `StyleID`, `Z*`, `SMS`, `StyleSSP`, `StyleGallery`

That ordering preserves the real comparison logic:

- `SaMST` is closest on amortized multi-style deployment and pluggable style representation;
- the state-space papers change the executor/backbone more than the representation object;
- the diffusion papers change the conditioning regime itself by using reference images, text prompts, or large generative priors.

## Concise recommendation

No new must-add baseline family is required for the tokenizer claim. The best literature refresh is simply to keep `SaMST` as the main representation comparator and, if the prose is touched later, make one explicit disambiguation sentence that `style tokenizer` here means a **style-id-conditioned control map**, not a diffusion or image-token tokenizer.
