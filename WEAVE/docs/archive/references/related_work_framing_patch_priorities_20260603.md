# Related-Work Framing Patch Priorities

Date: 2026-06-03  
Scope: paper-surface literature/framing gaps only; no manuscript edits

## Bottom line

There is still no major missing-citation hole. The most plausible next fixes are all **framing repairs** on the current paper surface.

## Top gaps still worth patching next

### 1. Remove internal probe jargon from related work

The current compact-style paragraph now mixes external comparison with internal evidence:

- `paper_aaai2026.tex`, line 62 mentions a "landed payload-backed L-family successor packet on Distinct5-512."

That is not related work. It is internal paper evidence and uses local project jargon that a reviewer cannot parse without the review docs. If one related-work patch is made first, this is the cleanest target.

### 2. Sharpen the split by inference-time conditioning regime

The intro and related-work story still compresses too many families into one flow:

- intro line 38 groups `AesPA/AesFA`, `HSI`, `SCSA`, and diffusion methods tightly;
- intro line 40 still relies on a broad "reference-image-guided" grouping;
- related-work line 68 already has the right ingredients, but the separation is not yet maximally crisp.

The highest-value clarification remains the same one identified earlier: separate

1. reference-guided arbitrary stylization,
2. compact multi-style / style-id-conditioned transfer,
3. training-free diffusion or large-prior stylization.

This is still a framing gap, not a citation gap.

### 3. Keep `style tokenizer` explicitly disambiguated from diffusion/image-token prior art

The tokenizer memo still points to a real terminology risk:

- `SaMST` remains the strongest direct compact style-representation anchor;
- adjacent papers such as `StyleTokenizer` are close in wording but belong to a different control regime.

Given the current paper surface, one explicit sentence is still worth preserving whenever the tokenizer story is presented: here, the tokenizer is a **style-id-conditioned control map**, not a target-image encoder, VQ image tokenizer, or diffusion-token controller.

### 4. Keep the `idt` / no-op point framed as a bounded paper contribution

The evaluation paragraph is strong, but the paper still needs to avoid sounding as if the exact unchanged-image-prior correction is already community-standard related work.

The current local literature surface still supports:

- metric fragility and calibration sensitivity (`yeh2020calibrated`, `wright2022artfid`, `zhou2024comprehensiveeval`);
- adjacent identity-style sanity-check precedent;
- but not a claim that the exact `idt` reporting protocol is already established practice.

So the safest patch is still to frame `idt` as a **paper-specific diagnostic/reporting contribution** layered on top of prior metric-caution literature.

## Priority order

If only two literature/framing fixes are made next, they should be:

1. remove the internal `L`-family successor jargon from related work;
2. sharpen the regime split between reference-guided, compact multi-style, and diffusion-based lines.
