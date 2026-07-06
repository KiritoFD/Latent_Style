# Exploratory Blind Audit

Updated: 2026-06-08

This file records a single-pass exploratory blind rubric audit over the prepared A/B packet in:

- [blind_pairwise_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/blind_pairwise_manifest.csv)

Important boundary:

- this is **not** a human study
- this is **not** an external VLM service evaluation
- this is a blinded rubric pass over the packet using the already prepared A/B panels and fixed scoring questions
- it is therefore supplementary support only

Detailed per-case votes:

- [exploratory_blind_audit.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/blind_pairwise_v1/exploratory_blind_audit.csv)

## Aggregate readout

### `LBM-Knee` vs `SaMST`

- target-style match:
  - `SaMST` preferred in `6/6`
- content preservation:
  - `LBM-Knee` preferred in `6/6`
- artifact / visual cleanliness:
  - `LBM-Knee` preferred in `6/6`

Interpretation:

- this matches the current paper story very closely
- `SaMST` is the stronger high-style baseline
- `LBM-Knee` is the cleaner and more content-preserving promoted point

### `LBM-Knee` vs `Seedream`

- target-style match:
  - `Seedream` preferred in `6/6`
- content preservation:
  - `Seedream` preferred in `4/6`
  - `LBM-Knee` preferred in `2/6`
- artifact / visual cleanliness:
  - `Seedream` preferred in `5/6`
  - `LBM-Knee` preferred in `1/6`

Interpretation:

- the external large-prior reference remains visually stronger on style match
- the current paper should therefore avoid any same-interface dominance wording against Seedream

### `LBM-PS-v2` vs `SaMST`

- target-style match:
  - `SaMST` preferred in `6/6`
- content preservation:
  - `SaMST` preferred in `6/6`
- artifact / visual cleanliness:
  - `SaMST` preferred in `6/6`

Interpretation:

- this supports the current decision to frame `LBM-PS-v2` as a style ceiling row rather than a globally preferable operating point
- it also shows why the paper should continue to headline `LBM-Knee`, not `PS-v2`

### `LBM-K` vs `IDT`

- target-style match:
  - `LBM-K` preferred in `6/6`
- content preservation:
  - `IDT` preferred in `6/6`
- artifact / visual cleanliness:
  - `IDT` preferred in `6/6`

Interpretation:

- this is exactly the intended compact-anchor tradeoff
- `LBM-K` moves past the no-op floor
- `IDT` remains the visually cleaner but non-transferring baseline

## Bottom line

Even though this blind audit is only exploratory, it supports the current manuscript reading:

- `LBM-Knee` is the main closed Pareto point
- `SaMST` remains the stronger high-style compact baseline
- `Seedream` remains the stronger external large-prior visual reference
- `LBM-PS-v2` is a style ceiling, not the preferred visual operating point
