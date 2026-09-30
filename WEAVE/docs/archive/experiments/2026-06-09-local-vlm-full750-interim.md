# Local VLM Full750 Interim

Date: 2026-06-09

Scope:

- local VLM final review over the current finalist trio:
  - `LBM-Knee_e13`
  - `LBM-PS-v2_e13`
  - `Seedream_repaired750`
- endpoint:
  - xf-yun OpenAI-compatible API
- model:
  - `xopqwen36v35b`

Runtime status:

- the first validation case succeeded after switching from multi-image payloads to one composed comparison panel per case
- the long run now continues in the background from:
  - [vlm_distinct5_finalists_full750_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.jsonl)
- transient `500` cases are now:
  - retried multiple times
  - logged to:
    - [vlm_distinct5_finalists_full750_20260609.errors.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.errors.jsonl)
  - skipped without killing the full batch

Current interim evidence:

- completed cases so far:
  - `713`
- interim winner summary:
  - [vlm_distinct5_finalists_interim_summary_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_interim_summary_20260609.csv)
- current aggregate read:
  - `Seedream_repaired750` currently wins `709 / 713`
  - `LBM-PS-v2_e13` currently has `3 / 713`
  - `LBM-Knee_e13` still has `0 / 713`

Runtime repair added after early failures:

- the batch runner now retries failed cases with:
  - progressively smaller comparison panels
  - progressively lower JPEG quality
- this was added because several:
  - `Ukiyo_e`
  - and some complex `Early_Renaissance`
  targets were triggering repeated `500` responses at the larger panel size
- the new adaptive retry path has already converted some earlier failures into successful completions

Ordering repair added:

- the initial long run was still front-loaded toward `Early_Renaissance` source cases
- the batch runner has now been restarted with a deterministic hashed case order
- current completed-source coverage now already includes:
  - `Early_Renaissance`
  - `Impressionism`
  - `Ukiyo_e`
  - `Minimalism`
  - `Rococo`

Interpretation:

- the current VLM panel review agrees with the `IntroStyle-DINO` direction rather than the old `CLIP`-only reading
- `Seedream` is still visually strongest on target-style specificity in the early sampled full750 cases
- `LBM-Knee` is not being rewarded as strongly for style, but its notes repeatedly preserve a relative structure advantage over `LBM-PS-v2`
- the new larger sample keeps the same qualitative conclusion as the smaller early snapshots

Next step:

- let the full batch continue
- once the completed case count is materially larger, summarize:
  - winner counts
  - average per-method style / structure / artifact scores
  - target-style-specific failure clusters
