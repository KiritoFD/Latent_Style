# Local VLM Full750 Review

Date: 2026-06-09

Purpose:

- move the most expensive paper-facing qualitative audit to local execution
- use the xf-yun OpenAI-compatible endpoint with `xopqwen36v35b`
- compare the strongest current finalist methods on all aligned `Distinct5-512` transfer cases

Current comparison set:

- `LBM-Knee_e13`
- `LBM-PS-v2_e13`
- `Seedream_repaired750`

Per-case protocol:

- one VLM request per aligned source-target pair
- each request receives a single composed panel image containing:
  - source image
  - target-style reference image
  - candidate outputs for the three compared methods
- the model returns structured JSON with:
  - `best_overall`
  - `best_style_specificity`
  - `best_structure`
  - `best_artifact_control`
  - per-method 1-5 scores and notes

Rationale for composed panels:

- the earlier multi-image payload shape caused server-side `500` errors
- one composed comparison image is much more stable while preserving the same visual information

Artifacts:

- batch script:
  - [eval_xf_qwen_vlm_distinct5.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_xf_qwen_vlm_distinct5.py)
- running outputs:
  - [vlm_distinct5_finalists_full750_20260609.jsonl](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.jsonl)
  - [vlm_distinct5_finalists_full750_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.csv)
  - [vlm_distinct5_finalists_full750_20260609.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.log)
  - [vlm_distinct5_finalists_full750_20260609.err.log](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_distinct5_finalists_full750_20260609.err.log)

Current status:

- single-case validation succeeded
- full run has been resumed and is progressing locally in the background
- observed early cases currently prefer `Seedream_repaired750`
