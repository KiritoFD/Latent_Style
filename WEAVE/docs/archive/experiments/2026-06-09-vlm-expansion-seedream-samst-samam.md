# VLM Expansion: Seedream, SaMST, SaMAM

Date: 2026-06-09

This note anchors the local CPU-only `VLM` expansion requested after the
original blind line was still only comparing against `Seedream`.

Formal intent of this branch:

- keep remote training independent
- keep local work on `CPU / network` only
- extend visual audit beyond `Seedream`
- explicitly compare current `LBM` against:
  - `Seedream`
  - `SaMST e15`
  - `SaMAM-2250`

## Current manifests

- `LBM-PS-v2 vs Seedream vs SaMST e15`
  - [vlm_manifest_lbmpsv2_vs_seedream_vs_samst_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_lbmpsv2_vs_seedream_vs_samst_20260609.csv)
- `LBM-PS-v2 vs Seedream vs SaMAM-2250`
  - [vlm_manifest_lbmpsv2_vs_seedream_vs_samam_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_manifest_lbmpsv2_vs_seedream_vs_samam_20260609.csv)

Current merged board:

- CSV:
  - [vlm_external_baseline_board_20260609.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_external_baseline_board_20260609.csv)
- Markdown:
  - [vlm_external_baseline_board_20260609.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_external_baseline_board_20260609.md)

## Current evaluator state

The earlier general-purpose `VLM` script was too brittle for this branch:

- larger comparison prompts were fine for some runs, but not stable enough here
- a substantial fraction of failures were not image I/O or auth failures
- there were two distinct failure families:
  - moderation / content filtering on some religious or sensitive cases
  - truncated JSON because the model stopped at `finish_reason=length`

To stabilize the branch, a dedicated simplified evaluator was added:

- [eval_xf_qwen_vlm_distinct5_simple.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_xf_qwen_vlm_distinct5_simple.py)

Current hardening in that script:

- simpler `3-way` panel protocol
- panel-letter supervision:
  - model returns `C/D/E`
  - local script remaps back to real run ids
- raw failure capture:
  - `raw_content`
  - `raw_response`
  - `finish_reason`
- wider generation budget than the first draft
- merged-board builder:
  - [build_vlm_external_baseline_board.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/build_vlm_external_baseline_board.py)

## Current read: `LBM-PS-v2 vs Seedream vs SaMST e15`

Authoritative running summary:

- [vlm_lbmpsv2_vs_seedream_vs_samst_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samst_20260609.method_summary.csv)

Current snapshot at this note:

- completed valid cases: `469`
- overall wins:
  - `Seedream = 386 / 469`
  - `SaMST e15 = 72 / 469`
  - `LBM-PS-v2 = 11 / 469`
- mean scores:
  - `LBM-PS-v2`
    - style `1.974`
    - structure `2.416`
    - artifact `2.072`
  - `SaMST e15`
    - style `3.618`
    - structure `3.557`
    - artifact `3.264`
  - `Seedream`
    - style `4.614`
    - structure `4.727`
    - artifact `4.699`

Current reading:

- `Seedream` is still the dominant visual winner
- `SaMST e15` is a real non-Seedream comparator, not just a dead baseline
- current `LBM-PS-v2` is visually behind both on the accumulated set so far

## Current read: `LBM-PS-v2 vs Seedream vs SaMAM-2250`

Authoritative running summary:

- [vlm_lbmpsv2_vs_seedream_vs_samam_20260609.method_summary.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/vlm_lbmpsv2_vs_seedream_vs_samam_20260609.method_summary.csv)

Current snapshot at this note:

- completed valid cases: `482`
- overall wins:
  - `Seedream = 326 / 482`
  - `SaMAM-2250 = 143 / 482`
  - `LBM-PS-v2 = 13 / 482`
- mean scores:
  - `LBM-PS-v2`
    - style `1.952`
    - structure `2.394`
    - artifact `1.952`
  - `SaMAM-2250`
    - style `3.817`
    - structure `4.133`
    - artifact `4.068`
  - `Seedream`
    - style `4.417`
    - structure `4.438`
    - artifact `4.301`

Current reading:

- `SaMAM-2250` is also visibly stronger than the current `LBM-PS-v2` packet
- relative to `SaMST`, current `SaMAM` looks more structure / artifact heavy
- `Seedream` still leads the branch overall

## Failure interpretation

Current expanded `VLM` failures should not be read as branch invalidation.

The current evidence says:

- moderation rejections are dataset-content dependent
- many non-moderation failures are length truncations, not semantic confusion
- the dominant non-moderation failure family is now directly audited as:
  - `finish_reason=length`
  - then truncated JSON rather than a semantically wrong winner
- therefore the branch is still usable as a final-review layer
- but it should not be used as the sole fast-loop selector

This remains aligned with the locked evaluation stack:

- fast screen:
  - `CLIP-S + LPIPS`
- paper-facing style axis:
  - `IntroStyle`
- local high-cost final review:
  - `VLM`

## Implication for the mainline

What this branch now already tells us:

- comparing only against `Seedream` was too narrow
- the current promoted `LBM-PS-v2` point is still not visually competitive with
  either `SaMST e15` or `SaMAM-2250` on the accumulated local audit
- with the larger accumulated set, this is no longer a tiny-sample artifact:
  - `SaMST` remains clearly above `LBM-PS-v2`
  - `SaMAM` also remains clearly above `LBM-PS-v2`
- the next mechanism push should not be framed as a minor polish step
- it needs to target the visual gap directly:
  - stronger target-style specificity
  - without collapsing structure
  - and without reverting into washed-out conservative outputs

## Next actions

1. Keep both local `VLM` lines accumulating in the background.
2. Periodically refresh the two method-summary CSVs and use them as the
   authoritative local visual board for `SaMST / SaMAM / Seedream`.
3. Keep remote GPU time on model-side mechanism work, not on local audit.
4. Use this expanded local read as a negative constraint on the next promoted
   mechanism family:
   - capacity-only continuation is not enough
   - visual competitiveness against external baselines still needs a stronger
     style-driving mechanism
