# Tokenizer Tail First-Pass Summary

Date: 2026-06-12

Purpose:

- summarize the first direct new-data round-1 attempts for the tokenizer families
- keep one concise node-level record before moving into tokenizer warmstart / reconstruction-pretrain packets

## Shared Infra Outcome

- DINO sidecar path:
  - matching new-data DINO cache is now available at:
    - `/mnt/i/Github/Latent_Style/eval_cache/offline_pairing/dinov2_wikiarts_5_full_notest_train_cache.pt`
- runtime bug fixed:
  - `semantic_tokenizer.py` now reconstructs the real DINO patch grid first
  - then interpolates to the latent target grid
- offline cache build path:
  - `build_offline_dino_pairing_cache.py`
  - `launch_remote_round1_dino_cache_build.py`
  - now support offline HF-cache-backed loading on the remote host

## Family Reads

- `tok_a_dino_dict`
  - first direct new-data run is now a clear strict bracket:
    - `batch=8`
      - enters training
      - later hits the hard cap around `11651MiB`
    - `batch=7`
      - enters training
      - but 180-second health read stays around `8546MiB`
  - current status:
    - `recalibration_needed`

- `tok_c_residual_adapter`
  - first direct new-data run is also a clear strict bracket:
    - `batch=8`
      - enters an in-band lane early
      - later falls to about `8313MiB`
      - under-band stop before first retained checkpoint
    - `batch=9`
      - later rises to about `11898MiB`
      - hard-cap stop before first retained checkpoint
  - current status:
    - `recalibration_needed`

- `tok_d_vlm_prompt`
  - direct strict bracket:
    - `batch=8`
      - overshoots to about `11896MiB`
    - `batch=7`
      - later health read around `8532MiB`
  - current status:
    - `recalibration_needed`

- `tok_b_cross_image`
  - first direct attempt is the most promising of the tokenizer tail:
    - `batch=8`
      - enters formal band early near `9793MiB`
      - later drifts down to about `8279MiB`
      - under-band stop before first retained checkpoint
    - `batch=9/10`
      - no clean second bracket point yet
      - retries were polluted by repeat `OSError: [Errno 5] Input/output error` during trainer log initialization
  - current status:
    - `recalibration_needed`

## Decision

- tokenizer tail direct family launches have now produced enough first-pass evidence
- the next useful step is not more brute-force direct family retries first
- move to:
  - tokenizer warmstart / reconstruction-pretrain packets
  - start with `tok_b_cross_image`

## Why `tok_b_cross_image` First

- it is the only tokenizer family that already showed a clean early in-band direct read
- its current blocker is mixed:
  - late under-band drift
  - plus run-root I/O instability on later retries
- warmstart / pretrain is more likely to convert that partial signal into a stable formal lane than continuing direct retries immediately
