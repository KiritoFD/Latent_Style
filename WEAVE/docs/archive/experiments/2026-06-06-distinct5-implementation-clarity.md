# Distinct5 Implementation Clarity Packet

Date: 2026-06-06

Purpose:

- answer the reviewer-side implementation questions with an auditable current
  contract
- record the exact data roots, latent/VAE contract, pairing cache contract, and
  full-eval contract for the current paper-facing Distinct5-512 family

Scope:

- current paper-facing `H` family as carried into:
  - `configs/distinct5_512_ema_variant_h_hard_explore_queue_e3.json`
  - `configs/aaai2027/mainline_h_seed42_b44_base.json`
  - `configs/aaai2027/executor_promotion_h_e1_seed42_b44.json`
  - `configs/aaai2027/mainline_h_softterm*.json`

## 1. Style domains

The current Distinct5-512 family inherits the following five style domains from
[distinct5_512_ema_baseline_direct_atom_residual.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/distinct5_512_ema_baseline_direct_atom_residual.json):

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

`model.num_styles = 5` is explicitly set in the same base config, and
`src/run.py` force-corrects `model.num_styles` to match the loaded dataset if a
mismatch is detected.

## 2. Current train/eval roots

Historical Distinct5 configs were first authored on `/mnt/f/...`, but the
current paper-facing AAAI packet overrides the active roots to `/mnt/i/...` in
[mainline_h_seed42_b44_base.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/mainline_h_seed42_b44_base.json).

Current authoritative train/eval roots:

- latent train root:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`
- latent cache dir:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache`
- full-eval image root:
  - `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- eval cache root:
  - `/mnt/i/Github/Latent_Style/eval_cache`
- CLIP HF cache root:
  - `/mnt/i/Github/Latent_Style/eval_cache/hf`

Interpretation:

- `/mnt/f/...` should now be treated as historical provenance only for the old
  Distinct5 config lineage
- the active AAAI packet is already normalized onto `/mnt/i/...`

## 3. Latent tensor contract

Authoritative code facts:

- `model.latent_channels = 4` in
  [config_schema.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)
- the latent dataset loader enforces per-sample tensor shape `[C,H,W]` and
  raises if the loaded tensor is not rank-3:
  [dataset.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/dataset.py)
- latent encode/decode are handled through `AutoencoderKL` in
  [inference.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py)

Current latent-scale contract:

- model latent scale factor:
  - `0.18215`
- encode path:
  - `latent = vae.encode(image).latent_dist.sample()`
  - `latent = latent * vae.config.scaling_factor`
- decode path:
  - `latent = latent / scaling_factor`
  - `image = vae.decode(latent).sample`

Inference from the current SD-style VAE contract:

- the paper-facing `512x512` line uses `4` latent channels at `1/8` spatial
  reduction, i.e. `4x64x64` for `512x512` images
- this is consistent with the runtime compile dummy in `load_vae`, which uses a
  decoder input shaped like `[1, 4, 64, 64]`

This last spatial-resolution statement is an inference from the active VAE
contract plus the current runtime path, not a separately hard-coded shape
assertion.

## 4. VAE contract

The Distinct5 base full-eval contract uses:

- `full_eval.vae_model = "ema"`

`src/utils/inference.py` resolves that preset through:

- `"ema" -> "stabilityai/sd-vae-ft-ema"`

Other supported presets currently include:

- `mse -> stabilityai/sd-vae-ft-mse`
- `sdxl -> stabilityai/sdxl-vae`
- `sdxl-fix -> madebyollin/sdxl-vae-fp16-fix`

So the current paper-facing Distinct5 family is still on the SD-EMA VAE line,
not the SDXL VAE line.

## 5. Pairing-cache contract

The current `H` family inherits the prototype-pairing queue from the
`E -> F -> H` chain:

- base latent pairing cache path from `E`:
  - `/mnt/f/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`
- current active AAAI override path:
  - `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt`

Current paper-facing settings after the active AAAI override:

- `pairing_cache_topk = 8`
- `pairing_cache_active_topk = 2`
- `pairing_cache_sample_mode = "rank_biased"` inherited from `F`
- `pairing_cache_rank_schedule = "fixed"` overridden in `H`
- `pairing_cache_min_topk = 2`
- `pairing_cache_curriculum_epochs = 0`
- `pairing_cache_rank_power = 1.5` inherited from `F`
- `pairing_cache_explore_prob = 0.15`
- `pairing_cache_explore_topk = 8`
- `pairing_cache_cross_only = true`

Operational meaning:

- the offline cache stores up to top-8 prototype-aware target candidates
- the active `H` packet normally draws from the top-2 band
- `15%` of samples may explore broader hard-ranked candidates from the top-8
- identity-only cache reuse is disabled by `pairing_cache_cross_only = true`

## 6. Full-eval contract for the current paper-facing family

Current full-eval defaults inherited from the Distinct5 base:

- `num_steps = 12`
- `step_size = 1.0`
- `style_strength = 1.0`
- `batch_size = 4`
- `target_chunk_size = 1`
- `vae_decode_batch_size = 4`
- `max_src_samples = 30`
- `max_ref_compare = 30`
- `max_ref_cache = 30`
- `ref_feature_batch_size = 16`
- `only_lpips_clip_style = true`

Relevant code path:

- [run_evaluation.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py)

Important behavior:

- `run_evaluation.py` now respects explicit CLI overrides and only backfills
  unset full-eval fields from config/defaults
- this matters for reproducibility because the batch-related knobs are no longer
  silently overwritten by checkpoint-side defaults

## 7. Current AAAI weekly packet surface

The current weekly `A1/A2` packet stays on the same Distinct5-512 latent/eval
contract above and changes only:

- checkpoint initialization / freeze policy in `A1`
- endpoint pressure and routing softness in `A2`

It does **not** change:

- dataset styles
- latent channel count
- VAE family
- full-eval step contract
- target image root

## 8. Reviewer-facing bottom line

The current paper-facing Distinct5 family is a `5`-style packed-latent
SchrodingerBridge line trained on `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train`,
decoded with the SD-EMA VAE preset, evaluated on
`/mnt/i/wikiart_distinct5_samam_512_classview/test`, and paired through a
prototype-aware top-8 cache whose active `H` packet normally samples from the
top-2 band with fixed hard exploration.
