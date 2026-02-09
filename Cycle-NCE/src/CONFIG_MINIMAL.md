# Overfit50 Minimal Config Guide

Use `overfit50_clean.json` for daily iteration. Keep only these sections:

- `model`: architecture and style-injection path
- `loss`: optimization targets
- `training`: runtime/infrastructure
- `data`: dataset root/subdirs
- `checkpoint`: output directory

## High-impact model keys

- `num_hires_blocks`: high-res 32x32 capacity (texture workspace)
- `use_style_texture_head`: explicit style-to-delta texture path
- `use_delta_highpass_bias`: suppress low-frequency brightness shortcut
- `use_content_skip_fusion`: decoder content skip path for structure retention
- `use_style_skip_gate`: style-conditioned gate for skip path

## High-impact loss keys

- `w_featmatch_hf`, `w_gram_hf`, `w_moment_hf`: style in high-frequency feature space
- `w_spatial_proto`: align style-id spatial priors to reference spatial style
- `w_prob`, `w_prob_margin`, `w_dir`: weak probability guidance (not hard argmax CE)
- `w_cycle`: low-frequency structure lock
- `w_ref_sep`, `w_proto_sep`: anti-collapse separation terms

## Infra keys (OOM first)

- `batch_size`: first lever for VRAM
- `full_eval_batch_size`: second lever for eval VRAM
- `auto_preload_gpu_budget_mb`: limit latent preload memory
- `channels_last`, `use_amp`, `amp_dtype`: throughput knobs

