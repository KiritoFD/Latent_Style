# Style-Covariant Noise Probe Round 1

Date: 2026-06-16

## Scope

Eval-only matched probe on retained parent `lowanchor050 e9`.

- Parent checkpoint:
  `/mnt/i/Github/Latent_Style/exp/aaai2027_phase2_i2sb_orthogonal_lowanchor050_k070_e3_sigma0p02_b8a2_vlen010/epoch_0009.pt`
- Fixed contract:
  - transfer-only `CLIP-S + LPIPS`
  - `max_src_samples=10`
  - `generation_batch_size=8`
  - `metric_batch_size=16`
  - `target_chunk_size=5`
  - ONNX VAE decode `ema_b16_32`
- Changed variables only:
  - `bridge_sigma in {0.0, 0.5, 0.8, 1.2}`
  - `i2sb_noise_family in {gaussian, style_covariant}`

## Launcher

- Script:
  [run_phase2_style_covariant_probe_round1.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase2_style_covariant_probe_round1.sh)
- Output root:
  `/mnt/i/Github/Latent_Style/exp/inmortal-exp/phase2_style_covariant_lowanchor050e9/`
- Log root:
  [logs/style_covariant_probe](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/style_covariant_probe)

## Status

- `2026-06-16`: configs, launcher, and extractor were synced to remote WSL and the full `7`-point sweep completed.
- Result table:
  [style_covariant_probe_round1_results.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/style_covariant_probe_round1_results.csv)
- Matched deltas:
  [style_covariant_probe_round1_control_delta.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/style_covariant_probe_round1_control_delta.csv)
- Homepage figure/data source updated:
  [plot_points.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/plot_points.csv)
  and [fig_wikiart5_page1_summary.png](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/figures/fig_wikiart5_page1_summary.png)

## Current Read

- Internal deterministic control under the new probe contract is `0.694798 / 0.351760`.
- This does **not** exactly match the earlier historical `lowanchor050 e9` fast10 point `0.701429 / 0.372203`.
- Decision: treat this sweep as a clean **internal matched probe**, not as a replacement for the historical parent curve until the contract mismatch is reconciled.

## Matched Results

### Sigma 0.5

- Gaussian control: `0.712437 / 0.589486`
- Style-covariant: `0.709468 / 0.580127`
- Delta vs. Gaussian:
  - style `-0.002969`
  - LPIPS `-0.009360`
- Read: slight LPIPS recovery against isotropic, but still catastrophically out of band versus the deterministic control.

### Sigma 0.8

- Gaussian control: `0.696612 / 0.664919`
- Style-covariant: `0.701423 / 0.665792`
- Delta vs. Gaussian:
  - style `+0.004810`
  - LPIPS `+0.000873`
- Read: mild style gain, but LPIPS is still fully reopened; not promotable.

### Sigma 1.2

- Gaussian control: `0.683506 / 0.707997`
- Style-covariant: `0.697291 / 0.722679`
- Delta vs. Gaussian:
  - style `+0.013785`
  - LPIPS `+0.014682`
- Read: style-covariant is stronger than isotropic here, but only by paying even more structure damage.

## Closure

- Provisional decision: `negative_eval_only`.
- Main reason:
  - the mechanism does not produce any usable point near the retained band; every noisy point lands around `LPIPS 0.58-0.72`.
- Secondary reason:
  - even when style-covariant beats isotropic at fixed sigma, the gain is only inside a region already too damaged to matter for promotion.
- Practical conclusion:
  - this does not rescue latent-noise SDE on the retained `lowanchor050 e9` parent.
  - if stochastic work continues, it needs a much tighter structural constraint than "change the raw noise spectrum only."

## Instrumentation Note

- `i2sb_style_noise_family_style_covariant` and `i2sb_style_noise_family_gaussian` are recorded correctly in the exported summaries.
- But `i2sb_style_noise_bank_active`, `i2sb_style_noise_amp_mean`, `i2sb_style_noise_amp_std`, and `i2sb_style_noise_post_std` remained `0.0` in the aggregated runtime summary despite the logs showing:
  - `Style-covariant latent templates ready: 5/5 styles`
- This means the family switch clearly executed, but the deeper style-noise observability is not yet trustworthy.
- Action for the next infra pass:
  - audit where those debug scalars are overwritten or dropped before summary aggregation.
