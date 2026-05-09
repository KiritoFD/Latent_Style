# 2026-05-10 Full-Dimensional Orthogonal Sweep Plan

This plan turns the current black-dot mitigation baseline into a compact 12-run, 20-epoch sweep that adds SWD receptive-field scale as a first-class axis.

## Design intent

We are using the numerically safer post-mitigation regime as the base and then sweeping five orthogonal control axes:

1. SWD patch scales
2. SWD / kinetic force balance
3. semantic attention temperature
4. low-frequency AdaIN kernel size
5. cycle penalty

The suite is generated under:

- config root: [experiments/active/full_dimensional_orthogonal_sweep_20](/g:/GitHub/Latent_Style/SchrodingerBridge/experiments/active/full_dimensional_orthogonal_sweep_20:1)
- manifest: [manifest.json](/g:/GitHub/Latent_Style/SchrodingerBridge/experiments/active/full_dimensional_orthogonal_sweep_20/manifest.json:1)
- plan table: [plan.csv](/g:/GitHub/Latent_Style/SchrodingerBridge/experiments/active/full_dimensional_orthogonal_sweep_20/plan.csv:1)

Run outputs are designed to land in-place under `exp/runs/fd20_*`, with each run writing its own `full_eval/`.

## G0 baseline

`G0` is the current universe center for this sweep:

- `terminal_swd_weight = 10.0`
- `w_kinetic = 0.45`
- `semantic_attn_temperature = 0.12`
- `w_cycle = 0.20`
- `w_low_freq = 1.0`
- `low_freq_kernel_size = 5`
- `swd_patch_sizes = [3, 5, 7, 15]`
- `batch_size = 32`
- `learning_rate = 2e-4`
- `num_epochs = 20`
- `save_interval = 10`
- `model.base_dim = 64`
- `model.num_res_blocks = 4`

## Evaluation protocol

Use the funnel the sweep was designed for:

1. Screen all runs by `all_content_lpips`, with `> 0.48` treated as likely unstable.
2. Among survivors, rank by `photo_to_art_clip_style`.
3. Compare the top two visually against `G0` and `G1`, focusing on large color transitions and edge-local texture behavior.

## Entry points

- generator: [scripts/generate_full_dimensional_orthogonal_sweep_20.py](/g:/GitHub/Latent_Style/SchrodingerBridge/scripts/generate_full_dimensional_orthogonal_sweep_20.py:1)
- train all: [run_all.bat](/g:/GitHub/Latent_Style/SchrodingerBridge/experiments/active/full_dimensional_orthogonal_sweep_20/run_all.bat:1)
- eval all: [eval_all.bat](/g:/GitHub/Latent_Style/SchrodingerBridge/experiments/active/full_dimensional_orthogonal_sweep_20/eval_all.bat:1)
