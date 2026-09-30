# SaMST Latent Distinct5-512 Same-Cost Closure

Date: 2026-06-06

Scope:

- method: latent `SaMST`
- dataset: `Distinct5-512`
- lane: `same-cost`
- machine: remote `RTX 3060 WSL`
- fast-screen protocol:
  - `transfer-only CLIP-S + LPIPS`
  - `--skip-art-fid`

## Why this note exists

The earlier preflight note only closed the launch and machine-contract side.

This note closes the first actual same-cost quality packet after the launcher
and wrapper contracts were repaired.

## Training packet

Run root:

- `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_041227`

Machine read:

- `30s` first-health passed at about `10728 MiB`
- the lane stayed stable under the hard `< 11.0 GiB` cap
- interval checkpoints were retained through:
  - `batch_id_50`
  - `batch_id_100`
  - `batch_id_150`
  - `batch_id_200`
  - `batch_id_250`
  - `batch_id_300`
  - `batch_id_350`
  - `batch_id_400`
  - `batch_id_450`
  - `batch_id_500`
  - `batch_id_550`
  - `batch_id_600`

Observed training pathology:

- the training log reports `content: nan`, `style: nan`, and `ae: nan`
  essentially from the beginning of the run
- therefore the checkpoint curve must be treated as quality-suspect even before
  evaluation

## Same-cost retained points evaluated

### Point A: nearest retained point to about `2 min`

Checkpoint:

- `ckpt_epoch_1_batch_id_50.pth`

Checkpoint timestamp:

- `2026-06-06 04:14:30 +0800`

Train wall:

- about `1.78 min` from launch

Fast eval packet:

- output:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_041227/eval_bundle_fast/batch050_fast`

Key transfer metrics:

- `clip_style = 0.6104`
- `content_lpips = 0.7296`
- `clip_dir = 0.0`
- `clip_content = 0.0`

Read:

- the outputs are collapsed
- every target style receives the same fixed `clip_style` constant
- directionality is zero everywhere

### Point B: nearest retained point to about `10 min`

Checkpoint:

- `ckpt_epoch_1_batch_id_300.pth`

Checkpoint timestamp:

- `2026-06-06 04:22:02 +0800`

Train wall:

- about `9.31 min` from launch

Fast eval packet:

- output:
  - `/mnt/i/Github/Latent_Style/Related_Works/baseline_pipeline/results/samst_latent_distinct5_512_samecost_20260606_041227/eval_bundle_fast/batch300_fast`

Key transfer metrics:

- `clip_style = 0.6104`
- `content_lpips = 0.7296`
- `clip_dir = 0.0`
- `clip_content = 0.0`

Read:

- the later point is operationally indistinguishable from the earlier point
- more training time did not move the packet toward a usable transfer regime

## Closure

Current closure:

- latent `SaMST` same-cost is now **operationally runnable** on the reviewed
  `3060`
- but the first closed same-cost packet is **quality-invalid**
- the packet collapses under the fast-screen metrics at both the early same-cost
  point and the later near-`10 min` point

What this means for the paper:

- this lane should not be promoted into a paper-facing positive baseline row
- it is valid negative evidence that the latent `SaMST` route, under the
  current packed-latent execution contract, is not competitive on
  `Distinct5-512`

Immediate consequence:

- keep the machine-side success as an engineering result
- keep the quality-side collapse as a negative baseline result
- do not spend more same-cost GPU time on latent `SaMST` unless a concrete
  stability fix for the `nan` loss pathology is identified first
