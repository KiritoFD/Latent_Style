# WikiArt512 Inference Speed Record - 2026-06-02

## Scope

This note records same-machine WSL generation-only throughput on the WikiArt512
5-style test protocol:

- Styles: `Realism`, `Impressionism`, `Post_Impressionism`, `Expressionism`, `Symbolism`
- Sources: 30 images per source style
- Transfers: 5 target styles per source image
- Total outputs: `30 * 5 * 5 = 750`
- Metrics are not computed in these runs; the timing is pure inference plus image write.
- Official timing artifact format is PNG. JPEG timings are not used for evaluation claims.

## Main Timing Table

| Method | Checkpoint | Output format | Outputs | Internal time | External wall | sec/img | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| LANCET | `epoch_0008.pt` | PNG | 750 | 46.79s | 54.80s | 0.073 | `summary_grid` disabled for pure throughput |
| LANCET | `epoch_0008.pt` | PNG | 750 | 70.12s | 77.49s | 0.103 | with `summary_grid.png` generation |
| SaMAM | step 10000 | PNG | 750 | 148.0s | 156.45s | 0.209 | official SaMAM curve script, generate-only |
| SaMST | epoch 15 | PNG | 750 | 319.97s | 320.19s | 0.427 | SaMST wrapper now converts outputs to PNG |

## From-Scratch LANCET Timing

This run uses `configs/archive/20260603_local_wsl_wikiart512/local_wsl_wikiart512_timing_from_scratch_20260602.json`
on local WSL with WikiArt512 EMA latents. It starts from no checkpoint, trains
8 epochs, evaluates `epoch_0008.pt`, and keeps the same 750-output PNG protocol.

| Stage | Wall time | Unit | sec/img or step | Notes |
|---|---:|---:|---:|---|
| Train from scratch | 66.56s | 8 epochs / 88 steps | 0.756s/step | batch 32, virtual length multiplier 0.02 |
| Generation only | 55.16s | 750 PNG | 0.0735s/img | no summary grid |
| Eval only, cold ref cache | 73.43s | 750 PNG | 0.0979s/img | reuses generated images, includes ref feature cache build |
| Eval only, hot ref cache | 62.54s | 750 PNG | 0.0834s/img | reuses generated images, ref cache hit |
| Direct full eval | 106.62s | 750 PNG | 0.1422s/img | one command: generate + metrics, hot ref cache |

Direct full eval metrics from this from-scratch `epoch_0008.pt`:

| Scope | clip_style | content_lpips |
|---|---:|---:|
| all 5x5 pairs | 0.7738 | 0.3941 |
| transfer only, excluding identity | 0.7679 | 0.3943 |

## Artifact Paths

- LANCET PNG no-grid:
  `SchrodingerBridge/exp/timing_20260602/run_eval_png750_b12_v2_w8_nogrid`
- LANCET PNG with grid:
  `SchrodingerBridge/exp/timing_20260602/run_eval_png750_b12_v2_w8_grid`
- SaMAM PNG:
  `Related_Works/baseline_pipeline/results/timing_20260602/samam_512_step10000_generate750_rerun_pngcomp`
- SaMST PNG:
  `Related_Works/baseline_pipeline/results/timing_20260602/samst_wikiart512_epoch15_generate750_png`
- LANCET from-scratch training:
  `SchrodingerBridge/exp/timing_20260602/lancet_from_scratch_b32_e8`
- LANCET from-scratch direct full eval:
  `SchrodingerBridge/exp/timing_20260602/lancet_from_scratch_e8_full_eval_direct750`

The generation/evaluation artifact directories were checked for `750`
generated PNG files.

## LANCET Settings

The current fastest official LANCET PNG command uses:

```text
--vae_model ema
--batch_size 12
--target_chunk_size 5
--vae_decode_batch_size 2
--image_save_workers 8
--image_save_backend pil_png
--no-save_summary_grid
--generation_only
```

The `summary_grid.png` path is intentionally optional. It is useful for visual
inspection but should not be included in pure model throughput comparisons.

## LANCET Breakdown

For the no-grid LANCET run, internal timing was:

| Stage | Time |
|---|---:|
| `uint8_cpu_copy` | 32.55s |
| `lancet_generation` | 4.52s |
| `source_load_to_device` | 3.52s |
| `vae_decode` | 2.58s |
| `encode_inversion` | 1.27s |
| `load_vae` | 1.25s |
| `load_lancet` | 0.69s |

The remaining bottleneck is still image materialization and CPU transfer, not
the LANCET field evaluation or VAE compute.

## Conclusions

1. LANCET is currently the fastest of the three under the same 750-image PNG
   output protocol.
2. Against SaMAM, LANCET is about `156.45 / 54.80 = 2.85x` faster in pure
   generation wall time.
3. Against SaMST PNG output, LANCET is about `320.19 / 54.80 = 5.84x` faster.
4. JPEG results should not be mixed into official evaluation timing because
   they change pixels and can affect CLIP/LPIPS. SaMST previously produced JPG
   by inheriting source suffixes; the wrapper now converts generated outputs to
   PNG for metric-compatible runs.

## Next Infra Work

The useful remaining optimization target is reducing `uint8_cpu_copy` and image
materialization overhead. Model-side generation is already below 5 seconds for
750 transfers, so further model kernel work is lower priority than output
pipeline work.
