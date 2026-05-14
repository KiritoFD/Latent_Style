# Advanced Metrics Toolchain

This directory documents the current protocol-750 evaluation stack. The strict image set is `5 source domains x 5 target domains x 30 images = 750 outputs`.

## Canonical Inputs

- Reference manifest: `SchrodingerBridge/exp/pareto_probe_4/S-add__K-3_C-2_W-10_Col-15/full_eval/epoch_0001/images`
- Ours row: `SchrodingerBridge/S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0007/images`
- Aggregated strict runs: `Related_Works/run_511/complete_750/<run>/images`

## Metric Scripts

Run these from the workspace root.

```bat
python Related_Works\run_511\eval\eval_750.py --help
python Related_Works\run_511\eval\eval_guard_750.py --help
python Related_Works\run_511\eval\eval_artifact_pack_750.py --help
python Related_Works\run_511\eval\eval_hf_patch_kid_750.py --help
python Related_Works\run_511\eval\eval_plain_kid_750.py --help
```

## Metric Blocks

| Block | Script | Output | Coverage now | Purpose |
| --- | --- | --- | --- | --- |
| Base style/content | `eval_750.py` | `eval_protocol750_sbmatch.json` | all strict 750 runs | LPIPS-content, CLIP-style, CLIP-content using SB-style target prototypes |
| Guard metrics | `eval_guard_750.py` | `eval_guard750.json` | all strict 750 runs | SSIM-Y, edge alignment, blur/downsample robustness, extra edges, chroma speckle |
| Artifact quality pack | `eval_artifact_pack_750.py` | `eval_artifact_pack750.json` | Ours, SaMST | MUSIQ, MANIQA, DISTS-content, denoise sensitivity, FFT shape, chroma-grain diagnostics |
| HF patch KID | `eval_hf_patch_kid_750.py` | `eval_hf_patch_kid750.json` | Ours, SaMST | High-pass patch distribution distance to real target styles |
| Plain KID | `eval_plain_kid_750.py` | `eval_plain_kid750.json` | Ours, SaMST | Standard image-level KID diagnostic |

## Summaries

```bat
python Related_Works\run_511\summaries\summarize_complete_750.py
python Related_Works\run_511\summaries\summarize_artifact_pack_750.py
python Related_Works\run_511\summaries\summarize_stroke_grain_750.py
python Related_Works\scripts\collect_repro_inventory.py
```

Main generated files:

- `Related_Works/run_511/complete_750/summary_complete_750.csv`
- `Related_Works/run_511/complete_750/summary_related_works_750.csv`
- `Related_Works/run_511/complete_750/summary_all_tested_metrics.csv`
- `Related_Works/results/metrics_summary/`
- `Related_Works/results/repro_data_inventory.csv`
- `Related_Works/docs/REPRO_DATA_INDEX.md`

## Reading The Current Metrics

The current standard metrics show that SaMST is structurally strong: CLIP-content, SSIM-Y, and Edge-F1 all favor it. The issue visible by eye is not content collapse but micro-grain artifacts: small color dithering, structured speckle, and high-frequency texture that does not resemble real brushstroke statistics.

Use the artifact pack for this failure mode:

- `MUSIQ` and `MANIQA`: no-reference perceptual quality.
- `HF-Patch-KID`: whether high-frequency patches resemble real target-style high-frequency patches.
- `FFT-Radial-KL-style` and `FFT-Slope-Error`: whether frequency falloff matches target paintings.
- `ChromaGrainIndex`: target-calibrated chroma residual coherence and blob statistics.

Plain KID is kept as a conventional distribution metric, but it currently does not penalize SaMST. Treat it as a supporting metric rather than an anti-grain detector.

## Dependencies

- Core: `torch`, `torchvision`, `Pillow`, `numpy`, `scipy`, `opencv-python`, `scikit-image`, `lpips`, `open_clip_torch`
- Strong diagnostics: `pyiqa`

Install missing metric dependencies in the active Python environment before running the full artifact pack.
