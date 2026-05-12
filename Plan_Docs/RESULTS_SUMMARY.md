# Quantitative Results Summary

Updated: 2026-05-12 | Protocol: 750 images (5 source x 5 target x 30)

## 1. Timing

| Method | Train (s) | Infer (s) | sec/image | Notes |
|--------|-----------|-----------|-----------|-------|
| Ours 1ep | 52.5 | 85.4 | 0.114 | SB training CSV |
| Ours 7ep | 309.9 | 85.4 | 0.114 | Same infer (diffusion steps fixed) |
| SaMST (100ep) | 6768.7 | 39.8 | 0.053 | Train extrapolated: 1ep probe 67.7s x 100 |
| StyleID | 0 | 603.3 | 0.804 | Training-free |
| AdaIN v32k | 9220.4 | 9.3 | 0.012 | 32k iter, batch 8 |
| AdaIN vgg19 | 262.8 | 9.1 | 0.012 | 2k iter, batch 4 |

## 2. Main Metrics (LPIPS / CLIP-style / CLIP-content)

Source: `summary.json all_pairs_overview` (Ours), `eval_protocol750_sbmatch.json` (baselines)

| Method | LPIPS↓ | CLIP-style↑ | CLIP-content↑ |
|--------|--------|-------------|---------------|
| **Ours 1ep** | 0.4272 | 0.7036 | **0.8392** |
| **Ours 7ep** | 0.4514 | 0.7161 | 0.8086 |
| SaMST (100ep) | 0.4664 | **0.7194** | 0.8193 |
| StyleID | 0.7497 | 0.7597 | 0.5519 |
| AdaIN v32k | 0.6298 | 0.7130 | 0.6990 |
| AdaIN vgg19 | 0.6870 | 0.6930 | 0.5991 |

### Per-target (Ours 7ep)

| Target | LPIPS↓ | CLIP-style↑ | CLIP-content↑ |
|--------|--------|-------------|---------------|
| photo | 0.4470 | 0.7219 | 0.8356 |
| monet | 0.4552 | 0.7538 | 0.8429 |
| vangogh | 0.3992 | 0.8244 | 0.8752 |
| cezanne | 0.4251 | 0.7909 | 0.8448 |
| Hayao | 0.5168 | 0.6200 | 0.8203 |

### Per-target (SaMST 100ep)

| Target | LPIPS↓ | CLIP-style↑ | CLIP-content↑ |
|--------|--------|-------------|---------------|
| photo | 0.5670 | 0.6799 | 0.8207 |
| monet | 0.4094 | 0.7418 | 0.8364 |
| vangogh | 0.4116 | 0.7716 | 0.8356 |
| cezanne | 0.4101 | 0.7533 | 0.8356 |
| Hayao | 0.5336 | 0.6504 | 0.7684 |

## 3. Complete 750 Summary (guard + main)

Source: `run_511/complete_750/summary_complete_750.md`

| Method | LPIPS↓ | CLIP-style↑ | Blur-drop↓ | Down-drop↓ | CLIP-content↑ | SSIM-Y↑ | Edge-F1↑ | Extra-edge↓ | Chroma-Z↓ | FlatChroma-Z↓ | Risk flags |
|--------|--------|-------------|-----------|-----------|---------------|---------|----------|------------|----------|--------------|------------|
| Ours 7ep | 0.4587 | 0.7041 | 0.0034 | -0.0014 | 0.8043 | 0.4545 | 0.311 | 0.0764 | -0.486 | -0.5239 | |
| SaMST | 0.4664 | 0.7194 | 0.0025 | -0.0012 | 0.8193 | 0.652 | 0.5162 | 0.1077 | 0.2636 | -0.3002 | grainy_but_structured |
| StyleID | 0.7497 | 0.7597 | -0.0005 | -0.012 | 0.5519 | 0.1466 | 0.1954 | 0.2886 | 0.7909 | 1.9891 | weak_content,semantic_drift,noisy_vs_style |
| AdaIN v32k | 0.6298 | 0.713 | 0.0391 | 0.0185 | 0.699 | 0.3246 | 0.0167 | 0.0071 | 1.1066 | -1.271 | washed_structure,blocky,style_not_blur_robust |
| AdaIN vgg19 | 0.687 | 0.693 | 0.0662 | 0.0361 | 0.5991 | 0.2897 | 0.0182 | 0.0197 | 0.3528 | -1.4401 | semantic_drift,washed_structure,blocky |

## 4. Guard Metrics (structural quality)

Source: `eval_guard750.json`

| Method | SSIM-Y↑ | Edge-F1↑ | Edge-IOU↑ | Blockiness↓ | HF-gen↓ |
|--------|---------|----------|-----------|-------------|---------|
| Ours 7ep | 0.4545 | 0.3110 | 0.1867 | 1.1770 | 0.2419 |
| SaMST | 0.6520 | 0.5162 | 0.3548 | 1.3059 | 0.1546 |

## 5. AntiHF Metrics (artifact scores)

Source: `eval_antihf750.json`

| Method | style_raw↑ | hf_z↓ | hf_artifact_index↓ | hf_artifact_index_pos↓ |
|--------|-----------|-------|-------------------|----------------------|
| Ours 7ep | 0.7040 | -0.8686 | -0.5152 | 0.2477 |
| SaMST | 0.7193 | -0.9363 | -0.5293 | 0.2027 |

## 6. Artifact Pack (perceptual + FFT + chroma)

Source: `eval_artifact_pack750.json`

| Method | MUSIQ↑ | MANIQA↑ | DISTS-content↓ | FFT-KL-style↓ | FFT-slope-error↓ |
|--------|--------|---------|----------------|--------------|-----------------|
| Ours 7ep | **49.21** | **0.4057** | **0.2477** | **0.0853** | **0.5473** |
| SaMST | 36.10 | 0.3139 | 0.2943 | 0.2419 | 1.0536 |

Ours significantly better on perceptual quality (MUSIQ +13) and FFT artifacts (KL 3x lower).

## 7. KID Metrics

Source: `eval_plain_kid750.json` + `eval_hf_patch_kid750.json`

| Method | KID↓ | HF-Patch-KID↓ |
|--------|------|--------------|
| Ours 7ep | 0.0524 | **4.1694** |
| SaMST | **0.0489** | 6.7598 |

SaMST slightly better on plain KID, but Ours much better on HF-Patch-KID (less high-frequency artifact).

## 8. Ours 1ep vs 7ep Comparison

| Metric | 1ep | 7ep | Delta |
|--------|-----|-----|-------|
| CLIP-style | 0.7036 | 0.7161 | +0.013 |
| CLIP-content | 0.8392 | 0.8086 | -0.031 |
| LPIPS | 0.4272 | 0.4514 | +0.024 |
| Train time | 52.5s | 309.9s | +257.4s |

7ep trades slight content loss for style gain. Training 6x longer for +0.013 CLIP-style.

## 9. Key Findings

1. **Ours vs SaMST**: Very close on CLIP-style (0.716 vs 0.720). Ours wins on LPIPS (0.451 vs 0.466), perceptual quality (MUSIQ 49 vs 36), and FFT artifacts. SaMST wins on structure (SSIM 0.65 vs 0.45, Edge-F1 0.52 vs 0.31).

2. **SaMST structural advantage**: SaMST preserves edges and global structure better (higher SSIM, Edge-F1) but produces visible grain/pointillist artifacts (positive Chroma-Z, high HF-Patch-KID).

3. **Ours perceptual advantage**: Higher MUSIQ/MANIQA scores, lower FFT artifacts, cleaner chroma (negative Chroma-Z).

4. **Training efficiency**: Ours 1ep (52.5s) already competitive with SaMST (6768.7s) on CLIP-style (0.704 vs 0.719). 130x faster training.

5. **AdaIN/StyleID weak**: AdaIN has severe structural loss (Edge-F1 ~0.02). StyleID has weak content preservation (CLIP-content 0.55).
