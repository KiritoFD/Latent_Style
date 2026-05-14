# Related Works 750 Evaluation Summary

Source folder: `run_511/complete_750`

| Method | Run | LPIPS down | CLIP-style up | Blur-drop down | Down-drop down | CLIP-content up | SSIM-Y up | Edge-F1 up | Extra-edge down | Chroma-Z down | FlatChroma-Z down | Risk flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SaMST strict | `samst_strict` | 0.4664 | 0.7194 | 0.0025 | -0.0012 | 0.8193 | 0.652 | 0.5162 | 0.1077 | 0.2636 | -0.3002 | grainy_but_structured |
| StyleID strict | `styleid_strict` | 0.7497 | 0.7597 | -0.0005 | -0.012 | 0.5519 | 0.1466 | 0.1954 | 0.2886 | 0.7909 | 1.9891 | weak_content,semantic_drift,noisy_vs_style,chroma_speckle |
| s2wat_strict | `s2wat_strict` | 0.5263 | 0.7139 | -0.013 | -0.0092 | 0.7465 | 0.507 | 0.2574 | 0.0776 | -0.4523 | -0.5766 |  |
| AdaIN v32k | `adain_v32k` | 0.6298 | 0.713 | 0.0391 | 0.0185 | 0.699 | 0.3246 | 0.0167 | 0.0071 | 1.1066 | -1.271 | washed_structure,blocky,style_not_blur_robust,style_not_scale_robust,chroma_speckle |
| AdaIN vgg19 | `adain_vgg19` | 0.687 | 0.693 | 0.0662 | 0.0361 | 0.5991 | 0.2897 | 0.0182 | 0.0197 | 0.3528 | -1.4401 | semantic_drift,washed_structure,blocky,style_not_blur_robust,style_not_scale_robust |
| AdaIN bad | `adain_bad` | 0.849 | 0.6308 | 0.007 | 0.0 | 0.5297 | 0.2868 | 0.0 | 0.0 | 410.2368 | -1.829 | weak_content,semantic_drift,washed_structure,chroma_speckle |

## Notes

- `Blur-drop` and `Down-drop` measure how much CLIP-style falls after mild blur or down-up sampling.
- Large positive `Chroma-Z` / `FlatChroma-Z` suggests color-speckle behavior stronger than the target style distribution.
- `Extra-edge` measures output edges that appear outside a dilated content-edge support mask.
