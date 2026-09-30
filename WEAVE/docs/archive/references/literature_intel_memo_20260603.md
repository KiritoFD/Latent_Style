# Literature Intelligence Memo for AAAI 2027

Date: 2026-06-03  
Scope: related-work intelligence only; no manuscript edits in this memo

## 1) Cited related works that are still enough

For an AAAI 2027 submission, the following core lines are already sufficient and should remain the spine of the narrative:

- **Classical / arbitrary style transfer**
  - `gatys2016image`
  - `huang2017adain`
  - `park2019sanet`
  - `deng2022stytr2`
  - `hong2023aespa`
  - `huang2024aesfa`

- **Efficient multi-style / compact deployable style families**
  - `liu2024samst`
  - `xia2024s2wat`

- **State-space / Mamba efficiency line**
  - `botti2025mambast`
  - `liu2025samam`

- **Training-free / diffusion style injection**
  - `gim2024cast`
  - `chung2024styleid`
  - `deng2024zstar`
  - `jiang2025sms`

- **Evaluation / calibration / protocol inconsistency**
  - `yeh2020calibrated`
  - `wright2022artfid`
  - `zhou2024comprehensiveeval`

These are enough to support the current five-lane story:
classical arbitrary transfer, compact multi-style transfer, state-space backbones, training-free diffusion methods, and evaluation calibration.

## 2) Missing or newer works that matter most

### A. Most useful missing work for the diffusion-style-transfer lane

**StyleSSP (CVPR 2025)** matters the most if one more method citation is added.  
Reason: it is a primary-source 2025 CVPR paper, directly on **training-free diffusion-based style transfer**, and makes the diffusion lane feel current rather than ending at `StyleID` / `Z*`.

### B. Most useful missing work for evaluation calibration

There is still **no direct primary-source paper I found that cleanly covers the exact `idt` / no-op prior issue in art-to-art style transfer**. That means the paper's current no-op argument remains mostly its own contribution.

If the paper wants one newer adjacent anchor for broader preference / metric calibration language, the safest options are:

1. **HPSv3 (ICCV 2025)**  
   Useful only as a general "human preference metrics keep evolving" citation.

2. **Image Quality Assessment: From Human to Machine Preference (CVPR 2025)**  
   Useful only as a broader IQA / preference-calibration anchor, not as a style-transfer-specific metric paper.

These two should stay peripheral. They do **not** replace `zhou2024comprehensiveeval` / `yeh2020calibrated` / `wright2022artfid` as the core style-transfer evaluation citations.

### C. Tokenizer / style-representation framing

For the **tokenizer-style-representation** framing, the literature signal is actually quite narrow:

- `liu2024samst` remains the strongest directly relevant anchor because it is explicitly about **pluggable style representation learning for multi-style transfer**.
- `liu2025samam` and `botti2025mambast` help on **backbone efficiency**, not really on tokenizer semantics.
- `zheng2024puffnet`, `zhang2025hsi`, and `shang2025scsa` help on **reference-guided style injection**, not on amortized style-code geometry.

So the right conclusion is: **there is no obviously stronger 2025-2026 primary-source replacement for SaMST as the main style-representation comparator**. The gap in the literature is real.

## 3) Exact bib entries or URLs to add

### Highest-priority add

```bibtex
@InProceedings{Xu_2025_CVPR,
  author    = {Xu, Ruojun and Xi, Weijie and Wang, XiaoDi and Mao, Yongbo and Cheng, Zach},
  title     = {StyleSSP: Sampling StartPoint Enhancement for Training-free Diffusion-based Method for Style Transfer},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {18260--18269}
}
```

Primary source:  
[CVPR 2025 OpenAccess - StyleSSP](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_StyleSSP_Sampling_StartPoint_Enhancement_for_Training-free_Diffusion-based_Method_for_Style_CVPR_2025_paper.html)

### Optional peripheral adds for evaluation language

```bibtex
@InProceedings{Ma_2025_ICCV,
  author    = {Ma, Yuhang and Wu, Xiaoshi and Sun, Keqiang and Li, Hongsheng},
  title     = {HPSv3: Towards Wide-Spectrum Human Preference Score},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {15086--15095}
}
```

Primary source:  
[ICCV 2025 OpenAccess - HPSv3](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html)

```bibtex
@InProceedings{Li_2025_CVPR,
  author    = {Li, Chunyi and Tian, Yuan and Ling, Xiaoyue and Zhang, Zicheng and Duan, Haodong and Wu, Haoning and Jia, Ziheng and Liu, Xiaohong and Min, Xiongkuo and Lu, Guo and Lin, Weisi and Zhai, Guangtao},
  title     = {Image Quality Assessment: From Human to Machine Preference},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {7570--7581}
}
```

Primary source:  
[CVPR 2025 OpenAccess - Image Quality Assessment: From Human to Machine Preference](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html)

## Bottom line

- **Enough already**: the core related-work spine is in place.
- **Most valuable missing add**: `StyleSSP` for the 2025 diffusion style-transfer lane.
- **Optional adds**: `HPSv3` and `Li 2025` only if the paper wants a slightly broader evaluation-calibration umbrella.
- **No stronger replacement found** for `SaMST` as the main tokenizer/style-representation anchor in compact multi-style transfer.
