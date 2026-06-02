# Evaluation Pathology / No-op Baseline Memo

Date: 2026-06-03  
Scope: literature-lane note only, focused on style-transfer evaluation pathologies, no-op / identity baselines, and 2024-2026 missing works relevant to metric hacking, content-style tradeoff evaluation, and benchmark hygiene.

## What already strengthens the current narrative

### 1. Protocol inconsistency and metric misalignment are already supported

The current narrative is well supported by three existing anchors:

- `yeh2020calibrated`  
  shows that automatic style-transfer metrics need calibration against perceptual judgment.
- `wright2022artfid`  
  provides a style-transfer-specific metric and explicitly motivates broader evaluation than naive perceptual/content scores alone.
- `zhou2024comprehensiveeval`  
  is the strongest current direct citation for benchmark hygiene: it argues that arbitrary style-transfer evaluation is protocol-sensitive, method-sensitive, and not reliably summarized by one metric.

These three are enough to justify writing that style-transfer evaluation is fragile and calibration-sensitive.

### 2. Identity-style sanity checks do have precedent

The best adjacent support for an identity/no-op sanity-check idea is:

- **Scaling Painting Style Transfer** (arXiv 2022, still relevant as a methodological reference)  
  explicitly introduces an **identity test for style transfer quality assessment**, i.e. using the same painting as both content and style to check whether the method can faithfully reproduce the source style image.

Primary source:  
[https://arxiv.org/abs/2212.13459](https://arxiv.org/abs/2212.13459)

This does **not** prove that our `idt` all-target copied-source baseline is already a community norm. But it does strengthen the argument that identity-style sanity checks are legitimate evaluation tools rather than an ad hoc invention.

## What still weakens the current narrative

### 1. No direct primary-source precedent for our exact `idt` diagnostic

I did **not** find a strong 2024-2026 primary-source paper that cleanly formalizes the exact pathology we care about:

- art-to-art transfer,
- high raw style score for the unchanged source,
- and the need to subtract an unchanged-image prior.

Implication:

- our current `idt` / no-op framing is still best written as a **paper-specific diagnostic contribution**;
- it should **not** be framed as if the community already recognizes this exact protocol as standard.

### 2. Broader preference / IQA papers are only peripheral support

There are newer adjacent evaluation papers, but they support the narrative only weakly:

- **HPSv3 (ICCV 2025)**: broad human-preference scoring, not style-transfer-specific
- **Image Quality Assessment: From Human to Machine Preference (CVPR 2025)**: broader IQA / preference alignment, not style-transfer-specific

These can support one sentence like "evaluation metrics continue to evolve toward human preference alignment," but they do not directly validate the no-op / metric-hacking claim.

## 2024-2026 missing works that matter most

### Must-care for diffusion / training-free freshness

1. **StyleSSP (CVPR 2025)**  
   Why it matters:
   - strengthens the 2025 diffusion-style-transfer lane;
   - useful if we want the diffusion comparison story to feel current beyond `StyleID` and `Z*`;
   - not directly about no-op pathology, but relevant to benchmark hygiene because stronger diffusion baselines increase the credibility of our evaluation discussion.

Primary source:  
[https://openaccess.thecvf.com/content/CVPR2025/html/Xu_StyleSSP_Sampling_StartPoint_Enhancement_for_Training-free_Diffusion-based_Method_for_Style_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_StyleSSP_Sampling_StartPoint_Enhancement_for_Training-free_Diffusion-based_Method_for_Style_CVPR_2025_paper.html)

### Optional adjacent evaluation anchors

2. **HPSv3 (ICCV 2025)**  
   Why it matters:
   - optional umbrella support for human-preference calibration language;
   - not style-transfer-specific.

Primary source:  
[https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html)

3. **Image Quality Assessment: From Human to Machine Preference (CVPR 2025)**  
   Why it matters:
   - optional support for benchmark-hygiene / human-preference alignment language;
   - still not style-transfer-specific.

Primary source:  
[https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html)

## Practical writing consequence

The safest current narrative remains:

1. style-transfer evaluation is known to be protocol-sensitive and metric-sensitive;
2. identity-style sanity checks have precedent in adjacent form;
3. our exact `idt` / unchanged-image-prior diagnostic is still relatively novel and should be claimed as a bounded contribution, not a standard community protocol;
4. if one more related work is added, `StyleSSP` is the highest-value 2025 add;
5. newer preference/IQA works are optional support, not core evidence for the no-op pathology itself.
