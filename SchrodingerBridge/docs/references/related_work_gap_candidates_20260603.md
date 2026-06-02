# Related-Work Gap Candidates

Date: 2026-06-03
Comparison target:

- `aaai_submission/refs.bib`
- `docs/references/literature_intel_memo_20260603.md`

## Missing works

### 1. StyleSSP (CVPR 2025) - landed

- **Status in `refs.bib`**: added in the current AAAI 2027 writing pass
- **Why it matters**: this is the strongest missing citation for the 2025 diffusion / training-free style-transfer lane. It keeps the diffusion-related narrative current beyond `StyleID` and `Z*` and is directly relevant to artistic style transfer rather than generic IQA.
- **Priority**: closed
- **Primary source URL**:
  [https://openaccess.thecvf.com/content/CVPR2025/html/Xu_StyleSSP_Sampling_StartPoint_Enhancement_for_Training-free_Diffusion-based_Method_for_Style_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_StyleSSP_Sampling_StartPoint_Enhancement_for_Training-free_Diffusion-based_Method_for_Style_CVPR_2025_paper.html)

### 2. HPSv3 (ICCV 2025) - optional

- **Status in `refs.bib`**: missing
- **Why it matters**: only useful if the paper wants one more modern human-preference or evaluation-calibration umbrella citation. It is not style-transfer-specific and should stay peripheral.
- **Priority**: optional
- **Primary source URL**:
  [https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_HPSv3_Towards_Wide-Spectrum_Human_Preference_Score_ICCV_2025_paper.html)

### 3. Image Quality Assessment: From Human to Machine Preference (CVPR 2025) - optional

- **Status in `refs.bib`**: missing
- **Why it matters**: another optional evaluation-side umbrella citation for metric calibration language. It helps only if the paper wants to broaden the IQA or preference backdrop; it is not necessary for the core style-transfer story.
- **Priority**: optional
- **Primary source URL**:
  [https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.html)

## No-action items

- No stronger 2025-2026 replacement was identified for `SaMST` as the main compact multi-style or style-representation comparator.
- No must-add primary-source paper was identified that directly supersedes the current `no-op / idt / metric-hacking` framing; that part still appears to remain mainly the paper's own contribution.

## Bottom line

- **Closed in this pass**: `StyleSSP`
- **Still optional**: `HPSv3`, `Image Quality Assessment: From Human to Machine Preference`
- **No other high-priority add-only gaps** were identified from the current memo comparison.
