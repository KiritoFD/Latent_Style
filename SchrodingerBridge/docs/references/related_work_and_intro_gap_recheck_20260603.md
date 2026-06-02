# Related-Work and Intro Gap Recheck

Date: 2026-06-03
Scope: literature / intro-gap audit only; no manuscript edits

## 1) What is still weak

There is no major missing-citation hole left in the current intro or related-work sections. The remaining weakness is mostly **framing compression**, not bibliography coverage.

- In the introduction, the current sweep from `AesPA/AesFA` to `HSI` / `SCSA` to diffusion methods is still a little too compressed. It can make heterogeneous method families read like one undifferentiated "recent methods" bucket rather than three distinct lanes: reference-guided arbitrary stylization, compact multi-style/style-id transfer, and training-free diffusion adaptation.
- The intro also needs the `idt` / unchanged-image-prior point to stay framed as a **bounded paper-specific diagnostic**, not as if the community already treats this exact protocol as standard. The existing evaluation citations justify metric fragility and calibration sensitivity, but not a claim that the exact no-op adjustment is already established practice.
- No stronger 2025-2026 replacement has surfaced for `SaMST` as the main compact multi-style / style-representation comparator, so that comparator is still right; the risk is over-explaining around it, not under-citing it.

## 2) What is already adequate and should not be churned

The current citation spine is already strong enough for AAAI review and should stay stable unless the paper's scope changes:

- **Classical / arbitrary style transfer**: `gatys2016image`, `huang2017adain`, `park2019sanet`, `deng2022stytr2`, `hong2023aespa`, `huang2024aesfa`
- **Reference-guided injection / semantic consistency**: `zheng2024puffnet`, `zhang2025hsi`, `shang2025scsa`
- **Compact multi-style / style representation**: `liu2024samst` remains the right anchor; no clearly stronger replacement was identified
- **State-space efficiency line**: `liu2025samam`, `botti2025mambast`
- **Training-free / diffusion line**: `gim2024cast`, `chung2024styleid`, `deng2024zstar`, `jiang2025sms`, `xu2025stylessp`
- **Evaluation / calibration / protocol fragility**: `yeh2020calibrated`, `wright2022artfid`, `zhou2024comprehensiveeval`

`StyleSSP` was the one high-value freshness add for the 2025 diffusion lane, and it has already landed. `HPSv3` and "Image Quality Assessment: From Human to Machine Preference" remain optional peripheral adds only; they should not trigger churn in the current intro/related-work story.

## 3) Single best next literature-side edit

If only one literature-side change is made next, it should be a **framing edit**, not a citation add:

> Rewrite the intro/related-work transition so methods are grouped by **inference-time conditioning mode** rather than by a broad "recent methods" chronology.

Concretely, the cleanest split is:

1. reference-guided arbitrary stylization methods that consume style evidence at test time;
2. compact multi-style methods that amortize a fixed style family and use style identity rather than a target style image at inference;
3. training-free diffusion methods that leverage large generative priors but usually still depend on inference-time style evidence and higher sampling cost.

That one edit would do more for reviewer clarity than adding another paper, because the current gap is mainly about sharper positioning boundaries, not missing literature coverage.
