# Weight Sweep 40 Scientific Conclusions

## Protocol

- Budget: 40 local experiments = 20 manual category-sampling recipes x `K={1,2}`.
- Base config: `S-add__K-1_C-0_W-20_Col-0/config.json`.
- Training: 8 epochs, checkpoint/evaluation at every epoch.
- Evaluation: strict 750-image protocol, `30 source images x 5 target styles x 5 source styles`, with local CLIP path injected by `run_evaluation.py`.
- Primary scalar score: `EC = CLIP-style * (1 - LPIPS)`. This rewards style strength only when content distortion is not too high.
- Secondary score: `0.65 * CLIP-style + 0.35 * (1 - LPIPS)`. This is less harsh and useful for sanity checking rank stability.

## Top Runs By Primary Score

| rank | experiment | epoch | style | content | LPIPS | EC | photo_style | photo_LPIPS | recipe |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | K2_r00_balanced_default | epoch_0003 | 0.6980 | 0.8727 | 0.3777 | 0.4343 | 0.6396 | 0.4036 | R00_balanced_default |
| 2 | K2_r18_cezanne_fix_prev | epoch_0001 | 0.6944 | 0.8584 | 0.3824 | 0.4289 | 0.6376 | 0.4138 | R18_cezanne_fix_prev |
| 3 | K2_r10_no_photo_target | epoch_0002 | 0.6972 | 0.8605 | 0.3875 | 0.4270 | 0.6384 | 0.4099 | R10_no_photo_target |
| 4 | K2_r02_prev_manual | epoch_0002 | 0.6990 | 0.8533 | 0.3891 | 0.4270 | 0.6404 | 0.4134 | R02_prev_manual |
| 5 | K2_r15_hard_art | epoch_0001 | 0.6980 | 0.8544 | 0.3897 | 0.4260 | 0.6413 | 0.4242 | R15_hard_art |
| 6 | K2_r06_monet_strong | epoch_0001 | 0.6978 | 0.8598 | 0.3897 | 0.4259 | 0.6401 | 0.4180 | R06_monet_strong |
| 7 | K2_r12_hayao_cezanne | epoch_0001 | 0.6955 | 0.8576 | 0.3880 | 0.4256 | 0.6380 | 0.4152 | R12_hayao_cezanne |
| 8 | K2_r11_photo_target_some | epoch_0001 | 0.6939 | 0.8585 | 0.3886 | 0.4243 | 0.6335 | 0.4049 | R11_photo_target_some |
| 9 | K2_r08_photo_content_high | epoch_0001 | 0.6962 | 0.8575 | 0.3910 | 0.4240 | 0.6424 | 0.4259 | R08_photo_content_high |
| 10 | K2_r16_photo_hayao_content_art_target | epoch_0001 | 0.6986 | 0.8541 | 0.3936 | 0.4236 | 0.6443 | 0.4269 | R16_photo_hayao_content_art_target |
| 11 | K2_r14_soft_art | epoch_0006 | 0.7040 | 0.8483 | 0.3996 | 0.4227 | 0.6466 | 0.4179 | R14_soft_art |
| 12 | K2_r03_hayao_strong | epoch_0001 | 0.6946 | 0.8555 | 0.3920 | 0.4223 | 0.6351 | 0.4110 | R03_hayao_strong |
| 13 | K2_r05_vangogh_strong | epoch_0001 | 0.6945 | 0.8562 | 0.3922 | 0.4222 | 0.6379 | 0.4197 | R05_vangogh_strong |
| 14 | K2_r17_art_content_art_target | epoch_0002 | 0.6977 | 0.8557 | 0.3953 | 0.4219 | 0.6381 | 0.4115 | R17_art_content_art_target |
| 15 | K2_r13_monet_vangogh | epoch_0001 | 0.6974 | 0.8562 | 0.3951 | 0.4219 | 0.6402 | 0.4215 | R13_monet_vangogh |

## Interpretation Guidance

- If a run beats SaMST on EC but not raw style, the claim should be content-preserving style transfer rather than stronger raw stylization.
- If a recipe improves `photo_clip_style` but hurts all-pairs EC, it is a candidate for domain-specific inference or category-conditioned weighting, not a global default.
- If K2 recipes dominate EC while K1 dominates style, the next stage should interpolate `K` or add per-target weights instead of changing all losses globally.
