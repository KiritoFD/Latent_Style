# Weight Sweep 40

Design: 20 category-sampling recipes x K={1,2}. All configs use the original K1 base config, `terminal_swd_weight=20`, `w_cycle=0`, `w_color=0`, `num_epochs=8`, `save_interval=1`.

Primary scalar score: `clip_style * (1 - LPIPS)`. This is intentionally simple and interpretable: style strength weighted by content preservation. The runner also reports `score_weighted_65_35 = 0.65 * clip_style + 0.35 * (1 - LPIPS)` and min-max normalized variants.

| experiment_id | K | recipe | content weights | target weights | note |
|---|---:|---|---|---|---|
| K1_r00_balanced_default | 1.0 | R00_balanced_default | balanced/default | balanced/default | Original balanced target sampler from base config. |
| K1_r01_uniform_unbalanced | 1.0 | R01_uniform_unbalanced | 1,1,1,1,1 | 1,1,1,1,1 | Uniform content and target weights, unbalanced sampler. |
| K1_r02_prev_manual | 1.0 | R02_prev_manual | 1.35,1.25,0.85,0.85,0.85 | 0.8,1.35,1.05,1.05,1.05 | Previous manual recipe: more photo/Hayao content and Hayao/art target. |
| K1_r03_hayao_strong | 1.0 | R03_hayao_strong | 1.3,1.4,0.85,0.85,0.85 | 0.75,1.7,1,1,1 | Strong Hayao target pressure. |
| K1_r04_cezanne_strong | 1.0 | R04_cezanne_strong | 1.35,1.05,0.85,0.85,1.25 | 0.75,1.1,1,1,1.6 | Repair Cezanne style drop by target emphasis. |
| K1_r05_vangogh_strong | 1.0 | R05_vangogh_strong | 1.35,1.05,0.85,1.25,0.85 | 0.75,1.1,1,1.6,1 | Van Gogh target emphasis. |
| K1_r06_monet_strong | 1.0 | R06_monet_strong | 1.35,1.05,1.25,0.85,0.85 | 0.75,1.1,1.6,1,1 | Monet target emphasis. |
| K1_r07_art_balanced | 1.0 | R07_art_balanced | 1.4,1,0.9,0.9,0.9 | 0.7,1.2,1.2,1.2,1.2 | All art targets boosted equally, photo target suppressed. |
| K1_r08_photo_content_high | 1.0 | R08_photo_content_high | 1.8,1.1,0.75,0.75,0.75 | 0.7,1.2,1.2,1.2,1.2 | Content-heavy recipe: frequent photo content, moderate art target. |
| K1_r09_photo_content_low | 1.0 | R09_photo_content_low | 0.9,1.2,1.1,1.1,1.1 | 0.7,1.25,1.25,1.25,1.25 | Less photo as content; stress art-to-art content retention. |
| K1_r10_no_photo_target | 1.0 | R10_no_photo_target | 1.5,1.1,0.9,0.9,0.9 | 0.3,1.35,1.35,1.35,1.35 | Nearly remove photo target to focus transfer style. |
| K1_r11_photo_target_some | 1.0 | R11_photo_target_some | 1.5,1.1,0.9,0.9,0.9 | 1,1.2,1.2,1.2,1.2 | Keep more photo target for identity/content stability. |
| K1_r12_hayao_cezanne | 1.0 | R12_hayao_cezanne | 1.35,1.25,0.8,0.8,1.1 | 0.65,1.45,1,1,1.45 | Joint Hayao plus Cezanne target repair. |
| K1_r13_monet_vangogh | 1.0 | R13_monet_vangogh | 1.35,1.05,1.1,1.1,0.8 | 0.65,1.1,1.45,1.45,1 | Joint Monet plus Van Gogh target pressure. |
| K1_r14_soft_art | 1.0 | R14_soft_art | 1.25,1.1,0.95,0.95,0.95 | 0.85,1.15,1.15,1.15,1.15 | Soft version of art target emphasis. |
| K1_r15_hard_art | 1.0 | R15_hard_art | 1.6,1.1,0.75,0.75,0.75 | 0.45,1.45,1.45,1.45,1.45 | Hard version of art target emphasis. |
| K1_r16_photo_hayao_content_art_target | 1.0 | R16_photo_hayao_content_art_target | 1.55,1.35,0.75,0.75,0.75 | 0.6,1.25,1.25,1.25,1.25 | Photo/Hayao content anchors with uniform art target pressure. |
| K1_r17_art_content_art_target | 1.0 | R17_art_content_art_target | 0.85,1.15,1.15,1.15,1.15 | 0.55,1.3,1.3,1.3,1.3 | More art content and more art target; less photo dominance. |
| K1_r18_cezanne_fix_prev | 1.0 | R18_cezanne_fix_prev | 1.35,1.25,0.85,0.85,0.95 | 0.75,1.3,1.05,1.05,1.45 | Previous manual recipe plus extra Cezanne target. |
| K1_r19_hayao_fix_prev | 1.0 | R19_hayao_fix_prev | 1.35,1.35,0.8,0.8,0.8 | 0.75,1.6,1,1,1 | Previous manual recipe plus extra Hayao target. |
| K2_r00_balanced_default | 2.0 | R00_balanced_default | balanced/default | balanced/default | Original balanced target sampler from base config. |
| K2_r01_uniform_unbalanced | 2.0 | R01_uniform_unbalanced | 1,1,1,1,1 | 1,1,1,1,1 | Uniform content and target weights, unbalanced sampler. |
| K2_r02_prev_manual | 2.0 | R02_prev_manual | 1.35,1.25,0.85,0.85,0.85 | 0.8,1.35,1.05,1.05,1.05 | Previous manual recipe: more photo/Hayao content and Hayao/art target. |
| K2_r03_hayao_strong | 2.0 | R03_hayao_strong | 1.3,1.4,0.85,0.85,0.85 | 0.75,1.7,1,1,1 | Strong Hayao target pressure. |
| K2_r04_cezanne_strong | 2.0 | R04_cezanne_strong | 1.35,1.05,0.85,0.85,1.25 | 0.75,1.1,1,1,1.6 | Repair Cezanne style drop by target emphasis. |
| K2_r05_vangogh_strong | 2.0 | R05_vangogh_strong | 1.35,1.05,0.85,1.25,0.85 | 0.75,1.1,1,1.6,1 | Van Gogh target emphasis. |
| K2_r06_monet_strong | 2.0 | R06_monet_strong | 1.35,1.05,1.25,0.85,0.85 | 0.75,1.1,1.6,1,1 | Monet target emphasis. |
| K2_r07_art_balanced | 2.0 | R07_art_balanced | 1.4,1,0.9,0.9,0.9 | 0.7,1.2,1.2,1.2,1.2 | All art targets boosted equally, photo target suppressed. |
| K2_r08_photo_content_high | 2.0 | R08_photo_content_high | 1.8,1.1,0.75,0.75,0.75 | 0.7,1.2,1.2,1.2,1.2 | Content-heavy recipe: frequent photo content, moderate art target. |
| K2_r09_photo_content_low | 2.0 | R09_photo_content_low | 0.9,1.2,1.1,1.1,1.1 | 0.7,1.25,1.25,1.25,1.25 | Less photo as content; stress art-to-art content retention. |
| K2_r10_no_photo_target | 2.0 | R10_no_photo_target | 1.5,1.1,0.9,0.9,0.9 | 0.3,1.35,1.35,1.35,1.35 | Nearly remove photo target to focus transfer style. |
| K2_r11_photo_target_some | 2.0 | R11_photo_target_some | 1.5,1.1,0.9,0.9,0.9 | 1,1.2,1.2,1.2,1.2 | Keep more photo target for identity/content stability. |
| K2_r12_hayao_cezanne | 2.0 | R12_hayao_cezanne | 1.35,1.25,0.8,0.8,1.1 | 0.65,1.45,1,1,1.45 | Joint Hayao plus Cezanne target repair. |
| K2_r13_monet_vangogh | 2.0 | R13_monet_vangogh | 1.35,1.05,1.1,1.1,0.8 | 0.65,1.1,1.45,1.45,1 | Joint Monet plus Van Gogh target pressure. |
| K2_r14_soft_art | 2.0 | R14_soft_art | 1.25,1.1,0.95,0.95,0.95 | 0.85,1.15,1.15,1.15,1.15 | Soft version of art target emphasis. |
| K2_r15_hard_art | 2.0 | R15_hard_art | 1.6,1.1,0.75,0.75,0.75 | 0.45,1.45,1.45,1.45,1.45 | Hard version of art target emphasis. |
| K2_r16_photo_hayao_content_art_target | 2.0 | R16_photo_hayao_content_art_target | 1.55,1.35,0.75,0.75,0.75 | 0.6,1.25,1.25,1.25,1.25 | Photo/Hayao content anchors with uniform art target pressure. |
| K2_r17_art_content_art_target | 2.0 | R17_art_content_art_target | 0.85,1.15,1.15,1.15,1.15 | 0.55,1.3,1.3,1.3,1.3 | More art content and more art target; less photo dominance. |
| K2_r18_cezanne_fix_prev | 2.0 | R18_cezanne_fix_prev | 1.35,1.25,0.85,0.85,0.95 | 0.75,1.3,1.05,1.05,1.45 | Previous manual recipe plus extra Cezanne target. |
| K2_r19_hayao_fix_prev | 2.0 | R19_hayao_fix_prev | 1.35,1.35,0.8,0.8,0.8 | 0.75,1.6,1,1,1 | Previous manual recipe plus extra Hayao target. |
