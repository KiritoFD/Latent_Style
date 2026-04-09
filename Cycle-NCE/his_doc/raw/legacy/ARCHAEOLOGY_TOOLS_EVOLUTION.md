# 🛠️ Tools & Infrastructure Evolution (Git History Analysis)

**Source**: `C:\Users\xy\repo.git` (Python-only filtered)

---

## 1. Complete Losses.py Evolution

| Date | SHA | Message | Lines | Key Functions | Delta |
|------|-----|---------|------:|---------------|-------|
| 2026-02-08 18:49 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的 | 236 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | INIT |
| 2026-02-08 21:29 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 303 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +67 |
| 2026-02-08 23:46 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 553 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +250 |
| 2026-02-09 01:43 | 6bd9c3f | 启动完整实验 | 592 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +39 |
| 2026-02-09 12:50 | 1368e32 | Run Step-B style-feature supervision exp | 607 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +15 |
| 2026-02-09 14:07 | 153166f | Add high-frequency multi-scale style sup | 678 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +71 |
| 2026-02-09 14:09 | d7a682a | Disable classifier CE by default and add | 681 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +3 |
| 2026-02-09 14:40 | 96c7c7d | Add style-collapse diagnostics and spati | 723 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +42 |
| 2026-02-09 14:40 | 2ba3f63 | Stabilize ref_cos metric with explicit n | 725 | calc_moment_loss, calc_gram_loss, calc_nce_loss, compute | +2 |
| 2026-02-09 18:52 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 274 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | -451 |
| 2026-02-09 21:52 | fc80895 | 终于推动了，但是有点过头了 | 265 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | -9 |
| 2026-02-10 00:42 | c05a187 | 分类完全学会了 | 299 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +34 |
| 2026-02-10 22:21 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 372 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +73 |
| 2026-02-10 23:40 | 7d376c0 | 分类器好像被hack掉了 | 399 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +27 |
| 2026-02-11 00:36 | 7e17d2d | 均衡了，跑全量实验 | 412 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +13 |
| 2026-02-11 11:10 | 6c45688 | 笔触风格 | 589 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +177 |
| 2026-02-12 12:27 | 89b3fe7 | 修复infra，风格略弱，加强 | 867 | calc_gram_loss, calc_moment_loss, calc_nce_loss, _active_hea | +278 |
| 2026-02-12 15:23 | 83ffe10 | 风格注入在map16中频大块，map32高频笔触 | 662 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | -205 |
| 2026-02-12 23:26 | 59eb2e5 | 针对3060进行infra优化 | 675 | calc_gram_loss, calc_moment_loss, calc_nce_loss, compute | +13 |
| 2026-02-14 18:05 | d992591 | 完整消融 | 330 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, comp | -345 |
| 2026-02-14 18:39 | a3a4167 | gram白化 | 366 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, _sel | +36 |
| 2026-02-15 09:45 | 2b0d391 | no-edge | 371 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, _sel | +5 |
| 2026-02-15 10:22 | ff580af | no-edge | 482 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, _sel | +111 |
| 2026-02-15 10:39 | 1882347 | semigroup=+5507.2MB逆天占用 | 570 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, _sel | +88 |
| 2026-02-15 15:14 | c0f538a | 修复infra，增加可视化 | 599 | calc_gram_loss_per_sample, calc_moment_loss_per_sample, _sel | +29 |
| 2026-02-15 21:58 | b8e2656 | 修正消融 | 472 | calc_gram_loss_per_sample, _self_similarity_loss_per_sample, | -127 |
| 2026-02-15 21:59 | fdc9d24 | ablation | 472 | calc_gram_loss_per_sample, _self_similarity_loss_per_sample, | 0 |
| 2026-02-16 00:54 | cef2299 | 非常好overfit50效果 | 505 | calc_gram_loss_per_sample, _self_similarity_loss_per_sample, | +33 |
| 2026-02-16 11:20 | 7d726f6 | 新的消融，删掉sturcutre loss | 458 | calc_gram_loss_per_sample, compute | -47 |
| 2026-02-16 14:51 | d598ad6 | ablation | 458 | calc_gram_loss_per_sample, compute | 0 |
| 2026-02-16 16:15 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不打断流水线 | 265 | calc_moment_loss, calc_swd_loss, compute | -193 |
| 2026-02-16 18:34 | 0db451d | 换SDXL，用统计出来的缩放因子0.154353 | 315 | calc_moment_loss, calc_swd_loss, compute | +50 |
| 2026-02-17 02:53 | e68bdc0 | 完全使用分类器，信号强度0.35 | 310 | compute | -5 |
| 2026-02-17 12:41 | f5c4754 | 简化 | 176 | compute_swd, compute | -134 |
| 2026-02-17 13:17 | 380c0a0 | 更新可视化 | 180 | compute_swd, compute | +4 |
| 2026-02-17 14:44 | d103db5 | SWD水平接近0但还是很差 | 281 | compute | +101 |
| 2026-02-17 15:14 | 364cb99 | 黎曼几何分开了 | 398 | compute_log_covariance, compute | +117 |
| 2026-02-17 15:54 | d619233 | 微分格拉姆，终于正了 | 466 | compute_log_covariance, compute_log_covariance, compute | +68 |
| 2026-02-17 18:44 | 5901e43 | fp32精度，moment锚定，更鲁棒的损失 | 0 |  | -466 |
| 2026-02-17 22:52 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 249 | compute | INIT |
| 2026-02-17 23:19 | 3b547bf | 改写loss，加上FP32 | 261 | compute | +12 |
| 2026-02-18 23:04 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 187 | compute | -74 |
| 2026-02-20 10:30 | c94d80e | 自动优化寻找参数 | 202 | compute | +15 |
| 2026-02-21 16:16 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 228 | compute | +26 |
| 2026-02-22 00:10 | 8c3edfd | 在style-8注入最好；做了一点算子融合 | 345 | calc_moment_loss, calc_swd_loss, _swd_features, compute | +117 |
| 2026-02-22 00:39 | fc72ca3 | 修复基本infra | 357 | calc_moment_loss, calc_swd_loss, _swd_features, compute | +12 |
| 2026-02-22 09:39 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 436 | calc_moment_loss, calc_swd_loss, _swd_features, compute | +79 |
| 2026-02-22 09:43 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 439 | calc_moment_loss, calc_swd_loss, _swd_features, compute | +3 |
| 2026-02-22 17:06 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 315 | calc_moment_loss, calc_swd_loss, compute | -124 |
| 2026-02-23 15:07 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x In | 287 | calc_feature_style_loss, calc_feature_content_loss, compute | -28 |
| 2026-02-24 10:10 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 270 | calc_swd_loss, compute | -17 |
| 2026-02-24 19:16 | adb274a | patch 1,3,5 | 286 | calc_swd_loss, compute | +16 |
| 2026-02-25 09:59 | fae58a0 | 消融实验，效果不佳，用conv1，信噪比明显提升 | 251 | calc_swd_loss, compute | -35 |
| 2026-02-26 19:06 | 0900fcf | 加颜色锚定 | 276 | calc_swd_loss, compute | +25 |
| 2026-02-27 16:23 | 7fdfec7 | updated color loss | 136 | calc_swd_loss, compute | -140 |
| 2026-03-05 23:40 | c619fda | evaluate cache added;  modified decoder  | 396 | calc_swd_loss, calc_hf_swd_loss, compute | +260 |
| 2026-03-08 17:29 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 478 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_loss, comput | +82 |
| 2026-03-08 17:39 | 2005377 | gate监控+正则 | 512 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_loss, comput | +34 |
| 2026-03-08 23:36 | dbcf851 | 梯度检查点真的要开，不然显存爆炸了 | 515 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_loss, comput | +3 |
| 2026-03-09 13:47 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 516 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_loss, comput | +1 |
| 2026-03-10 01:38 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 515 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_loss, comput | -1 |
| 2026-03-11 14:56 | 80ef230 | reverted to Decoder-D configs | 396 | calc_swd_loss, calc_hf_swd_loss, compute | -119 |
| 2026-03-19 03:56 | c9f81ad | Rebuild repository from local workspace | 396 | calc_swd_loss, calc_hf_swd_loss, compute | 0 |
| 2026-03-20 20:57 | ed52ecd | color loss有大问题，增加几种实现和消融 | 747 | _canonical_color_mode, calc_spatial_agnostic_color_loss, _hi | +351 |
| 2026-03-22 16:28 | fc2b5a9 | weight系列实验，TV可以扔了 | 560 | _canonical_color_mode, calc_spatial_agnostic_color_loss, cal | -187 |
| 2026-03-22 18:17 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 548 | _canonical_color_mode, calc_spatial_agnostic_color_loss, cal | -12 |
| 2026-03-25 08:23 | dd227e9 | 针对SWD消融，hf负收益 | 580 | _canonical_color_mode, calc_spatial_agnostic_color_loss, cal | +32 |
| 2026-03-26 15:08 | 015b68d | 权重尝试，但是亮度有大问题 | 391 | _canonical_color_mode, calc_spatial_agnostic_color_loss, cal | -189 |
| 2026-03-26 21:50 | c8577e0 | 加亮度约束，换cross_attn | 432 | _canonical_color_mode, _resolve_color_channel_weights, _calc | +41 |
| 2026-03-30 17:04 | 1e25659 | infra推进56s/epcoh | 475 | _canonical_color_mode, _resolve_color_channel_weights, _calc | +43 |

---

## 2. Loss System Evolution Timeline

### Phase 1: Gram+Moment+NCE (Jan 30 - Feb 16) — 236-725 lines
### Phase 2: SWD-Only Era (Feb 16 - Feb 22) — 176-466 lines
### Phase 3: SWD+Gram+NCE Return (Mar 8-26) — 400-700+ lines
### Phase 4: SWD+Color+Identity (Mar 26 - Apr 2) — ~475 lines
- Final state: 3-loss system (SWD + color + identity)

## 3. Trainer.py Evolution

| Date | SHA | Message | Lines | Delta |
|------|-----|---------|------:|-------|
| 2026-02-08 18:49 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的 | 522 | INIT |
| 2026-02-08 21:29 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 565 | +43 |
| 2026-02-08 23:46 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 699 | +134 |
| 2026-02-09 01:43 | 6bd9c3f | 启动完整实验 | 710 | +11 |
| 2026-02-09 13:09 | e79da29 | Fix style-config passthrough regressions | 841 | +131 |
| 2026-02-09 14:07 | 153166f | Add high-frequency multi-scale style sup | 855 | +14 |
| 2026-02-09 14:40 | 96c7c7d | Add style-collapse diagnostics and spati | 870 | +15 |
| 2026-02-09 15:09 | 70249e4 | Stabilize overfit infra, halve VRAM batc | 873 | +3 |
| 2026-02-09 15:51 | bc73153 | Strengthen style signal path and add flo | 909 | +36 |
| 2026-02-09 18:52 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 538 | -371 |
| 2026-02-09 21:52 | fc80895 | 终于推动了，但是有点过头了 | 602 | +64 |
| 2026-02-09 23:13 | ba8a914 | 风格分类很强，结构完全炸了 | 623 | +21 |
| 2026-02-10 00:42 | c05a187 | 分类完全学会了 | 639 | +16 |
| 2026-02-10 11:37 | 4cc5c9b | 风格确实好了，雾也解决了，就提升画质就行了 | 642 | +3 |
| 2026-02-10 22:21 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 631 | -11 |
| 2026-02-10 23:40 | 7d376c0 | 分类器好像被hack掉了 | 703 | +72 |
| 2026-02-11 11:10 | 6c45688 | 笔触风格 | 721 | +18 |
| 2026-02-12 12:27 | 89b3fe7 | 修复infra，风格略弱，加强 | 790 | +69 |
| 2026-02-12 15:23 | 83ffe10 | 风格注入在map16中频大块，map32高频笔触 | 785 | -5 |
| 2026-02-12 23:26 | 59eb2e5 | 针对3060进行infra优化 | 800 | +15 |
| 2026-02-12 23:50 | 44cd39e | 针对3060进行infra优化 | 808 | +8 |
| 2026-02-13 00:01 | 4067cc6 | 优化器状态从头加载 | 848 | +40 |
| 2026-02-13 00:08 | 57e9636 | 增强compile鲁棒性 | 862 | +14 |
| 2026-02-14 18:05 | d992591 | 完整消融 | 717 | -145 |
| 2026-02-14 18:39 | a3a4167 | gram白化 | 728 | +11 |
| 2026-02-15 09:45 | 2b0d391 | no-edge | 736 | +8 |
| 2026-02-15 10:22 | ff580af | no-edge | 843 | +107 |
| 2026-02-15 15:14 | c0f538a | 修复infra，增加可视化 | 966 | +123 |
| 2026-02-15 21:58 | b8e2656 | 修正消融 | 1135 | +169 |
| 2026-02-15 21:59 | fdc9d24 | ablation | 1135 | 0 |
| 2026-02-16 00:54 | cef2299 | 非常好overfit50效果 | 1158 | +23 |
| 2026-02-16 11:20 | 7d726f6 | 新的消融，删掉sturcutre loss | 1153 | -5 |
| 2026-02-16 14:51 | d598ad6 | ablation | 1153 | 0 |
| 2026-02-16 16:15 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不打断流水线 | 1153 | 0 |
| 2026-02-16 18:34 | 0db451d | 换SDXL，用统计出来的缩放因子0.154353 | 1141 | -12 |
| 2026-02-17 00:55 | 0b32631 | 消融搞得不太对，结构太强了，content到0.9了 | 1116 | -25 |
| 2026-02-17 02:53 | e68bdc0 | 完全使用分类器，信号强度0.35 | 1119 | +3 |
| 2026-02-17 12:41 | f5c4754 | 简化 | 407 | -712 |
| 2026-02-17 22:52 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 424 | +17 |
| 2026-02-18 23:04 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 538 | +114 |
| 2026-02-20 10:30 | c94d80e | 自动优化寻找参数 | 571 | +33 |
| 2026-02-21 16:16 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 584 | +13 |
| 2026-02-22 00:10 | 8c3edfd | 在style-8注入最好；做了一点算子融合 | 1161 | +577 |
| 2026-02-22 09:39 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 1181 | +20 |
| 2026-02-22 09:43 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 1206 | +25 |
| 2026-02-22 17:06 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 1173 | -33 |
| 2026-02-23 15:07 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x In | 1175 | +2 |
| 2026-02-24 10:10 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 1164 | -11 |
| 2026-02-24 19:16 | adb274a | patch 1,3,5 | 1398 | +234 |
| 2026-02-26 19:06 | 0900fcf | 加颜色锚定 | 1417 | +19 |
| 2026-02-27 16:13 | d943fab | Update trainer.py | 1398 | -19 |
| 2026-02-27 16:23 | 7fdfec7 | updated color loss | 1364 | -34 |
| 2026-03-05 23:40 | c619fda | evaluate cache added;  modified decoder  | 1315 | -49 |
| 2026-03-08 17:29 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 1403 | +88 |
| 2026-03-08 17:39 | 2005377 | gate监控+正则 | 1414 | +11 |
| 2026-03-09 13:47 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 1477 | +63 |
| 2026-03-10 01:38 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 1414 | -63 |
| 2026-03-11 14:56 | 80ef230 | reverted to Decoder-D configs | 1315 | -99 |
| 2026-03-19 03:56 | c9f81ad | Rebuild repository from local workspace | 1315 | 0 |
| 2026-03-20 20:57 | ed52ecd | color loss有大问题，增加几种实现和消融 | 1348 | +33 |
| 2026-03-22 16:28 | fc2b5a9 | weight系列实验，TV可以扔了 | 1364 | +16 |
| 2026-03-22 18:17 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 1359 | -5 |
| 2026-03-25 08:23 | dd227e9 | 针对SWD消融，hf负收益 | 1522 | +163 |
| 2026-03-26 15:08 | 015b68d | 权重尝试，但是亮度有大问题 | 1514 | -8 |
| 2026-03-26 21:50 | c8577e0 | 加亮度约束，换cross_attn | 1531 | +17 |
| 2026-03-29 14:49 | 426ae0a | 加入attention效果明显 | 1536 | +5 |
| 2026-03-30 17:04 | 1e25659 | infra推进56s/epcoh | 1548 | +12 |
| 2026-04-02 16:24 | 4e166f0 | micro batch效果大好 | 531 | -1017 |

---

## 4. The Great Simplification (Apr 2)

trainer.py: 1536 → 531 lines (-65%)
- Removed: teacher-student, NCE, classifier, full eval during training
- Kept: simple loop, micro-batch, AMP, gradient checkpointing
- Commit: 'micro batch效果大好'

---

## 5. Complete Losses.py Evolution (36 commits)

| Date | SHA | Msg | Lines | Key calc/compute Functions |
|------|-----|-----|------:|----------------------------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做 | 236 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | INIT |
| 2026-02-08 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 303 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +67 |
| 2026-02-08 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 553 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +250 |
| 2026-02-09 | 6bd9c3f | 启动完整实验 | 592 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +39 |
| 2026-02-09 | 1368e32 | Run Step-B style-feature supervisio | 607 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +15 |
| 2026-02-09 | 153166f | Add high-frequency multi-scale styl | 678 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +71 |
| 2026-02-09 | d7a682a | Disable classifier CE by default an | 681 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +3 |
| 2026-02-09 | 96c7c7d | Add style-collapse diagnostics and  | 723 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +42 |
| 2026-02-09 | 2ba3f63 | Stabilize ref_cos metric with expli | 725 | calc_moment_loss, calc_gram_matrix, calc_gram_loss | +2 |
| 2026-02-09 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 274 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | -451 |
| 2026-02-09 | fc80895 | 终于推动了，但是有点过头了 | 265 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | -9 |
| 2026-02-10 | c05a187 | 分类完全学会了 | 299 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +34 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 372 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +73 |
| 2026-02-10 | 7d376c0 | 分类器好像被hack掉了 | 399 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +27 |
| 2026-02-11 | 7e17d2d | 均衡了，跑全量实验 | 412 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +13 |
| 2026-02-11 | 6c45688 | 笔触风格 | 589 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +177 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 867 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +278 |
| 2026-02-12 | 83ffe10 | 风格注入在map16中频大块，map32高频笔触 | 662 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | -205 |
| 2026-02-12 | 59eb2e5 | 针对3060进行infra优化 | 675 | calc_gram_matrix, calc_gram_loss, calc_moment_loss | +13 |
| 2026-02-14 | d992591 | 完整消融 | 330 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | -345 |
| 2026-02-14 | a3a4167 | gram白化 | 366 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | +36 |
| 2026-02-15 | 2b0d391 | no-edge | 371 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | +5 |
| 2026-02-15 | ff580af | no-edge | 482 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | +111 |
| 2026-02-15 | 1882347 | semigroup=+5507.2MB逆天占用 | 570 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | +88 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 599 | calc_gram_matrix, calc_gram_loss_per_sample, calc_ | +29 |
| 2026-02-15 | b8e2656 | 修正消融 | 472 | calc_gram_matrix, calc_gram_loss_per_sample, _self | -127 |
| 2026-02-15 | fdc9d24 | ablation | 472 | calc_gram_matrix, calc_gram_loss_per_sample, _self | 0 |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 505 | calc_gram_matrix, calc_gram_loss_per_sample, _self | +33 |
| 2026-02-16 | 7d726f6 | 新的消融，删掉sturcutre loss | 458 | calc_gram_matrix, calc_gram_loss_per_sample, compu | -47 |
| 2026-02-16 | d598ad6 | ablation | 458 | calc_gram_matrix, calc_gram_loss_per_sample, compu | 0 |
| 2026-02-16 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不 | 265 | calc_moment_loss, calc_swd_loss, compute | -193 |
| 2026-02-16 | 0db451d | 换SDXL，用统计出来的缩放因子0.154353 | 315 | calc_moment_loss, calc_swd_loss, compute | +50 |
| 2026-02-17 | e68bdc0 | 完全使用分类器，信号强度0.35 | 310 | compute | -5 |
| 2026-02-17 | f5c4754 | 简化 | 176 | compute_swd, compute | -134 |
| 2026-02-17 | 380c0a0 | 更新可视化 | 180 | compute_swd, compute | +4 |
| 2026-02-17 | d103db5 | SWD水平接近0但还是很差 | 281 | compute | +101 |
| 2026-02-17 | 364cb99 | 黎曼几何分开了 | 398 | compute_log_covariance, compute | +117 |
| 2026-02-17 | d619233 | 微分格拉姆，终于正了 | 466 | compute_log_covariance, compute_log_covariance, co | +68 |
| 2026-02-17 | 5901e43 | fp32精度，moment锚定，更鲁棒的损失 | 237 | _calc_moments, compute | -229 |
| 2026-02-17 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 249 | _calc_moments, compute | +12 |
| 2026-02-17 | 3b547bf | 改写loss，加上FP32 | 261 | _calc_moments, compute | +12 |
| 2026-02-18 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 187 | _calc_moments, compute | -74 |
| 2026-02-20 | c94d80e | 自动优化寻找参数 | 202 | _calc_moments, compute | +15 |
| 2026-02-21 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 228 | _calc_moments, compute | +26 |
| 2026-02-22 | 8c3edfd | 在style-8注入最好；做了一点算子融合 | 345 | calc_moment_loss, calc_swd_loss, _swd_features, co | +117 |
| 2026-02-22 | fc72ca3 | 修复基本infra | 357 | calc_moment_loss, calc_swd_loss, _swd_features, co | +12 |
| 2026-02-22 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 436 | calc_moment_loss, calc_swd_loss, _swd_features, co | +79 |
| 2026-02-22 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 439 | calc_moment_loss, calc_swd_loss, _swd_features, co | +3 |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 315 | calc_moment_loss, calc_swd_loss, compute | -124 |
| 2026-02-23 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.1 | 287 | calc_feature_style_loss, calc_feature_content_loss | -28 |
| 2026-02-24 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 270 | calc_swd_loss, compute | -17 |
| 2026-02-24 | adb274a | patch 1,3,5 | 286 | calc_swd_loss, compute | +16 |
| 2026-02-25 | fae58a0 | 消融实验，效果不佳，用conv1，信噪比明显提升 | 251 | calc_swd_loss, compute | -35 |
| 2026-02-26 | 0900fcf | 加颜色锚定 | 276 | calc_swd_loss, compute | +25 |
| 2026-02-27 | 7fdfec7 | updated color loss | 136 | calc_swd_loss, compute | -140 |
| 2026-03-05 | c619fda | evaluate cache added;  modified dec | 396 | calc_swd_loss, calc_hf_swd_loss, compute | +260 |
| 2026-03-08 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 478 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_lo | +82 |
| 2026-03-08 | 2005377 | gate监控+正则 | 512 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_lo | +34 |
| 2026-03-08 | dbcf851 | 梯度检查点真的要开，不然显存爆炸了 | 515 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_lo | +3 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 516 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_lo | +1 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显 | 515 | calc_swd_loss, calc_hf_swd_loss, calc_patch_nce_lo | -1 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 396 | calc_swd_loss, calc_hf_swd_loss, compute | -119 |
| 2026-03-19 | c9f81ad | Rebuild repository from local works | 396 | calc_swd_loss, calc_hf_swd_loss, compute | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 747 | _canonical_color_mode, calc_spatial_agnostic_color | +351 |
| 2026-03-22 | fc2b5a9 | weight系列实验，TV可以扔了 | 560 | _canonical_color_mode, calc_spatial_agnostic_color | -187 |
| 2026-03-22 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 548 | _canonical_color_mode, calc_spatial_agnostic_color | -12 |
| 2026-03-25 | dd227e9 | 针对SWD消融，hf负收益 | 580 | _canonical_color_mode, calc_spatial_agnostic_color | +32 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 391 | _canonical_color_mode, calc_spatial_agnostic_color | -189 |
| 2026-03-26 | c8577e0 | 加亮度约束，换cross_attn | 432 | _canonical_color_mode, _resolve_color_channel_weig | +41 |
| 2026-03-30 | 1e25659 | infra推进56s/epcoh | 475 | _canonical_color_mode, _resolve_color_channel_weig | +43 |

## 6. Complete Trainer.py Evolution (38 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的 | 522 | INIT |
| 2026-02-08 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 565 | +43 |
| 2026-02-08 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 699 | +134 |
| 2026-02-09 | 6bd9c3f | 启动完整实验 | 710 | +11 |
| 2026-02-09 | e79da29 | Fix style-config passthrough regressions | 841 | +131 |
| 2026-02-09 | 153166f | Add high-frequency multi-scale style sup | 855 | +14 |
| 2026-02-09 | 96c7c7d | Add style-collapse diagnostics and spati | 870 | +15 |
| 2026-02-09 | 70249e4 | Stabilize overfit infra, halve VRAM batc | 873 | +3 |
| 2026-02-09 | bc73153 | Strengthen style signal path and add flo | 909 | +36 |
| 2026-02-09 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 538 | -371 |
| 2026-02-09 | fc80895 | 终于推动了，但是有点过头了 | 602 | +64 |
| 2026-02-09 | ba8a914 | 风格分类很强，结构完全炸了 | 623 | +21 |
| 2026-02-10 | c05a187 | 分类完全学会了 | 639 | +16 |
| 2026-02-10 | 4cc5c9b | 风格确实好了，雾也解决了，就提升画质就行了 | 642 | +3 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 631 | -11 |
| 2026-02-10 | 7d376c0 | 分类器好像被hack掉了 | 703 | +72 |
| 2026-02-11 | 6c45688 | 笔触风格 | 721 | +18 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 790 | +69 |
| 2026-02-12 | 83ffe10 | 风格注入在map16中频大块，map32高频笔触 | 785 | -5 |
| 2026-02-12 | 59eb2e5 | 针对3060进行infra优化 | 800 | +15 |
| 2026-02-12 | 44cd39e | 针对3060进行infra优化 | 808 | +8 |
| 2026-02-13 | 4067cc6 | 优化器状态从头加载 | 848 | +40 |
| 2026-02-13 | 57e9636 | 增强compile鲁棒性 | 862 | +14 |
| 2026-02-14 | d992591 | 完整消融 | 717 | -145 |
| 2026-02-14 | a3a4167 | gram白化 | 728 | +11 |
| 2026-02-15 | 2b0d391 | no-edge | 736 | +8 |
| 2026-02-15 | ff580af | no-edge | 843 | +107 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 966 | +123 |
| 2026-02-15 | b8e2656 | 修正消融 | 1135 | +169 |
| 2026-02-15 | fdc9d24 | ablation | 1135 | 0 |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 1158 | +23 |
| 2026-02-16 | 7d726f6 | 新的消融，删掉sturcutre loss | 1153 | -5 |
| 2026-02-16 | d598ad6 | ablation | 1153 | 0 |
| 2026-02-16 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不打断流水线 | 1153 | 0 |
| 2026-02-16 | 0db451d | 换SDXL，用统计出来的缩放因子0.154353 | 1141 | -12 |
| 2026-02-17 | 0b32631 | 消融搞得不太对，结构太强了，content到0.9了 | 1116 | -25 |
| 2026-02-17 | e68bdc0 | 完全使用分类器，信号强度0.35 | 1119 | +3 |
| 2026-02-17 | f5c4754 | 简化 | 407 | -712 |
| 2026-02-17 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 424 | +17 |
| 2026-02-18 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 538 | +114 |
| 2026-02-20 | c94d80e | 自动优化寻找参数 | 571 | +33 |
| 2026-02-21 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 584 | +13 |
| 2026-02-22 | 8c3edfd | 在style-8注入最好；做了一点算子融合 | 1161 | +577 |
| 2026-02-22 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 1181 | +20 |
| 2026-02-22 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 1206 | +25 |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 1173 | -33 |
| 2026-02-23 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x In | 1175 | +2 |
| 2026-02-24 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 1164 | -11 |
| 2026-02-24 | adb274a | patch 1,3,5 | 1398 | +234 |
| 2026-02-26 | 0900fcf | 加颜色锚定 | 1417 | +19 |
| 2026-02-27 | d943fab | Update trainer.py | 1398 | -19 |
| 2026-02-27 | 7fdfec7 | updated color loss | 1364 | -34 |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder  | 1315 | -49 |
| 2026-03-08 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 1403 | +88 |
| 2026-03-08 | 2005377 | gate监控+正则 | 1414 | +11 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 1477 | +63 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 1414 | -63 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 1315 | -99 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 1315 | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 1348 | +33 |
| 2026-03-22 | fc2b5a9 | weight系列实验，TV可以扔了 | 1364 | +16 |
| 2026-03-22 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 1359 | -5 |
| 2026-03-25 | dd227e9 | 针对SWD消融，hf负收益 | 1522 | +163 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 1514 | -8 |
| 2026-03-26 | c8577e0 | 加亮度约束，换cross_attn | 1531 | +17 |
| 2026-03-29 | 426ae0a | 加入attention效果明显 | 1536 | +5 |
| 2026-03-30 | 1e25659 | infra推进56s/epcoh | 1548 | +12 |
| 2026-04-02 | 4e166f0 | micro batch效果大好 | 531 | -1017 |

## 7. Complete utils/run_evaluation.py Evolution (29 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的 | 734 | INIT |
| 2026-02-08 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 765 | +31 |
| 2026-02-08 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 864 | +99 |
| 2026-02-09 | 93b0008 | Add conditional sensitivity metrics and  | 966 | +102 |
| 2026-02-09 | 458b4c7 | Fix eval CSV append pitfall and document | 969 | +3 |
| 2026-02-10 | c05a187 | 分类完全学会了 | 775 | -194 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 775 | 0 |
| 2026-02-11 | 6c45688 | 笔触风格 | 776 | +1 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 785 | +9 |
| 2026-02-12 | 0d13456 | 简化优化风格注入 | 904 | +119 |
| 2026-02-12 | 44cd39e | 针对3060进行infra优化 | 808 | -96 |
| 2026-02-14 | 40a26af | 消融实验结果 | 1041 | +233 |
| 2026-02-14 | d992591 | 完整消融 | 1036 | -5 |
| 2026-02-15 | 2b0d391 | no-edge | 1141 | +105 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 1141 | 0 |
| 2026-02-15 | b8e2656 | 修正消融 | 1291 | +150 |
| 2026-02-15 | fdc9d24 | ablation | 1291 | 0 |
| 2026-02-16 | 7c098a8 | structure loss完全没用，可以干掉了 | 1289 | -2 |
| 2026-02-16 | d598ad6 | ablation | 1289 | 0 |
| 2026-02-16 | 0db451d | 换SDXL，用统计出来的缩放因子0.154353 | 1289 | 0 |
| 2026-02-17 | 0b32631 | 消融搞得不太对，结构太强了，content到0.9了 | 1257 | -32 |
| 2026-02-17 | f5c4754 | 简化 | 1257 | 0 |
| 2026-02-17 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 1299 | +42 |
| 2026-02-20 | c94d80e | 自动优化寻找参数 | 1203 | -96 |
| 2026-02-21 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 1336 | +133 |
| 2026-02-22 | 4041398 | refactor(classify): replace heavy pipeli | 1443 | +107 |
| 2026-02-23 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x In | 1423 | -20 |
| 2026-02-24 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 1481 | +58 |
| 2026-02-24 | adb274a | patch 1,3,5 | 1481 | 0 |
| 2026-02-25 | fae58a0 | 消融实验，效果不佳，用conv1，信噪比明显提升 | 1704 | +223 |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder  | 1815 | +111 |
| 2026-03-08 | 9d1c0fe | 对比完成，差于CUT，需要把结构拉回来 | 1969 | +154 |
| 2026-03-08 | ccfbe39 | 修正评估 | 1938 | -31 |
| 2026-03-08 | a2fc1af | 整理实验 | 2014 | +76 |
| 2026-03-08 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 2143 | +129 |
| 2026-03-10 | 770ae3e | few shot | 2156 | +13 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 2157 | +1 |
| 2026-03-19 | 06764af | 结构消融 | 2160 | +3 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 2160 | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 2180 | +20 |
| 2026-03-21 | a6d096d | color 01效果极好，蒸馏后两方面都在进步 | 2182 | +2 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 2232 | +50 |
| 2026-03-29 | 426ae0a | 加入attention效果明显 | 2270 | +38 |
| 2026-03-30 | cfdbaba | 全部换用c-g-w的backbone | 2270 | 0 |

---

## 9. Complete Run.py Evolution (34 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的，就是要加 | 172 | INIT |
| 2026-02-08 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 178 | +6 |
| 2026-02-08 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 173 | -5 |
| 2026-02-09 | 6bd9c3f | 启动完整实验 | 254 | +81 |
| 2026-02-09 | d7a682a | Disable classifier CE by default and add sing | 335 | +81 |
| 2026-02-09 | 70249e4 | Stabilize overfit infra, halve VRAM batch, an | 366 | +31 |
| 2026-02-09 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 174 | -192 |
| 2026-02-09 | fc80895 | 终于推动了，但是有点过头了 | 177 | +3 |
| 2026-02-09 | ba8a914 | 风格分类很强，结构完全炸了 | 194 | +17 |
| 2026-02-10 | 4cc5c9b | 风格确实好了，雾也解决了，就提升画质就行了 | 256 | +62 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 270 | +14 |
| 2026-02-11 | 6c45688 | 笔触风格 | 276 | +6 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 275 | -1 |
| 2026-02-12 | 59eb2e5 | 针对3060进行infra优化 | 301 | +26 |
| 2026-02-13 | 4067cc6 | 优化器状态从头加载 | 301 | 0 |
| 2026-02-13 | 57e9636 | 增强compile鲁棒性 | 310 | +9 |
| 2026-02-14 | d992591 | 完整消融 | 304 | -6 |
| 2026-02-15 | ff580af | no-edge | 325 | +21 |
| 2026-02-15 | 1882347 | semigroup=+5507.2MB逆天占用 | 330 | +5 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 322 | -8 |
| 2026-02-15 | b8e2656 | 修正消融 | 340 | +18 |
| 2026-02-15 | fdc9d24 | ablation | 340 | 0 |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 341 | +1 |
| 2026-02-16 | 7d726f6 | 新的消融，删掉sturcutre loss | 337 | -4 |
| 2026-02-16 | d598ad6 | ablation | 337 | 0 |
| 2026-02-16 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不打断流水线 | 336 | -1 |
| 2026-02-17 | e68bdc0 | 完全使用分类器，信号强度0.35 | 358 | +22 |
| 2026-02-17 | f5c4754 | 简化 | 326 | -32 |
| 2026-02-17 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 335 | +9 |
| 2026-02-18 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 332 | -3 |
| 2026-02-20 | c94d80e | 自动优化寻找参数 | 331 | -1 |
| 2026-02-21 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 338 | +7 |
| 2026-02-22 | fc72ca3 | 修复基本infra | 345 | +7 |
| 2026-02-22 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 347 | +2 |
| 2026-02-22 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 352 | +5 |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 339 | -13 |
| 2026-02-23 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x Instanc | 346 | +7 |
| 2026-02-24 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 341 | -5 |
| 2026-02-24 | adb274a | patch 1,3,5 | 347 | +6 |
| 2026-02-26 | 0900fcf | 加颜色锚定 | 338 | -9 |
| 2026-02-27 | 7fdfec7 | updated color loss | 325 | -13 |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder block | 347 | +22 |
| 2026-03-08 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 385 | +38 |
| 2026-03-08 | 2005377 | gate监控+正则 | 388 | +3 |
| 2026-03-08 | af5f6cb | 新增空间门控 | 404 | +16 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 405 | +1 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 404 | -1 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 347 | -57 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 347 | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 377 | +30 |
| 2026-03-22 | fc2b5a9 | weight系列实验，TV可以扔了 | 352 | -25 |
| 2026-03-22 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 349 | -3 |
| 2026-03-25 | dd227e9 | 针对SWD消融，hf负收益 | 384 | +35 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 382 | -2 |
| 2026-03-26 | c8577e0 | 加亮度约束，换cross_attn | 383 | +1 |
| 2026-03-30 | 1e25659 | infra推进56s/epcoh | 385 | +2 |
| 2026-04-02 | 4e166f0 | micro batch效果大好 | 370 | -15 |

## 10. Complete Dataset.py Evolution (6 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的，就是要加 | 143 | INIT |
| 2026-02-09 | 1efb4fb | Record overfit50 diagnostics and harden exper | 157 | +14 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 174 | +17 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 180 | +6 |
| 2026-02-14 | d992591 | 完整消融 | 150 | -30 |
| 2026-02-15 | ff580af | no-edge | 148 | -2 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 148 | 0 |
| 2026-02-15 | b8e2656 | 修正消融 | 207 | +59 |
| 2026-02-15 | fdc9d24 | ablation | 207 | 0 |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 207 | 0 |
| 2026-02-16 | d598ad6 | ablation | 207 | 0 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 248 | +41 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 207 | -41 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 207 | 0 |

---

## 9. Complete Run.py Evolution (34 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的，就是要加 | 172 | INIT |
| 2026-02-08 | d916277 | 蒸馏把风格放进模型，推理不需要参考图 | 178 | +6 |
| 2026-02-08 | c7a14aa | 分类器引导，但是分类器很快被打穿了 | 173 | -5 |
| 2026-02-09 | 6bd9c3f | 启动完整实验 | 254 | +81 |
| 2026-02-09 | d7a682a | Disable classifier CE by default and add sing | 335 | +81 |
| 2026-02-09 | 70249e4 | Stabilize overfit infra, halve VRAM batch, an | 366 | +31 |
| 2026-02-09 | 9bafd54 | 从ckpt逆向，回滚到风格发挥作用的版本 | 174 | -192 |
| 2026-02-09 | fc80895 | 终于推动了，但是有点过头了 | 177 | +3 |
| 2026-02-09 | ba8a914 | 风格分类很强，结构完全炸了 | 194 | +17 |
| 2026-02-10 | 4cc5c9b | 风格确实好了，雾也解决了，就提升画质就行了 | 256 | +62 |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 270 | +14 |
| 2026-02-11 | 6c45688 | 笔触风格 | 276 | +6 |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 275 | -1 |
| 2026-02-12 | 59eb2e5 | 针对3060进行infra优化 | 301 | +26 |
| 2026-02-13 | 4067cc6 | 优化器状态从头加载 | 301 | 0 |
| 2026-02-13 | 57e9636 | 增强compile鲁棒性 | 310 | +9 |
| 2026-02-14 | d992591 | 完整消融 | 304 | -6 |
| 2026-02-15 | ff580af | no-edge | 325 | +21 |
| 2026-02-15 | 1882347 | semigroup=+5507.2MB逆天占用 | 330 | +5 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 322 | -8 |
| 2026-02-15 | b8e2656 | 修正消融 | 340 | +18 |
| 2026-02-15 | fdc9d24 | ablation | 340 | 0 |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 341 | +1 |
| 2026-02-16 | 7d726f6 | 新的消融，删掉sturcutre loss | 337 | -4 |
| 2026-02-16 | d598ad6 | ablation | 337 | 0 |
| 2026-02-16 | 12dfe7c | FP32转BF16保精度加速，和rand int代替rand1000不打断流水线 | 336 | -1 |
| 2026-02-17 | e68bdc0 | 完全使用分类器，信号强度0.35 | 358 | +22 |
| 2026-02-17 | f5c4754 | 简化 | 326 | -32 |
| 2026-02-17 | b231cb2 | diff-gram继续消融，换一下NORM方式，到0.07了 | 335 | +9 |
| 2026-02-18 | 1784944 | 平衡Loss权重，用回NCE，加入梯度的监控 | 332 | -3 |
| 2026-02-20 | c94d80e | 自动优化寻找参数 | 331 | -1 |
| 2026-02-21 | 873f271 | diff-gram在sdxl-fp32上表现极差 | 338 | +7 |
| 2026-02-22 | fc72ca3 | 修复基本infra | 345 | +7 |
| 2026-02-22 | 7d87a4e | style-8的SWD直接NAN了，换FP32 | 347 | +2 |
| 2026-02-22 | 89de09e | loss出问题直接跳过该batch，特征提取部分也FP32 | 352 | +5 |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 339 | -13 |
| 2026-02-23 | 1f818cc | 结果（5风格联合）：  Instance 1x1: Ratio 1.15x Instanc | 346 | +7 |
| 2026-02-24 | 80b5456 | infra与导出，CUDA上.plan跑到1-2ms/frame | 341 | -5 |
| 2026-02-24 | adb274a | patch 1,3,5 | 347 | +6 |
| 2026-02-26 | 0900fcf | 加颜色锚定 | 338 | -9 |
| 2026-02-27 | 7fdfec7 | updated color loss | 325 | -13 |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder block | 347 | +22 |
| 2026-03-08 | 4992e06 | 分类准确率有提升，NCE loss是有效的 | 385 | +38 |
| 2026-03-08 | 2005377 | gate监控+正则 | 388 | +3 |
| 2026-03-08 | af5f6cb | 新增空间门控 | 404 | +16 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 405 | +1 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 404 | -1 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 347 | -57 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 347 | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 377 | +30 |
| 2026-03-22 | fc2b5a9 | weight系列实验，TV可以扔了 | 352 | -25 |
| 2026-03-22 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 349 | -3 |
| 2026-03-25 | dd227e9 | 针对SWD消融，hf负收益 | 384 | +35 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 382 | -2 |
| 2026-03-26 | c8577e0 | 加亮度约束，换cross_attn | 383 | +1 |
| 2026-03-30 | 1e25659 | infra推进56s/epcoh | 385 | +2 |
| 2026-04-02 | 4e166f0 | micro batch效果大好 | 370 | -15 |

### Run.py Key Changes
- **Feb 22**: Added FP32 switch for SWD style-8 NAN fix
- **Mar 5-8**: Added gradient checkpointing, spatial gate config
- **Mar 10**: Added tokenizer distillation config
- **Mar 19**: Rebuilt with stable config pipeline
- **Mar 26**: Added cross_attn config, brightness constraint
- **Mar 30**: Added CGW backbone config
- **Apr 2**: Micro-batch optimization, simplified CLI

## 10. Complete Dataset.py Evolution (6 commits)

| Date | SHA | Msg | Lines | Notes |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的，就是要加 | 143 |  |
| 2026-02-09 | 1efb4fb | Record overfit50 diagnostics and harden exper | 157 |  |
| 2026-02-10 | 73248db | 相当不错的结果，就是还有一点雾，并不严重，把Cycle改到MSE是对的 | 174 |  |
| 2026-02-12 | 89b3fe7 | 修复infra，风格略弱，加强 | 180 |  |
| 2026-02-14 | d992591 | 完整消融 | 150 |  |
| 2026-02-15 | ff580af | no-edge | 148 |  |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 148 |  |
| 2026-02-15 | b8e2656 | 修正消融 | 207 |  |
| 2026-02-15 | fdc9d24 | ablation | 207 |  |
| 2026-02-16 | cef2299 | 非常好overfit50效果 | 207 |  |
| 2026-02-16 | d598ad6 | ablation | 207 |  |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 248 |  |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 207 |  |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 207 |  |

**Dataset.py is remarkably stable**: only 6 commits, always ~200 lines.
Key features: hflip, preload-to-GPU, VRAM reservation ratio, virtual length multiplier.

## 11. Complete utils/style_classifier.py Evolution (5 commits)

| Date | SHA | Msg | Lines | Delta |
|------|-----|-----|------:|-------|
| 2026-02-08 | ae596d1 | 小规模验证，分类成绩很好但是画面有点崩，增大通道数，在4个以上的通道做风格是对的，就是要加 | 1157 | INIT |
| 2026-02-14 | 40a26af | 消融实验结果 | 1168 | +11 |
| 2026-02-14 | d992591 | 完整消融 | 1164 | -4 |
| 2026-02-15 | c0f538a | 修复infra，增加可视化 | 1164 | 0 |
| 2026-02-17 | 0b32631 | 消融搞得不太对，结构太强了，content到0.9了 | 1194 | +30 |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 0 | -1194 |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder block | 1240 | INIT |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 1240 | 0 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 1479 | +239 |

## 12. Complete utils/classify.py Evolution (2 commits)

| 2026-02-17 | e68bdc0 | 完全使用分类器，信号强度0.35 | 250 lines |
| 2026-02-17 | f5c4754 | 简化 | 319 lines |
| 2026-02-22 | 4041398 | refactor(classify): replace heavy pipeline wi | 485 lines |
| 2026-02-22 | 1a3f2d0 | tune(classify): improve recall with proven de | 509 lines |
| 2026-02-22 | 0cf4938 | feat(classify): add generated-dir target-styl | 631 lines |
| 2026-02-22 | c105ef2 | fix(classify): default to test-only when test | 638 lines |
| 2026-02-22 | abe0e53 | fix(classify): only overwrite ckpt when impro | 688 lines |
| 2026-02-22 | cce6a2b | feat(classify): print 4-style art-only accura | 714 lines |
| 2026-02-22 | c0c58e0 | feat(classify): report full 5x5 source-target | 745 lines |
| 2026-02-22 | f5474c3 | 修正moment的实现，通道解耦合保护亮度+协方差对齐恢复色彩 | 751 lines |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 485 lines |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 655 lines |

## 13. Complete ablate.py Evolution (20 commits)

| 2026-02-27 | 7fdfec7 | updated color loss | 157 | INIT |
| 2026-03-05 | c619fda | evaluate cache added;  modified decoder block | 97 | -60 |
| 2026-03-07 | 7148a5e | 增加数值限制 | 111 | +14 |
| 2026-03-08 | 9d1c0fe | 对比完成，差于CUT，需要把结构拉回来 | 106 | -5 |
| 2026-03-08 | dbcf851 | 梯度检查点真的要开，不然显存爆炸了 | 163 | +57 |
| 2026-03-09 | 5c7c2a2 | 投影会引入cllip先验污染，干扰评估 | 171 | +8 |
| 2026-03-09 | dc341ae | clear src | 151 | -20 |
| 2026-03-10 | 4699637 | 单独蒸馏tokenizer，优化style_embedding，有明显指标提升 | 171 | +20 |
| 2026-03-11 | 80ef230 | reverted to Decoder-D configs | 106 | -65 |
| 2026-03-19 | 06764af | 结构消融 | 194 | +88 |
| 2026-03-19 | c9f81ad | Rebuild repository from local workspace | 194 | 0 |
| 2026-03-20 | ed52ecd | color loss有大问题，增加几种实现和消融 | 176 | -18 |
| 2026-03-22 | fc2b5a9 | weight系列实验，TV可以扔了 | 144 | -32 |
| 2026-03-22 | ed596c0 | 通道映射回RGB的缩略图color loss大赢 | 150 | +6 |
| 2026-03-25 | dd227e9 | 针对SWD消融，hf负收益 | 90 | -60 |
| 2026-03-26 | 015b68d | 权重尝试，但是亮度有大问题 | 97 | +7 |
| 2026-03-26 | c8577e0 | 加亮度约束，换cross_attn | 138 | +41 |
| 2026-03-29 | 426ae0a | 加入attention效果明显 | 392 | +254 |
| 2026-03-30 | cfdbaba | 全部换用c-g-w的backbone | 190 | -202 |
| 2026-03-30 | 1e25659 | infra推进56s/epcoh | 108 | -82 |
| 2026-04-02 | 4e166f0 | micro batch效果大好 | 198 | +90 |

## 14. Final File Inventory (as of Apr 2)



---

## 5. Signal Separation & Plotting Era (Feb 17 - Feb 21)

This was the critical period where the project shifted from "Does SWD work?" to "**Which operator best separates style signals in latent space?**"

### 5.1 The `plot-swd.py` Evolution (The Grid Search Script)
Around **Feb 17**, a massive analysis script `plot-swd.py` (later moved to `Cycle-NCE/plot-swd.py`) was written. It evolved from a simple visualizer to a rigorous 300+ line grid search tool.

#### 1. The Core Engine: `ParametricExtractor`
A frozen convolutional projector designed to probe the latent space (SDXL VAE latents, 4 channels).
```python
class ParametricExtractor(nn.Module):
    def __init__(self, config: dict):
        # Lift Layer: 1x1 or 3x3 Conv (Orthogonal Init)
        self.projector = nn.Conv2d(4, dim=cfg['dim'], kernel=cfg['kernel'], ...)
        
        # Activation Space studied:
        acts = ['relu', 'leaky_relu', 'gelu', 'silu', 'tanh']
        
        # Normalization (Whitening) strategies tested:
        # - InstanceNorm (Eliminates "cone effect")
        # - LayerNorm, GroupNorm (1, 2, 4 groups)
        
    def forward(self, x):
        x = x * 0.13025 # SDXL VAE scaling factor
        h = self.projector(...)
        h = F.{act}(h)
        h = F.{norm}(h)
        
        # Gram Matrix Extraction (The Style Signal)
        # MODE 1: raw Gram(h)
        # MODE 2: diff Gram (Differential Gram) -> h[:, :, :, s:] - h[:, :, :, :-s]
        # Tested scales: [1], [2], [1, 2]
```

#### 2. The Separability Metrics (The "Loss" Alternatives)
To find the best operator without training, the script evaluated feature vectors using 5 distinct metrics:
- **Fisher Ratio (J, J*)**: `trace(Sb) / trace(Sw)`. Measures class separation vs intra-class variance.
- **Silhouette Score**: Measures cluster cohesion and separation.
- **Distance AUC**: Can a simple distance metric distinguish same-style vs different-style pairs?
- **Inter/Intra Margin**: `mean(dist_diff) - mean(dist_same)`.
- **Linear Probe Accuracy / Macro F1**:
  - `probe_random`: Standard StratifiedKFold.
  - `probe_group`: GroupKFold (split by content image ID to ensure style-only generalization).

#### 3. Key Configuration Sweep (The "Operator Loss" Experiments)
| Variable | Search Space | Finding |
|----------|-------------|---------|
| **Dim** | 32, 48, 64, 128 | 64-128 was optimal. |
| **Kernel** | 1, 3, 7 | 1x1 was surprisingly best (preserves raw texture). |
| **Feature Mode** | `raw`, `diff` (delta) | **`diff` won.** Raw Gram captured structure, Diff Gram captured *texture/style gradients*. |
| **Act** | `relu`, `leaky`, `gelu`, `silu` | `leaky` and `silu` performed best. |
| **Norm** | `instance`, `layer`, `group` | **`group_norm`** (with `gram_norm=chw`) gave the highest separation. |
| **Scales** | `[1]`, `[1, 2]` | `[1, 2]` captured both fine and coarse texture. |

### 5.2 Visualization Outputs
- **PCA Scatter Plots**: `best_config_pca-fp32.png`
  - Used `torch.pca_lowrank` to project the best Gram feature vectors to 2D.
  - Resulted in clean 5-style clusters (Photo, Hayao, Vangogh, Monet, Cezanne).
  - Proved that **Diff-Gram + GroupNorm** linearly separates styles in latent space!

### 5.3 Subsequent Plotting Tools
- `export_summary_history_to_csv_and_plot.py` (Mar 8): Automated training curve plotting (Loss vs Epoch, Style Score vs Epoch) across 50+ experiment folders.
- `plot_experiments_and_related_works_scatter.py` (Mar 19): Compared Cycle-NCE results against related works in 2D scatter plots.

**Conclusion**: This era proved that style doesn't need complex cross-attention; a simple differentiable Gram matrix extractor (Diff-Gram) combined with the right normalization is enough to linearly classify 5 art styles from SDXL latents with high accuracy. This solidified the decision to use SWD/Gram over heavier attention mechanisms.

