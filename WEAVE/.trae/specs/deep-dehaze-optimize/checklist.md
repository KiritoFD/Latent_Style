# Checklist: 深度去雾化优化 (Phase 2) — 最终版

## Task 1: AdaIN 后处理验证 ✅
- [x] AdaIN postprocess 函数实现（model620.py L290-297）
- [x] inference_adain config 参数添加（config_schema.py）
- [x] 独立 eval 脚本 tools/p2_adain_eval.py
- [x] AdaIN OFF eval 完成 + summary_grid.png
- [x] AdaIN ON eval 完成 + summary_grid.png
- [x] **查看对比图**: 饱和度提升 40-70%，雾化显著减轻 ⭐⭐⭐⭐⭐
- [x] 结论: **AdaIN 是 Phase 2 最大突破，零成本显著去雾**

## Task 2: HSV Saturation Loss ❌
- [x] w_hsv_saturation / hsv_sat_threshold config 参数
- [x] losses620.py KL 散度 saturation proxy loss 实现
- [x] 向后兼容验证通过
- [x] 训练 5 epoch 完成
- [x] Full eval + 图片生成
- [x] **查看图片**: 基本无变化，LPIPS 反而恶化
- [x] 结论: **无效，latent-space KL 代理信号太间接**

## Task 3: 长 epoch + 最终验证 ✅（有重要发现）
- [x] 10 epoch 训练完成（显存 < 7.3G，稳定）
- [x] velocity_ratio 曲线: 0.50→0.64（未突破 0.95）
- [x] **关键发现: 过拟合！** LPIPS 从 0.47→0.604（+22%恶化）
- [x] 10ep + AdaIN ON/OFF 对比图生成
- [x] **查看图片**: 10ep+AdaIN 不如 3ep+AdaIN（基础质量下降）
- [x] 结论: **3 epoch = optimal early stopping point**

## Task 4: 最优组合最终验证 ✅
- [x] 完整指标对比表完成（6 组实验）
- [x] 最优方案确定: **R4-D1 (3ep) + AdaIN ON**
- [x] 所有图片已查看并记录
- [x] clip_style / LPIPS / 雾化评分明确

## 最终交付物 ✅
- [x] **最优配置 JSON**: R4-D1 config + inference_adain=true
- [x] **最佳视觉证据**: `r4d1_velmag_high/p2_adain/comparison.png`
- [x] **完整指标表**: Phase 1(12轮) + Phase 2(4轮) 共 16 轮实验
- [x] **修改文件清单**: model620.py, config_schema.py, losses620.py
- [x] **剩余问题清单**: 5 个方向（见 tasks.md）

## 总实验统计
- **Phase 1**: 12 轮迭代训练实验 (R1-A → R6)
- **Phase 2**: 4 轮实验 (AdaIN, HSV Loss, 10ep, Final)
- **总计**: **16 轮实验**，全部在本地 GPU 完成，显存始终 < 7.3GB
