# Checklist: Phase 3 深度突破平凡解

## Task 1: 降低 FM 权重实验
- [ ] w_flow_scale config 参数实现（config_schema.py + losses620.py）
- [ ] 向后兼容验证（w_flow_scale=1.0 时 loss 不变）
- [ ] P3-A (w_flow=0.5) 训练 3 epoch + eval + 图片
- [ ] P3-B (w_flow=0.3) 训练 3 epoch + eval + 图片
- [ ] P3-C (w_flow=0.5 + SWD boost) 训练 3 epoch + eval + 图片
- [ ] **查看所有图片**：风格区分度是否提升？
- [ ] velocity_ratio 和 style_cross_sim 指标记录
- [ ] 结论：最优 w_flow_scale 是多少？

## Task 2: Style Contrastive Loss
- [ ] w_style_contrastive / contrastive_margin config 参数
- [ ] losses620.py 中对比损失实现（InfoNCE 或 cosine margin）
- [ ] style_cross_sim_mean debug metric 输出
- [ ] P3-D (w=0.1, margin=0.1) 训练 3 epoch + eval + 图片
- [ ] P3-E (w=0.5, margin=0.05) 训练 3 epoch + eval + 图片
- [ ] **查看图片**：不同 target 列的视觉差异是否增大？
- [ ] style_cross_sim 是否从 ~0.99 下降到 < 0.9？

## Task 3: FiLM 大初始化
- [ ] film_init_std config 参数（默认 0.0）
- [ ] FiLM 层初始化逻辑修改（支持非零 std）
- [ ] P3-F (std=0.05) 训练 3 epoch + eval
- [ ] P3-G (std=0.10) 训练 3 epoch + eval
- [ ] film_gamma_abs 初始值和收敛值记录
- [ ] 训练稳定性确认（无 NaN/explosion）

## Task 4: 最优组合 + 完整评估
- [ ] 至少 2 个最终组合实验完成（含 AdaIN ON）
- [ ] **完整量化评估**：clip_style, LPIPS, FID 等全部指标
- [ ] 所有 summary_grid.png 生成并**逐张查看**
- [ ] 与 Phase 2 基线的数字对比表
- [ ] 目标达成检查：
  - [ ] clip_style > 0.72 ? 或明确差距
  - [ ] LPIPS < 0.40 ? 或明确差距
  - [ ] 雾化 < 3/10 ? 或明确剩余

## 最终交付物
- [ ] 最优配置 JSON（可直接用于生产/远程训练）
- [ ] 最终 summary_grid.png（最佳结果视觉证据）
- [ ] **完整数字指标表**（Phase 1 vs Phase 2 vs Phase 3）
- [ ] 剩余问题清单和下一步建议
