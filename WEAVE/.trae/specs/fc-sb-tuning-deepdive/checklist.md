# FC-SB 深度调优攻坚 — 验证清单

## 基础设施验证
- [x] Task 1: smoke test 能完整跑完 3 个 epoch 不中断
- [x] Task 1: 训练完成后生成 epoch_0001.pt, epoch_0002.pt, epoch_0003.pt
- [ ] Task 1: 统一评估脚本能对 checkpoint 跑出 summary.json
- [x] Task 1: 远程 GPU 环境 (WSL /mnt/i/...) 可正常访问

## 阶段 1: Sigma 扫描验证
- [ ] Task 2: F1 (sigma=0.04) 成功复现 E2 基线 (clip≈0.708, LPIPS≈0.540)
- [ ] Task 2: F2-F5 全部训练完成 (3 epoch each)
- [ ] Task 2: sigma 扫描结果有明确的趋势（单调或存在甜点）
- [ ] Task 2: 已确定最优 sigma 值，作为后续实验基线

## 阶段 2: SDE 配方验证
- [ ] Task 3: 代码支持 `training_sde_noise_mode` (subtractive/additive)
- [ ] Task 3: F6-F8 全部训练完成 (3 epoch each)
- [ ] Task 3: F6 (推理不加噪) 的 LPIPS 比基线低（验证去噪假说）
- [ ] Task 3: 至少一个 SDE 配方在帕累托前沿上优于阶段1最优
- [ ] Task 3: 已确定最佳 SDE 配方，作为后续实验基线

## 阶段 3: FC-SB 增量验证
- [ ] Task 4: F9-F13 全部训练完成 (3 epoch each)
- [ ] Task 4: F9: Fiber Vel Proj 的增量效果已测量 (style%, LPIPS%)
- [ ] Task 4: F10: Highpass Noise 的增量效果已测量
- [ ] Task 4: F11: Base Locking 的 LPIPS 相对下降 > 10% (AC-4)
- [ ] Task 4: F12: Fiber-Only Endpoint 的增量效果已测量
- [ ] Task 4: F13: Wavelet Lowpass vs AvgPool 对比完成
- [ ] Task 4: 已确定最佳 FC-SB 特性组合

## 阶段 4: 长训练验证
- [ ] Task 5: 代码支持按 epoch 调度训练 sigma (curriculum)
- [ ] Task 5: F14 (curriculum, 5ep) 训练完成
- [ ] Task 5: F15 (constant, 5ep) 训练完成
- [ ] Task 5: 每个 epoch 都有评估数据，可画学习曲线
- [ ] Task 5: 已找到自然收敛的最优停止点 (类似 E4-long 的 epoch 5)

## 阶段 5: CFG 外推验证
- [ ] Task 6: CFG scale 扫描完成 (1.0, 1.5, 2.0, 2.5, 3.0)
- [ ] Task 6: cfg_scale=2.0 比 1.0 的 clip_style 提升 > 5% (AC-5)
- [ ] Task 6: 最佳组合"终极版"实验已跑（如果时间允许）
- [ ] Task 6: Dashboard 已更新所有新数据点
- [ ] Task 6: 新帕累托前沿清晰可见

## 总体验收
- [ ] 至少一个实验超越 E2 帕累托 (clip > 0.708 AND LPIPS < 0.540) (AC-3)
- [ ] 所有实验数据有完整记录 (config + metrics + checkpoint)
- [ ] 初始三大假设已验证/证伪
- [ ] 下一步实验建议清晰明确
