# FC-SB 深度调优攻坚 — 实施任务清单

> 按阶段顺序执行，每个阶段完成后分析结果再决定下一阶段具体参数

---

## [x] Task 1: 基础设施核查与修复
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 确认 `full_eval_each_epoch` 配置是否存在，没有则添加
  - 确认 `train_only.sh` 脚本可用（或新建 robust 版本）
  - 确认统一评估脚本可用
  - 确认远程 GPU 环境可访问
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 启动一个 smoke test 训练，能完整跑完 3 个 epoch 不中断
  - `programmatic` TR-1.2: 训练完成后有 epoch_0001.pt, epoch_0002.pt, epoch_0003.pt
  - `programmatic` TR-1.3: 统一评估脚本能对 checkpoint 跑出 summary.json
- **Notes**: 这是所有后续实验的基础，必须先搞定

---

## [/] Task 2: 阶段1 — Sigma 精细扫描（F1-F5）
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 从 E2 基线配置出发，生成 5 个 sigma 梯度配置
  - F1: sigma=0.04 (精确复现 E2)
  - F2: sigma=0.030
  - F3: sigma=0.035
  - F4: sigma=0.045
  - F5: sigma=0.050
  - 全部训练 3 个 epoch，统一评估
- **Acceptance Criteria Addressed**: AC-2
- **Test Requirements**:
  - `programmatic` TR-2.1: 5 个实验全部有 epoch_0003.pt
  - `programmatic` TR-2.2: F1 的 clip_style ∈ [0.69, 0.72], LPIPS ∈ [0.50, 0.58] (复现E2)
  - `programmatic` TR-2.3: 有 sigma-LPIPS 和 sigma-style 趋势曲线（单调或有甜点）
- **Notes**: 用 batch=24 确保 12GB 显存安全

---

## [ ] Task 3: 阶段2 — 训练-推理对齐实验（F6-F8）
- **Priority**: high
- **Depends On**: Task 2
- **Description**:
  - 基于阶段1找到的最优 sigma，做 3 个 SDE 配方实验
  - F6: 推理时不加噪 (sigma_infer=0.0) — 验证"训练去噪"假说
  - F7: 训练 target 加噪 (additive noise mode) — 训练推理对齐
  - F8: 训练高σ=0.08 + 推理低σ=0.04 — 退火策略
  - F7 需要代码改动：新增 `training_sde_noise_mode` 配置 (subtractive/additive)
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-3.1: 3 个实验全部训练完成
  - `programmatic` TR-3.2: F6 的 LPIPS 比基线低（验证去噪假说）
  - `programmatic` TR-3.3: 至少一个实验在帕累托前沿上优于阶段1最优
- **Notes**: F7 需要修改 losses620.py 中的 SDE 噪声注入逻辑

---

## [ ] Task 4: 阶段3 — FC-SB 增量叠加（F9-F13）
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - 基于阶段2找到的最佳 SDE 配方，逐个叠加 FC-SB 特性
  - F9: + Fiber Velocity Projection (i2sb_fiber_project_endpoint=true)
  - F10: + Highpass Noise (i2sb_fiber_project_noise=true)
  - F11: + Base Locking (bridge_path_mode=vertical)
  - F12: + Fiber-Only Endpoint (fiber_only_endpoint=true + pure_vertical_flow_wavelet)
  - F13: + Wavelet Lowpass (lowpass_mode=wavelet)
  - 每个都和前一个基线对比，确定是增益还是损伤
- **Acceptance Criteria Addressed**: AC-3, AC-4
- **Test Requirements**:
  - `programmatic` TR-4.1: 5 个实验全部训练完成
  - `programmatic` TR-4.2: F11 (Base Locking) 的 LPIPS 比其前一个基线下降 > 10%
  - `programmatic` TR-4.3: 记录每个特性的增量贡献（style 变化%, LPIPS 变化%）
- **Notes**: 如果某个特性导致显著退化，可以跳过后续的叠加，保留最优组合

---

## [ ] Task 5: 阶段4 — 课程学习 + 长训练（F14-F15）
- **Priority**: medium
- **Depends On**: Task 4
- **Description**:
  - 基于阶段3找到的最佳组合，做长训练对比
  - F14: Curriculum σ 调度，5 个 epoch
    - Epoch 0-1: σ = 0.0
    - Epoch 1-3: σ = 最优的 50%
    - Epoch 3-5: σ = 最优的 100%
  - F15: 恒常 σ = 最优值，5 个 epoch（对照）
  - 需要代码改动：支持按 epoch 调度训练 sigma
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-5.1: 两个实验都跑完 5 epoch
  - `programmatic` TR-5.2: 对比每个 epoch 的评估曲线
  - `programmatic` TR-5.3: 找到自然收敛的最优停止点
- **Notes**: Curriculum 调度需要修改 trainer.py 或 losses620.py

---

## [ ] Task 6: 阶段5 — CFG 外推 + 组合爆破
- **Priority**: medium
- **Depends On**: Task 5
- **Description**:
  - 用最佳模型做 CFG scale 扫描 [1.0, 1.5, 2.0, 2.5, 3.0]
  - 把所有正向特性组合在一起，做一个"终极版"实验（如果时间够）
  - 更新 dashboard，画出新的帕累托前沿
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-6.1: CFG 扫描有完整数据点
  - `programmatic` TR-6.2: cfg_scale=2.0 比 1.0 的 clip_style 提升 > 5%
  - `human-judgement` TR-6.3: Dashboard 更新，新帕累托前沿清晰可见
- **Notes**: CFG 外推不需要重新训练，直接用已有的 checkpoint

---

## [ ] Task 7: 结果汇总与报告
- **Priority**: medium
- **Depends On**: Task 6
- **Description**:
  - 汇总所有 15+ 个实验的结果
  - 标注每个特性的增量贡献
  - 验证/证伪初始假设
  - 输出下一步建议
- **Acceptance Criteria Addressed**: AC-3, AC-4, AC-5
- **Test Requirements**:
  - `human-judgement` TR-7.1: 完整的实验结果表格
  - `human-judgement` TR-7.2: 清晰的结论和下一步建议
  - `programmatic` TR-7.3: Dashboard 已更新所有数据点
- **Notes**: 重点回答：训练 SDE 噪声是帮倒忙吗？Base Locking 有用吗？
