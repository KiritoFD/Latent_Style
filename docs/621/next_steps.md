# 621 下一步行动计划

> 建立日期: 2026-06-21  
> 时间预算: 3天

---

## Day 1: 基线诊断 + 白化修复验证

### 上午 (09:00-12:00)

1. **运行基线探针** (本地 RTX 4070)
   - WFI probe on current最优 checkpoint
   - Endpoint alpha probe
   - Style sensitivity probe
   - Layer statistics probe
   - 输出: `docs/621/probe_results/baseline_*.json`

2. **建立指标基线**
   - 记录所有指标的当前值
   - 与Seedream IDT对比
   - 输出: `docs/621/whitening_baseline.md`

### 下午 (13:00-18:00)

3. **实现无GN endpoint head**
   - 修改 `model620.py`: FiLMEndpointHead 移除 GroupNorm(1)
   - 添加新的配置选项 `endpoint_film_no_norm`
   - 1 epoch smoke test
   - 运行探针验证

4. **实现velocity_scale_loss**
   - 修改 `losses620.py`: 添加 velocity magnitude constraint
   - `loss_velocity_scale = MSE(velocity_abs / target_velocity_abs, 1.0)`
   - 权重: 0.1
   - 1 epoch smoke test

### 晚上 (19:00-22:00)

5. **远程3060环境搭建**
   - 同步代码到远程
   - 安装依赖
   - 运行 smoke test 验证环境
   - 输出: `docs/621/remote_3060_validation.md`

---

## Day 2: 消融实验 (远程3060)

### 上午 (09:00-12:00)

6. **Block A: Style Encoder 多尺度** (~2h)
   - A1: DINO单层 (基线)
   - A2: DINO layers [4,8,11] concat
   - A3: DINO + Trainable LocalCNN
   - 每个配置: 3 epochs, 记录CLIP-S, LPIPS, WFI

### 下午 (13:00-18:00)

7. **Block B: Per-Region SWD** (~2h)
   - B1: 全局SWD (基线)
   - B2: 2-scale SWD (64+32)
   - B3: 3-scale SWD (64+32+16)
   - B4: Attention-weighted SWD

8. **Block C: Skip 信号比** (~2h)
   - C1: α=1.0 (基线)
   - C2: α=[1.0, 0.7, 0.5, 0.3] per-layer
   - C3: α=0.5 all layers
   - C4: 可学习门控 α=sigmoid(w)

### 晚上 (19:00-22:00)

9. **运行完整探针**
   - 对每个实验配置运行探针
   - 记录所有指标
   - 输出: `docs/621/ablation_results.md`

---

## Day 3: 理论修正 + 文档完善

### 上午 (09:00-12:00)

10. **Block D: Cross-Attention 架构** (~2h)
    - D1: Q=concat (基线)
    - D2: Q=bottleneck only
    - D3: Q=proj(DINO features)
    - D4: 仅细尺度CrossAttn

11. **Block E: Attention 稀疏化** (~2h)
    - E1: Softmax (基线)
    - E2: Top-k attention (k=16)
    - E3: Attention entropy 正则

### 下午 (13:00-18:00)

12. **理论修正**
    - 根据实验结果更新数学模型
    - 修正shrinkage系数分解
    - 更新可证伪预测
    - 输出: `docs/621/theory/updated_theory.md`

13. **决策台账更新**
    - 确认哪些实验有用
    - 确认哪些该删除
    - 更新优先级
    - 输出: `docs/621/decisions.md`

### 晚上 (19:00-22:00)

14. **文档完善**
    - 完善所有文档
    - 添加实验图片对比
    - 总结关键发现
    - 输出: `docs/621/final_summary.md`

---

## 关键里程碑

| 时间 | 里程碑 | 验收标准 |
|------|--------|----------|
| Day 1 完成 | 基线诊断完成 | 所有探针运行完毕，指标基线建立 |
| Day 1 完成 | 白化修复验证 | WFI < 0.40 (已有 FiLM hd512) |
| Day 2 完成 | 消融实验完成 | 至少15个配置完成，结果记录 |
| Day 3 完成 | 理论修正完成 | 数学模型与实验结果一致 |
| Day 3 完成 | 文档完善 | 所有文档完成，可交付 |

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 3060 VRAM不足 | 中 | 训练慢 | 降低batch_size, 使用gradient checkpointing |
| WSL环境问题 | 低 | 无法训练 | 使用Windows直接运行 |
| 实验结果不达预期 | 中 | 需要额外时间 | 优先完成P0修复, P1/P2可推迟 |
| 理论与实验不一致 | 低 | 需要修正理论 | 以实验为准，修正理论 |

---

## 交付物清单

| 文件 | 内容 | 状态 |
|------|------|------|
| `docs/621/README.md` | 总索引 | ✅ 完成 |
| `docs/621/architecture_audit.md` | 架构审计 | ✅ 完成 |
| `docs/621/whitening_metrics.md` | 指标体系 | ✅ 完成 |
| `docs/621/probe_design.md` | 探针设计 | ✅ 完成 |
| `docs/621/experiment_inventory.md` | 实验清单 | ✅ 完成 |
| `docs/621/decisions.md` | 决策台账 | ✅ 完成 |
| `docs/621/theory/whitening_mechanism.md` | 白化理论 | ✅ 完成 |
| `docs/621/theory/endpoint_shrinkage.md` | 收缩分析 | ✅ 完成 |
| `docs/621/remote_3060_setup.md` | 远程环境 | ✅ 完成 |
| `docs/621/next_steps.md` | 行动计划 | ✅ 完成 |
| `docs/621/ablation_results.md` | 消融结果 | ⏳ Day 2 |
| `docs/621/final_summary.md` | 最终总结 | ⏳ Day 3 |
| `docs/621/probe_results/` | 探针结果 | ⏳ Day 1-2 |
