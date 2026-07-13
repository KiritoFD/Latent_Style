# SchrodingerBridge Documentation

**最后更新**: 2026-07-13 (代码清理 + 文档归档)
**状态**: 所有失败实验代码已删除，文档已归档到 archive/

---

## 1. 当前文档结构

| 目录 | 用途 | 内容简述 |
|------|------|---------|
| [method.md](method.md) | **核心方法文档** | WEAVE 架构完整数学描述、组件消融、DINO-S 天花板验证 |
| [delivery/](delivery/) | **交付文档** | 当前结论、最佳 checkpoint、运行命令、代码状态 |
| [710/](710/) | **710 阶段结论** | DWT routing、AdaLN 消融、ASG 突破、DINO 协议、infra 优化 |
| [dino_s_break/](dino_s_break/) | **DINO-S 突破跟踪** | 所有 Round 实验进度、findings、Pareto 前沿 |
| [latent_migration/](latent_migration/) | **全分辨率评估** | 256/512 最终指标表、所有 baseline 对比 |
| [79/](79/) | **数据注册表** | 主表数据源、baseline 结果溯源、统一评估 CSV |
| [baseline/](baseline/) | **Baseline 方法** | 12 baseline 完整收敛证据 |
| [math/](math/) | **理论框架** | FC-SB 完整理论、Haar DWT、频域解耦、ODE 求解器 |
| [tools/](tools/) | **工程参考** | 数据库、评估协议、infra、调用命令、实验经验 |
| [archive/](archive/) | **历史归档** | 所有历史文档（不再维护，仅追溯） |

---

## 2. 关键文档速查

### 2.1 论文核心

- **方法描述**: [method.md](method.md) — 架构数学公式、有效组件、DINO-S 天花板分析
- **交付文档**: [delivery/DELIVERY_SUMMARY.md](delivery/DELIVERY_SUMMARY.md) — 最佳 checkpoint、结论、运行命令
- **710 结论**: [710/710_CONCLUSIONS.md](710/710_CONCLUSIONS.md) — DWT routing、ASG 突破、AdaLN 失败、infra 优化

### 2.2 实验数据

- **最终指标表**: [latent_migration/final_metrics_table.md](latent_migration/final_metrics_table.md) — 256/512 所有方法对比
- **DINO-S 进度**: [dino_s_break/state/progress.json](dino_s_break/state/progress.json) — 各 Round 实验状态
- **Baseline 溯源**: [79/](79/) — 主表数据来源、统一评估 CSV

### 2.3 实验清单

- **远程实验**: [archive/exp/remote_experiments.md](archive/exp/remote_experiments.md) — I 盘所有实验
- **本地实验**: [archive/exp/local_experiments.md](archive/exp/local_experiments.md) — G 盘所有实验
- **方法审计**: [archive/exp/method_audit_2026-07-11.md](archive/exp/method_audit_2026-07-11.md) — 有效组件独立审计

---

## 3. 核心结论摘要

### DINO-S 天花板

经过 **30+ 轮实验**（Round 1-12）验证：
- DINO-S ≈ 0.48 ± 0.003 是当前 SAT 范式 fundamental limit
- 所有上游风格通路（11 方向）和下游 Decoder AdaLN（10 方向）均无法突破
- 唯一突破方式：推理时 Endpoint AdaIN 缩放（α=1.5 → 0.4843, α=2.0 → 0.4859）

### 有效组件

WEAVE 实际有效组件仅 3 个：
1. **Rectified Flow Matching** — 核心传输引擎
2. **Haar Wavelet Decomposition** — 正交频域分离
3. **Endpoint AdaIN** — 推理时风格注入

### 最佳 Checkpoint

| 配置 | 文件 | DINO-S | CLIP-S | 说明 |
|------|------|--------|--------|------|
| WEAVE-m (α=1.5) | `I:/checkpoints/brk_a_ll03_10ep/epoch_0010.ckpt` | 0.4843 | 0.7180 | 主表主点 |
| WEAVE-q (α=2.0) | same | 0.4859 | 0.7075 | DINO-S 天花板 |

---

## 4. 历史归档

所有历史文档已归档到 [archive/](archive/)，包括：
- 19 个日期型目录 (612-630)
- 15 个主题型目录 (theory, timing, model, reviews, plans, cleanup, exp, 710, 72, SWD, 630, refactor_task, model_probe, baseline_256, root)
- 26 个根级历史文件

归档文档不再维护，仅作历史追溯。详见 [archive/](archive/)。

---

## 5. 文档维护原则

1. **新增实验**: 记录到对应 exp 目录，更新 `delivery/DELIVERY_SUMMARY.md`
2. **新增结论**: 更新 `method.md` + `delivery/DELIVERY_SUMMARY.md`
3. **过时文档**: 移动到 `archive/`，git commit 保留历史
4. **无效代码**: 确认无效后直接删除，git commit + 文档记录
5. **数值变更**: 所有数值需在 `method.md`、`delivery/`、`710/` 同步更新

---

**文档维护**: 2026-07-13 代码清理 + 文档归档
**代码状态**: ✅ 所有失败实验代码已删除，src/ 仅含有效文件
**数据状态**: ✅ 所有数值与 deliver/DELIVERY_SUMMARY.md 对齐