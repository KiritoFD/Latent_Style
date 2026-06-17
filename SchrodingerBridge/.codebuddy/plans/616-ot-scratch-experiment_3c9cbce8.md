---
name: 616-ot-scratch-experiment
overview: 完成 616 OT scratch 长训练实验：审查修复 OT 实现与 infra 问题，创建 self_affinity_gw + tokenizer_entropy_affinity_gw 两种结构代价模式 matched scratch 配置，编写远程 3060 启动 .sh 脚本，固化实验计划文档。按 unified-plan 从头训练（无 warmstart），24 epochs，b8a2。
todos:
  - id: fix-ot-losses
    content: 修复 src/losses.py 中 OT 实现问题：抽取 _vertical_lowpass 方法消除重复代码（问题B），删除 _project_training_target 死代码（问题C），在 __init__ 中读取 bridge_vertical_base_stride 消除冗余 getattr（问题D）。使用 [subagent:code-explorer] 验证无遗漏调用点。
    status: completed
  - id: fix-scratch-config
    content: 修复 configs/aaai2027/phase616_ot_vertical_scratch_b8a2_e24.json：添加 unbalanced Sinkhorn 参数（tau_src=0.5, tau_tgt=0.5, dummy_cost=1.0, dummy_offdiag_cost=8.0）激活真正 OT 截断（问题A）
    status: completed
    dependencies:
      - fix-ot-losses
  - id: create-tokentropy-config
    content: 新建 configs/aaai2027/phase616_ot_tokentropy_scratch_b8a2_e24.json：以 self_affinity_gw 配置为模板，改 coupling_structure_cost_mode 为 tokenizer_entropy_affinity_gw，同样添加 unbalanced 参数，更新 save_dir 和 ablation.name
    status: completed
    dependencies:
      - fix-scratch-config
  - id: write-launch-script
    content: 编写 tools/experiments/run_phase616_ot_scratch_matched.sh：通过 launch_remote_experiment_train.py 顺序远程启动两种模式 scratch 实验，含 smoke 跳过和 GPU 内存守卫参数
    status: completed
    dependencies:
      - create-tokentropy-config
  - id: write-plan-doc
    content: 编写 docs/616/ot_scratch_experiment_plan.md：固化实验计划，含背景、配置矩阵、修复清单、远程环境参数、预期指标和结论文档路径
    status: completed
    dependencies:
      - fix-scratch-config
---

## 产品概述

完成 616 OT 阶段的 scratch 长训练实验：审查并修复 OT 实现与 infra 问题，创建两种结构代价模式的 matched 对比配置，编写远程 3060 启动脚本，固化实验计划文档。

## 核心功能

- 审查并修复 `src/losses.py` 中 OT 实现的 4 个问题（配置语义、DRY 违规、死代码、冗余 getattr）
- 修复现有 scratch 配置中 unbalanced Sinkhorn 参数失效问题（tau=1.0 等效 balanced，dummy_cost=0 跳过 dummy 列）
- 新建 `tokenizer_entropy_affinity_gw` 模式的 scratch 配置，与 `self_affinity_gw` 做 matched 对比
- 评估并修复 full_eval_in_process 在 12GB 3060 上的内存风险
- 编写远程 3060 启动 .sh 脚本，支持两种模式顺序运行
- 将实验计划固化为 `docs/616/` 下的文档

## 技术栈

- 训练框架：PyTorch + 自定义 OTFlowMatchingObjective（src/losses.py）
- OT 求解器：自实现 Sinkhorn（balanced + unbalanced）+ Hungarian（CPU 后备）
- 结构代价：7 种模式（self_affinity_gw, tokenizer_entropy_affinity_gw 等），基于 Gromov-Wasserstein 风格的自亲和矩阵
- 远程基础设施：SSH + WSL2（Ubuntu-26.04）+ nvidia-smi 监控 + nohup fallback
- 配置系统：JSON + `_base` 继承链（config_schema.py load_config）

## 实现方案

### 问题 A 修复 - Unbalanced Sinkhorn 参数（严重）

**现状**：`phase616_ot_vertical_scratch_b8a2_e24.json` 设 `coupling_solver=sinkhorn_unbalanced` 但 `tau_src=tau_tgt=1.0, dummy_cost=0.0`。代码中 `rho = tau / (tau + eps)`，tau=1.0 时 rho≈0.952 几乎 balanced；`_augment_cost_with_source_dummies`（losses.py:1078）在 `dummy_cost <= 0.0` 时直接返回不扩展 dummy 列。

**修复**：在两个 scratch 配置中显式设置：

- `sinkhorn_unbalanced_tau_src: 0.5`（rho≈0.5，允许源端质量截断）
- `sinkhorn_unbalanced_tau_tgt: 0.5`
- `sinkhorn_unbalanced_dummy_cost: 1.0`（激活 dummy 列，允许不匹配样本被丢弃）
- `sinkhorn_unbalanced_dummy_offdiag_cost: 8.0`（保持默认，dummy 对角线廉价、非对角昂贵）

### 问题 B 修复 - DRY 违规（中等）

**现状**：`_bridge_state_and_velocity`（losses.py:1245-1249）和 `_bridge_path_state`（losses.py:1305-1312）各自定义相同的 `_lowpass` 闭包：

```python
k = max(1, int(getattr(self, "bridge_vertical_base_stride", 2)))
def _lowpass(x):
    d = F.avg_pool2d(x.float(), kernel_size=k, stride=k)
    return F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False)
```

**修复**：抽取为 `_vertical_lowpass(self, x: torch.Tensor) -> torch.Tensor` 方法，两处调用统一引用。

### 问题 C 修复 - 死代码（轻微）

**现状**：`_project_training_target`（losses.py:874）`projected_low = anchor_low` 赋值后在 pure_vertical_flow 分支未被任何后续代码使用（只有 `projected` 被返回，`projected_low_eval` 从 `projected` 重新 split）。

**修复**：删除 `projected_low = anchor_low` 行。

### 问题 D 修复 - 冗余 getattr（轻微）

**现状**：losses.py:1245, 1305 用 `getattr(self, "bridge_vertical_base_stride", 2)`，但 `bridge_vertical_base_stride` 已是 BridgeConfig 已知字段（config_schema.py:497，默认 2）。在 `OTFlowMatchingObjective.__init__` 中未显式读取该字段。

**修复**：在 `__init__` 中添加 `self.bridge_vertical_base_stride = max(1, int(bridge_cfg.bridge_vertical_base_stride))`，两处改用 `self.bridge_vertical_base_stride`。

### 问题 E 评估 - full_eval_in_process 内存风险（中等）

**现状**：scratch 配置继承 base 的 `full_eval_in_process=true`。run.py:286-293 在同进程内 `from utils.run_evaluation import main` 后执行评估。trainer 已有 offload_for_full_eval/restore_after_full_eval（trainer.py:308-340）在评估前将模型移至 CPU。

**分析**：offload 机制已有效缓解 VRAM 压力。in_process 模式的优势是避免子进程启动开销和 VAE/CLIP 模型重复加载（`full_eval_runtime_model_cache=true` 配合）。12GB 3060 上 b8a2 + channels_last + bf16 AMP 训练峰值约 8-9GB，offload 后评估可用约 10GB，transfer_only 快速评估（max_src_samples=10）足够。

**结论**：保留 `full_eval_in_process=true`，但添加防御性检查——在 `run.py` 的 `_run_full_eval_for_checkpoint` 中，确保评估后 `gc.collect()` + `torch.cuda.empty_cache()` 被调用（现有代码 run.py:291-293 已有此逻辑）。无需改动。

### 问题 G - 新建 tokenizer_entropy_affinity_gw 配置

以 `phase616_ot_vertical_scratch_b8a2_e24.json` 为模板，仅改：

- `coupling_structure_cost_mode: "tokenizer_entropy_affinity_gw"`
- `checkpoint.save_dir` 和 `ablation.name` 改为对应名称
- 同样添加问题 A 的 unbalanced 参数修复

## 实现注意事项

- **性能**：vertical lowpass 使用 stride=k 的 avg_pool + bilinear interpolate，复杂度 O(N) 不引入瓶颈。tokenizer_entropy_affinity_gw 需要额外前向调用 structured_style_tokenizer 获取 aux_map，每 step 增加 1 次前向，在 b8a2 下约 +15% 训练时间，可接受。
- **Blast radius**：问题 B/C/D 修改是纯重构，不改变任何计算结果。问题 A 修改配置参数会改变 OT 计划行为（从近 balanced 变为真正 unbalanced），这是实验目标的一部分。
- **向后兼容**：不修改 BridgeConfig 字段定义，不修改 `_normalize_phase616_bridge_ot_defaults` 默认值。新配置通过显式设置覆盖默认。
- **日志**：现有 OT debug 指标（ot_source_truncation, ot_target_truncation, ot_dummy_mass 等）已完整，unbalanced 激活后这些指标将有非零值，可用于验证修复效果。

## 目录结构

```
SchrodingerBridge/
├── src/
│   └── losses.py                                          # [MODIFY] 修复问题B/C/D：抽取_vertical_lowpass方法，删除死代码，消除冗余getattr
├── configs/aaai2027/
│   ├── phase616_ot_vertical_scratch_b8a2_e24.json         # [MODIFY] 修复问题A：添加unbalanced参数(tau=0.5, dummy_cost=1.0)
│   └── phase616_ot_tokentropy_scratch_b8a2_e24.json       # [NEW] tokenizer_entropy_affinity_gw模式，matched对比配置
├── tools/experiments/
│   └── run_phase616_ot_scratch_matched.sh                 # [NEW] 远程3060顺序启动两种模式scratch实验
└── docs/616/
    └── ot_scratch_experiment_plan.md                       # [NEW] 固化实验计划文档
```

## 关键代码结构

`_vertical_lowpass` 方法签名（losses.py 新增）：

```python
def _vertical_lowpass(self, x: torch.Tensor) -> torch.Tensor:
    """Strided avg-pool + bilinear upsample for vertical flow base manifold."""
    k = self.bridge_vertical_base_stride
    d = F.avg_pool2d(x.float(), kernel_size=k, stride=k)
    return F.interpolate(d, size=x.shape[-2:], mode='bilinear', align_corners=False)
```

Unbalanced Sinkhorn 配置片段（两份 scratch 配置的 bridge 段）：

```
{
  "coupling_solver": "sinkhorn_unbalanced",
  "sinkhorn_unbalanced_tau_src": 0.5,
  "sinkhorn_unbalanced_tau_tgt": 0.5,
  "sinkhorn_unbalanced_dummy_cost": 1.0,
  "sinkhorn_unbalanced_dummy_offdiag_cost": 8.0
}
```

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 在实现阶段深入搜索 losses.py 中所有 `getattr(self, "bridge_vertical_base_stride"` 调用点，确保重构无遗漏；验证 `_augment_cost_with_source_dummies` 在 dummy_cost>0 时的代码路径完整性
- Expected outcome: 确认所有修改点已覆盖，无遗漏的 vertical lowpass 重复代码或未处理的 edge case