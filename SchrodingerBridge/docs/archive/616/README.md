# 616 项目状态 — 可交接文档

> 最后更新: 2026-06-17

## 当前状态

**Active experiment**: `exp/20250618_lite_ot_vertical/` — b24, vl=0.1, legacy_factorized tokenizer + topogate, 7 hypothesis tests.

结构保持（LPIPS 0.31）已通过 TopoGate 完美解决。**唯一瓶颈是 Style 推不到 0.72+**。

## 已确认的关键决策

| # | 决策 | 状态 |
|---|------|------|
| 1 | PureLatentSpatial tokenizer ZERO ROI | ❌ 确认废弃 — style/LPIPS 不变，白耗 ~1.2GB VRAM |
| 2 | 代码改用 legacy_factorized + `ablation_disable_spatial_prior=true` | ✅ 已落实 |
| 3 | OT 结构代价不依赖 tokenizer 输出或 encoder 特征 | ✅ 改用 TopoGate 内生 attention |
| 4 | 新模式 `topogate_attention_gw` 替代 `tokenizer_entropy_affinity_gw` | ✅ 代码已实现 |
| 5 | `virtual_length_multiplier` 配置位置修复（从 training 移到 data） | ✅ 已修复 |
| 6 | Pairing cache 禁用（它禁用了 OT 的候选池） | ✅ 已修复 |
| 7 | `_prepare_style_maps` 未检查 `ablation_disable_spatial_prior` | ✅ 已修复 |
| 8 | WSL 后台进程 SSH 断开后不存活 | ⚠️ 已知，需 nohup/screen/tmux |

## 根本原因诊断

ODE 确定性路径收敛到纤维上的条件期望 $\mathbb{E}[X \mid c]$ ——"平均笔触"而非"锐利画笔"。

4 个理论-实现差距:

| # | 问题 | 状态 |
|---|------|------|
| 1 | OT 匹配用欧氏距离 → 退化为颜色匹配 (Gini 升高) | ✅ 已切换到 `topogate_attention_gw` |
| 2 | 垂直 FM 被 validation 拒绝 | ✅ 已修复 (`bridge_path_mode="vertical"`) |
| 3 | SMoE tok_delta=0.019，被水平分量 loss 压制 | 📋 废弃 SMoE，legacy_factorized 替代 |
| 4 | 非平衡 OT 未启用 (哑类匹配) | 📋 代码已有 `coupling_solver="sinkhorn_unbalanced"` |

## 运行中的实验

```json
{
  "exp_dir": "exp/20250618_lite_ot_vertical/",
  "bridge": {
    "bridge_path_mode": "vertical",
    "coupling_structure_cost_mode": "topogate_attention_gw",
    "coupling_cost_composition": "appearance_plus_structure",
    "coupling_structure_cost_weight": 0.3
  },
  "tokenizer": {
    "tokenizer_name": "legacy_factorized",
    "ablation_disable_spatial_prior": true
  },
  "training": {
    "batch_size": 24,
    "virtual_length_multiplier": 0.1
  }
}
```

7 个假说测试: h0 (vertical baseline), h1 (linear对照), h2 (appearance_only), h3 (SDE噪声), h4 (unbalanced), h5 (topogate_attention_gw), h6 (全组合)。

## 已知的关键实验数据

- topogate e2: transfer 0.671/0.314, all-pairs 0.703/0.312
- SMoE e8: 0.670/0.318
- I2SB orthogonal e1: style=0.705 (最高) 但 LPIPS 0.447
- 所有 ODE 方法 style 均稳定在 0.67-0.70

## 文档索引

| 文件 | 内容 |
|------|------|
| `design.md` | 垂直 FM + AffineConnectionTokenizer 的数学推导 |
| `ot_theory.md` | OT 失效分析 + 7 种结构代价模式 |
| `unbalanced_ot.md` | 非平衡 OT / 哑类匹配 |
| `debug.md` | 4 维度诊断探针 + 自动熔断 |
| `infra.md` | WSL/GPU 优化 + eval 耗时分布 |
| `tools.md` | 可用脚本和轮子 |
| `launch.md` | 实验启动指南 |
