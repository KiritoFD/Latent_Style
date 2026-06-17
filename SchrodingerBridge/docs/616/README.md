# 616 项目状态 — 可交接文档

> 最后更新: 2026-06-17

## 当前瓶颈

结构保持（LPIPS 0.31）已通过 TopoGate 完美解决。**唯一瓶颈是 Style 推不到 0.72+**。

## 根本原因诊断

ODE 确定性路径收敛到纤维上的条件期望 $\mathbb{E}[X \mid c]$ ——"平均笔触"而非"锐利画笔"。

具体的 4 个理论-实现差距:

| # | 问题 | 状态 |
|---|------|------|
| 1 | OT 匹配用欧氏距离 → 退化为颜色匹配 (Gini 升高) | 📋 代码已支持结构 OT，需切换到 `tokenizer_entropy_affinity_gw` |
| 2 | 垂直 FM 被 validation 拒绝 | ✅ 已修复 (`bridge_path_mode="vertical"`) |
| 3 | SMoE tok_delta=0.019，被水平分量 loss 压制 | 📋 需配合垂直 FM + 提高 tokenizer lr |
| 4 | 非平衡 OT 未启用 (哑类匹配) | 📋 代码已有 `coupling_solver="sinkhorn_unbalanced"` |

## 推荐的下一个实验

```json
// 单次训练跑 3 个修改，一步到位
{
  "bridge": {
    "bridge_path_mode": "vertical",
    "coupling_solver": "sinkhorn_unbalanced",
    "sinkhorn_unbalanced_tau_src": 0.5,
    "coupling_structure_cost_mode": "tokenizer_entropy_affinity_gw",
    "coupling_cost_composition": "appearance_plus_structure",
    "coupling_structure_cost_weight": 0.3
  }
}
```

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
