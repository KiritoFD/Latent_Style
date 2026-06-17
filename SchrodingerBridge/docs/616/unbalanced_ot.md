# 非平衡最优传输 (Unbalanced OT) — 哑类匹配

## 问题

标准 OT 要求 $\sum_j \Pi_{ij} = \mu_i$ 和 $\sum_i \Pi_{ij} = \nu_j$——每张源图必须找到匹配，每张目标图必须被匹配。
在无配对风格迁移中，batch 内的源图和目标图**可能根本没有合理的配对**——极简风格的源图面对的全是巴洛克目标，
或者目标库里根本没有"大面积留白的画作"。强行 1-to-1 匹配会引入巨大的噪声梯度。

## 数学

非平衡 OT 放松了质量守恒约束，允许部分概率质量被"丢弃"到哑类：

$$\min_{\Pi} \langle \Pi, C \rangle + \epsilon H(\Pi) + \tau_{\text{src}} \text{KL}(\Pi \mathbf{1} \| \mu) + \tau_{\text{tgt}} \text{KL}(\Pi^T \mathbf{1} \| \nu)$$

- $\tau_{\text{src}}$: 源侧的"放弃惩罚"——越大 → 越不能放弃 → 越接近标准 Sinkhorn
- $\tau_{\text{tgt}}$: 目标侧的"放弃惩罚"
- 哑类的概率质量 = $1 - \sum_i \Pi_{ij}$（未被任何源图匹配的目标）或 $1 - \sum_j \Pi_{ij}$（放弃匹配的源图）

**直觉**: $\tau$ 小 → 模型可以选择"这张源图在目标库里找不到好匹配，我先不学它"→ 降噪梯度。

## 配置

```json
{
  "bridge": {
    "coupling_solver": "sinkhorn_unbalanced",
    "sinkhorn_unbalanced_tau_src": 0.5,
    "sinkhorn_unbalanced_tau_tgt": 0.8
  }
}
```

- `tau_src=0.5` — 源侧较放松（允许放弃不合适的匹配）
- `tau_tgt=0.8` — 目标侧较严格（避免所有目标被放弃）

## 如何判断哑类是否工作

观察诊断指标:
- `ot_raw_total_mass` < 1.0: 哑类吸收了质量
- `ot_source_truncation` > 0: 源侧有质量被放弃
- `ot_target_truncation` > 0: 目标侧有质量被放弃
- `ot_target_gini` < 0.4: 没有枢纽现象

## 与结构感知 OT 的配合

推荐组合:
- `coupling_cost_composition = "appearance_plus_structure"`
- `coupling_structure_cost_weight = 0.3` (给外观 70% 权重)
- `coupling_solver = "sinkhorn_unbalanced"`
- `sinkhorn_unbalanced_tau_src = 0.5`
- `sinkhorn_unbalanced_tau_tgt = 0.8`

结构代价防止"颜色匹配"的平凡解，Unbalanced OT 防止"找不到匹配还硬凑"的噪声梯度。
