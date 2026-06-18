# OT.md — 最优传输层设计: 从在线 Sinkhorn 到确定性配对

> 承接 math.md C1+C3. 本文件只回答一个问题: **训练时, 一张内容图应该往哪张目标风格图走**.
> 在线 Sinkhorn / Minibatch OT 已被"均值坍缩定理"判死, 这里给出替代方案及其代价分析.

---

## 1. 现状诊断: 仓库内已有的 OT 路线全表

来自 `losses.py:64-150`, `ot_cost.py`, `docs/616/ot_theory.md`, `docs/experiments/`:

| 路线 | 代价矩阵 | 求解器 | 状态 | 真实结果 |
|------|---------|-------|------|---------|
| Minibatch Sinkhorn (legacy default) | latent Euclidean² | GPU Sinkhorn | 主线 | `ot_target_gini` 高, hub 现象 |
| Hungarian OT (allow_cpu_hungarian) | latent Euclidean² | scipy CPU | 仅诊断 | 训练不可用 |
| Unbalanced Sinkhorn | latent Euclidean² + KL松约束 | sinkhorn_unbalanced | 已实现 | 没改善, `tau_src/tgt` 调不动 |
| structure_only / `self_affinity_gw` | latent self-affinity cdist² | Sinkhorn | 默认配置 | Var≈常量, 退化为随机匹配 |
| `topogate_attention_gw` | TopoGate complexity profile cdist² | Sinkhorn | 616 推荐 | 仅靠最后1个 body block, 信号弱 |
| `encoder_self_affinity_gw` | UNet encoder 特征 GW | Sinkhorn | 待评估 | 需额外 forward, 不划算 |
| 垂直流内 OT | latent Euclidean (垂直分量) | Sinkhorn | `bridge_path_mode=vertical` | **LPIPS 极好, style 0.669 卡死** |

### 1.1 OT 路线失败的统一根因

无论选哪个代价矩阵, 它们都共享同一个"通病":
**目标 $z_s$ 在每个 batch 内浮动的可能性是 O(B) 离散的 (B=16), 单一 $(z_c, z_s)$ 对的稳定性 $\Pr[\text{same pair next epoch}]\approx O(B/|\mathcal{D}|)$**.

由均值坍缩定理 (math.md §2), 这等价于"训练目标在跳变中走条件期望".

OT 路线**改善代价矩阵救不了这个**. 因为这是 OT 在线计算自身的本质 — 它只能在 batch 闭包里求最优, 跨 batch 必然抖动.

→ **唯一出路: 离线决策, 训练时只回归.**

---

## 2. 替代方案 A: Independent Coupling (无 OT)

直接对每个 batch 内 $(z_c, z_s)$ 做随机配对, 不做任何 OT. 但每个 epoch **同一对**$(z_c, z_s)$要么始终配对, 要么完全随机每 epoch 重抽.

### 2.1 两种 sub-variant

**A1. Epfixed Independent** (619 推荐):
- 每个 epoch 开始时, 把内容图打乱顺序, 与风格图打乱顺序对齐 — 但同一 epoch 内, 第 $i$ 个 (content, style) 对**保持不变**.
- 下一 epoch 重新打乱, 但 epoch 内目标稳定.
- 这种"per-epoch stable"是不够的 — `v_target` 在 cross-epoch 仍跳变.

**A2. Hard-Pair Independent** (本方案推荐):
- **离线一次性**指定每张内容图 $z_c^{(i)}$ 的固定目标 $\hat z_s^{(i)}$, 整个训练不变.
- 即使目标不"最优", 但 $v_{target}$ 是确定函数, epoch 间稳定.
- 这才是真正满足 math.md C1 的方案.

### 2.2 A2 的副作用 — 真正的不安

确定性配对意味着 $v_{target}^{(i)}=\hat z_s^{(i)}-z_c^{(i)}$ 是固定的.
模型在数学上会从"学分布"退化为"学实例复制".
若训练集大小 $|\mathcal{D}|=$ 数万, 这种"实例复制"反而能学到 style 的局部纹理 — 接近 paired I2I 的训练动力学.

但有个隐患: **同一 content 在所有 epoch 见到的 target 是同一张图, 模型会过拟合到这特定 pair 的非纹理细节** (例如 target 里那个特定的物体形状, 而不是 target 的笔触).

**缓解 (重要)**: 选多张 target, 每 content 固定 5–20 张候选 $\{\hat z_s^{(i,k)}\}_{k=1}^K$, 训练时按 epoch 轮转或采样. 这把"跨 epoch 抖动"控制在 $K$ 内, 又提供了多样性. math.md §2 的均值坍缩在 $K=5\sim 20$ 时被显著缓解 (因为 attractor 集合从 $O(B)$ 缩到 $O(K)$).

---

## 3. 替代方案 B: 离线 DINOv2 语义弱配对 (本方案)

### 3.1 流程

```
[离线, 一次性]
1. 所有内容图 → DINOv2 CLS feature  (frozen, 384-D)
2. 所有风格图 → DINOv2 CLS feature  (frozen, 384-D)
3. 对每张内容图 i:
   - 算余弦相似度 top-20 候选风格图
   - 随机抽其中 K=8 张作为该 content 的固定候选集
4. 保存:  mapping.json = {content_idx: [style_idx_1..8]}
```

训练时: 每 step, 对 batch 内的 content 从其候选集随机抽 1 张作为 target.

### 3.2 为什么用 DINO 而不是 latent Euclidean

616/ot_theory.md §2.2 的"度量空间错配定理"在这里直接适用:
- Latent Euclidean² 被 DC 分量 (亮度/对比度) 主导 → 退化为亮度匹配 → hub.
- DINOv2 CLS 是 *语义/结构* 表示, 余弦距离反映"这张风景照 vs 这幅风景油画"的语义对齐, 不是颜色距离.
- 这给"内容图找结构相似的风格图"提供了正确度量.

> 注意: DINO 在这里只参与**匹配选择**, 不参与训练前向. 它不是 tokenizer.
> tokenizer.md 会单独设计 forward 用的风格表征.

### 3.3 为什么 top-20 候选 + K=8 抽取

- top-20 保证候选都"语义说得过去" (不是把一只猫匹配到抽象画).
- K=8 抽取给训练目标引入足够多样性, 同时把均值吸引子集合限制在 8 张 — math.md §2.1 的"高频抵消推论"在 $K=8$ 时高频期望仍有显著方差, 不会清零.
- 这是 $K\to\infty$ (Minibatch OT) 与 $K=1$ (hard pair, 过拟合) 之间的 sweet spot.

### 3.4 与 619/prematched_ot_evaluation.md 的差异

619 外审建议在 latent 上做**像素级 Sinkhorn 重排** (`Z_aligned = plan @ Z_style`). 619/problem.md 已正确指出这是**致命错误** (VAE decoder checkerboard artifacts).

本方案的关键差异:
- **不做任何 latent 重排**. 离线只算"哪张图配哪张图", 是 instance-level 决策.
- 训练 target 直接是 $\hat z_s$ 本身 (它的原始 VAE latent), 让 forward 中的 Cross-Attention 自己学纹理搬运.

### 3.5 与 616/unbalanced_ot.md 的关系

Unbalanced OT 在 math.md 框架下是"减害不减因":
- 减害: 允许"找不到匹配就跳过", 降低噪声梯度.
- 不减因: 在线计算本质未变, $\mathbb{E}[z_s\mid z_t]$ 仍不稳, 均值坍缩仍在.

→ 在 A2/B (确定性配对) 已经做到"$\mathbb{E}[z_s\mid z_t]\!=\!z_s$"的情况下, Unbalanced OT 完全多余. **废弃**.

---

## 4. 替代方案 C: Gromov-Wasserstein 离线配对 (可选升级)

如果方案 B 在新 tokenizer 上仍然 style 不够, 升级到 GW 离线:

- 把每张图的 DINOv2 spatial feature (16×16×384 = 256 tokens) 算 self-affinity 矩阵 $A\in\mathbb{R}^{256\times 256}$.
- 用 GW 距离比较 $A_{content}$ 与 $A_{style}$: 拓扑同构的距离, 不是颜色距离.
- 这保证"复杂图配复杂画"(616/ot_theory.md §7 的洞察).

但 GW 计算昂贵 ($O(N^3)$ 内存, $N$=数据集大小). 实施建议:
1. 先用 DINOv2 CLS top-20 缩小候选集 (方案 B).
2. 在 top-20 内做 GW ranking 重排, 取 top-8.

→ 这是 B 的"二次精细化", 不进入第一轮实验.

---

## 5. 替代方案 D: 合成强配对 (Paired Synthetic)

用 ControlNet + 风格 LoRA 离线生成"同内容不同风格"的强配对数据.

- 优势: 直接降维为 supervised I2I, 训练最稳定, 几乎不可能均值坍缩.
- 代价: 上限被合成器的风格质量锁定; 引入合成器自身伪影.
- 用法: 作为方案 B 的"补充种子" — 一小部分用合成强配对 (解决 cold start), 大部分用 DINOv2 弱配对.

→ 方案 D 不优先. 只有在 B + 新 tokenizer 跑完仍到不了 0.72 时再考虑.

---

## 6. 选定方案与配置 (推荐 B 起步)

**OT 配置**:

```jsonc
{
  "pairing": {
    "mode": "offline_dino_topk",
    "encoder": "dinov2_vits14",        // frozen
    "feature": "cls_token",
    "topk_pool": 20,                     // 离线召回候选池大小
    "k_per_content": 8,                  // 每 content 固定候选
    "sample_strategy": "epoch_rotate",   // 每 epoch 轮转 1 张
    "cache_path": "data/pairing_dinov2_top20_k8.json"
  },
  "bridge": {
    "objective_mode": "fm_velocity",                  // math.md C4 不展开
    "coupling_solver": "independent",                  // 在线不再做 OT
    "coupling_structure_cost_mode": "none",            // OT 模块整体退役
    "bridge_path_mode": "vertical",                    // math.md C3 保留
    "terminal_swd_weight": 0.0,                        // 训练态不再算 SWD (移到单步假想终点)
    "single_step_endpoint_weight": 12.0,               // 在 $\hat z_1$ 上算 SWD
    "w_flow": 1.0
  }
}
```

**保留 vertical FM** 是 math.md C3 的直接落地: 防止水平泄漏.
**退役所有在线 OT 模块** (`sinkhorn`, `self_affinity_gw`, `topogate_attention_gw` 等): math.md §2 已证明它们救不了均值坍缩, 是冗余计算.

---

## 7. 验证指标 (Phase A, 1-2 epoch smoke)

| 指标 | 老路线 (Minibatch OT) 期望 | 本方案期望 | 通过线 |
|------|------|------|------|
| `v_target_stability` (跨 epoch 同一 content 的 target 余弦方差) | <0.3 (跳变) | >0.85 (稳定) | >0.7 |
| `clip_style` 训练曲线上限 | 0.70 (坍缩) | 0.72+ | >0.71 |
| `LPIPS` 训练曲线 | 0.31-0.36 | < 0.40 | < 0.40 |
| 含 PureLatent 老 tokenizer (消融控制) | 同样卡 0.70 | 同样卡 0.70 | — |

最后一行的"含老 tokenizer"是关键: **此 OT 修复必须配合 tokenizer 修复才能见效**.
若 OT 改对了但 tokenizer 不动, style 应该有 0.02 左右的提升; 真正的突破要等 tokenizer.md.

---

## 8. 失败回退路径

如果方案 B 在 8 epoch 内 style 仍 <0.71:
1. 检查 `pairing_dinov2_top20_k8.json` 是否真的稳定 (即同一 content 跨 epoch 抽到的 target 是否在 8 张里).
2. 若稳定但 style 不涨 — 问题在 tokenizer 没有 capacity 接住新信号, 转 tokenizer.md 的"训练 capacity 提升"分支.
3. 若 style 涨到 0.72 以上但 LPIPS >0.45 — 数学 C3 (vertical FM) 没生效, 检查 `bridge_path_mode=vertical` 是否真的进 forward.

如果完全卡住, 升级到方案 C (GW 离线精排) 或方案 D (合成强配对) — 这是路线图上的后手.
