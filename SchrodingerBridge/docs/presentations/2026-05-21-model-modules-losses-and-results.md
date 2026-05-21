# LANCET / LBM 当前模型、模块、Loss 与实验结果完整说明

日期：2026-05-21  
范围：整理当前仓库中已经确定保留的模型架构、模块、Loss、评价指标、实验结果与复现入口。  
定位：这不是论文正文，也不是花哨 PPT，而是一份给项目成员、新接手者或组会汇报前阅读的技术说明文档。

## 0. 摘要

当前项目的主线已经比较清楚：

1. 早期 LBM 主线，即 **OT-coupled latent flow matching**，已经被实验确认为稳定基础。
2. 论文结果显示，在 strict-750 协议下，Ours 与 SaMST 的 CLIP-style 接近，但 LPIPS、EC 和 artifact-sensitive 指标更好，并且训练成本低很多。
3. 1024 本地协议下，Ours 明显优于本地 SaMST epoch-50 CSV，是当前主表中最亮的结果之一。
4. 256 分辨率下，原始 OMF/LBM 虽然结构稳定，但风格强度偏保守。
5. 5 通道 diffeomorphic stroke 证明了“颜色残差 + 坐标形变”比单纯颜色 residual 更有风格表达力。
6. 自由坐标形变会破坏结构；texture-tangent warp 通过内容梯度的切向投影显著修复结构，是当前 256 路线最重要的新模块。
7. physical loss tree 已完成负结果记录；下一轮回到切向形变和 5 通道输出代数本身，探索更精巧的 stroke 参数化。

一句话版本：

> LANCET 现在的核心不是“再堆一个风格模块”，而是把风格迁移写成 latent 流场里的受约束物理过程：OT 提供端点，flow matching 学运输，terminal SWD 拉风格分布，kinetic 守路径，diffeomorphic stroke 提供笔触形变，texture-tangent projection 守住结构边界。

## 1. 当前项目的问题定义

我们面对的是 domain-level 多风格迁移，不是 arbitrary single-reference style transfer。模型输入是内容图像和目标风格 domain id，输出对应风格化图像。

核心约束是三角关系：

| 目标 | 指标/现象 | 风险 |
|---|---|---|
| 风格强度 | CLIP-style、Gram、CMMD、视觉笔触 | 过强会产生结构漂移或脏色 |
| 内容保持 | LPIPS、CLIP-content、DINO-SSM | 过强会退化为颜色滤镜 |
| 效率 | 参数量、训练时间、推理时间 | diffusion 或重 attention 方法成本高 |

项目目前的策略是：不要用重模型硬堆风格，而是在 compact latent space 里设计更好的运输动力学。

## 2. 主线模型：Latent Bridge Matching / LANCET

### 2.1 总体数据流

模型工作在 VAE latent 空间。令内容 latent 为：

```text
z0 in R^{C x H x W}
```

目标风格 domain id 为：

```text
s
```

训练时从目标风格域采样一批 target-style latents，通过 OT coupling 给每个 content latent 找一个 matched endpoint：

```text
z1_tilde ~ pi*(target | content)
```

模型学习一个时间条件速度场：

```text
v_theta(z_t, t, s)
```

推理时通过 Euler integration 得到 stylized latent，再用 VAE decoder 解码。

### 2.2 主线 objective

当前主线 objective 可以理解为：

```text
L = L_flow + lambda_term * L_terminal_swd + lambda_kin * L_kinetic + optional_losses
```

其中：

- `L_flow`：学习从 content 到 OT matched endpoint 的路径。
- `L_terminal_swd`：约束积分终点靠近目标风格分布。
- `L_kinetic`：约束速度场能量，避免为了风格大幅破坏结构。

主线配置中，历史启发式 Loss 如 PatchNCE、cycle、repulsive、强 color transport 不再作为主项。

### 2.3 代码入口

| 功能 | 文件 |
|---|---|
| 配置 schema | `src/config_schema.py` |
| 模型构建 | `src/model.py` |
| LANCET backbone | `src/lancet_backbone.py` |
| runtime / integrate | `src/lancet_runtime.py` |
| objective / loss | `src/losses.py` |
| SWD / OT cost | `src/ot_cost.py` |
| diffeomorphic stroke | `src/utils/diffeomorphic.py` |
| modern metrics | `src/utils/modern_metrics.py` |

## 3. 已确认保留的架构模块

### 3.1 OT-coupled endpoint construction

状态：确定保留。

作用：

- 在无配对数据中构造合理 target endpoint。
- 避免直接用随机 style sample 当作监督造成过强噪声。
- 保持 LBM 的“latent transport”叙事。

风险：

- mini-batch OT 只是一种局部近似，不应在论文中宣称全局最优传输。
- cost oracle 的尺度敏感性曾经造成低分辨率下风格测度偏差。

### 3.2 Terminal SWD

状态：确定保留。

作用：

- 是当前最明确的风格驱动力。
- 直接对 integrated endpoint 做分布匹配。

实验依据：

| Variant | CLIP-S | CLIP-C | LPIPS | 解释 |
|---|---:|---:|---:|---|
| D0 full | 0.7014 | 0.8022 | 0.4593 | 基础稳定 |
| D1 w/o terminal SWD | 0.6708 | 0.8989 | 0.3490 | 内容更强但风格明显不足 |

结论：

> terminal SWD 不是装饰项，而是风格写入的主力。

### 3.3 Kinetic regularization

状态：确定保留。

作用：

- 控制路径长度和速度能量。
- 避免模型通过大幅 latent 位移硬追风格。

实验依据：

| Variant | CLIP-S | CLIP-C | LPIPS | 解释 |
|---|---:|---:|---:|---|
| D0 full | 0.7014 | 0.8022 | 0.4593 | 基础稳定 |
| D2 w/o kinetic | 0.7159 | 0.6624 | 0.6375 | 风格上升但结构崩坏 |

结论：

> kinetic 是当前 style-content frontier 的主要结构守卫。去掉它会让 style 看起来接近 SaMST，但不是可接受解。

### 3.4 Semantic routing and spatial style prior

状态：确定保留。

作用：

- 用 style id 和 learnable spatial style prior 提供 domain-level conditioning。
- 避免每张图都做 reference-specific optimization。
- 保持推理速度和参数量优势。

当前判断：

- 这个模块已经足够作为轻量风格写入通道。
- 下一步主要不是继续堆 attention，而是改善 loss 和输出代数结构。

## 4. 5 通道 Diffeomorphic Stroke

### 4.1 为什么需要它

原始模型主要输出 latent residual：

```text
endpoint = x + delta
```

这种形式很容易变成颜色滤镜：

- 结构稳定；
- 但风格像“贴颜色”；
- 缺少真实笔触中的空间拉扯、边缘挤压、局部形变。

因此引入 5 通道 stroke：

```text
raw_out = [color_delta, warp_x, warp_y]
```

组装：

```text
spatial_warp = tanh(raw_out[:, C:C+2]) * warp_strength
x_warped = grid_sample(x, grid + spatial_warp)
endpoint = x_warped + color_delta
```

### 4.2 物理解释

| 输出部分 | 含义 |
|---|---|
| `color_delta` | 颜料/latent 颜色残差 |
| `warp_x, warp_y` | 局部坐标形变场 |
| `grid_sample` | 连续可微的图像/latent 重采样 |

这让模型第一次具备“移动局部空间”的能力。

### 4.3 实验结果

| 组别 | CLIP-S | LPIPS | CLIP-C | 结论 |
|---|---:|---:|---:|---|
| 256 OMF 基线 | 0.7167 | 0.4615 | 0.7977 | 结构强，风格偏保守 |
| 自由 5 通道 stroke | 0.7265 | 0.6318 | 0.6471 | 风格上升，结构崩坏 |

结论：

> 5 通道 stroke 确实提高风格表达能力，但自由 warp 不是最终答案。

## 5. Texture-Tangent Warp

### 5.1 问题诊断

自由 warp 的主要问题不是颜色，而是跨边界扩散：

- 五官、轮廓、背景容易被互相拉扯；
- 局部边缘会发生不受控相位偏移；
- LPIPS 和 CLIP-content 都明显变差。

### 5.2 机制

用内容图的梯度场构造法向和切向：

```text
n = normalize(grad(content))
tau = (-n_y, n_x)
```

将 warp 投影到切向：

```text
warp_effective = proj_tau(warp) + normal_leak * proj_n(warp)
```

直觉：

- 法向移动穿过边界，破坏拓扑。
- 切向移动沿着边界/纹理刷动，更像笔触。
- `normal_leak` 是小阀门，允许非常少的法向移动补充风格。

### 5.3 实验结果

| 组别 | CLIP-S | LPIPS | CLIP-C | DINO-SSM | 结论 |
|---|---:|---:|---:|---:|---|
| 自由 5 通道 stroke | 0.7265 | 0.6318 | 0.6471 | -- | 风格强但结构崩 |
| 单点 tangent stroke | 0.7287 | 0.5526 | 0.7120 | 0.0335 | 256 style 最高，但仍偏激进 |
| `t01` balanced | 0.7264 | 0.5170 | 0.7570 | 0.0263 | 最好 style 平衡点 |
| `t00` structure-best | 0.7259 | 0.5166 | 0.7602 | 0.0259 | 当前稳态基线 |

结论：

> texture-tangent warp 是已确权模块。它显著降低结构损伤，同时保留大部分 5 通道 stroke 的风格收益。

## 6. 当前 256 与 SaMST 的关系

SaMST 参照来自：

```text
clip_lpips_eval_epoch_50.csv
```

| 方法 | 统计口径 | CLIP-S | LPIPS |
|---|---|---:|---:|
| SaMST | 25 组全表平均 | 0.6574 | 0.6780 |
| SaMST | 对角线平均 | 0.6735 | 0.6672 |
| SaMST | Cubism 单点峰值 | 0.8473 | 0.3996 |
| Ours 256 tangent `t01` | 当前平衡点 | 0.7264 | 0.5170 |
| Ours 256 tangent `t00` | 当前稳态点 | 0.7259 | 0.5166 |

需要谨慎解释：

- 平均口径下，我们的 256 tangent 分支显著强于 SaMST 平均。
- SaMST 的 Cubism 单点峰值非常高，但不能代表整体平均能力。
- 论文中不应只用单点峰值做主张，应强调协议、公平性、artifact-sensitive 指标和视觉质量。

## 7. 论文主表结果

主表来自：

```text
aaai_submission/paper_aaai2026.tex
```

| Method | Params | CLIP-S ↑ | LPIPS ↓ | EC ↑ | Infer (s) | ms/img | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours ep7 | 3.9M | 0.716 | 0.451 | 0.393 | 40.0 | 53 | 309.9 |
| SaMST | 6.0M | 0.719 | 0.466 | 0.384 | 39.8 | 53 | 6768.7 |
| Ours 1024 | 3.9M | 0.741 | 0.426 | 0.425 | 40.8 | 54 | 141.4 |
| SaMST 1024 | 6.0M | 0.657 | 0.678 | 0.212 | -- | -- | 106800 |
| S2WAT | ~7M | 0.714 | 0.526 | 0.338 | 45.1 | 60 | 10600 |
| AdaIN v32k | ~5M | 0.713 | 0.630 | 0.264 | 9.3 | 12 | 9220.4 |
| StyleID | SD-based | 0.760 | 0.750 | 0.190 | 603.3 | 804 | free |
| CAST | 7.0M | 0.665 | 0.726 | 0.182 | 75.5 | 101 | free |

主表结论：

1. Ours 与 SaMST 在 256 strict-750 CLIP-S 上接近。
2. Ours LPIPS 和 EC 更好。
3. Ours 训练时间远低于 SaMST。
4. Ours 1024 结果显著强于 SaMST 1024。
5. StyleID 虽然 CLIP-S 高，但 LPIPS 崩坏，说明 raw CLIP-S 不能单独作为目标。

## 8. Artifact-sensitive 指标

论文中的 artifact 表：

| Metric | Ours e7 | SaMST | S2WAT | 越优方向 |
|---|---:|---:|---:|---|
| MUSIQ | 49.2059 | 36.0950 | 36.5256 | ↑ |
| MANIQA | 0.4057 | 0.3139 | 0.1754 | ↑ |
| DISTS-content | 0.2477 | 0.2943 | 0.2942 | ↓ |
| HF-Patch-KID | 4.1694 | 6.7598 | 12.6623 | ↓ |
| FFT slope error | 0.5473 | 1.0536 | 0.7017 | ↓ |
| Gram micro | 0.0798 | 0.0947 | -- | ↓ |

解释：

- SaMST 可能在标准 style/content 指标上很强，但局部纹理有泥、脏、颗粒化问题。
- Ours raw style 略温和，但 artifact-sensitive 指标更好。
- 这支撑当前论文叙事：不单卷 CLIP-S，而是讲质量-效率 Pareto。

相关图：

```text
aaai_submission/fig_quality_tradeoff.png
aaai_submission/fig_qual_grid_ours_vs_samst.png
aaai_submission/fig_zoom_ours_vs_samst.png
aaai_submission/fig_artifact_diagnostics.png
```

## 9. 评价体系升级

当前不再只看 LPIPS。

| 指标 | 用途 | 问题 |
|---|---|---|
| CLIP-style | 风格语义强度 | 可能奖励贴图式或指标作弊式风格 |
| LPIPS | 像素/感知相位差 | 会惩罚真实笔触造成的局部相位偏移 |
| CLIP-content | 语义保持 | 对局部结构不够敏感 |
| DINO-SSM | 结构拓扑保持 | 更适合评价强笔触/局部形变 |
| CMMD | CLIP 分布距离 | 判断生成分布是否接近 style domain |
| VGG Gram micro/macro | 纹理统计 | 可区分局部高频与深层风格统计 |

对 256 diffeomorphic stroke，DINO-SSM 非常重要：

- 自由 warp 的 LPIPS 很差，视觉也确实结构崩。
- tangent sweep 中 DINO-SSM 从 `0.0335` 降到 `0.0259`，说明结构拓扑明显修复。

## 10. 已剥离或不作为主线的 Loss

以下 Loss 或机制不作为主线：

- PatchNCE
- cycle loss
- repulsive loss
- 强 local color loss
- naive color transport
- 大量历史低频启发式损失

原因：

1. 很多已经长期置零。
2. 强颜色损失会带来 palette drift，而不是可控风格。
3. 它们和当前“latent flow + diffeomorphic tangent stroke”的主线不够一致。

## 11. Physical Loss 探索结果

状态：已完成一轮自动决策树探索，并从主线代码中回退。

结果文件保留在：

```text
exp/physical_loss_tree/physical_loss_tree_frontier.csv
exp/physical_loss_tree/physical_loss_tree_ledger.jsonl
```

这轮探索覆盖了四类 loss：

- `impasto divergence`
- `gradient-anchored style energy`
- `curl style-field`
- `latent self-similarity content`

核心结论：**没有任何一组超过 `t00/t01` tangent 基线，physical loss family 暂不进入主线。**

### 11.1 与 tangent 基线对比

| 组别 | CLIP-S | LPIPS | CLIP-C | DINO-SSM | 判断 |
|---|---:|---:|---:|---:|---|
| `t00` tangent baseline | 0.72591 | 0.51660 | 0.76021 | 0.02590 | 当前稳态主线 |
| `t01` tangent balanced | 0.72636 | 0.51697 | 0.75699 | 0.02631 | 当前 style 平衡点 |
| `p1_03_t00_grad_high` | 0.72449 | 0.52326 | 0.74537 | 0.02817 | physical loss 最佳综合分，但全面弱于 `t00/t01` |
| `p1_00_t00_imp_low` | 0.72309 | 0.52819 | 0.74397 | 0.02857 | 厚涂假设未兑现 |
| `p2_01_t00_imp_low__curl_low` | 0.72560 | 0.54643 | 0.71513 | 0.03128 | style 接近但结构明显崩坏 |

### 11.2 负结果解释

1. `gradient_style` 是最不坏的一支，但仍然降低 `clip_content`、提高 DINO-SSM。
2. `impasto` 没有产生稳定厚涂收益，更像把压缩区域变成额外结构噪声。
3. `curl` 当前版本基本否决：它没有显著提高风格，却明显拉坏 LPIPS、CLIP-content 和 DINO-SSM。
4. `self_similarity` 的 latent surrogate 没能替代 DINO-SSM 的结构约束。
5. Phase 3 释放更强 warp 已无必要，因为 Phase 1/2 已经显示 loss 本身没有带来结构余量。

阶段结论：

> 下一步不继续 physical loss tree，而是回到 texture-tangent warp 与 5 通道输出代数本身，做更精细的形变参数化。

## 12. 下一轮：更精巧的 5 通道切向设计

当前最值得推进的不是增加外部 loss，而是改善 5 通道 stroke head 的代数约束。

候选方向：

1. **切向标量化 warp**：网络不直接输出 `(warp_x, warp_y)`，而是输出 `alpha_tangent` 和可选小 `alpha_normal`，再由内容结构张量生成方向场。这样从输出空间上禁止跨边界自由向量。
2. **幅度-方向解耦**：输出 `color_delta + tangent_magnitude + texture_gate_delta + normal_leak_delta`，方向完全来自内容梯度，模型只学“刷多大”和“哪里刷”。
3. **低梯度区域自由场，高梯度边界切向场**：平坦区域允许轻度二维自由 warp，边界区域强制切向投影，避免背景太死。
4. **多尺度 tangent warp**：低频 latent 做轻微形变，高频 residual 只沿切向注入，减少局部撕裂。
5. **Jacobian/area diagnostic 只做指标不做 loss**：先记录形变压缩/拉伸分布，避免再次把不成熟的物理直觉硬塞进 objective。

已经验证并回退的最小推进：

```text
model.diffeomorphic_warp_mode = projected_xy | scalar_tangent
```

- `projected_xy`：旧行为，网络输出二维 warp，再投影到内容切向场。
- `scalar_tangent`：临时行为，网络输出切向幅度和法向泄露幅度，方向由内容梯度场给定。
- 结论：未超过旧 `t00/t01`，相关代码和脚本已清理。
- 结果文件：`exp/diffeomorphic_tangent_head_sweep/tangent_head_frontier.csv`

## 13. 可复现入口

### 13.1 256 tangent sweep

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge
python tools\experiments\run_diffeomorphic_tangent_sweep.py
```

结果：

```text
exp/diffeomorphic_tangent_sweep/tangent_grid_frontier.csv
exp/diffeomorphic_tangent_sweep/tangent_grid_ledger.jsonl
```

### 13.2 Physical loss negative ablation

这轮脚本已从主线回退，只保留结果文件作为 negative ablation：

```text
exp/physical_loss_tree/physical_loss_tree_frontier.csv
exp/physical_loss_tree/physical_loss_tree_ledger.jsonl
```

### 13.3 论文主表和图

```text
aaai_submission/paper_aaai2026.tex
aaai_submission/paper_aaai2026.pdf
aaai_submission/fig_quality_tradeoff.png
aaai_submission/fig_qual_grid_ours_vs_samst.png
aaai_submission/fig_zoom_ours_vs_samst.png
aaai_submission/fig_ablation_pareto.png
aaai_submission/fig_train_efficiency_pareto.png
aaai_submission/fig_weight_sweep_summary.png
```

## 14. 建议汇报结构

如果拿这份材料讲组会，建议顺序如下：

1. 任务约束：style / content / cost 三角问题。
2. LBM 主线：OT endpoint + flow matching + terminal SWD + kinetic。
3. 主表结果：Ours vs SaMST/S2WAT/StyleID，强调效率和 LPIPS/EC。
4. Artifact-sensitive 分析：为什么 SaMST headline 强但视觉局部脏。
5. 256 新路线：5 通道 stroke，为什么 style 上升。
6. 结构崩坏诊断：自由 warp 的失败。
7. Texture-tangent warp：为什么切向约束修复结构。
8. 当前 256 Pareto：`t00/t01`。
9. 负结果：physical loss tree 为什么不进入主线。
10. 下一轮：更精巧的 5 通道切向参数化。

## 15. 后续待补

1. 设计并实现切向标量化 / 幅度-方向解耦的 5 通道 head。
2. 增加 warp Jacobian、切向/法向能量比例、低梯度自由场占比等 diagnostic。
3. 与 `t00/t01` 做同源图像 qualitative grid。
4. 若新 5 通道设计有效，再更新论文 ablation 或 appendix。
