# 256 分辨率 Diffeomorphic Stroke / Tangent Warp 实验进展

日期：2026-05-20  
范围：只讨论 256x256 训练与评估，不混入 1024x1024 结果。  
主线目标：在 256 分辨率下继续提高 `clip_style`，同时把结构破坏控制住。这里的结构不只看 LPIPS，也重点看 `clip_content` 和新增的 DINO-SSM 结构距离。

## 1. 背景与问题

前一轮 256 实验已经说明：纯粹依靠端点 SWD、kinetic、FFT amplitude 等损失，模型可以稳定保持结构，但风格化上限较明显。视觉上更像“颜色滤镜 + 局部纹理”，不是有物理笔触的风格迁移。

因此我们转向 5 通道 diffeomorphic stroke：

- 前 `C` 通道仍然表示颜色/latent 残差。
- 额外 2 通道表示空间 warp。
- 推理/训练时用 `grid_sample` 对当前 latent 做连续重采样，再叠加颜色残差。

这个机制的直觉是：真正的笔触不是只改变颜色，它会造成局部空间拉扯、边缘挤压、方向性纹理。只做颜色映射很难得到强风格；允许可微坐标场后，`clip_style` 明显抬升。

但自由 warp 也引入了新问题：它会跨越物体边界扩散，把局部结构拉散。用户观察生成图后指出“内容物体/结构崩坏”，这和指标一致：style 上来了，但 LPIPS 和 CLIP content 明显变差。

## 2. 代码机制位置

当前 diffeomorphic stroke 主体已经做成开关，相关位置：

- 配置开关：`src/config_schema.py`
  - `use_diffeomorphic_stroke`
  - `diffeomorphic_color_strength`
  - `diffeomorphic_warp_strength`
  - `diffeomorphic_texture_gate_strength`
  - `diffeomorphic_normal_leak`
- 输出通道：`src/lancet_backbone.py`
  - 开启后 decoder 输出 `latent_channels + 2`，额外 2 通道作为 warp。
- 组装位置：
  - `src/model.py`
  - `src/lancet_runtime.py`
- 工具实现：`src/utils/diffeomorphic.py`
  - `apply_diffeomorphic_stroke`
  - `apply_texture_aligned_diffeomorphic_stroke`
  - `_texture_tangent_warp`

关键改进是 `_texture_tangent_warp`：用内容图的局部梯度构造切向方向，只允许 warp 主要沿纹理/边缘方向移动，尽量禁止跨边界法向扩散。

直觉上：

- 边缘法向移动会穿过物体边界，容易造成器官、轮廓、背景互相污染。
- 边缘切向移动更像沿着物体表面刷笔触，能制造纹理而不直接破坏拓扑。
- `normal_leak` 是一个很小的法向泄露阀门，用来测试完全禁止法向移动是否太死。

## 3. 当前对照结果

下面只列 256 相关结果。

| 组别 | 结果文件 | `clip_style` ↑ | LPIPS ↓ | `clip_content` ↑ | DINO-SSM ↓ | 结论 |
|---|---|---:|---:|---:|---:|---|
| 256 基线 OMF | `S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json` | 0.7167 | 0.4615 | 0.7977 | 未记录 | 结构保留强，但风格上限明显，视觉上偏保守。 |
| 自由 5 通道 stroke | `exp/diffeomorphic_stroke_local/full_eval/epoch_0008/summary.json` | 0.7265 | 0.6318 | 0.6471 | 未记录 | 风格明显增强，但结构崩坏严重。 |
| 单点 tangent stroke | `exp/diffeomorphic_stroke_tangent_local/full_eval/epoch_0008/summary.json` | 0.7287 | 0.5526 | 0.7120 | 0.0335 | 当前 256 style 最强点；切向约束比自由 warp 更稳，但结构还不够。 |
| 12 组 tangent sweep 最平衡 | `exp/diffeomorphic_tangent_sweep/tangent_grid_frontier.csv` 的 `t01_ws0p03_g6_nl0p05` | 0.7264 | 0.5170 | 0.7570 | 0.0263 | 风格保留大部分，结构明显修复。 |
| 12 组 tangent sweep 结构最好 | `exp/diffeomorphic_tangent_sweep/tangent_grid_frontier.csv` 的 `t00_ws0p03_g6_nl0` | 0.7259 | 0.5166 | 0.7602 | 0.0259 | 目前最适合作为下一轮稳态基线。 |

### 与 SaMST 的 256 指标参照

SaMST 指标来自 `clip_lpips_eval_epoch_50.csv`。需要注意，SaMST 的数据集/风格标签体系与我们的 5 类 overfit50 评估不完全同构，所以这里只作为 256 量级参照，不直接等价成主表胜负。

| 方法 | 统计口径 | `clip_style` ↑ | LPIPS ↓ | 备注 |
|---|---|---:|---:|---|
| SaMST | 25 组全表平均 | 0.6574 | 0.6780 | 平均风格和结构都不如当前 256 tangent 分支。 |
| SaMST | 对角线平均 | 0.6735 | 0.6672 | 同类风格测试仍低于我们的 256 tangent 分支。 |
| SaMST | Cubism 单点峰值 | 0.8473 | 0.3996 | 单点非常高，属于它的优势/异常峰值；不能用均值路线解释。 |
| Ours 256 tangent balanced | `t01` | 0.7264 | 0.5170 | 平均风格强于 SaMST 平均，LPIPS 也更低。 |
| Ours 256 tangent structure-best | `t00` | 0.7259 | 0.5166 | 当前推荐稳定基线。 |

阶段性判断：在 256 平均口径上，我们已经明显超过 SaMST 的平均表现；但 SaMST 的 Cubism 单点峰值仍是一个“极端风格注入”参照，说明如果论文要讲强风格峰值，还需要解释我们选择的是更平衡、更物理的连续流，而不是单风格指标作弊。

## 4. 12 组 tangent sweep 结果

实验脚本：`tools/experiments/run_diffeomorphic_tangent_sweep.py`  
结果汇总：`exp/diffeomorphic_tangent_sweep/tangent_grid_frontier.csv`  
完整 ledger：`exp/diffeomorphic_tangent_sweep/tangent_grid_ledger.jsonl`

固定项：

- 训练 8 epoch。
- 每组在 epoch 4/6/8 做 full eval，并从中选当前最优。
- `diffeomorphic_color_strength = 0.85`

扫描维度：

- `warp_strength`: 0.03 / 0.05 / 0.07
- `texture_gate_strength`: 6 / 8
- `normal_leak`: 0 / 0.05

按 `clip_style` 排序的前几名：

| Rank | 组别 | best epoch | `clip_style` ↑ | LPIPS ↓ | `clip_content` ↑ | DINO-SSM ↓ | 参数 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `t01_ws0p03_g6_nl0p05` | 8 | 0.7264 | 0.5170 | 0.7570 | 0.0263 | warp 0.03, gate 6, leak 0.05 |
| 2 | `t04_ws0p05_g6_nl0` | 8 | 0.7261 | 0.5170 | 0.7587 | 0.0259 | warp 0.05, gate 6, leak 0 |
| 3 | `t03_ws0p03_g8_nl0p05` | 8 | 0.7260 | 0.5165 | 0.7575 | 0.0262 | warp 0.03, gate 8, leak 0.05 |
| 4 | `t00_ws0p03_g6_nl0` | 8 | 0.7259 | 0.5166 | 0.7602 | 0.0259 | warp 0.03, gate 6, leak 0 |
| 5 | `t02_ws0p03_g8_nl0` | 8 | 0.7257 | 0.5171 | 0.7594 | 0.0259 | warp 0.03, gate 8, leak 0 |

最差趋势集中在 `warp_strength = 0.07`：

- `clip_style` 没有继续提高，反而降到约 0.7228-0.7247。
- LPIPS 上升到约 0.521-0.528。
- `clip_content` 降到约 0.740-0.749。
- DINO-SSM 也变差到约 0.027-0.028。

这说明当前模型不是“warp 越大越风格化”。超过 0.05 之后，额外空间形变主要变成结构噪声，没有转化为有效风格。

## 5. 阶段性结论

### 5.1 5 通道 stroke 是值得留下的机制

从 256 基线到自由 stroke：

- `clip_style`: 0.7167 -> 0.7265，提升约 +0.0097。
- 说明空间重采样确实给了模型新的风格表达能力。

这不是单纯调参能解释的；它改变了模型输出的代数结构：从“只加颜色残差”变成“坐标形变 + 颜色残差”。

### 5.2 自由 warp 的主要问题是跨边界扩散

自由 stroke 的 LPIPS 达到 0.6318，`clip_content` 只有 0.6471。结合图像观察，结构崩坏不是简单的颜色偏差，而是局部物体边界和纹理方向被错位拉扯。

这解释了为什么后续不应该继续粗暴提高 warp，而应该限制 warp 的方向和作用区域。

### 5.3 切向约束是正确方向

单点 tangent stroke：

- `clip_style`: 0.7287，是目前 256 最高。
- LPIPS 从自由 stroke 的 0.6318 降到 0.5526。
- `clip_content` 从 0.6471 升到 0.7120。

12 组 sweep 进一步把结构打下来：

- 最好 DINO-SSM 从 0.0335 降到 0.0259。
- LPIPS 从 0.5526 降到 0.5166。
- `clip_content` 从 0.7120 升到 0.7602。
- 风格仍保持在 0.7259-0.7264。

换句话说，切向约束没有完全牺牲风格，却显著修复结构。这是当前最有价值的工程/理论交汇点。

### 5.4 当前最佳点不是 style 最高点，而是 Pareto 点

如果只看 `clip_style`，旧的 `diffeomorphic_stroke_tangent_local` 仍最高：0.7287。  
但它的 LPIPS 和 DINO 结构更差。

如果看平衡：

- `t00_ws0p03_g6_nl0` 是结构最好点。
- `t01_ws0p03_g6_nl0p05` 是 style 稍好点。
- `t04_ws0p05_g6_nl0` 说明 warp 0.05 在无 leak 时仍可用。

推荐把 `t00` 作为下一轮默认基线，把 `t01` 作为“更激进一点”的对照。

## 6. 参数规律

### `warp_strength`

当前最有用区间：0.03-0.05。

- 0.03：结构最稳，style 已经接近最高。
- 0.05：个别组能持平 style，但结构没有明显胜出。
- 0.07：基本确认过强，结构变差且 style 没有收益。

下一轮不应该继续扫 0.07 以上。更合理的是在 0.025/0.035/0.045 附近细扫。

### `texture_gate_strength`

gate 6 与 gate 8 差异很小。

- gate 6 的 `t00/t01/t04` 表现略稳。
- gate 8 没有带来明确 style 收益。

下一轮可以保留 5/6/7，而不是继续扩大到 8 以上。

### `normal_leak`

`normal_leak = 0.05` 有时会微弱提高 style，例如 `t01` 是当前 sweep style 第一；但它也可能伤结构，例如高 warp 下的 `t09/t11`。

结论：低 warp 下可以保留 0.03-0.05 的小 leak 作为风格阀门；中高 warp 下应优先设为 0。

### `color_strength`

本轮固定为 0.85，没有消融。当前结构修复主要来自 warp 约束，不是颜色强度变化。下一轮可测试 0.80/0.85/0.90，目标是看看能否用颜色残差承担更多风格，把空间 warp 再降一点。

## 7. 评价指标解释

当前建议的 256 指标优先级：

1. `clip_style`：主目标，衡量风格接近度。
2. `clip_content`：必须保留，高于 0.75 说明语义仍基本稳定。
3. DINO-SSM：新的结构主指标，越低越好；它比 LPIPS 更能容忍合理笔触变形。
4. LPIPS：仍记录，但不作为唯一结构结论。因为 LPIPS 会惩罚真实笔触造成的局部相位偏移。
5. `cmmd`、VGG Gram：辅助判断生成分布与纹理统计。

这次 sweep 中，DINO-SSM 的改善尤其关键：它说明结构修复不是简单“图像变得更像原图颜色滤镜”，而是在允许笔触形变的情况下，语义拓扑更稳了。

## 8. Physical Loss Tree 负结果补记

日期：2026-05-21。

在 tangent sweep 之后，我们做了一轮更激进的 loss 原理探索，脚本和配置曾临时实现为 physical loss decision tree。探索对象包括：

- `impasto divergence`：希望把 warp 压缩区转化为厚涂/高频能量。
- `gradient-anchored style energy`：希望在低梯度区域增强风格，在边界区域收手。
- `curl style-field`：希望让 warp 学到旋涡/流线式笔法。
- `latent self-similarity content`：希望用轻量 surrogate 守住结构拓扑。

结果保留在：

```text
exp/physical_loss_tree/physical_loss_tree_frontier.csv
exp/physical_loss_tree/physical_loss_tree_ledger.jsonl
```

关键对照如下：

| 组别 | `clip_style` ↑ | LPIPS ↓ | `clip_content` ↑ | DINO-SSM ↓ | 判断 |
|---|---:|---:|---:|---:|---|
| `t00` tangent baseline | 0.7259 | 0.5166 | 0.7602 | 0.0259 | 当前稳态基线。 |
| `t01` tangent balanced | 0.7264 | 0.5170 | 0.7570 | 0.0263 | 当前 style 平衡点。 |
| `p1_03_t00_grad_high` | 0.7245 | 0.5233 | 0.7454 | 0.0282 | physical loss 最佳综合分，但全面弱于 `t00/t01`。 |
| `p1_00_t00_imp_low` | 0.7231 | 0.5282 | 0.7440 | 0.0286 | 厚涂假设未兑现。 |
| `p2_01_t00_imp_low__curl_low` | 0.7256 | 0.5464 | 0.7151 | 0.0313 | style 接近，但结构明显崩坏。 |

结论：

- 这轮 physical loss 没有带来 Pareto 改进。
- `gradient_style` 是最不坏的一支，但仍然损失结构。
- `curl` 和 `impasto` 更像把形变变成额外结构噪声，而不是有效笔法或厚涂。
- `latent self-similarity` 没能替代 DINO-SSM 的结构判断。

因此这轮 physical loss family 已从主线代码回退，只作为 negative ablation 记录。下一步不再沿着外加 loss 做 Phase 3，而是回到 **5 通道 stroke head 的参数化设计**：让网络更少输出自由二维向量，更多输出沿内容切向场的标量幅度、gate 和小法向泄露。

## 9. 下一步建议

下一轮不建议继续泛化 physical loss tree。更合理的是先做 5 通道切向参数化升级，再配小网格验证：

- `A0`: 当前 5 通道 baseline：`color_delta + warp_x + warp_y`，再投影到 tangent。
- `A1`: 切向标量化：`color_delta + alpha_tangent + alpha_normal`，方向由内容梯度决定。
- `A2`: 幅度-门控解耦：`color_delta + alpha_tangent + gate_delta + normal_leak_delta`。
- `A3`: 区域混合：低梯度区域允许小自由 warp，高梯度边界强制切向。
- `A4`: 多尺度：低频 latent 做轻微形变，高频 residual 只沿切向注入。

`A1` 已做最小实现并验证，随后从主线代码回退：

- 临时配置：`model.diffeomorphic_warp_mode`
- 默认值：`projected_xy`，保持旧行为不变。
- 临时模式：`scalar_tangent`
  - 第 4 通道解释为切向幅度 `alpha_tangent`。
  - 第 5 通道解释为法向泄露幅度 `alpha_normal`。
  - 切向/法向方向由内容梯度场决定，不再让网络自由输出二维向量后再投影。
- 临时脚本：`tools/experiments/run_diffeomorphic_tangent_head_sweep.py`
  - 输出：`exp/diffeomorphic_tangent_head_sweep/tangent_head_frontier.csv`
  - ledger：`exp/diffeomorphic_tangent_head_sweep/tangent_head_ledger.jsonl`

### A1 head sweep 结果补记

日期：2026-05-21。

8 组 `projected_xy` vs `scalar_tangent` 小扫已完成。需要注意，这个脚本中的 `h00/h05` 是同脚本内控制组，最适合用来判断 head 参数化相对差异；最终主线仍然要和旧 `t00/t01` tangent sweep 对齐比较。由于结果未超过旧基线，相关代码已清理，结果文件保留为 negative ablation 记录。

按 `clip_style` 选 best epoch 的前几组：

| 组别 | mode | best epoch | `clip_style` ↑ | LPIPS ↓ | `clip_content` ↑ | DINO-SSM ↓ | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| `h03_scalar_wide` | `scalar_tangent` | 6 | 0.7256 | 0.5443 | 0.7183 | 0.0313 | style 最高，但结构差。 |
| `h04_scalar_strong_color` | `scalar_tangent` | 8 | 0.7255 | 0.5317 | 0.7374 | 0.0290 | 本轮最平衡点。 |
| `h06_scalar_t01_leak` | `scalar_tangent` | 6 | 0.7255 | 0.5449 | 0.7169 | 0.0310 | 小 leak 回收 style，但伤结构。 |
| `h05_projected_t01` | `projected_xy` | 6 | 0.7252 | 0.5442 | 0.7173 | 0.0310 | 同脚本 projected 控制组。 |
| `h00_projected_t00` | `projected_xy` | 6 | 0.7248 | 0.5457 | 0.7157 | 0.0311 | 同脚本 t00 控制组。 |

如果改按结构更好的 epoch 8 看，`h04_scalar_strong_color` 仍是本轮最值得看的点：

| 组别 | epoch | `clip_style` ↑ | LPIPS ↓ | `clip_content` ↑ | DINO-SSM ↓ |
|---|---:|---:|---:|---:|---:|
| `h04_scalar_strong_color` | 8 | 0.7255 | 0.5317 | 0.7374 | 0.0290 |
| `h01_scalar_t00` | 8 | 0.7244 | 0.5348 | 0.7299 | 0.0300 |
| `h00_projected_t00` | 8 | 0.7246 | 0.5347 | 0.7303 | 0.0301 |

结论：

- `scalar_tangent` 没有带来主线级 Pareto 改进。
- 它相对同脚本 projected control 有一些局部信号：`h04` 说明“更低 warp + 更强颜色 + 标量切向”比普通 scalar 更稳。
- 但与旧 `t00/t01` tangent sweep 相比，`h04` 仍然明显落后：旧 `t00` 是 `clip_style=0.7259, LPIPS=0.5166, clip_content=0.7602, DINO=0.0259`。
- 因此 `scalar_tangent` 不替代 `projected_xy` 主线；代码已回退，主线回到旧 `t00/t01` tangent 配置。

筛选规则：

- 首先淘汰 `clip_content < 0.755` 或 DINO-SSM > 0.0265 的组。
- 在剩余组里追 `clip_style`。
- 如果 `clip_style >= 0.728` 且 LPIPS <= 0.52、DINO-SSM <= 0.026，则作为 256 新主线结果。

推荐起始候选：

| 候选 | 目的 |
|---|---|
| `A2_tangent_gate_delta` | 让网络学习局部 gate 增减，而不是只靠固定 `texture_gate_strength`。 |
| `A3_mixed_flat_free_edge_tangent` | 平坦区域给小自由 warp，边界区域保持切向，解决背景太死的问题。 |
| `A4_multiscale_lowfreq_warp` | 只让低频 latent 被 warp，高频 residual 沿切向注入，减少局部撕裂。 |

## 10. 当前可复现入口

训练/评估脚本：

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge
python tools\experiments\run_diffeomorphic_tangent_sweep.py
```

远程启动相关脚本：

- `start_remote_diffeomorphic_tangent_sweep.bat`
- `run_diffeomorphic_tangent_sweep_ssh.cmd`

结果检查：

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge
Get-Content exp\diffeomorphic_tangent_sweep\tangent_grid_frontier.csv
Get-Content exp\diffeomorphic_tangent_sweep\tangent_grid_ledger.jsonl -Tail 20
```

## 11. 一句话总结

256 这条线目前最清楚的结论是：**diffeomorphic stroke 能带来真实风格增益，但必须被内容纹理的切向场约束；自由空间 warp 是风格强、结构散，切向 warp 是当前最好的 Pareto 方向。**
