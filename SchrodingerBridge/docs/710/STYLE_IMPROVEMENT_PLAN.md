# WEAVE Style 提升实验计划

日期：2026-07-10

## 1. 目标

在统一 750 图、8-step Heun、无后处理协议下，提高 `CLIP-S` 和 `DINO-S`，同时约束
`LPIPS` 与 `DINO-C`。目标不是通过增加任意纹理噪声换取单一 style 分数，而是找到同一模型上的
可解释 Pareto 改善。

主基线为 `configs/710_b0_weave_d5.json`。所有候选必须从同一个 checkpoint、同一批生成图计算
`CLIP-S / LPIPS / DINO-S / DINO-C`；同时保留 all-pairs 与 off-diagonal 分组，避免对角 identity pair
抬高 style 均值。

## 2. 当前配置诊断

当前基线已经不是“style loss 太弱”的配置：

- `single_step_swd_weight=8.0`；
- `w_endpoint_style=8.0`，而 `w_endpoint_content=1.0`；
- `spectral_w_lh=spectral_w_hl=1.0`，`spectral_w_ll=0.3`；
- endpoint alignment 已启用，`endpoint_adain_scale=1.0`；基线未显式设置
  `endpoint_adain_mode`，因此 effective mode 是 schema 默认的 `spatial_fiber`，不是 per-subband；
- `style_extrap_alpha=0.1` 同时进入训练 style embedding 与推理 style fiber；
- `style_strength_max=1.0` 且 `allow_style_overdrive=false`。

因此，继续整体放大 style loss 或 velocity 的风险很高：它更可能增加 latent 幅度和 VAE grain，而不是增加
有目标方向的风格。合理方向应优先增加**目标风格方向性**或只释放**特定频带**，而不是增加全局能量。

另一个需要先验证的契约问题是：评估 CLI 会传入 `style_strength`，但当前
`SpectralODEBridge620.integrate_transport` 通过 `**_` 接收后没有使用该值。现阶段不能把
`--style_strength` 当成有效旋钮。

## 3. 总体策略

按自由度和风险从低到高执行：

1. 先验证推理控制参数确实进入活跃路径；
2. 用现有 checkpoint 做 endpoint 分频强度诊断；
3. 只对表现出正确方向的推理点做 matched retraining；
4. 再扫描训练长度和少量 loss 权重；
5. 最后才考虑增加 HH head、FiLM 或新模块。

禁止同时修改两个机制轴。每轮只允许一个主要变量，并使用相同 seed、相同数据顺序和相同评估图集。

## 4. Phase S0：闭环基线

### S0.1 重训 B0

- 配置：`configs/710_b0_weave_d5.json`；
- seed：42；
- epochs：10；
- 推理：8-step Heun；
- 输出：750 图，无后处理；
- 验收参考：历史不可复现输出约为
  `0.7292 / 0.3239 / 0.4874 / 0.7688`，但不要求逐项完全复现。

若 B0 与历史候选偏差过大，先检查代码精简、数据 pairing、checkpoint contract 和随机性，不进入 style 扫描。

### S0.2 三 seed 方差

完成 seed 42 后，至少补 seed 43/44。后续所有“改善”必须超过基线 seed 标准差，不能以单 seed 的
`0.001--0.003` 波动作为模块结论。

## 5. Phase S1：控制契约与零训练成本诊断

### S1.1 `style_strength` 契约测试

先添加单元测试和 runtime debug：

- 在固定 latent、固定 style、固定 checkpoint 下比较 `style_strength=0, 1, 1.2`；
- 验证输出 tensor 和 endpoint debug 确实变化；
- 若该参数只是 velocity scale，必须明确命名并确保 endpoint WCT 同步受控；
- 若无法形成单调、可解释的统一强度，删除 CLI 入口，避免伪控制参数。

在该测试通过前，不运行 style-strength 大矩阵。

### S1.2 Endpoint mode 契约

先将 effective endpoint 配置写入运行 manifest。当前基线的真实路径是：

- `endpoint_adain_mode=spatial_fiber`；
- `endpoint_adain_only_last_step=false`，即每个 solver step 都应用；
- `endpoint_adain_scale=1.0`；
- `lowpass_mode=dwt_haar`；
- `style_extrap_alpha=0.1`。

因此 `endpoint_adain_scale_ll/lh/hl/hh` 在基线上并不生效。第一项实验必须是严格 matched
`spatial_fiber` 对 `per_subband` mode A/B，而不是直接宣称在现有路径上扫描分频增益。若 mode A/B 本身不能改善
canonical DINO-S，分频扫描立即停止。

### S1.3 Endpoint 分频增益

只有 `per_subband` mode A/B 晋级后，才固定 LL 为 0 并提高高频：

| ID | LL | LH | HL | HH | 目的 |
|---|---:|---:|---:|---:|---|
| S1-A0 | 0.0 | 1.0 | 1.0 | 1.0 | 原始 endpoint |
| S1-A1 | 0.0 | 1.0 | 1.0 | 1.15 | 只增强细粒度纹理 |
| S1-A2 | 0.0 | 1.05 | 1.05 | 1.15 | 轻度增强笔触与纹理 |
| S1-A3 | 0.0 | 1.10 | 1.10 | 1.25 | 风格上限探针，不直接作为主点 |

判定重点：

- `DINO-S` 是否提高，而不是只有 CLIP-S 提高；
- `DINO-C` 与 LPIPS 是否呈平滑退化，而非突然崩塌；
- 五个 target style 是否普遍改善，避免由单一纹理强 style 驱动均值；
- 若 HH-only 有收益，优先保留 endpoint 增益，不增加 HH velocity head。

### S1.4 Style extrapolation

`style_extrap_alpha` 当前为 0.1，并同时影响训练与推理。只允许将 eval-only 扫描作为敏感度诊断：

- `alpha = 0.0, 0.1, 0.2`；
- 任何候选最终必须用同一 alpha 重训，避免训练/推理分布不一致；
- 若 CLIP-S 上升但 DINO-S 不升，判定为幅度放大而非有效风格迁移；
- 若 LPIPS 快速上升或出现 VAE grain，停止更大 alpha。

## 6. Phase S2：最小训练改动

只有 S1 出现可重复的正确方向后才进入本阶段。

### S2.1 训练长度

在同一结构上比较 5/10/15 epochs。10 epochs 是主基线，15 epochs 是第一优先的 style 候选，因为它不增加
模型自由度，也不会改变理论结构。

### S2.2 Endpoint style 权重

固定其他参数，仅比较：

- `w_endpoint_style = 8, 12, 16`；
- `w_endpoint_content = 1` 保持不变；
- 优先看 off-diagonal `DINO-S` 和每风格 breakdown。

若 12 与 16 只增加 LPIPS、CLIP-S，未提高 DINO-S，则保持 8，并删除更大权重方向。

### S2.3 SWD 权重

只在 endpoint 权重无效时比较 `single_step_swd_weight = 8, 12`。不同时扫描 endpoint style weight，避免无法判断
是哪条监督产生收益。历史 semantic-region SWD 的同源结果没有证明 canonical DINO 改善，因此不作为第一优先。

### S2.4 LL 监督

比较 `spectral_w_ll = 0.3, 0.1, 0.0`。预期目标不是直接增加 style，而是释放一部分色调空间，同时依靠
endpoint LL=0 保持结构。若 DINO-C 不升且 CLIP-S/DINO-S无收益，保留 0.3。

## 7. Phase S3：最后才增加自由度

优先级如下：

1. 低维 endpoint color transform：只预测 channel mean/std 或低秩 `4x4` 颜色矩阵；
2. hard-region SWD：仅做 region 4/8 的小矩阵，并用 canonical DINO 验证；
3. HH velocity head：只做 matched false/true；
4. style-FiLM heads：最后验证。

历史记录显示 HH head 大致中性、FiLM heads 和过强 extrapolation 容易产生 artifact。若它们不能在三 seeds 上进入
四指标 Pareto 前沿，应删除对应 flag 和代码，而不是继续调参。

## 8. 晋级与停止标准

单 seed 筛选阶段，候选至少满足以下之一：

- `CLIP-S` 与 `DINO-S` 同时提高，且 `LPIPS`、`DINO-C` 退化不超过一个基线标准差；
- `DINO-S` 明显提高，`DINO-C` 基本持平；
- 在相同 style 水平下显著降低 LPIPS 或提高 DINO-C。

以下情况立即停止该方向：

- 只有 CLIP-S 上升，DINO-S 不升；
- style 增益完全由 LPIPS 增长解释；
- 对角 pair 改善、off-diagonal 不改善；
- 少数 style 提升、其余 style 系统性退化；
- 需要后处理才能得到改善。

## 9. 推荐执行顺序

1. 重训并闭合 B0；
2. 修复/验证 `style_strength` 契约；
3. 运行 `spatial_fiber`/`per_subband` endpoint mode A/B；
4. 仅对晋级的 per-subband mode 运行分频小矩阵并 matched retraining；
5. 比较 10/15 epochs；
6. 依次验证 endpoint style weight、SWD weight、LL weight；
7. 只有前述方向均失败时，进入低维色调模块和结构增量。
