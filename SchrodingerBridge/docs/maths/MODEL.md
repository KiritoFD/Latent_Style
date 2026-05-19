# 当前模型的数学建模

更新日期：`2026-05-19`。

这份文档只描述当前代码仍然真实存在、并且实验上仍在使用的建模假设。已经在代码里退休或权重恒为 0 的 loss 不再进入 active objective。

## 1. 状态空间

模型工作在 Stable Diffusion latent 空间。记：

- `z0`：输入内容 latent；
- `s`：目标风格 id；
- `z1`：模型预测的终点 latent；
- `v_theta(z, t, s)`：风格条件速度场或残差场。

训练和推理都可以写成残差流：

```text
z1 = z0 + integral_0^1 v_theta(z_t, t, s) dt
```

在当前主线配置中，模型更像一个学习到的 endpoint corrector，而不是一个依赖高精度数值积分的 ODE。已有 step size / step count sweep 基本是平的，所以提升 `clip_style` 的第一优先级不是增加推理步数，而是改变学习到的更新方向和更新幅度。

## 2. 架构分解

当前 LANCET 主干可以拆成三条有效通道。

### 2.1 内容主干

内容 latent 被 lift 到特征空间，经过高分辨率卷积块和 body 块，形成：

- body 分辨率的内容特征；
- 解码器需要的 skip 特征；
- 最终 latent 残差。

内容主干决定了结构信息如何保留下来。

### 2.2 语义风格绘制

body 中的核心风格注入是 semantic cross-attention。对内容特征 `x` 和风格空间先验 `y`：

```text
q = Wq(IN(x))
k = Wk(IN(y))
v = Wv(IN(y))
A = softmax(q k^T / (sqrt(C) tau))
P = A v
x_out = x + g_global * g_local(x) * (1 + gamma) * P
```

如果 routing mode 改为 `sinkhorn`，`A` 会被近似归一化为双随机矩阵。历史结果显示 Sinkhorn 更像内容保护器：它降低 LPIPS，但通常也压低 `clip_style`。因此它不是当前第一风格增益手段，除非某个高风格分支已经出现内容滑坡。

### 2.3 Skip 路径

skip fusion / skip routing 决定干净内容结构可以绕过 body 的程度。当前参考配置使用 `skip_fusion_mode=add_proj` 与 `skip_routing_mode=add_proj`。已有实验说明 skip 路径过强可以制造很高的 raw style，但会造成 LPIPS 崩坏，所以 skip 不应作为第一风格提高手段。

## 3. 当前有效目标函数

当前 OMF 主线目标可以近似写成：

```text
min_theta  lambda_swd * SWD_sem(z1, Z_style; K_style)
         + lambda_kin * E ||v_theta||^2
```

其中：

- `lambda_swd = terminal_swd_weight`；
- `lambda_kin = w_kinetic`；
- `SWD_sem` 是终点 SWD，当前代码允许用 semantic keys 对投影方向做语义引导；
- semantic SWD 必须作用在完整 batch 上，不能只作用在非 identity 子集上，否则目标会偏离参考实现。

在最近清理后，以下分支不属于 active objective：

- PatchNCE；
- local color / contextual color；
- low-frequency anchor；
- cycle consistency；
- repulsive anti-collapse。

这些分支要么已被实验否定，要么在参考配置中恒为 0，并且已经从当前 loss 实现中移除。

## 4. 两个主要控制量

### 4.1 `w_kinetic`

`w_kinetic` 控制运动预算：

```text
lambda_kin * E ||v_theta||^2
```

降低它通常提高风格，但过低会造成内容漂移。`D2_no_kinetic` 已经证明：去掉 kinetic 能抬高 style，但 LPIPS 会严重恶化。因此正确做法是连续降低，而不是直接关掉。

### 4.2 `terminal_swd_weight`

`terminal_swd_weight` 控制终点分布压力：

```text
lambda_swd * SWD(z1, Z_style)
```

它是不可替代项。没有 terminal SWD，`clip_style` 明显下降。增加它可能提高风格，但收益会饱和，过高可能把模型推向失真方向。

## 5. 残差幅度和推理增强

推理时可以把终点写成：

```text
z1(a) = z0 + a * Delta_theta(z0, s)
```

已有 residual scale sweep 显示：

- `a=1.25` 可以明显提高 `clip_style`；
- `a=1.5` 开始更明显牺牲内容；
- `a=2.0` 已经越过可用区域。

这说明模型不是没有风格容量，而是 baseline 附近的交付幅度偏保守。推理残差 sweep 是低成本验证手段，但如果训练目标本身没变，它通常只是沿着同一条 style-content frontier 移动。

## 6. 当前经验锚点

参考目录：

```text
S-add__K-1_C-0_W-20_Col-0/full_eval/epoch_0008/summary.json
```

参考指标：

```text
clip_style_all    = 0.7167235834
content_lpips_all = 0.4615265376
clip_content_all  = 0.7977139172
```

最近 batch=64 复现分支：

```text
exp/refactor_clean_batch64_e10_fix/full_eval/epoch_0008/summary.json
clip_style_all    = 0.7128111729
content_lpips_all = 0.4613536712
```

结论：速度已经满足目标，LPIPS 也接近参考；当前主要短板是 `clip_style` 仍低约 `0.004`。所以下一轮实验应优先提高有效风格压力，同时用 LPIPS 阈值防止内容滑坡。

## 7. 提升 `clip_style` 的建模假设

当前最可信的风格公式是：

```text
style_gain ~= endpoint_pressure
           * semantic_style_activation
           * delivered_residual_amplitude
```

内容保持则近似由：

```text
content_preservation ~= kinetic_pressure
                     * skip_retention
                     * routing_smoothness
```

因此下一轮实验的合理顺序是：

1. 保持 K1 / W20 参考谱系可复现；
2. 轻微降低 `w_kinetic`，检查 style 是否上升；
3. 若 LPIPS 仍稳，再提高 `terminal_swd_weight`；
4. 若 style 高但 LPIPS 滑坡，再用 Sinkhorn / 更保守 kinetic 修复；
5. 最后才测试 SWD projection 从 64 降到 32 的速度分支。

这个顺序比直接加新 loss 更干净，也更符合已有证据。
