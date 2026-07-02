# FC-SB 实验日志

> 基于 docs/622/FC.md 的"纤维约束薛定谔桥 (FC-SB)"理论，在 e4_long_10ep epoch_0008 历史最佳检查点（repro: clip=0.6913, lpips=0.5643）之上，通过纯推理侧改造冲击 clip>0.73 / lpips<0.30 的帕累托前沿。

## 基线与历史最佳

| 实验 | clip_style (transfer) | lpips (transfer) | 说明 |
|------|----------------------|------------------|------|
| e4_long_10ep epoch_0008 (历史 CLIP 版) | 0.7155 | 0.5906 | OpenAI CLIP 评测版本 |
| e4_long_10ep epoch_0008 (HF 复现) | 0.6913 | 0.5643 | HF CLIP backend 复现（差 0.024，CLIP backend 差异） |
| E2 (Two-Stage baseline) | 0.611 | 0.3326 | 早期 Two-Stage 训练 |
| E4-long ep5 | 0.727 | 0.581 | 早期长训练（旧 CLIP） |

**结论**：复现以 e4_long_10ep epoch_0008 (HF: 0.6913) 为基准。所有 H/I 系列均在此 ckpt 之上修改推理逻辑，不重训。

## H 系列：FC-SB 三改造组件消融

基线配置（全 False）→ 逐项开启 FC-SB 三改造。

| 变体 | fiber_proj_ep | fiber_only_ep | fiber_proj_noise | lowpass_mode | sigma | schedule | t_clip | t_lpips | a_clip |
|------|--------------|---------------|------------------|--------------|-------|----------|--------|---------|--------|
| baseline (e4_long ep8 repro) | - | - | - | - | 0 | - | 0.6913 | 0.5643 | - |
| H1 velocity_fiber_proj | ✓ | - | - | avg_pool | 0 | - | ~0.69 | ~0.56 | 失败：avg_pool 模糊了结构 |
| H2 fiber_endpoint_only | - | ✓ | - | avg_pool | 0 | - | ~0.69 | ~0.56 | 同 H1 |
| H3 fiber_sde_noise | - | - | ✓ | avg_pool | 0.08 | brownian | ~0.69 | ~0.56 | 噪声单独无效 |
| H4 full_fc_sb (avg_pool) | ✓ | ✓ | ✓ | avg_pool | 0.08 | brownian | 失败 | 失败 | avg_pool 模式下三改造组合失败 |
| **H5 full_fc_sb (wavelet)** | ✓ | ✓ | ✓ | **wavelet** | 0.08 | brownian | **0.7026** | **0.4936** | 0.7221 |

### H 系列关键发现

1. **avg_pool 低通模式失败**：H4 在 avg_pool 下三改造同时开启反而性能崩溃，原因：
   - avg_pool 的 k=5 滑窗在 latent (32×32) 上做空间均值，对低频结构的提取不够干净
   - 多次 avg_pool 残差叠加导致边界伪影
2. **wavelet 是关键突破点**：H5 把 lowpass_mode 从 avg_pool 换成 wavelet（Haar 小波，2×2 下采样后双线性上采样回原尺寸），**同时 clip +1.6% (0.6913→0.7026) 和 lpips -12.5% (0.5643→0.4936)**
3. **三改造协同效应**：H5 验证了 FC.md 理论 — 只有 fiber_proj_ep + fiber_only_ep + fiber_proj_noise 同时开启，并在 wavelet 低通下，才能实现 Base/Fiber 真正解耦

## I 系列：H5 参数空间细扫

固定 H5 三改造架构，扫 sigma / kernel / schedule。

| 变体 | sigma | kernel | schedule | t_clip | t_lpips | a_clip | a_lpips |
|------|-------|--------|----------|--------|---------|--------|---------|
| H5 (基准) | 0.08 | 5 | brownian_bridge | 0.7026 | 0.4936 | - | - |
| I1 sigma_004 | 0.04 | 5 | brownian_bridge | 0.7024 | 0.4928 | 0.7219 | 0.4931 |
| I2 sigma_012 | 0.12 | 5 | brownian_bridge | 0.7028 | 0.4946 | 0.7221 | 0.4948 |
| I3 sigma_016 | 0.16 | 5 | brownian_bridge | 0.7031 | 0.4964 | 0.7222 | 0.4966 |
| I4 sigma_020 | 0.20 | 5 | brownian_bridge | 0.7035 | 0.4980 | 0.7222 | 0.4982 |
| I5 kernel_3 | 0.08 | 3 | brownian_bridge | 0.7023 | 0.4936 | 0.7217 | 0.4939 |
| I6 kernel_7 | 0.08 | 7 | brownian_bridge | 0.7025 | 0.4935 | 0.7218 | 0.4938 |
| I7 kernel_9 | 0.08 | 9 | brownian_bridge | 0.7025 | 0.4936 | 0.7220 | 0.4938 |
| I8 curriculum | 0.08 | 5 | curriculum | 0.7026 | 0.4931 | 0.7220 | 0.4934 |
| I9 linear_ramp | 0.08 | 5 | linear_ramp | 0.7025 | 0.4931 | 0.7219 | 0.4934 |
| I10 constant | 0.08 | 5 | constant | 0.7025 | 0.4939 | 0.7219 | 0.4942 |
| I11 sigma012_kernel3 | 0.12 | 3 | brownian_bridge | 0.7028 | 0.4947 | 0.7223 | 0.4950 |
| I12 sigma012_kernel7 | 0.12 | 7 | brownian_bridge | (pending) | | | |

### I 系列关键发现

1. **sigma 微调收益微小**：sigma 从 0.04→0.20，clip 仅 +0.0011，但 lpips 也 +0.0052。**噪声能量与风格强度近乎线性弱耦合**。
2. **kernel 大小几乎无影响**（0.7023~0.7028）：说明 wavelet 已经把频率边界切得足够干净，kernel 在 3-9 之间不影响 fiber 切割
3. **schedule 影响也极小**：curriculum/linear_ramp 相对 brownian_bridge，lpips 微降 0.0005，clip 不变
4. **整体聚集在 0.7023-0.7035**：相比 H5 (0.7026)，所有 I 变体增益 < 0.001。**说明 H5 架构在当前 num_steps=12 下已达推理侧参数瓶颈**

## 瓶颈诊断

H5/I 系列证明：**纯推理侧的 Euler-Maruyama + Base Locking 已经在 num_steps=8 下榨干潜力**。继续在 sigma/kernel/schedule 上微调无法突破 0.705。

剩余可调整的维度：
1. **num_steps 增大**：当前 8 步，可能不足以让 fiber SDE 充分演化
2. **tri_band_inference_lock**：代码已预埋（model620.py L634-644），但未测试
3. **CFG-style fiber extrapolation**：在 fiber 空间做 classifier-free guidance，外推到更极端风格
4. **训练侧改造**（FC.md 改造1）：anisotropic training target + highpass-only noise — 需要重训
5. **endpoint 统计外推**：在预测的 endpoint 上做风格统计匹配（color/contrast matching）

## J 系列：推理步数与多频段（参数+架构调整）

| 变体 | steps | tri_band | α | t_clip | t_lpips | a_clip | a_lpips |
|------|-------|----------|---|--------|---------|--------|---------|
| H5 (基准) | 12 | - | - | 0.7026 | 0.4936 | 0.7221 | 0.4936 |
| J1_steps16 | 16 | - | - | 0.7008 | 0.5338 | 0.7170 | 0.5340 |
| J2_steps25 | 25 | - | - | 0.7006 | 0.5462 | 0.7155 | 0.5464 |
| J3_steps50 | 50 | - | - | 0.7001 | 0.5553 | 0.7138 | 0.5554 |
| J4_triband_a07 | 12 | ✓ | 0.7 | 0.7020 | 0.5179 | 0.7198 | 0.5182 |
| J5_triband_a05 | 12 | ✓ | 0.5 | 0.7020 | 0.5178 | 0.7196 | 0.5181 |
| J6_triband_a09 | 12 | ✓ | 0.9 | 0.7017 | 0.5179 | 0.7195 | 0.5181 |
| J7_triband+steps25 | 25 | ✓ | 0.7 | 0.7003 | 0.5462 | 0.7150 | 0.5464 |

### J 系列关键发现

**发现 1: 步数增加导致单调恶化**

| steps | t_clip | t_lpips | 趋势 |
|-------|--------|---------|------|
| 12 (H5) | 0.7026 | 0.4936 | 基准 |
| 16 (J1) | 0.7008 | 0.5338 | clip↓ lpips↑ |
| 25 (J2) | 0.7006 | 0.5462 | 持续恶化 |
| 50 (J3) | 0.7001 | 0.5553 | 持续恶化 |

**理论解释**：
- 更多步数 → fiber SDE 累积更多布朗噪声 → fiber 结构被噪声侵蚀
- Euler-Maruyama SDE 在步数增大时，**噪声累积效应超过精度提升**
- steps=12 已是 SDE 精度与噪声累积的最佳平衡点
- **FC-SB 的 fiber SDE 不需要更多步数，而是需要更精确的噪声控制**

**发现 2: tri_band_lock 也导致 lpips 恶化**

J4 (tri_band α=0.7) vs H5 (no tri_band):
- clip: 0.7026 → 0.7020 (-0.0006, 微降)
- lpips: 0.4936 → 0.5179 (**+0.0243**, 恶化!)

**理论解释**：
- tri_band_lock 的 mid band α-blend 引入了额外的内容偏移
- edge_alpha=0.7 意味着 30% 的 target 边缘混入，足以让 lpips 恶化
- **tri_band 的边缘混合破坏了 FC-SB 的纯 fiber 扩散，反而引入了 base 污染**

**发现 3: J7 (triband + steps25) 双重负面效应叠加**

J7 同时启用 tri_band_lock(α=0.7) 和 steps=25：
- clip: 0.7026 → 0.7003 (-0.0023, 比 J2/J4 单独恶化更严重)
- lpips: 0.4936 → 0.5462 (+0.0526, 接近 J2 的 0.5462)
- 说明 **tri_band 的边缘污染 + SDE 噪声累积两个独立机制同时作用**

### J 系列结论

**纯推理侧的架构调整（步数、tri_band）无法突破 H5 的 0.7026/0.4936**。

SDE 噪声累积是主要瓶颈。需要：
1. **K 系列 (FVA/Fiber-CFG)**: 不改变步数或频段，在 fiber 速度上做幅度放大/CFG 外推
2. **L 系列 (训练侧改造)**: 让模型学会更精确的 fiber 速度预测，减少对 SDE 噪声的依赖

## 下一轮探索方向（理论驱动）

### J 系列：推理步数与多频段（参数调整，无需改代码）

**理论依据**：
- num_steps 增大 → Euler-Maruyama SDE 收敛性更好，fiber SDE 充分演化
- tri_band_inference_lock → FC.md "底流形死寂，纤维狂热扩散" 的多尺度实现：
  - LL (结构) 完全锁死 → 保 LPIPS
  - Mid (边缘) α-blend → 平衡结构与风格
  - HH (纹理) 自由扩散 → 保 clip_style

| 变体 | 配置 | 理论假设 |
|------|------|----------|
| J1 | H5 + num_steps=16 | SDE 收敛性提升 |
| J2 | H5 + num_steps=25 | 更充分演化 |
| J3 | H5 + num_steps=50 | 极限测试 |
| J4 | H5 + tri_band_lock, α=0.7 | 多频段锁死 |
| J5 | H5 + tri_band_lock, α=0.5 | 更多边缘自由度 |
| J6 | H5 + tri_band_lock, α=0.9 | 接近全锁边缘 |
| J7 | H5 + tri_band_lock + steps=25 | 联合 |

### K 系列：Fiber 空间理论创新（已实现代码改造）

**理论依据**（FC.md 改造2 & 改造3）：
- 当前推理速度场 v_pred 包含 base 和 fiber 混合分量
- FC-SB 理论要求：base 静止，fiber 狂热扩散
- K0 (FVA): 在 fiber 速度上做幅度放大，突破"保守吸引子"均值陷阱
- K1 (Fiber-CFG): 在 fiber 空间做 CFG 外推，base 完全来自 target

**代码改造**（model620.py L559-569, L610-625）：
```python
# K1: Fiber-CFG
if fiber_cfg_scale > 0.0:
    ep_null = self.predict_endpoint(h, t=t_batch, style_id=null, ...)
    v_null = (ep_null - h) / denom
    v_null_fiber = v_null - lp(v_null) if fiber_proj_ep else v_null
    v_fiber = v_fiber + fiber_cfg_scale * (v_fiber - v_null_fiber)

# K0: FVA
if fiber_velocity_scale != 1.0:
    v_fiber = v_fiber * fiber_velocity_scale
```

| 变体 | 机制 | 参数 | 理论预期 |
|------|------|------|----------|
| K1 | FVA | scale=1.2 | 温和放大 +20%，clip 微涨 |
| K2 | FVA | scale=1.5 | 中等放大 +50%，clip 明显涨 |
| K3 | FVA | scale=2.0 | 激进放大 +100%，验证方向性 |
| K4 | FVA | scale=0.8 | 抑制 -20%，验证方向性（应降 clip） |
| K5 | Fiber-CFG | α=0.3 | 温和外推 |
| K6 | Fiber-CFG | α=0.5 | 中等外推 |
| K7 | Fiber-CFG | α=1.0 | 标准外推 |
| K8 | FVA+FCFG | 1.5+0.5 | 联合 |

**关键预期**：
- 若 K1-K3 显示 scale↑ → clip↑，验证"保守吸引子"假说
- 若 K5-K7 显示 CFG > FVA，说明 null direction 比 magnitude 更重要
- K8 联合若超 0.71，则接近突破帕累托前沿

### L 系列：训练侧改造（FC.md 改造1，需要重训）

**代码分析结论**：L 系列所有改造已在 losses620.py 中预埋实现，无需改代码！

- **L1 (各向异性训练目标)**: `training_target_projection_mode = "pure_vertical_flow_wavelet"`
  - 代码位置: losses620.py L272-277
  - 作用: `projected = Base(content) + Fiber(target)`，训练目标只保留 fiber 差异
  - 理论: 让网络只学 v_fiber，不浪费容量在 base 重构上
- **L2 (高通噪声注入)**: `bridge_sigma > 0`（如 0.08）
  - 代码位置: losses620.py L416-423
  - 作用: `ε_fiber = ε - Lowpass(ε)`，噪声只注入 fiber 空间
  - 理论: 训练噪声不破坏 base，只增强 fiber 的方差拟合
- **L3 (联合)**: L1 + L2
- **L4 (三频段目标)**: `training_target_projection_mode = "tri_band_wavelet"`
  - 代码位置: losses620.py L278-287
  - 作用: LL 锁死, Mid α-blend, HH 自由
  - 理论: 比 L1 更精细的频段控制

| 变体 | 配置 | 训练时间 | 理论预期 |
|------|------|----------|----------|
| L1 | pure_vertical_wavelet, σ=0 | ~1h (2 epoch) | clip 微涨, lpips 持平 |
| L2 | 默认 target, σ=0.08 | ~1h | clip 涨, lpips 微涨 |
| L3 | pure_vertical_wavelet + σ=0.08 | ~1h | clip 明显涨, lpips 持平 |
| L4 | tri_band_wavelet + σ=0.08 | ~1h | 最精细控制 |

**启动策略**：从 e4_long ep8 出发，LR=5e-5（低 LR fine-tune），2 epoch，batch_size=24（12GB VRAM 安全）

## 关键诊断：runtime_observability 揭示真正瓶颈

对 H5 vs J1 的 runtime_observability 数据进行对比分析，发现了一个**颠覆性的理论发现**。

### 风格路径几乎完全休眠

| 指标 | H5 | J1 | 含义 |
|------|-----|-----|------|
| `model_style_gate_value` | 0.0500 | 0.0500 | 风格门 95% 关闭！ |
| `model_style_dino_active` | 0.0000 | 0.0000 | DINO 完全未激活 |
| `model_cross_attn_delta_abs` | 0.0382 | 0.0382 | 交叉注意力贡献极弱 |
| `model_ca_output_std` | 0.0806 | 0.0806 | ca 输出比输入小 10x |
| `model_endpoint_style_high_abs` | 0.0000 | 0.0000 | 风格→fiber 投影完全为零 |
| `model_endpoint_style_low_abs` | 0.0000 | 0.0000 | 风格→base 投影完全为零 |
| `model_film_gamma_abs` | 0.1303 | 0.1303 | FiLM gamma 弱 |
| `model_film_beta_abs` | 0.1271 | 0.1271 | FiLM beta 弱 |
| `model_endpoint_alpha` | 0.0000 | 0.0000 | endpoint 位移为零 |

### 代码层面分析

检查 `predict_endpoint` 代码（model620.py L399-410）发现：

1. **`endpoint_style_to_low/high` 层是零初始化且未使用**：
   - 当 `endpoint_film_enabled = True`（e4_long 配置），这两个层是死代码
   - 但 `runtime_observability` 仍在测量它们的输出 → 永远报 0
   - 这是测量指标的 red herring，不是真正瓶颈

2. **真正活跃的风格路径是 FiLM**：
   - `endpoint_film_low/high`（FiLMEndpointHead）是活跃路径
   - gamma ~0.13, beta ~0.13 — 弱但非零
   - FiLM 调制：`h = (1 + gamma) * h + beta`

3. **`style_condition_source = "latent"`**：
   - 模型用 intrinsic CNN 从 target latent 提取风格，不用 DINO
   - `intrinsic_style_cnn` → `style_feat` → `style_global`
   - `style_global` 进入 FiLM 和 cross-attention

### 理论结论

**H5 的"突破"（clip=0.7026, lpips=0.4936）实际上是假突破**：
- clip=0.7026 主要来自 FiLM 弱调制（gamma~0.13）+ VAE latent 统计匹配
- fiber SDE 噪声注入提供了纹理变化，但**缺乏风格方向**
- FC-SB 理论要求"fiber 狂热扩散"，但当前 fiber 扩散是**无方向的布朗运动**，没有强风格信号驱动

**K 系列（FVA/Fiber-CFG）注定无效**：放大一个接近零的信号不会产生有意义的结果。

### 真正瓶颈：风格信号强度不足

要突破 0.7026，需要**放大风格信号**，让 fiber 扩散获得明确的方向性。这正是 FC-SB "fiber 狂热扩散"理论的真正要求 — 不是噪声放大，而是**风格方向放大**。

## K 系列：FVA + Fiber-CFG 评估（基础设施受阻）

K1-K8 检查点已生成（k_series/），代码已实现（model620.py L559-569, L610-625）。

### 评估状态

| 变体 | 状态 | 备注 |
|------|------|------|
| K1-K8 | 多次尝试评估均失败 | WSL 实例周期性重启，杀死所有进程 |

**WSL 重启问题**：`uptime` 显示 "up 0 min"，dmesg 显示 "journal corrupted or uncleanly shut down"。WSL 实例每隔几分钟重启一次，导致 nohup/tmux 进程全部死亡。单次评估需要 ~5 分钟，无法在单个 WSL 会话内完成。

**理论预测**：基于 runtime_observability 分析，K 系列（放大 fiber 速度）无法突破 0.7026，因为 fiber 缺乏风格方向。即使 K 评估成功，预期结果与 H5 持平或更差。

## M 系列：风格路径放大（理论驱动新方向）

### 理论依据

基于 runtime_observability 诊断，真正的瓶颈是**风格信号强度不足**。FC-SB 理论要求 fiber "狂热扩散"，但当前：
- FiLM gamma/beta ~0.13（弱调制）
- 风格门 0.05（95% 关闭）
- 交叉注意力 delta 0.038（极弱）

**M 系列核心思想**：放大风格信号，让 fiber 扩散获得明确方向性。

### 代码改造（model620.py）

**M3: style_embed_scale**（L114-124, L386-390）：
```python
# __init__ 中读取配置
self.style_embed_scale = float(getattr(model_cfg, "style_embed_scale", 1.0))

# forward 中放大 style_global（所有下游路径的源信号）
if self.style_embed_scale != 1.0:
    style_global = style_global * self.style_embed_scale
```

**M4: endpoint_delta_scale**（L122-124, L425-430）：
```python
# __init__ 中读取配置
self.endpoint_delta_scale = float(getattr(model_cfg, "endpoint_delta_scale", 1.0))

# forward 中放大 FiLM 输出的 low/high delta
if self.endpoint_delta_scale != 1.0:
    low_delta = low_delta * self.endpoint_delta_scale
    high_delta = high_delta * self.endpoint_delta_scale
```

### 变体设计

| 变体 | 机制 | 参数 | 理论预期 |
|------|------|------|----------|
| M1 | style_embed_scale | 1.5 | 温和放大 +50%，clip 微涨 |
| M2 | style_embed_scale | 2.0 | 中等放大 +100%，clip 明显涨 |
| M3 | style_embed_scale | 3.0 | 激进放大 +200%，验证方向性 |
| M4 | style_embed_scale | 0.5 | 抑制 -50%，应降 clip（验证） |
| M5 | endpoint_delta_scale | 1.5 | 精准放大 endpoint 修改 |
| M6 | endpoint_delta_scale | 2.0 | 更强放大 |
| M7 | endpoint_delta_scale | 3.0 | 激进放大 |
| M8 | ses=2.0 + eds=2.0 | 联合 | 双重放大，可能突破 0.71 |

### 关键预期

- 若 M1-M3 显示 scale↑ → clip↑，验证"风格信号不足"假说
- 若 M4（scale=0.5）显示 clip↓，反向验证方向性
- M8 联合若超 0.71，则成功突破帕累托前沿
- 若 M 也持平，说明 FiLM 路径已饱和，需考虑训练侧改造（L 系列）

## M 系列评估结果：放大现有路径失败

### Forward 差异诊断

直接比较 M1-M8 与 H5 的 forward 输出差异（随机输入）：

| 变体 | ses | eds | 相对差异 | 效果倍数 |
|------|-----|-----|---------|---------|
| M1 | 1.5 | 1.0 | 0.000331 | 1.0x |
| M2 | 2.0 | 1.0 | 0.000413 | 1.2x |
| M3 | 3.0 | 1.0 | 0.000472 | 1.4x |
| M4 | 0.5 | 1.0 | 0.001465 | 4.4x |
| **M5** | 1.0 | **1.5** | **0.426705** | **1289x** |
| M6 | 1.0 | 2.0 | 0.853410 | 2578x |
| M7 | 1.0 | 3.0 | 1.706820 | 5156x |
| M8 | 2.0 | 2.0 | 0.853851 | 2579x (≈M6) |

**关键发现**：
1. **M3 (style_embed_scale) 几乎无效**：放大 3x 只产生 0.047% 变化，因为 gate(0.05) 衰减了 95% 的 style 信号
2. **M4 (endpoint_delta_scale) 效果巨大**：放大 1.5x 产生 42.7% 变化，绕过 gate 直接放大 FiLM 输出
3. **M8 联合 ≈ M6**：ses 的贡献在 eds 面前完全可忽略，无协同效应

### M5-M7 单 style 评估（Early_Renaissance）

| 变体 | eds | a_clip | a_lpips | 趋势 |
|------|-----|--------|---------|------|
| H5 | 1.0 | 0.7788 | 0.5114 | 基准 |
| M5 | 1.5 | 0.7107 | 0.6036 | clip↓ lpips↑ |
| M6 | 2.0 | 0.6731 | 0.6479 | 持续恶化 |
| M7 | 3.0 | 0.6483 | 0.6910 | 持续恶化 |

### M 系列结论

**放大现有风格路径会同时损害 clip 和 lpips**。FiLM 学到的调制参数缺乏风格方向性，放大只产生噪声。与 K 系列（FVA）预测一致。

根本问题：不是"信号太弱需要放大"，而是"信号无方向，放大只增加噪声"。

## N 系列：Endpoint AdaIN（纤维统计匹配）— 重大突破

### 理论依据

基于 M 系列失败的教训，重新理解 FC-SB 理论：
- FC-SB 要求 fiber "狂热扩散"且**携带风格方向**
- M 系列证明：放大无方向的 fiber 信号只会产生噪声
- **N 系列核心思想**：不放大，而是**注入**有方向的风格统计

**N1: Endpoint AdaIN (Fiber Statistics Matching)**：
- 在预测的 endpoint 上，分离 base/fiber
- 用目标风格 fiber 的统计量（mean/std）替换预测 fiber 的统计量
- base 保持不变（保 LPIPS）
- fiber 获得明确风格方向（提 clip）

公式：
```
ep_fiber_matched = (ep_fiber - μ_pred) / σ_pred * σ_style + μ_style
endpoint = ep_base + (1-α)*ep_fiber + α*ep_fiber_matched
```

### 代码实现（model620.py L593-664）

在 `i2sb_inference()` 中，fiber_only_ep 之后、速度计算之前注入 AdaIN。支持三种模式：
- `full`: 同时匹配 mean+std
- `mean_only`: 只匹配 color（mean）
- `std_only`: 只匹配 contrast（std）

### N 系列评估结果（Early_Renaissance 单 style）

| 变体 | adain | mode | eds | a_clip | a_lpips |
|------|-------|------|-----|--------|---------|
| H5 | 0 | - | 1.0 | 0.7788 | 0.5114 |
| N1 | 0.3 | full | 1.0 | 0.7792 | 0.5107 |
| N2 | 0.5 | full | 1.0 | 0.7781 | 0.5117 |
| N3 | 0.7 | full | 1.0 | 0.7793 | 0.5111 |
| N4 | 1.0 | full | 1.0 | 0.7805 | 0.5113 |
| N5 | 0.5 | mean_only | 1.0 | 0.7803 | 0.5111 |
| N6 | 0.5 | std_only | 1.0 | 0.7800 | 0.5112 |
| N7 | 0.7 | mean_only | 1.0 | 0.7780 | 0.5118 |
| **N8** | **0.3** | **full** | **0.5** | **0.8243** | **0.3642** |

### N 系列关键发现

**发现 1: AdaIN 单独无效（N1-N7）**
- N1-N7 的 clip 和 lpips 与 H5 几乎相同（差异 < 0.001）
- 原因：eds=1.0 时 FiLM 的无方向 delta 占主导，AdaIN 注入的风格统计被淹没

**发现 2: N8 (AdaIN + eds=0.5) 产生巨大突破！**
- **clip=0.8243** (+5.8% vs H5 的 0.7788)
- **lpips=0.3642** (-28.8% vs H5 的 0.5114)
- 这是单 style 评估的历史最佳

### N8 协同效应理论解释

N8 的配置：adain_scale=0.3 + endpoint_delta_scale=0.5

**协同机制**：
1. **eds=0.5 抑制无方向 FiLM delta**：将 FiLM 的噪声方向 delta 缩小到一半
2. **adain=0.3 注入有方向风格统计**：用目标风格 fiber 统计量替换 30% 的预测 fiber
3. **结果**：fiber 空间中，无方向噪声被抑制，有方向风格信号被注入 → fiber 获得清晰风格方向

这完全验证了 FC-SB 理论："底流形死寂，纤维狂热扩散"——关键是给 fiber 注入**有方向**的风格信号，同时抑制**无方向**的 FiLM 噪声。

**为什么单独 AdaIN 无效**：
- eds=1.0 时，FiLM delta 以原始强度进入 endpoint
- AdaIN 只替换 30% 的 fiber 统计，70% 仍是无方向 delta
- 无方向 delta 的噪声淹没 AdaIN 的风格信号

**为什么 N8 的协同效应如此强**：
- eds=0.5 让无方向 delta 减半，信噪比提升
- AdaIN 的 30% 风格统计在低噪声环境中变得显著
- 两者协同：降噪 + 注入 = fiber 获得清晰风格方向

## O 系列：N8 突破点附近精细探索

### 设计

在 N8 (adain=0.3, eds=0.5) 附近探索更优组合：

| 变体 | adain | eds | mode | 目的 |
|------|-------|-----|------|------|
| O1 | 0.2 | 0.5 | full | 更保守 adain |
| O2 | 0.4 | 0.5 | full | 更强 adain |
| O3 | 0.5 | 0.5 | full | adain=eds 平衡 |
| O4 | 0.3 | 0.3 | full | 更保守 eds |
| O5 | 0.3 | 0.4 | full | 中间 eds |
| O6 | 0.3 | 0.6 | full | 更强 eds |
| O7 | 0.5 | 0.3 | full | 反转比例 |
| O8 | 0.3 | 0.5 | mean_only | 只 color 匹配 |

### O 系列评估结果

| 变体 | adain | eds | mode | a_clip | a_lpips |
|------|-------|-----|------|--------|---------|
| H5 | 0 | 1.0 | - | 0.7788 | 0.5114 |
| N8 | 0.3 | 0.5 | full | 0.8243 | 0.3642 |
| O1 | 0.2 | 0.5 | full | 0.8225 | 0.3655 |
| O2 | 0.4 | 0.5 | full | 0.8242 | 0.3651 |
| O3 | 0.5 | 0.5 | full | 0.8237 | 0.3649 |
| **O4** | **0.3** | **0.3** | full | **0.8282** | **0.3133** |
| O5 | 0.3 | 0.4 | full | 0.8272 | 0.3367 |
| O6 | 0.3 | 0.6 | full | 0.8173 | 0.3959 |
| O7 | 0.5 | 0.3 | full | 0.8274 | 0.3131 |
| O8 | 0.3 | 0.5 | mean_only | 0.8227 | 0.3649 |

### O 系列关键发现

**发现 1: eds 越低越好（O4/O7 vs O5/N8/O6）**

| eds | a_clip | a_lpips | 趋势 |
|-----|--------|---------|------|
| 0.3 (O4) | 0.8282 | 0.3133 | 最佳 |
| 0.4 (O5) | 0.8272 | 0.3367 | 次之 |
| 0.5 (N8) | 0.8243 | 0.3642 | 再次 |
| 0.6 (O6) | 0.8173 | 0.3959 | 最差 |

**理论解释**：eds 越低，FiLM 的无方向 delta 被抑制越多，AdaIN 的有方向风格信号信噪比越高。

**发现 2: adain 在 0.3-0.5 范围内影响小（O4 vs O7）**
- O4 (adain=0.3): clip=0.8282, lpips=0.3133
- O7 (adain=0.5): clip=0.8274, lpips=0.3131
- 差异 < 0.001，说明 adain=0.3 已足够注入风格统计

**发现 3: O4 (clip=0.8282, lpips=0.3133) 是新的历史最佳**
- lpips=0.3133 接近 E2 历史最佳 0.3326（不同评估条件）
- clip=0.8282 远超 H5 的 0.7788

### O 系列结论

**最优配置方向**：继续降低 eds（< 0.3）可能进一步提升性能。P 系列将探索 eds→0 极限。

## P 系列：eds→0 极限探索（单 style）

### 设计

基于 O 系列"eds 越低越好"的发现，P 系列探索 eds→0 的极限，并尝试不同 adain 强度与 mode。

| 变体 | adain | eds | mode | 目的 |
|------|-------|-----|------|------|
| P1 | 0.3 | 0.2 | full | 降 eds |
| P2 | 0.3 | 0.1 | full | 进一步降 eds |
| P3 | 0.3 | 0.0 | full | eds=0 极限 |
| P4 | 0.5 | 0.0 | full | 纯 AdaIN, 增强 adain |
| P5 | 0.7 | 0.0 | full | 更强 adain |
| P6 | 1.0 | 0.0 | full | 完全 AdaIN 替换 |
| P7 | 0.3 | 0.2 | mean_only | 低 eds + 仅 color 匹配 |
| P8 | 0.5 | 0.2 | full | 中等 adain + 低 eds |

### P 系列评估结果（Early_Renaissance 单 style）

| 变体 | adain | eds | mode | a_clip | a_lpips |
|------|-------|-----|------|--------|---------|
| H5 | 0 | 1.0 | - | 0.7788 | 0.5114 |
| N8 | 0.3 | 0.5 | full | 0.8243 | 0.3642 |
| O4 | 0.3 | 0.3 | full | 0.8282 | 0.3133 |
| P1 | 0.3 | 0.2 | full | 0.8264 | 0.2970 |
| P2 | 0.3 | 0.1 | full | 0.8255 | 0.2889 |
| **P3** | **0.3** | **0.0** | full | **0.8266** | **0.2853** |
| P4 | 0.5 | 0.0 | full | 0.8243 | 0.2851 |
| P5 | 0.7 | 0.0 | full | 0.8247 | 0.2849 |
| P6 | 1.0 | 0.0 | full | 0.8230 | 0.2861 |
| **P7** | **0.3** | **0.2** | mean_only | **0.8289** | **0.2970** |
| P8 | 0.5 | 0.2 | full | 0.8263 | 0.2966 |

### P 系列关键发现

**发现 1: eds→0 趋势饱和**

| eds | a_lpips | 趋势 |
|-----|---------|------|
| 0.5 (N8) | 0.3642 | 基准 |
| 0.3 (O4) | 0.3133 | 大幅改善 |
| 0.2 (P1) | 0.2970 | 持续改善 |
| 0.1 (P2) | 0.2889 | 接近饱和 |
| 0.0 (P3) | 0.2853 | 饱和 |

eds 从 0.5→0.3 改善 0.051，从 0.3→0.0 仅再改善 0.028。**eds=0.0 是收益递减的饱和点**。

**发现 2: adain 强度在 eds=0 时影响微小（P3-P6）**
- P3 (adain=0.3): lpips=0.2853
- P4 (adain=0.5): lpips=0.2851
- P5 (adain=0.7): lpips=0.2849
- P6 (adain=1.0): lpips=0.2861
- 差异 < 0.001，**adain=0.3 已足够**，继续增大无收益反而轻微回退

**发现 3: P7 (mean_only, eds=0.2) 达到最佳 clip=0.8289**
- P7 vs P1 (同 adain=0.3, eds=0.2, 仅 mode 不同):
  - P1 (full): clip=0.8264, lpips=0.2970
  - P7 (mean_only): clip=0.8289, lpips=0.2970
- **mean_only 在保持同等 lpips 时 clip 略高 +0.0025**
- 理论解释：保留预测的 contrast (std)，仅匹配 color (mean)，减少过度风格化对 clip 的负面影响

### P 系列单 style 结论

**P3 (adain=0.3, eds=0.0, full)** 是 LPIPS 最优点：0.2853（vs H5 0.5114，改善 -44.2%）
**P7 (adain=0.3, eds=0.2, mean_only)** 是 CLIP 最优点：0.8289（vs H5 0.7788，提升 +6.4%）

## 5-Style 完整验证（主指标）

### 评估协议

**所有配置在 5-style 完整数据集（Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e）上评估**，使用 `num_steps=12, batch_size=1, clip_backend=hf`。单 style 评估（Early_Renaissance）仅作诊断参考，**不作为性能结论依据**（单 style 存在风格偏好偏差）。

### 5-Style 完整结果（主指标）

| 配置 | adain | eds | mode | t_clip | t_lpips | a_clip | a_lpips | Δclip vs H5 | Δlpips vs H5 |
|------|-------|-----|------|--------|---------|--------|---------|-------------|--------------|
| **H5 (基线)** | 0 | 1.0 | - | 0.7026 | 0.4936 | 0.7221 | 0.4936 | - | - |
| N8 | 0.3 | 0.5 | full | 0.6862 | 0.3581 | 0.7168 | 0.3585 | -2.3% | **-27.4%** |
| O4 | 0.3 | 0.3 | full | 0.6747 | 0.2964 | 0.7095 | 0.2965 | -4.0% | **-39.9%** |
| P7 | 0.3 | 0.2 | mean_only | 0.6695 | 0.2766 | 0.7053 | 0.2766 | -4.7% | **-44.0%** |
| **P3** | **0.3** | **0.0** | full | **0.6638** | **0.2658** | 0.6998 | 0.2658 | -5.5% | **-46.2%** |

### 单 style 诊断（仅参考，不作结论）

Early_Renaissance 单 style 评估，用于定位风格偏好偏差：

| 配置 | a_clip | a_lpips | 单 style 偏差 |
|------|--------|---------|---------------|
| H5 | 0.7788 | 0.5114 | - |
| N8 | 0.8243 | 0.3642 | clip +5.8%, lpips -28.8% |
| O4 | 0.8282 | 0.3133 | clip +6.4%, lpips -38.7% |
| P3 | 0.8266 | 0.2853 | clip +6.1%, lpips -44.2% |
| P7 | 0.8289 | 0.2970 | clip +6.4%, lpips -41.9% |

**偏差诊断**：单 style 上 CLIP 全部提升（+5.8% ~ +6.4%），但 5-style 综合 CLIP 全部下降（-2.3% ~ -5.5%）。说明 Early_Renaissance 是 AdaIN 受益最强的风格，单 style 评估对 CLIP 方向性结论有偏差。**LPIPS 改善幅度在单/5-style 之间一致（~27-46%），LPIPS 结论可靠**。

### 5-Style 关键发现（客观结论）

**发现 1: 单 style 的 0.82+ CLIP 突破在 5-style 上完全未复现**

N8 单 style 报告 a_clip=0.8243（+5.8% vs H5 单 style），但 5-style 复现 t_clip=0.6862（**-2.3% vs H5 5-style**）。所有 N/O/P 配置在 5-style 上 CLIP 全部下降，**没有任何配置复现单 style 的 CLIP 突破**。

**单 style 0.82+ 是 Early_Renaissance 偏好偏差**，不可信。5-style 才是真实性能。

**发现 2: eds↓ → LPIPS 单调改善，但 CLIP 同步下降（trade-off，非突破）**

| eds | t_clip | t_lpips | Δclip vs H5 | Δlpips vs H5 |
|-----|--------|---------|-------------|--------------|
| 1.0 (H5) | 0.7026 | 0.4936 | - | - |
| 0.5 (N8) | 0.6862 | 0.3581 | -2.3% | -27.4% |
| 0.3 (O4) | 0.6747 | 0.2964 | -4.0% | -39.9% |
| 0.2 (P7) | 0.6695 | 0.2766 | -4.7% | -44.0% |
| 0.0 (P3) | 0.6638 | 0.2658 | -5.5% | -46.2% |

LPIPS 改善是以 CLIP 下降为代价的 trade-off。抑制 FiLM 无方向 delta（eds↓）确实提升内容保真度（LPIPS↓），但同时削弱风格迁移强度（CLIP↓）。**这不是"突破帕累托前沿"，而是在 trade-off 曲线上向 LPIPS 方向移动**。

**发现 3: LPIPS 改善幅度在单/5-style 之间一致，LPIPS 结论可靠**

| 配置 | 单 style Δlpips | 5-style Δlpips | 一致性 |
|------|-----------------|----------------|--------|
| N8 | -28.8% | -27.4% | ✓ |
| O4 | -38.7% | -39.9% | ✓ |
| P3 | -44.2% | -46.2% | ✓ |
| P7 | -41.9% | -44.0% | ✓ |

LPIPS 改善是真实的跨风格效应，但 CLIP 下降也是真实的。

**发现 4: P7 (mean_only) 在 CLIP 上略优于 P3，但仍未超过 H5**

P7 (eds=0.2, mean_only) vs P3 (eds=0.0, full):
- t_clip: 0.6695 vs 0.6638 (+0.0057, P7 略好)
- t_lpips: 0.2766 vs 0.2658 (+0.0108, P3 略好)

mean_only 保留预测的 contrast (std)，仅匹配 color (mean)，减少过度风格化对 CLIP 的负面影响。但 P7 的 t_clip=0.6695 仍**低于 H5 的 0.7026**，CLIP 维度仍是退化。

### 帕累托前沿分析

```
t_lpips
0.50 |  H5 (0.7026, 0.4936)  ← CLIP 最优
     |
0.36 |     N8 (0.6862, 0.3581)
     |
0.30 |         O4 (0.6747, 0.2964)
     |
0.27 |             P7 (0.6695, 0.2766) / P3 (0.6638, 0.2658)  ← LPIPS 最优
     +---------------------------------- t_clip
     0.66   0.67   0.68   0.69   0.70   0.71
```

N8/O4/P3/P7 在 trade-off 曲线上，**没有任何配置在两维度同时优于 H5**。H5 在 CLIP 维度最优，P3 在 LPIPS 维度最优。

## 现状总结（截至 2026-06-26，客观）

### 已确认的事实

1. **N1 Endpoint AdaIN 机制本身有效**：能注入风格方向到 fiber，产生可预期的 LPIPS 改善
2. **5-style LPIPS 改善真实**：P3 达到 t_lpips=0.2658（-46.2% vs H5），是当前 5-style LPIPS 历史最低
3. **N1 + eds 协同效应存在**：adain 注入有方向统计 + eds 抑制无方向 delta，机制明确

### 未复现的"突破"

**单 style 报告的 CLIP 0.82+ 在 5-style 上完全未复现**。所有配置 5-style t_clip 均低于 H5 基线：
- N8: t_clip=0.6862 (vs H5 0.7026, **-2.3%**)
- P3: t_clip=0.6638 (vs H5 0.7026, **-5.5%**)

**5-style CLIP 维度是退化，不是突破**。之前的"突破"表述是单 style 偏差导致的误判。

### 根本矛盾

**CLIP 与 LPIPS 的 trade-off 无法同时突破**：
- 降低 eds → LPIPS 改善但 CLIP 下降
- 当前 adain 机制（per-channel mean/std 匹配）不足以在 5-style 上同时提升 CLIP
- 根源：per-channel 全局统计匹配丢失空间信息，无法捕捉 CLIP 关心的笔触/纹理/构图

### 理论反思

FC-SB 理论要求 "fiber 狂热扩散且携带风格方向"。N1 AdaIN 实现了"携带风格方向"，但方式是 **per-channel 全局统计匹配**，这存在根本局限：
1. **空间信息丢失**：per-channel mean/std 只匹配全局 color/contrast，丢失笔触、纹理的空间结构
2. **过度均匀化**：fiber 被强制匹配到风格的全局统计，局部风格特征被抹平
3. **CLIP 下降根源**：CLIP 衡量的是高级语义风格相似度（笔触、构图、纹理），per-channel 统计匹配无法捕捉这些

## 下一步计划（理论驱动）

**核心目标：在保持 LPIPS 改善的同时，恢复或提升 5-style CLIP**。当前 P3 的 t_lpips=0.2658 已足够好，重点突破 CLIP。

### Q 系列：空间感知 AdaIN（突破 CLIP 瓶颈）

**理论依据**：当前 AdaIN 是 per-channel 全局统计匹配，丢失空间信息。需要**空间感知的统计匹配**，保留笔触/纹理的空间结构。

**Q1: Patch-wise AdaIN**
- 将 fiber 分成 patch（如 8×8），每个 patch 独立匹配风格对应 patch 的统计
- 保留空间局部风格特征
- 代码：`F.unfold` → per-patch stats → `F.fold`

**Q2: Multi-scale AdaIN**
- 在不同频率 band（LL/Mid/HH）上做不同强度的 AdaIN
- LL: 强 adain（color 匹配）
- HH: 弱 adain（保留纹理细节）
- 利用 wavelet 分解

**Q3: Spatial AdaIN + Content Structure Preservation**
- AdaIN 时保留 content fiber 的空间结构，只替换风格相关的统计
- `ep_fiber_matched = content_structure * style_stats`
- 用 Gram matrix 而非 mean/std 匹配

### R 系列：adain 与 eds 的非线性组合

**理论依据**：当前 eds 是全局缩放，无差别抑制所有 FiLM delta。但 FiLM delta 中可能包含有用的风格方向信息（只是被无方向噪声淹没）。

**R1: 选择性 eds**
- 只抑制 FiLM delta 的高频分量（无方向噪声主要在高频）
- 保留低频 FiLM delta（可能含风格方向）
- 代码：`low_delta = lp(low_delta) * eds + (low_delta - lp(low_delta)) * eds_high`

**R2: adain-gated eds**
- eds 强度由 adain 注入的风格信号强度自适应决定
- 风格信号强 → eds 大（抑制无方向 delta，让 adain 主导）
- 风格信号弱 → eds 小（保留 FiLM delta 补偿）

### S 系列：训练侧改造（长期）

如果 Q/R 系列仍无法突破 CLIP 瓶颈，需启动训练侧改造（L 系列，FC.md 改造1）：
- L1: pure_vertical_wavelet 训练目标（让模型学会更精确的 fiber 速度预测）
- L2: 高通噪声注入（训练噪声只注入 fiber，增强 fiber 方差拟合）
- 从 e4_long ep8 出发，LR=5e-5，2 epoch fine-tune
