# FC-SB 调优与架构创新实施计划

> **范围**：4 个高 ROI 调优方向（A1-A4）+ 4 个架构大创新（B1-B4）= 8 方向路线图
> **基线**：I7 → U4(α0.1, clip=0.7225/lpips=0.3660) → V6(k32, clip=0.7262/lpips=0.3722) → V3(k16, clip=0.7295/lpips=0.3963)
> **硬约束**：显存 ≤ 11GB（RTX 3060 12GB），Windows native Python，数据集 I 盘路径
> **核心理论**：BASE LOCKING 锁死 clip↑/lpips↓ 三难困境是结构性约束，需要工程调优压榨 + 架构创新重构双管齐下

---

## 一、当前状态分析

### 1.1 帕累托死结的物理本质

6 个月 645+ 实验证实：在 **单流网络 + MSE 损失 + 欧氏空间积分** 范式下，clip↑/lpips↓/WFI↓ 三难困境是 BASE LOCKING（[model620.py:957](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L957) `h = x_base_lock + (h - lp(h))`) 的结构性约束，不是优化问题。

- **保守吸引子** D = {g≈0.05, α≈0.16, η≈0.997, R_style≈0.001} 是 L_flow 主导 + SWD 平坦 + style 梯度衰减 三条件下的 loss 最优解
- 任何 clip 提升必然伴随 lpips 代价（V3: clip+0.0278/lpips+0.0338）
- 与 SaMST per-style 天花板（0.7597）的 5% gap 可能是通用模型理论上限

### 1.2 已验证的代码能力（避免重复造轮子）

| 能力 | 位置 | 状态 | 备注 |
|------|------|------|------|
| N1 Endpoint AdaIN 块 | [model620.py:676-874](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L676) | ✅ 生效 | 推理期 fiber 统计匹配 |
| K1 Fiber-CFG | [model620.py:893-903](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L893) | ✅ 生效 | fiber 空间 CFG 外推 |
| Tri-directional CFG (source-repulsion) | [model620.py:964-1028](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L964) | ✅ 已存在 | 默认关闭，ep_null 非真正 unconditional |
| Beta 时间采样 | [losses620.py:207-214](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L207) | ✅ 生效 | 需加 Logit-Normal 分支 |
| W2 anti_input hinge loss | [losses620.py:669-673](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L669) | ⚠️ 仅 step=1 生效 | margin 失效，需替换 |
| Block-level soft MoE | [blocks620.py:143-156](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py#L143) | ✅ 已存在 | K/V experts + router |
| Haar 局部闭包 | [model620.py:756-771](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L756) | ✅ 存在 | 需提取为独立模块（B2） |
| multiband_adain (T) | [model620.py:744-812](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L744) | ✅ 生效 | mid/hh 静态混合 |
| patch_adain (V) | [model620.py:813-852](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L813) | ✅ 生效 | kernel 必须 2 幂次 |

### 1.3 静态参数的局限

当前 mid_adain_scale/hh_adain_scale/patch_adain_kernel/style_extrap_alpha 都是**静态全局参数**，对所有时间步 t 等价作用。这导致：
- 早期 t∈[0,0.3]（应建低频轮廓）和高频注入冲突
- 晚期 t∈[0.7,1.0]（应注高频笔触）强度不足
- 时间步采样均匀浪费 99% 精力在 t∈[0,0.3] 无意义区间

---

## 二、实施路线图

### 优先级与依赖关系

```
Week 1 (高 ROI, 低风险):
  A1 时频交叉调度 (推理期, 复用 U4 ckpt)  ← 最优先
  A4 输出方差匹配 (训练期, 替换失效 W2)   ← 与 A1 并行
  A2 Step1 评估现有 source-repulsion       ← 零代码, 仅评估

Week 2 (中等风险):
  A2 Step2 fiber 空间 source-repulsion 改造
  A3 Logit-Normal 时间采样
  B4 Fiber-MoE Adapters (复用现有 MoE)

Week 3-4 (架构创新, 高风险高回报):
  B2 原生频域 ODE POC (wavelet620 + spectral_bridge)
  B1 Dual-Stream Flow Matching (可选, 依赖 B2 结果)
  B3 Energy-Guided SB (可选, 依赖 B2 结果)
```

---

## 三、Phase A：高 ROI 调优方向

### A1：时频交叉调度（Time-Frequency Coupled Scheduling）

**洞察**：mid_adain_scale 和 hh_adain_scale 当前是静态的（[model620.py:791-792](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L791)）。应让它们成为时间 t 的函数：
- t∈[0, 0.5]：锁死 mid 和 hh，让模型专心把 content 推到目标域色彩基调
- t∈[0.5, 1.0]：指数级放大 hh_adain_scale，最后阶段爆发式注入高频笔触

**类型**：推理期改动，可复用 U4(α0.1) checkpoint，ROI 最高

#### 实施步骤

**步骤 1: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)** — ModelConfig 新增 6 个字段（在 L484 附近 `tri_band_edge_lock_alpha` 后）

```python
# === FC-SB Phase 4 A1: Time-Frequency Coupled Scheduling ===
tf_schedule_enabled: bool = False       # 总开关
tf_hh_ramp_start: float = 0.5           # hh 开始升温的 t 阈值
tf_hh_ramp_end: float = 1.0             # hh 达到 max_scale 的 t
tf_hh_max_scale: float = 1.5            # hh 在 t=1.0 时的最大倍数（相对静态 hh_adain_scale）
tf_mid_lock_threshold: float = 0.5      # mid 锁死阈值（t < 此值时 mid_scale=0）
tf_mid_max_scale: float = 1.0           # mid 在 t=1.0 时的最大倍数
```

**步骤 2: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) L596-631** — 读取 A1 配置（在现有 `patch_adain_kernel` 读取后追加）

```python
# === FC-SB Phase 4 A1: Time-Frequency Coupled Scheduling ===
tf_schedule_enabled = bool(_cfg_get('tf_schedule_enabled', False))
tf_hh_ramp_start = float(_cfg_get('tf_hh_ramp_start', 0.5))
tf_hh_ramp_end = float(_cfg_get('tf_hh_ramp_end', 1.0))
tf_hh_max_scale = float(_cfg_get('tf_hh_max_scale', 1.5))
tf_mid_lock_threshold = float(_cfg_get('tf_mid_lock_threshold', 0.5))
tf_mid_max_scale = float(_cfg_get('tf_mid_max_scale', 1.0))
```

**步骤 3: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) L642 时间循环内** — 在 `for idx in range(steps):` 后、N1 块前，新增动态 scale 计算

```python
# === FC-SB Phase 4 A1: 动态时频调度 ===
if tf_schedule_enabled:
    # mid: t < threshold 时锁死（0），t >= threshold 时线性升到 max
    if t_curr < tf_mid_lock_threshold:
        mid_scale_dyn = 0.0
    else:
        mid_progress = (t_curr - tf_mid_lock_threshold) / max(1e-6, (1.0 - tf_mid_lock_threshold))
        mid_scale_dyn = tf_mid_max_scale * min(1.0, mid_progress)
    # hh: t < ramp_start 时保持原值，t >= ramp_start 时指数爆发
    if t_curr < tf_hh_ramp_start:
        hh_scale_dyn = hh_adain_scale  # 保持静态值
    else:
        ramp_progress = (t_curr - tf_hh_ramp_start) / max(1e-6, (tf_hh_ramp_end - tf_hh_ramp_start))
        ramp_progress = min(1.0, ramp_progress)
        # 指数爆发: 1.0 → tf_hh_max_scale
        hh_scale_dyn = hh_adain_scale * (1.0 + (tf_hh_max_scale - 1.0) * (ramp_progress ** 2))
else:
    mid_scale_dyn = mid_adain_scale
    hh_scale_dyn = hh_adain_scale
```

**步骤 4: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) L791-792** — two_level 分支用动态值替换静态值

```python
# 原: mid_final = mid_adain_scale * mid_matched + (1.0 - mid_adain_scale) * f_mid
# 原: hh_final = hh_adain_scale * hh_matched + (1.0 - hh_adain_scale) * f_hh_band
mid_final = mid_scale_dyn * mid_matched + (1.0 - mid_scale_dyn) * f_mid
hh_final = hh_scale_dyn * hh_matched + (1.0 - hh_scale_dyn) * f_hh_band
```

**步骤 5: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) N1 块末尾** — 新增 probe 写入（在 L806 `n1_hh_contribution_ratio` 后）

```python
self.last_debug["tf_mid_scale_dyn"] = float(mid_scale_dyn)
self.last_debug["tf_hh_scale_dyn"] = float(hh_scale_dyn)
self.last_debug["tf_t_curr"] = float(t_curr)
```

#### 验证
1. probe 验证：`tf_hh_scale_dyn` 在 t<0.5 时等于静态值，t>0.5 时指数升到 max
2. 复用 U4(α0.1) checkpoint，仅改 config 字段 `tf_schedule_enabled=True`
3. 5-style 评估对比 U4 baseline（clip=0.7225, lpips=0.3660）
4. **预期**：LPIPS 下降（早期锁死 mid 减少结构扰动），CLIP 持平或微升（晚期 hh 爆发注入笔触）

---

### A4：输出方差匹配（Output Variance Loss，W 方向重生）

**洞察**：W2 hinge loss 失效根因是约束 input-target 距离与风格迁移目标冲突（[losses620.py:669-673](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L669)）。白化的真正原因是输出 fiber 的方差被洗掉。应改为约束输出 fiber 的标准差对齐 target style fiber 的标准差。

**类型**：训练期改动，需重新训练（但用 I7 初始化 + 2 epoch 即可验证）

#### 实施步骤

**步骤 1: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) L128-134** — 初始化段新增 A4 字段

```python
# === FC-SB Phase 4 A4: Output Variance Matching (W 方向重生) ===
self.w_output_variance = float(getattr(self.bridge_cfg, "w_output_variance", 0.0))
self.output_variance_band: str = str(getattr(self.bridge_cfg, "output_variance_band", "hh")).strip().lower()
# "hh" = 仅匹配 HH 频带方差; "mid" = 仅匹配 Mid; "all" = 匹配全 fiber
```

**步骤 2: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) L635-676 W loss 段** — 在 `anti_input_loss` 计算后新增 variance matching loss

```python
# === FC-SB Phase 4 A4: Output Variance Matching ===
output_variance_loss = content.new_tensor(0.0)
if self.w_output_variance > 0:
    # 计算生成 fiber 和 target fiber 的 per-channel 标准差
    # f_gen 已在 L647 计算: z_hat1 - z_low
    f_target = projected_target - t_low  # target fiber（t_low 在后续计算，需提前）
    # 实际上 t_low 在 L764 才计算，这里需用 _lowpass(projected_target, self.lowpass_kernel)
    f_target = projected_target - _lowpass(projected_target, self.lowpass_kernel)
    if self.output_variance_band == "hh":
        # Haar HH 频带方差匹配（与推理期 N1 的 hh 一致）
        def _haar_hh(x):
            return (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
        f_gen_hh = _haar_hh(f_gen.float())
        f_target_hh = _haar_hh(f_target.float())
        gen_std = f_gen_hh.std(dim=[2, 3], keepdim=False)
        target_std = f_target_hh.std(dim=[2, 3], keepdim=False)
    elif self.output_variance_band == "mid":
        def _haar_mid(x):
            lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
            hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
            return lh + hl
        gen_std = _haar_mid(f_gen.float()).std(dim=[2, 3], keepdim=False)
        target_std = _haar_mid(f_target.float()).std(dim=[2, 3], keepdim=False)
    else:  # "all"
        gen_std = f_gen.float().std(dim=[2, 3], keepdim=False)
        target_std = f_target.float().std(dim=[2, 3], keepdim=False)
    # L2 距离 between per-channel stds
    output_variance_loss = ((gen_std - target_std) ** 2).mean()
    if _w_debug_print:
        print(f"[A4-debug] step={self._w_debug_counter} gen_std_mean={gen_std.mean().item():.4f} target_std_mean={target_std.mean().item():.4f} loss={output_variance_loss.item():.6f}", flush=True)
```

**步骤 3: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) L724-757 总 loss 组装** — 两个分支都加 A4 项

```python
# 在 "+ self.w_anti_input_style * anti_input_loss" 后追加
+ self.w_output_variance * output_variance_loss
```

**步骤 4: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) metrics 上报** — 在 metrics dict 中加入 A4

```python
"output_variance_loss": output_variance_loss.detach(),
```

**步骤 5: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) BridgeConfig** — 新增 2 字段（在 L131 `anti_input_margin` 后）

```python
w_output_variance: float = 0.0            # FC-SB Phase 4 A4: 输出方差匹配 loss 权重
output_variance_band: str = "hh"          # "hh" | "mid" | "all"
```

#### 验证
1. 训练 I7 初始化 + `w_output_variance=0.5, output_variance_band="hh"`，2 epoch
2. probe 验证：`output_variance_loss` 非零且持续下降（非 step=1 归零）
3. 5-style 评估对比 I7 baseline（clip=0.7017, lpips=0.3625）
4. **预期**：方差匹配直接惩罚"发灰平凡解"，clip 提升且 lpips 不恶化（因只约束 HH 频带方差，不约束均值）

---

### A2：Manifold CFG / Source-Repulsion（推理期无分类器流形引导）

**洞察**：模型预测速度 v 倾向均值（保守退化）。训练时不敢加大 loss（结构炸裂），就在推理时借力打力。公式：
$$v_{final} = v_{pred} + \omega_{style}(v_{pred} - v_{uncond}) - \omega_{source}(v_{source\_style} - v_{uncond})$$

**关键发现**：[model620.py:964-1028](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L964) `integrate_transport_cfg` **已存在 source-repulsion 实现**（L1008-1014），但默认关闭且 ep_null 不是真正 unconditional（L1006 仍传 style_latent）。

**类型**：分两步。Step1 评估现有实现，Step2 改造为 fiber 空间 source-repulsion

#### Step 1：评估现有 integrate_transport_cfg source-repulsion（零代码）

**步骤 1**: 在评估脚本中新增 `integrate_transport_cfg` 调用分支
- 提取 source content 的 DINO patches 作为 `idt_dino_patches`
- 配置 `cfg_target_scale ∈ {0.5, 1.0, 2.0}`, `cfg_repulse_scale ∈ {0.0, 0.5, 1.0}`
- 修复 ep_null：L1006 `ep_null = self.predict_endpoint(h, t=t_batch, style_id=style_id, style_latent=style_latent)` 应改为 `style_latent=None`（真正 unconditional）

**步骤 2**: 5-style 评估 9 组配置（3×3 grid）
- baseline: cfg_target=0, cfg_repulse=0
- 对比 K1 Fiber-CFG（[model620.py:893](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L893)）的效果

#### Step 2：fiber 空间 source-repulsion 改造

**洞察**：现有 source-repulsion 在全空间作用，会扰动 base。应在 fiber 空间作用（与 K1 一致），base 完全来自 target 保 LPIPS。

**步骤 1: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) L893-903 K1 段扩展** — 新增 fiber 空间 source-repulsion

```python
# === FC-SB Phase 4 A2: Fiber-Space Source-Repulsion ===
# v_fiber_final = v_fiber + ω_style*(v_fiber - v_null_fiber) - ω_source*(v_source_fiber - v_null_fiber)
fiber_source_repulse_scale = float(_cfg_get('fiber_source_repulse_scale', 0.0))
if fiber_source_repulse_scale > 0.0 and source_style_latent is not None:
    ep_source = self.predict_endpoint(
        h, t=t_batch, style_id=None,
        style_dino_patches=None, style_dino_cls=None,
        style_text_tokens=None, style_latent=source_style_latent,
    )
    v_source = (ep_source - h) / denom
    v_source_fiber = v_source - lp(v_source) if fiber_proj_ep else v_source
    v_fiber = v_fiber - fiber_source_repulse_scale * (v_source_fiber - v_null_fiber)
```

**步骤 2: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) integrate_transport 签名** — 新增 `source_style_latent` 参数

```python
def integrate_transport(
    self,
    x: torch.Tensor,
    ...
    target_style_latent: torch.Tensor | None = None,
    source_style_latent: torch.Tensor | None = None,  # 🆕 A2
    **_: object,
) -> torch.Tensor:
```

**步骤 3: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) ModelConfig** — 新增字段

```python
fiber_source_repulse_scale: float = 0.0   # FC-SB Phase 4 A2: fiber 空间 source-repulsion 强度
```

**步骤 4: [src/utils/inference.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/inference.py) L519-559** 和 **[src/utils/run_evaluation.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py) L3177-3248** — 构造 source_style_latent（VAE encode 源 content 图）并传递

#### 验证
1. Step1 评估现有实现：观察 cfg_target_scale 和 cfg_repulse_scale 对 clip/lpips 的影响
2. Step2 fiber 空间改造后，验证 base 不受影响（lpips 不恶化）
3. **预期**：source-repulsion 强行把 α（Endpoint 移动率）从 0.16 拉伸到 0.5+，打破白化

---

### A3：Logit-Normal 时间采样（非均匀时间步采样）

**洞察**：模型 99% 精力浪费在 t∈[0,0.3]（画面全是底噪/纯低频）。应改用 Logit-Normal 分布将 70% 样本集中在 t∈[0.6,0.95] 笔触生成关键期。

**类型**：训练期改动，需重新训练

#### 实施步骤

**步骤 1: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) L203-220 `_sample_t`** — 新增 Logit-Normal 分支

```python
def _sample_t(self, content: torch.Tensor) -> torch.Tensor:
    lo = max(0.0, min(1.0, self.t_min))
    hi = max(lo + 1e-4, min(1.0, self.t_max))

    if self.t_sampling_mode == "logit_normal":
        # Logit-Normal: u = sigmoid(N(μ, σ²))
        # 集中在 μ 附近，σ 控制集中度
        u_normal = torch.randn(content.shape[0], device=content.device, dtype=content.dtype)
        u_normal = u_normal * self.t_sampling_logit_std + self.t_sampling_logit_mean
        u = torch.sigmoid(u_normal).clamp(1e-6, 1.0 - 1e-6)
    elif self.t_sampling_beta_a > 0 and self.t_sampling_beta_b > 0:
        # Beta distribution sampling (existing)
        a = torch.tensor(self.t_sampling_beta_a, device=content.device)
        b = torch.tensor(self.t_sampling_beta_b, device=content.device)
        dist = torch.distributions.Beta(a, b)
        u = dist.sample([content.shape[0]]).to(dtype=content.dtype)
        u = u.clamp(1e-6, 1.0 - 1e-6)
    else:
        # Original uniform power sampling (backward compatible)
        u = torch.empty(content.shape[0], device=content.device, dtype=content.dtype).uniform_(0.0, 1.0)
        u = u.pow(self.t_sampling_power)

    return lo + (hi - lo) * u
```

**步骤 2: [src/losses620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py) L85-141 初始化** — 新增字段

```python
self.t_sampling_mode = str(getattr(self.bridge_cfg, "t_sampling_mode", "uniform_power")).strip().lower()
self.t_sampling_logit_mean = float(getattr(self.bridge_cfg, "t_sampling_logit_mean", 0.0))  # 负值偏向前段，正值偏向后段
self.t_sampling_logit_std = float(getattr(self.bridge_cfg, "t_sampling_logit_std", 1.0))    # 越小越集中
```

**步骤 3: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) BridgeConfig L526-530** — 新增 3 字段

```python
t_sampling_mode: str = "uniform_power"     # "uniform_power" | "beta" | "logit_normal"
t_sampling_logit_mean: float = 0.0          # Logit-Normal 的 μ（正值偏向 t→1）
t_sampling_logit_std: float = 1.0           # Logit-Normal 的 σ（越小越集中）
```

#### 验证
1. 训练 I7 初始化 + `t_sampling_mode="logit_normal", t_sampling_logit_mean=1.0, t_sampling_logit_std=0.5`（70% 样本集中在 t>0.6），2 epoch
2. probe 验证：训练时打印 t 分布直方图确认集中度
3. 5-style 评估对比 I7 baseline
4. **预期**：高频笔触 MSE 权重提升，clip 提升且 lpips 持平或微降

---

## 四、Phase B：架构大创新

### B4：Fiber-MoE Adapters（纤维层级的混合专家路由）

**洞察**：通用模型（0.70）与单风格模型（0.76）的 gap 是因为一个网络难同时拟合莫奈油画和浮世绘木版画。在 N1 块插入轻量 MoE，根据 DINO style global token 路由到 4-8 个小 expert。

**关键发现**：[blocks620.py:143-156](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/blocks620.py#L143) **已有 block-level soft MoE**（K/V experts + router）。B4 是将 MoE 应用到 N1 endpoint AdaIN 块（推理路径），不是 block-level。

**类型**：推理期 + 训练期混合（MoE router 需训练，但可复用 I7 ckpt 做冷启动 fine-tune）

#### 实施步骤

**步骤 1: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) N1 块（L676-874）** — 在 multiband_adain 和 patch_adain 分支后新增 MoE 分支

```python
elif n1_moe_enabled and style_latent is not None:
    # === FC-SB Phase 4 B4: Fiber-MoE Adapters ===
    # Router: style_dino_cls (global token) → expert weights
    # Experts: per-expert AdaIN 参数 (target_mean, target_std 的轻量变换)
    B_c, C_c, H_c, W_c = ep_fiber_curr.shape
    # style_global: (B, dim) — 从 style_dino_cls 或 style_latent 池化
    style_global = style_dino_cls.mean(dim=1) if style_dino_cls is not None else style_latent.mean(dim=[2,3])
    router_logits = self.n1_moe_router(style_global.float())  # (B, num_experts)
    router_weights = F.softmax(router_logits, dim=-1)  # (B, num_experts)
    # 每个 expert 产生 (target_mean_offset, target_std_scale) of shape (B, C)
    # 聚合: 加权平均
    target_mean_offset = torch.zeros(B_c, C_c, 1, 1, device=endpoint.device, dtype=endpoint.dtype)
    target_std_scale = torch.ones(B_c, C_c, 1, 1, device=endpoint.device, dtype=endpoint.dtype)
    for e_idx in range(self.n1_moe_num_experts):
        w_e = router_weights[:, e_idx].view(B_c, 1, 1, 1)
        offset_e = self.n1_moe_experts_offset[e_idx](style_global.float())  # (B, C)
        scale_e = F.softplus(self.n1_moe_experts_scale[e_idx](style_global.float()))  # (B, C)
        target_mean_offset = target_mean_offset + w_e * offset_e.unsqueeze(-1).unsqueeze(-1)
        target_std_scale = target_std_scale + w_e * (scale_e - 1.0).unsqueeze(-1).unsqueeze(-1)
    # 应用 MoE 调制的 AdaIN
    target_mean = style_fiber.mean(dim=[2, 3], keepdim=True) + target_mean_offset
    target_std = style_fiber.std(dim=[2, 3], keepdim=True).clamp_min(1e-6) * target_std_scale
    pred_mean = ep_fiber_curr.mean(dim=[2, 3], keepdim=True)
    pred_std = ep_fiber_curr.std(dim=[2, 3], keepdim=True).clamp_min(1e-6)
    ep_fiber_norm = (ep_fiber_curr - pred_mean) / pred_std
    ep_fiber_matched = ep_fiber_norm * target_std + target_mean
    self.last_debug["n1_moe_router_entropy"] = -(router_weights * router_weights.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
```

**步骤 2: [src/model620.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py) __init__** — 新增 MoE 模块定义

```python
# === FC-SB Phase 4 B4: N1 Fiber-MoE ===
self.n1_moe_enabled = bool(getattr(model_cfg, 'n1_moe_enabled', False))
self.n1_moe_num_experts = int(getattr(model_cfg, 'n1_moe_num_experts', 4))
if self.n1_moe_enabled:
    self.n1_moe_router = nn.Sequential(
        nn.LayerNorm(self.dim),
        nn.Linear(self.dim, 64),
        nn.SiLU(),
        nn.Linear(64, self.n1_moe_num_experts),
    )
    self.n1_moe_experts_offset = nn.ModuleList([
        nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.latent_channels))
        for _ in range(self.n1_moe_num_experts)
    ])
    self.n1_moe_experts_scale = nn.ModuleList([
        nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.latent_channels))
        for _ in range(self.n1_moe_num_experts)
    ])
```

**步骤 3: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py) ModelConfig** — 新增字段

```python
n1_moe_enabled: bool = False              # FC-SB Phase 4 B4: N1 块 MoE 路由
n1_moe_num_experts: int = 4               # expert 数量
```

#### 验证
1. 冷启动 I7 ckpt + 新增 MoE 模块（随机初始化），fine-tune 1 epoch
2. probe 验证：`n1_moe_router_entropy` 非零且不同 style 路由权重不同
3. 5-style 评估对比 I7 baseline
4. **预期**：用极小参数代价换取接近 per-style 模型的上限能力，clip 突破 0.75

---

### B2：原生频域微分方程（Native Spectral/Wavelet ODE）POC

**洞察**：当前 FC-SB 是在欧氏空间算完速度 v 再人工投影/剥离低频（[model620.py:889](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L889) `v_fiber = v_pred - lp(v_pred)`），数学上是事后补救。应在进入网络前用 2D DWT 把 latent 拆成 LL/LH/HL/HH，主干直接输出 4 个独立速度场，推理时在频域独立 Euler 积分，最后 iDWT 合成。

**类型**：架构大创新，POC 阶段独立模块，不污染现有代码

#### 实施步骤

**步骤 1: 新建 `src/wavelet620.py`** — 提取 Haar 小波为独立模块

```python
import torch
import torch.nn.functional as F

class HaarTransform2D(nn.Module):
    """2D Haar 离散小波变换 (单级)。
    输入: (B, C, H, W) — H, W 必须是偶数
    输出: 4 个 (B, C, H/2, W/2) 张量: LL, LH, HL, HH
    """
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ll = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
        lh = (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] - x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
        hl = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] + x[..., 1::2, 0::2] - x[..., 1::2, 1::2]) / 2.0
        hh = (x[..., 0::2, 0::2] - x[..., 0::2, 1::2] - x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) / 2.0
        return ll, lh, hl, hh

class InverseHaarTransform2D(nn.Module):
    """2D 逆 Haar 变换 (单级)。
    输入: 4 个 (B, C, H/2, W/2) 张量
    输出: (B, C, H, W) — 用 nearest 上采样近似 iDWT
    """
    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        H, W = target_size
        ll_up = F.interpolate(ll, size=(H, W), mode='nearest')
        lh_up = F.interpolate(lh, size=(H, W), mode='nearest')
        hl_up = F.interpolate(hl, size=(H, W), mode='nearest')
        hh_up = F.interpolate(hh, size=(H, W), mode='nearest')
        return ll_up + lh_up + hl_up + hh_up
```

**步骤 2: 新建 `src/spectral_bridge620.py`** — 频域 ODE 主干 POC

```python
class SpectralBridge620(nn.Module):
    """原生频域流匹配 POC。
    主干输出 4 个独立速度场 v_LL, v_LH, v_HL, v_HH。
    训练时 v_LL loss 权重极小（保 content 结构），v_HH 权重大（学笔触）。
    推理时频域独立 Euler 积分，最后 iDWT 合成。
    """
    def __init__(self, model_cfg, bridge_cfg=None):
        super().__init__()
        self.haar = HaarTransform2D()
        self.ihaar = InverseHaarTransform2D()
        # 4 个独立速度场 (共享 backbone, 4 个 head)
        # POC 阶段: 复用 SpatialBridge620 的 blocks, 加 4 个 velocity head
        # ... (具体实现细节在 POC 阶段细化)

    def forward(self, x_t, t, style_id, ...):
        # 频域分解
        ll, lh, hl, hh = self.haar(x_t)
        # 各频带独立 velocity 预测
        v_ll = self.v_head_ll(self.backbone(ll, t, style_id, ...))
        v_lh = self.v_head_lh(self.backbone(lh, t, style_id, ...))
        v_hl = self.v_head_hl(self.backbone(hl, t, style_id, ...))
        v_hh = self.v_head_hh(self.backbone(hh, t, style_id, ...))
        return v_ll, v_lh, v_hl, v_hh

    @torch.no_grad()
    def integrate_transport(self, x, style_id, num_steps=8, ...):
        h = x
        for idx in range(num_steps):
            t_curr = idx / num_steps
            t_next = (idx + 1) / num_steps
            dt = t_next - t_curr
            v_ll, v_lh, v_hl, v_hh = self.forward(h, t_curr, style_id, ...)
            # 频域独立 Euler
            ll, lh, hl, hh = self.haar(h)
            ll_new = ll + v_ll * dt
            lh_new = lh + v_lh * dt
            hl_new = hl + v_hl * dt
            hh_new = hh + v_hh * dt
            # iDWT 合成
            h = self.ihaar(ll_new, lh_new, hl_new, hh_new, h.shape[-2:])
        return h
```

**步骤 3: 新建 `src/spectral_losses620.py`** — 频域独立 loss

```python
class SpectralLoss620:
    """频域独立 loss。
    v_LL loss 权重极小（保 content），v_HH 权重大（学笔触）。
    """
    def __init__(self, w_ll=0.01, w_lh=0.5, w_hl=0.5, w_hh=2.0):
        self.w_ll = w_ll  # content 结构几乎不学
        self.w_lh = w_lh
        self.w_hl = w_hl
        self.w_hh = w_hh  # 笔触权重最大

    def __call__(self, v_pred, v_target):
        v_ll_p, v_lh_p, v_hl_p, v_hh_p = v_pred
        v_ll_t, v_lh_t, v_hl_t, v_hh_t = v_target
        loss_ll = F.mse_loss(v_ll_p, v_ll_t)
        loss_lh = F.mse_loss(v_lh_p, v_lh_t)
        loss_hl = F.mse_loss(v_hl_p, v_hl_t)
        loss_hh = F.mse_loss(v_hh_p, v_hh_t)
        return self.w_ll * loss_ll + self.w_lh * loss_lh + self.w_hl * loss_hl + self.w_hh * loss_hh
```

**步骤 4: [src/config_schema.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/config_schema.py)** — 新增 spectral mode 字段

```python
spectral_ode_enabled: bool = False        # FC-SB Phase 4 B2: 原生频域 ODE
spectral_w_ll: float = 0.01               # LL 频带 loss 权重
spectral_w_lh: float = 0.5
spectral_w_hl: float = 0.5
spectral_w_hh: float = 2.0
```

**步骤 5: POC 实验**
- 在 5-style 数据集上训练 SpectralBridge620 POC（2 epoch）
- 对比 FC-SB baseline 的 clip/lpips
- **预期**：特征解耦在网络第一层物理完成，LPIPS 锁死，Style 评分无上限飙升

#### 验证
1. 单元测试：HaarTransform2D + InverseHaarTransform2D 可逆性（重建误差 < 1e-6）
2. POC 训练：v_LL loss 极小（content 几乎不动），v_HH loss 主导
3. 推理评估：与 FC-SB baseline 对比

---

### B1：双流独立流匹配（Dual-Stream Flow Matching / MMDiT）— 可选

**洞察**：Cross-Attention 的 Gate Collapse（收敛到 0.048）是因为把 style 强行塞入 content 特征流，content 流为保结构主动关闭水管。应设计两个平行求解流：Stream A 专跑 Content（H×W 空间特征），Stream B 专跑 Style（DINO token 序列），每 block 互相做 Cross-Attention 交换信息。

**类型**：架构大创新，依赖 B2 POC 结果决定是否实施

**实施步骤**：仅在 B2 POC 成功后展开。POC 阶段仅做理论设计文档，不写代码。

---

### B3：能量引导薛定谔桥（Energy-Guided SB）— 可选

**洞察**：MSE 损失假设分布高斯单峰（导致均值平凡解）。训练极轻量"风格打分器" E_φ(x, style)，推理时加入能量梯度 Langevin 动力学：
$$dx = v_{fiber}dt + \sigma dW_t - \nabla_x E_\phi(x, style)dt$$

**类型**：架构大创新，依赖 B2 POC 结果决定是否实施

**实施步骤**：仅在 B2 POC 成功后展开。POC 阶段仅做理论设计文档，不写代码。

---

## 五、假设与决策

### 5.1 核心假设

1. **A1 复用 U4 ckpt**：时频调度是推理期改动，不需重新训练
2. **A4 替换 W2**：W2 hinge loss 已证实失效（step=1 归零），A4 是 W 方向重生
3. **A2 Step1 零代码评估**：现有 `integrate_transport_cfg` 已实现 source-repulsion，只需配置评估
4. **A3 Logit-Normal 优于 Beta**：Logit-Normal 在 SD3 中已验证优于 Beta，集中度更可控
5. **B4 复用现有 MoE 代码模式**：blocks620.py 已有 soft MoE，B4 是将其应用到 N1 块
6. **B2 POC 独立模块**：不污染现有代码，POC 失败可整体丢弃

### 5.2 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| A1 mid 调度方式 | 线性升温（非指数） | mid 对 lpips 敏感，线性更保守 |
| A1 hh 调度方式 | 指数爆发（二次方） | hh 对 lpips 不敏感，可激进 |
| A4 方差匹配频带 | 默认 "hh" | hh 对 lpips 不敏感，安全起步 |
| A2 ep_null 修复 | style_latent=None | 现有 L1006 仍传 style_latent，非真正 unconditional |
| A3 Logit-Normal 参数 | μ=1.0, σ=0.5 | 70% 样本集中在 t>0.6 |
| B4 MoE 位置 | N1 块（推理路径） | block-level MoE 已存在，N1 是 fiber 统计匹配点 |
| B2 POC 独立模块 | spectral_bridge620.py | 不污染现有 model620.py |

### 5.3 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| A1 mid 早期锁死导致色彩基调学不到 | 中 | tf_mid_lock_threshold 可调，先用 0.5 再降到 0.3 |
| A4 方差匹配仍梯度失效 | 低 | 方差是连续值，不会像 hinge 一步归零 |
| A2 source-repulsion 扰动 base | 中 | Step2 改造为 fiber 空间，base 不受影响 |
| A3 集中后段导致前段欠拟合 | 中 | σ=0.5 保证前段仍有 30% 样本 |
| B4 MoE 路由坍缩 | 中 | probe 监控 router_entropy，加 load balancing loss |
| B2 POC 训练不稳定 | 高 | POC 阶段独立模块，失败可丢弃 |

---

## 六、验证步骤

### 6.1 每个方向的验证协议

1. **probe-first 原则**：任何改动后，先用 probe 验证开关生效（runtime_observability 非零），再做完整评估
2. **5-style 标准评估**：所有方向在 5-style all_pairs_overview（25 对含 identity）上评估
3. **baseline 对比**：
   - 推理期改动（A1/A2/B4）对比 U4(α0.1) baseline（clip=0.7225, lpips=0.3660）
   - 训练期改动（A3/A4/B2）对比 I7 baseline（clip=0.7017, lpips=0.3625）
4. **WFI 检查**：任何优化必须先通过 WFI < 0.40 白化验收

### 6.2 实施顺序与里程碑

**Week 1**：
- A1 完成 → probe 验证 tf_hh_scale_dyn 动态 → 5-style 评估
- A4 完成 → probe 验证 output_variance_loss 非零 → 训练 2 epoch → 5-style 评估
- A2 Step1 完成 → 9 组配置评估 → 决定是否进入 Step2

**Week 2**：
- A2 Step2 完成 → fiber 空间 source-repulsion → 5-style 评估
- A3 完成 → 训练 2 epoch → 5-style 评估
- B4 完成 → fine-tune 1 epoch → 5-style 评估

**Week 3-4**：
- B2 POC 完成 → 单元测试 → POC 训练 → 对比 FC-SB baseline
- 根据 B2 结果决定是否展开 B1/B3

### 6.3 成功标准

| 方向 | 成功标准 | 失败处理 |
|------|---------|---------|
| A1 | clip 持平 + lpips 下降 ≥0.005 | 调整 tf_hh_ramp_start |
| A4 | clip 提升 ≥0.01 + lpips 不恶化 | 调整 w_output_variance 权重 |
| A2 | α（Endpoint 移动率）从 0.16 提升到 0.3+ | 放弃 source-repulsion，仅用 K1 |
| A3 | clip 提升 ≥0.01 | 调整 logit_mean/std |
| B4 | clip 突破 0.75 | 增加专家数或加 load balancing |
| B2 | LPIPS 锁死 + clip 无上限 | POC 失败则归档，不进入 B1/B3 |

---

## 七、文件改动清单

### 7.1 新增文件

| 文件 | 方向 | 用途 |
|------|------|------|
| `src/wavelet620.py` | B2 | Haar 小波独立模块 |
| `src/spectral_bridge620.py` | B2 | 频域 ODE 主干 POC |
| `src/spectral_losses620.py` | B2 | 频域独立 loss |

### 7.2 修改文件

| 文件 | 方向 | 改动 |
|------|------|------|
| `src/config_schema.py` | A1/A2/A3/A4/B2/B4 | 新增配置字段 |
| `src/model620.py` | A1/A2/B4 | 动态调度 + fiber CFG + N1 MoE |
| `src/losses620.py` | A3/A4 | Logit-Normal 采样 + 方差匹配 loss |
| `src/utils/inference.py` | A2 | source_style_latent 传递 |
| `src/utils/run_evaluation.py` | A2 | source_style_latent 构造 |

---

## 八、总结

本计划在 **工程调优**（A1-A4 压榨现有架构极限）和 **架构创新**（B1-B4 重构底层数学流形）两个层面双管齐下：

- **短期突围（Week 1）**：A1 时频调度 + A4 方差匹配，最低风险最高 ROI
- **中期突破（Week 2）**：A2 fiber CFG + A3 时间采样 + B4 MoE，挑战 0.75 clip
- **长期护城河（Week 3-4）**：B2 原生频域 ODE POC，如成功则可冲击顶会 Oral 亮点

所有改动遵循 **probe-first 原则** 和 **5-style 标准评估协议**，确保每步可验证、可回退。
