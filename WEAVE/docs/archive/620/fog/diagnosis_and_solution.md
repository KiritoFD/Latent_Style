# 620 白化/雾化 深度诊断与解决方案

Date: 2026-06-21

## 1. 问题定义

620模型生成的风格迁移图片呈现系统性白化/雾化：
- 低对比度、低饱和度、高亮度
- 视觉上像蒙了一层白雾
- identity和style_transfer都受影响

**量化基准** (Seedream repaired750, 30 images):
- avg_contrast_ratio ≈ 0.4244
- avg_dynamic_range ≈ 0.6218
- avg_saturation_mean ≈ 0.3642
- avg_wfi_score ≈ 0.1581

**620当前状态** (intrinsic_v2 e8, lowswd_formal e3, film_formal e8):
- clip_style ≈ 0.702-0.707 (超过IDT但白化严重)
- metrics.csv中clip_image_vector的cos_sim_to_mean ≈ 0.79-0.81 (方向高度一致=白化信号)

## 2. 已确认的机理 (Probe-Backed)

### 2.1 白化起源于endpoint预测，不是solver

| 阶段 | 结论 |
|------|------|
| predict_endpoint(t=0) | **白化已存在** |
| integrate_transport | 不是白化来源，甚至部分补偿 |
| VAE decode | 不放大白化 |

### 2.2 核心病理: Endpoint Shrinkage

Hypothesis probe (targetlinear e8) 在 t=0.0:
- **latent_alpha_mean ≈ 0.1633** — endpoint只移动了目标方向的16%
- **high_alpha_mean ≈ -0.0501** — 高频endpoint方向错误/消失
- **low_alpha_mean ≈ 0.4263** — 低频弱但仍有目标方向
- **style_sensitivity_latent ≈ 8.75** — style信号存在但未转化为endpoint位移

**数学表述**:
```
delta = y - x  (target方向)
move = y_hat - x  (预测位移)
alpha = <move, delta> / ||delta||^2 ≈ 0.16
```
即endpoint严重收缩，只走了16%的目标方向路程。

### 2.3 时间趋势

| t | latent_alpha | high_alpha | img_std |
|---|-------------|------------|---------|
| 0.0 | 0.163 | -0.050 | 0.069 |
| 0.5 | 0.561 | 0.499 | 0.199 |
| 0.875 | 0.904 | 0.893 | 0.275 |

**关键发现**: t越大，alpha越接近1（正确）。白化集中在t≈0的source端。

## 3. 已尝试的修复及失败原因

| 修复 | 结果 | 失败原因 |
|------|------|----------|
| lowfreqfix (endpoint lowpass loss) | velocity_abs从0.15降到0.016 | 惩罚了低频动态，endpoint更靠近source |
| target_linear vertical path | 早期有效(e1-3)，晚期回归(e6-8) | 晚期endpoint动态范围再次坍缩 |
| endpointaux (source-endpoint loss) | img_std恢复但to_source_rms=0.055 | endpoint坍回source，不是真正修复 |
| tlow (低t采样偏好) | 同上 | 强调低t导致模型选择"不动"来避免惩罚 |
| endpoint_lowhigh (分离低高频head) | style_sensitivity降到0.003 | 无style注入，endpoint坍回source |
| endpoint_stylehead (style注入endpoint head) | style_sensitivity恢复到0.23 | 仍不够，alpha仍为负 |
| FiLM endpoint head (formal 8 epoch) | alpha=0.12, style_sens=10.1 | style信号存在但endpoint位移仍不足 |
| direction loss | alpha=-0.007, style_sens=0.002 | 完全坍回source |

## 4. 根因分析

### 4.1 为什么所有修复都失败？

**统一解释**: 当前架构的优化landscape存在一个"shrinkage basin"——

1. **velocity参数化是根本问题**: `velocity = (endpoint - x) / (1-t)`
   - 当t→0时，denominator→1，velocity≈endpoint-x
   - 但训练时velocity的L2正则/weight decay倾向于让velocity变小
   - 小velocity = 小endpoint位移 = shrinkage

2. **style gate初始化过小** (0.05): cross-attention的style信号被gate压制
   - 即使style_sensitivity高，gate=0.05意味着style信息只贡献5%

3. **endpoint head的零初始化**: 所有endpoint head的最后一层weight初始化为0或1e-3
   - 训练初期endpoint ≈ x (identity)
   - 优化从identity出发，倾向于找到"少动"的局部最优

4. **GroupNorm(1) = LayerNorm**: 在endpoint head中使用GroupNorm(1)
   - 这会归一化feature map的均值和方差
   - 直接抑制了endpoint预测的动态范围

### 4.2 为什么晚期训练回归？

target_linear早期有效是因为：
- 低频路径允许endpoint向target移动
- 训练初期velocity还不大

晚期回归是因为：
- velocity_abs持续增长(0.15→0.40)
- 但endpoint的image-space对比度反而下降
- 说明velocity增长的方向不完全是target-facing的
- 模型找到了一个"高频抖动+低频不动"的局部最优

## 5. 解决方案

### 5.1 核心修复: Endpoint-First Parameterization with Style-FiLM

**设计原则**:
1. 直接预测endpoint，velocity从endpoint推导
2. Style通过FiLM调制endpoint head的feature map（不是加性偏移）
3. 去掉endpoint head中的GroupNorm（避免动态范围被压缩）
4. 增大style gate初始值

**具体实现**:

```python
class StyleFiLMEndpointHead(nn.Module):
    """FiLM-modulated endpoint head WITHOUT GroupNorm."""
    def __init__(self, dim, latent_channels, style_dim, style_hidden_dim):
        super().__init__()
        # 不用GroupNorm！直接用conv trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        # FiLM: style → gamma, beta
        self.film = nn.Sequential(
            nn.LayerNorm(style_dim),
            nn.Linear(style_dim, style_hidden_dim),
            nn.SiLU(),
            nn.Linear(style_hidden_dim, dim * 2),
        )
        # 输出投影
        self.proj = nn.Conv2d(dim, latent_channels, kernel_size=3, padding=1)
        # 关键: 非零初始化，让初始endpoint偏向target方向
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, h, style_embed):
        # Trunk feature
        feat = self.trunk(h)
        # FiLM modulation
        film_params = self.film(style_embed.float()).to(dtype=h.dtype)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        feat = (1.0 + gamma) * feat + beta
        feat = F.silu(feat)
        return self.proj(feat)
```

### 5.2 Velocity Scaling约束

在loss中加入velocity magnitude约束，防止shrinkage:

```python
# 在losses620.py中
velocity_target_ratio = velocity_abs / (target_velocity_abs + 1e-8)
loss_velocity_scale = F.mse_loss(velocity_target_ratio, torch.ones_like(velocity_target_ratio))
# 权重: 0.1
```

### 5.3 Style Gate初始化调整

```python
# 从0.05改为0.3
style_gate_init = 0.3
```

### 5.4 Endpoint Head非零初始化

让endpoint head的初始输出偏向target方向:
```python
# 不用zero-init，用稍大的normal init
nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
```

## 6. 实验计划

### Phase A: 白化修复验证 (优先级最高)

| ID | 实验 | 预期时间 |
|----|------|----------|
| A1 | StyleFiLM endpoint head (无GroupNorm) + gate=0.3 | 1 epoch smoke |
| A2 | A1 + velocity_scale_loss (w=0.1) | 1 epoch smoke |
| A3 | A1 + target_linear + velocity_scale_loss | 3 epoch formal |
| A4 | 最优组合 + WFI benchmark vs Seedream | 8 epoch formal |

**验收标准**:
- WFI score ≤ Seedream IDT水平 (wfi ≤ 0.20)
- clip_style ≥ 0.70 (不降)
- endpoint alpha(t=0) ≥ 0.5

### Phase B: DINO评估 (白化修复后)

| ID | 实验 | 判断标准 |
|----|------|----------|
| B1 | intrinsic_style (无DINO) vs DINO | clip_style差 < 0.01 → 砍DINO |
| B2 | DINO多尺度 [4,8,11] | style +0.005 → 保留，否则砍 |

### Phase C: 620后续优化 (白化修复后)

| ID | 实验 | 来源 |
|----|------|------|
| C1 | 加入text conditioning | phase4_plan |
| C2 | Cross-attention Q=bottleneck only | phase4_plan D2 |
| C3 | Per-region SWD | phase4_plan B2 |
| C4 | Skip α per-layer | phase4_plan C2 |

## 7. WFI指标体系

### 图像空间指标 (需要生成图片)

| 指标 | 公式 | 健康值(Seedream) | 白化信号 |
|------|------|-----------------|----------|
| contrast_ratio | luminance_std / luminance_mean | ≈0.42 | <0.30 |
| dynamic_range | (p95-p5)/(p95+p5) | ≈0.62 | <0.45 |
| saturation_mean | HSV saturation mean | ≈0.36 | <0.25 |
| wfi_score | 1-(0.4*cr+0.3*sr+0.3*dr) | ≈0.16 | >0.35 |

### 潜空间指标 (从metrics.csv可计算)

| 指标 | 健康值 | 白化信号 |
|------|--------|----------|
| cos_sim_to_mean (clip vectors) | <0.6 | >0.75 |
| latent_alpha_mean (t=0) | >0.5 | <0.3 |
| high_alpha_mean (t=0) | >0.3 | <0.0 |

## 8. 当前阻塞项

1. **远程eval图片未保存**: images/目录为空，需要重新跑eval with --save_generated_images
2. **WFI benchmark需要图片**: 无法在无图片情况下计算图像空间WFI
3. **下一步**: 先在远程跑一个带图片保存的eval，获取WFI基准数据
