# 去除 DINO + 突破白化问题 Spec

## Why

当前模型输出存在明显的**白化/雾化**现象（从 summary_grid.png 可直观确认）：
- Early Renaissance 行：所有目标风格输出几乎完全相同，呈现统一的灰白色调
- 跨风格转换时颜色饱和度严重不足，对比度塌缩
- 模型倾向于输出"所有风格的平均值"而非特定目标风格

**核心假设**：DINO 作为外部先验，其语义特征可能**加剧了白化**：
1. DINO ViT 特征经过 adapter → 256 memory tokens → cross-attention K/V 的长链路，每一步都可能平均化风格特异性
2. DINO 特征是"语义级"的（物体、场景），不是"风格级"的（笔触、色彩分布、纹理统计），用它驱动风格迁移可能引入了错误的对齐信号
3. 去掉 DINO 后改用 latent 空间直接作为 style condition，可能保留更多原始风格信息

## What Changes

### 1. 去除 DINO 条件路径
- 将 `style_condition_source` 从 `"target_dino_patches"` 切换为 `"latent"`（使用 intrinsic_style_cnn 从 latent 提取 style tokens）
- 移除 `style_dino_adapter_enabled` / `style_dino_adapter_hidden_dim` / `style_dino_adapter_scale` 相关逻辑（配置保留但默认关闭）
- StyleConditioner620 不再接收 DINO patches/cls 输入

### 2. 目标函数调整（针对白化）
基于图片观察到的具体症状，调整 loss 以对抗白化：

**症状 A：全局对比度塌缩** → 加入 **对比度保持损失**
```python
# 对 generated 和 target 都计算 std，惩罚生成的 std 过低
gen_std = z_hat1.std(dim=[1,2,3]).mean()
tgt_std = projected_target.std(dim=[1,2,3]).mean()
contrast_loss = F.relu(tgt_std * 0.8 - gen_std)  # 允许略低但不允许太低
```

**症状 B：颜色向灰色坍缩** → 加入 **通道方差保持损失**
```python
# 每个通道的方差不应都趋于相同值
gen_ch_var = z_hat1.var(dim=[2,3])  # (B, C)
ch_var_loss = -gen_ch_var.log().mean()  # 鼓励各通道有不同方差（最大化熵）
```

**症状 C：高频细节丢失** → 增强 edge loss 权重或加入高频能量约束
```python
high_freq_energy = (z_hat1 - avg_pool(z_hat1)).pow(2).mean()
target_high_freq = (projected_target - avg_pool(projected_target)).pow(2).mean()
hf_ratio_loss = F.relu(target_high_freq * 0.5 - high_freq_energy)
```

### 3. 向后兼容
- 所有新增参数默认值为 0 或 False，不影响现有行为
- `style_condition_source` 切换为 latent 时自动跳过 DINO 相关代码路径

## Impact

- Affected code: `src/model620.py`（style_condition_source 切换）、`src/style_encoder620.py`（StyleConditioner620）、`src/losses620.py`（新增 anti-whitening loss）、`src/config_schema.py`（新参数）、实验 config
- Affected specs: 继承 `620-whitening-fix` 的 E4.3（DINO 去留决策）

## ADDED Requirements

### Requirement: 支持 latent-based style condition（无 DINO）

系统 SHALL 支持通过 `style_condition_source = "latent"` 完全绕过 DINO，使用 CNN 从 latent 空间提取 style tokens。

#### Scenario: 无 DINO 训练
- **WHEN** 配置 `style_condition_source = "latent"`
- **THEN** 不加载 DINO 模型、不计算 DINO features、不传入 dino patches/cls 给 bridge
- **THEN** 使用 intrinsic_style_cnn + intrinsic_style_proj 生成 style tokens 和 global embedding
- **THEN** 训练和推理均正常工作，无额外依赖

### Requirement: Anti-whitening contrast preservation loss

系统 SHALL 提供可配置的全局对比度保持损失，防止生成图像动态范围塌缩。

#### Scenario: 对比度损失生效
- **WHEN** `w_contrast_preserve > 0`
- **THEN** 计算 generated 和 target 的 pixel-level std ratio
- **THEN** 当 gen_std 显著低于 target_std 时产生梯度惩罚
- **THEN** 默认权重为 0（向后兼容）

### Requirement: Anti-whitening channel variance loss

系统 SHALL 提供通道方差多样性损失，防止所有通道退化为相同方差。

#### Scenario: 通道方差损失生效
- **WHEN** `w_channel_variance > 0`
- **THEN** 计算每个空间位置的通道方差分布的熵/对数方差
- **THEN** 惩罚通道方差过于均匀的情况
- **THEN** 默认权重为 0

### Requirement: Anti-whitening high-frequency energy loss

系统 SHALL 提供高频能量保持损失，防止高频细节被过度平滑。

#### Scenario: 高频能量损失生效
- **WHEN** `w_hf_energy > 0`
- **THEN** 通过高通滤波（x - avg_pool(x)）计算高频分量能量
- **THEN** 惩罚生成的高频能量显著低于目标的情况
- **THEN** 默认权重为 0

### Requirement: 图片级白化诊断

系统 SHALL 在每次 full eval 时输出关键的白化相关指标和可视化对比。

#### Scenario: 白化诊断输出
- **WHEN** 执行 full evaluation
- **THEN** 输出以下指标：global_std（生成 vs 目标）、channel_std_distribution、hf_energy_ratio、亮度均值、饱和度均值
- **THEN** 保存 source / target / generated 并排对比图
- **THEN** 明确标注当前白化程度评分

## MODIFIED Requirements

### Requirement: DINO 去留决策（来自 620-whitening-fix E4.3）

将原计划中的"DINO 对照实验"升级为核心任务：不仅做对照，而是以**无 DINO 为主力方向**进行突破。

- 原：先做无 DINO 对照，收益不显著则砍掉
- 新：**默认使用无 DINO（latent）模式**，验证是否能改善白化；仅在有明确证据表明 DINO 有帮助时才恢复

## Open Questions

- [ ] intrinsic_style_cnn 的容量是否足以替代 DINO 256 tokens 的表达能力？是否需要增大？
- [ ] 无 DINO 后 style tokens 的维度/数量如何设置？当前是 256 memory tokens，是否需要调整？
- [ ] 三个 anti-whitening loss 的最优权重比例是什么？是否需要自适应调度？
- [ ] 去掉 DINO 是否会导致 clip_style 下降（因为 DINO 特征本身携带语义信息）？
