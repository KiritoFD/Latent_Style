# E5-E12 反退化硬核突破实验 Spec

## Why

E4 实验证明 RMSNorm + vmag=2.0 成功将 velocity_std 从 ~0.05 提升到 ~0.896（18倍），但 LPIPS 从 E2 的 0.3326 恶化到 0.3735。**反退化方向正确但需要更精细的多轴协同修复。**

基于四大退化吸引子（Gate Collapse / GN白化 / Training-Output Mismatch / Endpoint Shrinkage）的数学诊断，需要在**流路径、注意力机制、归一化层、训练目标**四个维度同时动刀，打破"自强化退化吸引子"。

## What Changes

### 核心修改范围

1. **流路径改造** (`src/losses620.py`): 新增 VP-Flow 球面插值路径模式
2. **Cross-Attention 改造** (`src/blocks620.py`): 新增 Top-K 截断掩码注意力
3. **归一化层** (`src/blocks620.py`): E4 已完成 RMSNorm，新增 AdaLN-Zero 初始化
4. **Loss 函数** (`src/losses620.py`): 新增方向余弦损失 + 高频L1回归
5. **时间采样** (`src/losses620.py`): 新增 Late-Stage Beta 采样分布
6. **Endpoint Head** (`src/model620.py`): 新增方差注入支路
7. **门控机制** (`src/blocks620.py`): 新增 Residual-First Gating 模式
8. **CFG 训练** (`src/losses620.py` / `src/trainer.py`): 条件丢弃正则化
9. **配置扩展** (`src/config_schema.py`): 所有新参数的 schema 定义

### 实验矩阵（按优先级排序）

| 实验 | 组合方案 | 核心改动 | 预期效果 |
|------|---------|---------|----------|
| **E5** | VP-Flow (#1) | `_vertical_state` 用 cos/sin 替代线性插值 | 方差恒定，消灭中间步发灰 |
| **E6** | Top-K Attention (#7) | Cross-Attention 加 Top-K 截断掩码 | 注意力锐利，笔触清晰 |
| **E7** | **VP-Flow + Top-K + RMSNorm** (1+7+5) | 三斧组合 | **推荐组合，预期相变** |
| **E8** | Directional Cosine Loss (#2) | Loss 加余弦方向惩罚 | 速度场精确指向目标 |
| **E9** | Late-Stage Sampling (#4) | t 采样改为 Beta(3,1) 偏向后期 | 后期风格注入增强 |
| **E10** | Variance Injection (#6) | Endpoint 预测 exp(σ_style) 缩放 | 对比度撑开，抗白化 |
| **E11** | Residual-First Gate (#9) | 门控逻辑反转 | 吸收风格成阻力最小路径 |
| **E12** | CFG Training (#10) | 15%概率条件丢弃 | 推理时可暴力外推 |

## Impact

- Affected specs: phase3-breakout-deep (继承), trivial-solution-breakout (继承)
- Affected code:
  - `src/losses620.py` — 流路径、Loss函数、时间采样
  - `src/blocks620.py` — 注意力机制、门控、归一化
  - `src/model620.py` — Endpoint head 方差注入
  - `src/config_schema.py` — 新参数定义
  - `src/trainer.py` — CFG 条件丢弃

## ADDED Requirements

### Requirement: VP-Flow 球面插值路径

系统 SHALL 提供 `bridge_path_mode = "spherical_vp"` 模式，将标准线性插值：
$$x_t = (1-t)x_0 + tx_1$$
替换为球面插值：
$$x_t = \cos(\frac{\pi}{2}t) \cdot x_0 + \sin(\frac{\pi}{2}t) \cdot x_1$$

#### Scenario: VP-Flow 训练与推理
- **WHEN** 配置 `bridge_path_mode = "spherical_vp"`
- **THEN** `_vertical_state()` 使用 cos/sin 插值，特征图方差在全程保持恒定
- **THEN** target_velocity 相应调整为 $v_t = -\frac{\pi}{2}\sin(\frac{\pi}{2}t) \cdot x_0 + \frac{\pi}{2}\cos(\frac{\pi}{2}t) \cdot x_1$
- **THEN** 默认 `"linear"` 模式行为不变（向后兼容）

### Requirement: Top-K 截断掩码 Cross-Attention

系统 SHALL 提供 `style_attn_topk > 0` 模式，在 softmax 前对每行 attention logits 强行截断：

```python
if self.style_attn_topk > 0:
    _, topk_indices = logits.topk(self.style_attn_topk, dim=-1)
    mask = torch.zeros_like(logits)
    mask.scatter_(-1, topk_indices, 1.0)
    logits = logits.masked_fill(mask == 0, float('-inf'))
```

#### Scenario: Top-K 注意力锐利化
- **WHEN** 配置 `style_attn_topk = 4`
- **THEN** 每个 Query 只关注 4 个最相关的 Style Token
- **THEN** cross_attn_entropy 显著下降（从 ~5.53 降至更低）
- **THEN** 默认 `topk = 0` 表示不截断（向后兼容）

### Requirement: 方向余弦损失

系统 SHALL 在 Flow Matching MSE 基础上增加可选的方向余弦惩罚：

$$L_{total} = \|v_{pred} - v_{target}\|^2 + \lambda_{dir} \cdot (1 - \cos(v_{pred}, v_{target}))$$

#### Scenario: 方向约束生效
- **WHEN** 配置 `w_directional_cosine > 0`
- **THEN** 即使模型缩短向量模长，仍被强制指向正确方向
- **THEN** endpoint_alpha 应提升（方向准确后模长可逐步恢复）

### Requirement: Late-Stage 时间采样

系统 SHALL 支持 Beta 分布时间采样：

$$t \sim \text{Beta}(\alpha, \beta), \quad \text{clamp to } [t_{min}, t_{max}]$$

#### Scenario: 后期偏重采样
- **WHEN** 配置 `t_sampling_beta_a=3, t_sampling_beta_b=1`
- **THEN** 70%+ 的采样集中在 t ∈ [0.7, 1.0]
- **THEN** 默认 uniform 采样不变（向后兼容）

### Requirement: Endpoint 方差注入

系统 SHALL 在 endpoint_lowhigh 模式下可选地预测通道级标准差缩放因子：

$$x_{final} = x_{base} \odot \exp(\sigma_{style}) + \Delta_{style}$$

#### Scenario: 对比度恢复
- **WHEN** 配置 `endpoint_variance_injection = true`
- **THEN** Endpoint Head 输出额外的 sigma 分支
- **THEN** 高频能量指标应显著提升

### Requirement: Residual-First 门控

系统 SHALL 提供 `style_gate_mode = "residual_first"` 模式，反转门控语义：

$$h_{new} = \sigma(gate) \cdot h_{old} + Attn(Q,K,V), \quad gate_{init} = 4.0$$

#### Scenario: 风格吸收成为默认路径
- **WHEN** 使用 residual_first 门控
- **THEN** 不主动抑制风格时，gate ≈ 0.98，内容几乎原样通过
- **THEN** 要消除风格必须主动降低 gate（成本更高）

### Requirement: CFG 条件丢弃训练

系统 SHALL 在训练时以可配置概率将 style tokens 替换为 null tokens。

#### Scenario: CFG 正则化
- **WHEN** 配置 `cfg_dropout_prob > 0`
- **THEN** 每步以该概率用 null tokens 替换 style input
- **THEN** 推理时可使用 $v_{final} = v_{uncond} + \omega(v_{cond}-v_{uncond})$ 外推

## MODIFIED Requirements

### Requirement: RMSNorm (E4 已实现，E5-E12 继承)

RMSNorm 在 E4 已验证可用（velocity_std 从 0.05→0.896）。所有后续实验均基于 `body_norm_type = "rms_norm"` 继续优化。

### Requirement: 评估流程

所有实验 MUST：
1. 训练 3 epoch（保持与 E1-E4 一致）
2. 每个 epoch 结束自动运行 full_eval
3. 生成 summary_grid.png 用于目视检查
4. 记录 clip_style, LPIPS, WFI, velocity_std, cross_attn_entropy 五大核心指标

## REMOVED Requirements

无移除项。所有现有功能保持向后兼容。
