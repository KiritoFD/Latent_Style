# 621 完整模型架构审计

> 审计日期: 2026-06-21  
> 覆盖: 620 SpatialBridge + Legacy LANCET

---

## 1. 620 SpatialBridge 架构

### 1.1 整体流程

```
Input: content latent x ∈ R^{B×4×64×64}
  ↓ input_proj (Conv2d 4→dim)
  ↓ N × SpatialBridgeBlock620
    ↓ AdaLN(time) → Self-Attention → Cross-Attention(style) → StyleFiLM → FFN
  ↓ Endpoint Head
  ↓ velocity 或 endpoint_lowhigh 模式
Output: velocity v ∈ R^{B×4×64×64} 或 endpoint ẑ₁
```

### 1.2 SpatialBridgeBlock620 详解

每个block包含:

1. **GroupNorm(1)** (affine=False) + **AdaLN(time_emb)**
   - scale, shift, gate = Linear(SiLU(time_emb))
   - h_time = GN(x) * (1 + scale) + shift

2. **Self-Attention** (content Q/K/V)
   - sa_qkv = Linear(h_time, 3*dim)
   - sa_out = SDPA(sa_q, sa_k, sa_v)
   - sa_delta = sigmoid(gate) * sa_out

3. **Cross-Attention** (content Q × style K/V)
   - Pre-FiLM: style_global → gamma_q, beta_q → modulate ca_in
   - Q = Linear(ca_in), K = Linear(style_tokens), V = Linear(style_tokens)
   - Attention modes: softmax/gated/gated_raw/relu2/style_select/sparsemax
   - style_bias = Linear(style_global) → bias per token
   - style_delta = tanh(style_gate) * out_proj(attended)

4. **StyleFiLM** (post-cross-attention)
   - gamma, beta = Linear(style_global, 2*dim)
   - x = (1 + gamma) * x + beta

5. **FFN** with GroupNorm
   - GN(x) → Conv(dim→4dim) → SiLU → Conv(4dim→dim)

6. **Style MoE** (可选)
   - Multiple expert K/V projections with router network

### 1.3 Endpoint Head 模式

**模式A: velocity**
```
h → Conv(dim→2dim) → SiLU → Conv(2dim→dim) → SiLU → Conv(dim→4) → velocity
endpoint = x + (1-t) * velocity
```
无GroupNorm, 非零初始化(std=0.02)

**模式B: endpoint_lowhigh** (当前最优)
```
style_low = MLP(style_global) → R^4
style_high = MLP(style_global) → R^4
low_delta = endpoint_film_low(h, style_global)  # FiLM head
high_delta = endpoint_film_high(h, style_global) * high_scale
endpoint = (x_low + low_delta) + (x_high + high_delta)
velocity = (endpoint - x) / (1-t)
```
FiLM head: GroupNorm(1) + FiLM modulation + Conv

### 1.4 StyleConditioner620

```
DINO patches [B, 256, 384]
  ↓ adapter (residual MLP)
  ↓ patch_proj (LN→Linear→SiLU→Linear) → [B, 256, dim] (style tokens)
  ↓ cls_proj (LN→Linear→SiLU→Linear) → [B, dim] (style_global)
  ↓ Optional: local_cnn + local_pool → additional tokens
  ↓ Optional: text_proj + text tokens
  ↓ Modality dropout (15% each)
Output: style_tokens [B, 256+?, dim], style_global [B, dim]
```

### 1.5 损失函数 (losses620.py)

```
L = w_FM * MSE(v_pred, v_target)
  + w_SWD * SWD(ẑ₁, projected_target)  [64 random projections]
  + w_edge * L1(high_freq(ẑ₁), high_freq(target))
  + w_low * L1(lowpass(ẑ₁), lowpass(target))
  + w_aux * source_endpoint_aux
  + w_energy * endpoint_energy_band
  + w_entropy * attention_entropy_reg
```

关键设计:
- **单步SWD**: SWD(ẑ₁, y_proj) 而非多步ODE展开
- **投影目标**: source_low + target_high (可配置)
- **target_linear**: 低频路径线性插值
- **SWD noise**: σ=0.02 打破排序稳定性

---

## 2. Legacy LANCET 架构

### 2.1 整体流程

```
Input: content latent x
  ↓ encoder (LatentAdaCUT)
  ↓ body (style-modulated residual blocks)
  ↓ decoder
  ↓ style injection (carrier gate / spatial carrier / direct)
  ↓ proximal attention
Output: predicted latent
```

### 2.2 关键差异
- U-Net encoder-decoder (vs 纯transformer blocks)
- 多种style tokenizer (factorized, pure_latent_spatial, smoe_translator)
- OT coupling with Sinkhorn solver
- Terminal SWD (multi-step)
- 更复杂的loss (cycle consistency, target teacher, structure descriptors)

---

## 3. 文件依赖图

```
src/model620.py
  ├── src/blocks620.py (SpatialBridgeBlock620)
  ├── src/style_encoder620.py (StyleConditioner620)
  └── src/config_schema.py (ModelConfig, BridgeConfig)

src/losses620.py
  └── src/config_schema.py (ExperimentConfig)

src/trainer.py
  ├── src/model.py or src/model620.py
  ├── src/losses.py or src/losses620.py
  └── src/utils/inference.py

src/utils/inference.py
  └── src/model620.py (build_spatial_bridge620_from_config)
```

---

## 4. 配置参数清单 (关键参数)

| 参数 | 默认值 | 当前最优 | 影响 |
|------|--------|----------|------|
| base_dim | 128 | 128 | 模型容量 |
| num_res_blocks | 4 | 4 | 深度 |
| style_cross_attn_gate_init | 0.05 | 0.3 | style信号强度 |
| endpoint_head_mode | velocity | endpoint_lowhigh | endpoint预测方式 |
| endpoint_film_enabled | false | true | endpoint FiLM调制 |
| endpoint_style_hidden_dim | 128 | 512 | FiLM容量 |
| endpoint_film_init_std | 0.0 | 0.0 | FiLM初始化 |
| style_film_enabled | false | true | block内FiLM |
| style_attn_mode | softmax | gated | attention模式 |
| single_step_swd_weight | 8.0 | 8.0 | SWD loss权重 |
| swd_noise_sigma | 0.0 | 0.02 | SWD噪声 |
| training_target_projection_mode | source_low_target_high | target_linear | 训练目标 |
| training_target_projection_low_mode | all | target_linear | 低频路径 |
| bridge_sigma | 0.02 | 0.02 | 推理噪声 |
