# 619 解决方案 — 逐项修复5个致命缺陷

> 基于代码审计被全部验证属实的5个问题, 按优先级给出具体修复方案

---

## 修复1: 时间/风格解耦 (优先级最高)

### 当前代码 (错误)

```python
# model.py:1459-1460
return style_code + time_code  # 混在一起

# lancet_blocks.py:128 — 下游同时接收混合信号
scale, shift = self.global_proj(style_code).chunk(2, dim=1)
```

### 修复方案

**步骤A**: 分离 `_compute_style_code` 的输出

```python
# model.py — 修改 _compute_style_code
def _compute_style_code(self, x, style_id, t):
    style_code = self.encode_style_id(style_id, t=t)
    time_code  = self.time_mlp(sinusoidal_time_embedding(t, ...))
    # 不再相加, 分开返回
    return style_code, time_code
```

**步骤B**: 所有调用处接收两个 code, 分别传递

**步骤C**: AdaGN 只接收 `time_code`:

```python
# lancet_blocks.py — CrossAttnAdaGN
class CrossAttnAdaGN(nn.Module):
    def forward(self, x, style_code, time_code, gate):
        # AdaGN: 只用 time_code
        scale, shift = self.global_proj(time_code).chunk(2, dim=1)
        x = x * (1 + scale) + shift
        
        # CrossAttn: 只用 style_code
        style_bias = self.style_proj(style_code)
        style_tokens = self.style_tokens_basis + style_bias
        # ... attention with style tokens
```

**影响范围**: `model.py` 的 `_compute_style_code` + 所有 `_run_block` / `_run_style_blocks` 调用处 (约 10 处)

**预期**: 训练稳定性提升, style 梯度不再被 time 信号污染

---

## 修复2: 训练中移除 ODE 展开 (优先级最高)

### 当前代码 (错误)

```python
# losses.py:2082 — 训练时展开 ODE
endpoint = model.integrate(content, style_id=..., num_steps=self.terminal_num_steps)
```

### 修复方案

**改为单步预测**:

```python
# losses.py — _terminal_swd 修改
def _terminal_swd(self, model, *, content, matched_target, ...):
    # 方案A: 用单步预测替代 ODE 展开
    t_full = torch.ones((content.shape[0],), device=content.device)
    pred_endpoint = model.predict_transport_base(content, t=t_full, style_id=target_style_id)
    
    # 方案B: 如果必须多步, 用 torch.no_grad() + 短路径
    # with torch.no_grad():
    #     endpoint_detached = model.integrate(...)
    # pred_endpoint = model.predict_transport_base(content, t=t_full, ...)
    
    return self.transport_cost.swd(pred_endpoint, matched_target)
```

**备选**: 完全移除 `_terminal_swd`, 只用 `MSE(pred_endpoint, matched_target)`. Flow Matching 理论已证明 MSE 足够.

**预期**: 消除梯度爆炸, 释放 style 梯度, 不再需要大量 clamp

---

## 修复3: 真实交叉注意力 (中优先级)

### 当前代码 (错误)

```python
# lancet_blocks.py:130-131 — 全局 learned tokens + 1D 偏移
style_bias = self.style_proj(style_code).unsqueeze(1)
style_tokens = self.style_tokens_basis.unsqueeze(0) + style_bias
k = self.k_proj(style_tokens)
v = self.v_proj(style_tokens)
```

### 修复方案

**方案A: 从 matched_target 编码空间特征** (无需参考图)

```python
# losses.py — OT匹配后立即编码
matched_target = self._ot_match_targets(...)
# 用一个共享的轻量 StyleEncoder
style_spatial_feat = self.style_encoder(matched_target)  # [B, C, H, W]
# 展平为序列
style_tokens = style_spatial_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
k = self.k_proj(style_tokens)
v = self.v_proj(style_tokens)
```

**方案B: 保留当前 learned tokens 但增加多样性** (最小改动)

```python
# 增加 token 数量: 128 → 512
# 用多个 style_code 副本生成多样化的 token bias
style_tokens = self.style_tokens_basis.unsqueeze(0) + \
    self.style_proj(style_code).view(B, -1, self.num_tokens, C).mean(dim=2)
```

**方案A 更适合预匹配 OT 管线** — matched_target 已经对齐, 编码的 style features 天然对应内容结构。

**预期**: 风格细节可学, 不再是全局偏置

---

## 修复4: Minibatch OT → 离线预配对 (中优先级)

### 当前代码

```python
# losses.py — 每 batch 内动态 OT 匹配
matched_target = self._ot_match_targets(content, target_style, ...)
```

### 修复方案

**短期: 独立耦合 (Independent Coupling)**

```python
# 不做 OT, 直接随机配对
matched_target = target_style[torch.randperm(B)]
```

Flow Matching 的 Independent Coupling 理论保证: 即使不做 OT, 模型也能学到正确的速度场。只是需要更多训练步数。

**中期: 离线预配对**

预处理管线:
1. 提取所有图像的 DINOv2 feature
2. 对每张 content, 在每个 style 中找 top-20 相似 target
3. Sinkhorn pixel 级对齐 → 保存 (z_c, z_tgt_aligned, cond)
4. 训练时直接加载, 固定配对

见 `prematched_ot_evaluation.md` 完整方案.

**预期**: 目标稳定, 速度场学习一致

---

## 修复5: 闭集查表 → 实例级编码 (低优先级, 但长期最重要)

### 当前代码

```python
# semantic_tokenizer.py:283
self.style_values = nn.Embedding(self.num_styles, D)
```

### 修复方案

**短期: style_condition bank**

```python
# 为每个 style 预计算平均 style condition
style_bank = {style_id: compute_mean_style_embedding(style_images)}
```
训练/推理时: `cond = style_bank[style_id]`. 仍然是查表, 但从"随机初始化"升级为"有意义的特征".

**中期: 从风格参考图编码**

```python
# 训练时也训练一个 StyleEncoder
class StyleEncoder(nn.Module):
    def forward(self, style_image):
        return self.encoder(style_image)  # → cond

# 训练:
style_image = sample_from_style_pool(style_id)
cond = style_encoder(style_image)
# 推理:
cond = style_encoder(user_provided_reference_image)  # 可选
cond = style_bank[style_id]  # fallback
```

---

## 修复顺序与时间估算

| 优先级 | 修复 | 代码量 | 时间 |
|:---:|------|:---:|:---:|
| 1 | time/style解耦 | ~50行改 | 半天 |
| 2 | 移除ODE展开 | ~10行改 | 1h |
| 3 | ODEl展开移除后, 重训验证 | — | 半天 |
| 4 | 离线预配对(DINOv2) | ~200行新 | 2天 |
| 5 | 真实cross-attention | ~100行改 | 1天 |
| 6 | 实例级style编码 | ~100行新 | 1天 |

**最小可行修复 (1+2+3)**: 1天. 预期 style 从 0.67 → ? 取决于 time/style 纠缠的影响程度.

**完整重构 (1+2+4+5+6)**: 5天. 预期可达到 state-of-the-art.
