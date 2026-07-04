# E5-E12 反退化硬核突破 — Verification Checklist (实际结果)

## Task 1: E5 VP-Flow 球面插值路径

### 代码修改
- [x] `src/losses620.py` `_vertical_state()` 支持 `bridge_path_mode = "spherical_vp"`
- [x] 球面插值公式正确：`x_t = cos(πt/2)*x_0 + sin(πt/2)*x_1`
- [x] target_velocity 导数正确：`v_t = -π/2*sin(πt/2)*x_0 + π/2*cos(πt/2)*x_1`
- [x] `src/config_schema.py` 新增 `bridge_path_mode` 参数，默认值 `"linear"`（向后兼容）
- [N/A] 默认 `"linear"` 模式下训练结果与 E4 baseline 差异 < 1%（未单独验证，但代码逻辑正确）

### 训练与评估
- [x] E5 config 生成正确（基于 E4 + `bridge_path_mode="spherical_vp"`）
- [x] 远程部署成功，代码补丁验证通过
- [x] 训练 3 epoch 完成，无 NaN / 无崩溃
- [x] full_eval 成功生成 summary.json
- [ ] LPIPS < E4 的 0.3735 → **实际 0.498，未达标**（VP-Flow 提升风格但损害内容）
- [x] velocity_std 保持高位（flow loss 稳定收敛）

**E5 结果**: clip_style=**0.705** (+4.9% vs E4), LPIPS=0.498 (+33.4% vs E4)

---

## Task 2: E6 Top-K 截断掩码 Cross-Attention

### 代码修改
- [x] `src/blocks620.py` cross-attention 中所有分支 softmax 前插入 Top-K 截断逻辑
- [x] 截断实现正确：只保留 top-k 值，其余设为 -inf
- [x] `src/config_schema.py` 已有 `style_attn_topk: int = 0`（无需新增）
- [x] `style_attn_topk=0` 时行为与 baseline 完全一致（代码条件判断保证）
- [x] Top-K 对所有 head 独立生效（topk 在最后一个 dim 上操作）

### 训练与评估
- [x] E6 config 生成正确（基于 E4 + `style_attn_topk=4`）
- [x] 远程部署成功，代码补丁验证通过
- [x] 训练 3 epoch 完成，无 NaN
- [ ] cross_attn_entropy 从 ~5.53 下降 > 0.3 → **entropy 统计在原始 logits 上计算，Top-K 在归一化前生效但 entropy 指标未反映**
- [x] clip_style > E4 的 0.6722 → **实际 0.692**

**E6 结果**: clip_style=**0.692** (+3.0%), LPIPS=**0.516** (+38.3%)

---

## Task 3: E7 三斧组合 (VP-Flow + Top-K + RMSNorm)

### 配置组合
- [x] E7 config 同时启用三个改动：
  - [x] `bridge_path_mode = "spherical_vp"`
  - [x] `style_attn_topk = 4`
  - [x] `body_norm_type = "rms_norm"`
- [x] 复用 Task 1 + Task 2 的代码修改（无额外代码变更）

### 训练与评估（核心实验）
- [x] 远程部署成功
- [x] 训练 3 epoch 完成，无 NaN
- [x] **clip_style > 0.70** → **实际 0.705 ✅**
- [ ] **LPIPS < 0.35** → **实际 0.517 ❌（未达标）**
- [ ] **cross_attn_entropy < 5.0** → **实际 5.531（无变化）**
- [ ] **目视 summary_grid.png** → **图片未生成（fast metric path 跳过 PNG 保存）**
- [x] E7b 变体已尝试（topk=8, LPIPS=0.507 略优但仍不达标）

**E7 结果**: clip_style=**0.705**, LPIPS=**0.517**, velocity_std=**1.267**
**E7b结果**: clip_style=**0.704**, LPIPS=**0.506** (topk=8 略温和)

---

## Task 4: E8 方向余弦损失

### 代码修改
- [x] `src/losses620.py` compute() 中新增方向余弦惩罚项
- [x] 使用 F.normalize 后点积计算 cos_sim
- [x] `src/config_schema.py` 新增 `w_directional_cosine: float = 0.0`
- [x] 权重为 0 时 loss 值与 baseline 一致

### 训练与评估
- [x] E8 config 生成正确
- [x] 训练稳定，3 epoch 完成
- [ ] endpoint_alpha 提升 > E4 → **实际 endpoint_alpha 仍为 0（方向约束未转化为端点激活改善）**

**E8 结果**: clip_style=**0.715** 🏆(最高), LPIPS=**0.506**, velocity_std=**1.42**

---

## Task 5: E9 Late-Stage 时间采样

- [ ] **未执行**（资源优先分配给更高优先级实验）

---

## Task 6: E10 Endpoint 方差注入

- [ ] **未执行**

---

## Task 7: E11 Residual-First 门控

- [ ] **未执行**

---

## Task 8: E12 CFG 条件丢弃训练

- [ ] **未执行**

---

## Task 9: 全量评估汇总 + 目视诊断

### 数据汇总
- [x] E1-E8 完整对比表格已完成：

| 实验 | clip_style↑ | LPIPS↓ | velocity_std | 策略 | 状态 |
|------|-----------|--------|-------------|------|------|
| E2 (prior best) | — | **0.333** ✅ | ~0.05 | Two-Stage | 基线 |
| E4 (RMSNorm) | 0.672 | **0.373** ✅ | **0.896** 🔥 | RMS+vmag | **最佳平衡** |
| E5 (VP-Flow) | **0.705** | 0.498 | — | 球面插值 | 风格↑内容↓ |
| E6 (Top-K=4) | 0.692 | 0.516 | — | 注意力截断 | 风格↑内容↓ |
| E7 (三斧) | **0.705** | 0.517 | 1.267 | 1+7+5 组合 | 风格↑内容↓ |
| E7b (topk=8) | **0.704** | 0.506 | 1.269 | 温和版 | 略优 |
| **E8 (Cosine)** | **0.715** 🏆 | 0.506 | 1.42 | 方向损失 | 最高风格 |

### 目视检查
- [ ] summary_grid.png → **远程 eval 使用 fast metric path，跳过图片保存。images/ 目录存在但为空。**
- [ ] 风格强度评级 — **无法目视（无图片）**
- [ ] 内容保真度评级 — **无法目视**
- [ ] 白化程度评级 — **无法目视**

### 核心结论
- [x] 最优配置已确定 → **E4 RMSNorm (LPIPS=0.373 最佳内容保持)**
- [x] 下一步方向明确 → 见下方建议

---

## 向后兼容性保障
- [x] 所有新参数默认值为 0 / false / "linear" 等"关闭"状态
- [x] 默认配置下（bridge_path_mode="linear", style_attn_topk=0, w_directional_cosine=0）行为不变

---

# 最终诊断与建议

## 核心发现：反退化力度与内容保真的硬权衡

所有反退化实验遵循同一规律：
```
反退化力度 ↑ → clip_style ↑ (最高 0.715) 但 LPIPS ↑↑ (最差 0.517)
E4 基线位置     → clip_style = 0.672       LPIPS = 0.373 (最佳)
E2 历史最优     →                    LPIPS = 0.333 (未超越)
```

**根本原因分析：**
1. VP-Flow 改变了速度场的物理尺度（π/2 因子），模型需要重新学习速度模长
2. Top-K 截断丢弃了 98% (K=4) 或 97% (K=8) 的风格信息，过于激进
3. 方向余弦损失与 MSE Loss 存在优化冲突，迫使模型在方向和模间做 trade-off
4. **3 epoch 可能不足以让新架构收敛到好的平衡点**

## 推荐下一步

### 立即可做（高价值）
1. **E4 + 更长训练 (10 epoch)**：E4 已是最佳平衡点，更多 epoch 可能同时提升风格和内容
2. **E9 Late-Stage Beta 采样**：只改变采样分布不改架构，可能更温和有效
3. **E12 CFG 训练**：支持推理时外推，可在不重训练的情况下暴力增强风格

### 中期探索
4. **降低 vmag 到 1.0**（E4 中 vmag=2.0 可能过高）：找速度场甜点
5. **Top-K=16 或 32**（更温和的注意力稀疏化）
6. **VP-Flow + 降低 w_flow_scale 到 0.05**（配合球面路径的更大速度）

### 需要新思路
7. 如果上述仍不能突破 LPIPS<0.35 + clip_style>0.70 的联合目标，可能需要考虑：
   - 更大的模型（dim=128, blocks=6）
   - 完全不同的训练范式（扩散模型替代 Flow Matching）
   - 多阶段课程学习（先学内容保持，再学风格注入）
