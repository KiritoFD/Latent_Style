# Tasks: E5-E12 反退化硬核突破实验

## 前置条件（E4 继承）

```
E4 Anti-Degeneration 已完成:
  body_norm_type = "rms_norm"  (已验证: velocity_std=0.896)
  w_flow_scale = 0.15
  w_velocity_magnitude = 2.0
  two_stage_s1_w_endpoint_style = 24.0
  style_gate_mode = "fixed_one"
  
E4 结果:
  clip_style = 0.6722, LPIPS = 0.3735, velocity_std = 0.896
  问题: LPIPS 较高(内容保真度下降)，需要多轴协同修复

目标:
  clip_style > 0.70 且 LPIPS < 0.35 (超越 E2 的 0.3326)
  WFI < 0.40 (白化验收标准)
  cross_attn_entropy < 5.0 (注意力锐利化)
```

---

## Task 1: E5 — VP-Flow 球面插值路径 (#1)

- **Priority**: P0 (推荐组合第一把斧)
- **Depends On**: None
- **Description**:

  a) 在 `src/losses620.py` 的 `_vertical_state()` 中新增 `spherical_vp` 路径模式：
     ```python
     # 当前线性: x_t = (1-t)*x_0 + t*x_1
     # 新增球面: x_t = cos(pi/2*t)*x_0 + sin(pi/2*t)*x_1
     # target_velocity = d/dt[x_t] = -pi/2*sin(pi/2*t)*x_0 + pi/2*cos(pi/2*t)*x_1
     ```
     
  b) 新增 config 参数：
     ```python
     bridge_path_mode: str = "linear"  # "linear" | "spherical_vp"
     ```

  c) 基于 E4 配置生成 E5 config，唯一改动：`bridge_path_mode = "spherical_vp"`

  d) 部署到远程，训练 3 epoch + full_eval

  e) 重点观察：
     - 中间步(t≈0.5)的特征方差是否保持稳定
     - LPIPS 是否改善（减少中间步发灰）
     - velocity_std 是否保持高位

- **Test Requirements**:
  - 训练稳定，无 NaN
  - `bridge_path_mode="linear"` 时行为与 baseline 完全一致
  - VP-Flow 模式下 LPIPS < E4 的 0.3735

---

## Task 2: E6 — Top-K 截断掩码 Cross-Attention (#7)

- **Priority**: P0 (推荐组合第二把斧)
- **Depends On**: None（可与 Task 1 并行）
- **Description**:

  a) 在 `src/blocks620.py` 的 cross-attention 计算中（约 line 336-350），在 softmax 前插入 Top-K 截断：
     ```python
     if self.style_attn_topk > 0:
         topk_val, topk_idx = logits.topk(self.style_attn_topk, dim=-1)
         mask = torch.full_like(logits, float('-inf'))
         mask.scatter_(-1, topk_idx, 0.0)
         logits = logits + mask
     ```

  b) 新增 config 参数：
     ```python
     style_attn_topk: int = 0  # 0=不截断(默认), 4=只保留top4
     ```

  c) 基于 E4 配置生成 E6 config，改动：`style_attn_topk = 4`

  d) 部署到远程，训练 3 epoch + full_eval

  e) 重点观察：
     - cross_attn_entropy 是否从 ~5.53 显著下降
     - 笔触是否更锐利（目视 summary_grid.png）
     - clip_style 是否提升

- **Test Requirements**:
  - `style_attn_topk=0` 时行为与 baseline 一致
  - Top-K 模式下 cross_attn_entropy 下降 > 0.3
  - 训练稳定，无 NaN

---

## Task 3: E7 — 三斧组合 VP-Flow + Top-K + RMSNorm (1+7+5)

- **Priority**: P0 (**用户推荐的最优组合**)
- **Depends On**: Task 1, Task 2（代码修改完成后组合配置）
- **Description**:

  a) 复用 Task 1 和 Task 2 的代码修改
  
  b) 基于 E4 配置生成 E7 config，同时启用：
     ```python
     bridge_path_mode = "spherical_vp"    # VP-Flow
     style_attn_topk = 4                   # Top-K Attention
     body_norm_type = "rms_norm"           # RMSNorm (E4已验证)
     ```

  c) 部署到远程，训练 3 epoch + full_eval

  d) 这是**预期产生相变的核心实验**

  e) 如果结果不理想，尝试调参变体：
     - E7b: `style_attn_topk = 8`（更宽松的截断）
     - E7c: `w_flow_scale = 0.1`（进一步降低 FM 压力配合 VP-Flow）

- **Test Requirements**:
  - clip_style > 0.68 且 LPIPS < 0.35（超越 E2 基线）
  - cross_attn_entropy < 5.0
  - 目视 summary_grid.png 确认风格锐利度和内容保持

---

## Task 4: E8 — 方向余弦损失 (#2)

- **Priority**: P1
- **Depends On**: None
- **Description**:

  a) 在 `src/losses620.py` 的 compute() 中，在 MSE loss 后追加方向余弦惩罚：
     ```python
     if self.w_directional_cosine > 0:
         v_pred_n = F.normalize(pred_velocity.float(), dim=[1,2,3])
         v_tgt_n = F.normalize(target_velocity.float(), dim=[1,2,3])
         cos_sim = (v_pred_n * v_tgt_n).sum(dim=[1,2,3]).mean()
         dir_loss = (1.0 - cos_sim).clamp(min=0.0)
         fm = fm + self.w_directional_cosine * dir_loss
     ```

  b) 新增 config 参数：`w_directional_cosine: float = 0.0`

  c) 实验：基于 E4 + `w_directional_cosine = 1.0`

- **Test Requirements**:
  - endpoint_alpha 提升 > E4 baseline
  - 方向余弦相似度指标可观测

---

## Task 5: E9 — Late-Stage 时间采样 (#4)

- **Priority**: P1
- **Depends On**: None
- **Description**:

  a) 修改 `src/losses620.py` 的 `_sample_t()` 支持 Beta 分布：
     ```python
     if self.t_sampling_beta_a > 0 and self.t_sampling_beta_b > 0:
         u = torch.distributions.Beta(
             torch.tensor(self.t_sampling_beta_a),
             torch.tensor(self.t_sampling_beta_b)
         ).sample([content.shape[0]], device=content.device).to(content.dtype)
     else:
         u = torch.empty(...).uniform_(0, 1)
         u = u.pow(self.t_sampling_power)
     ```

  b) 新增 config 参数：
     ```python
     t_sampling_beta_a: float = 0.0  # 0=uniform(默认), 3=偏向后期
     t_sampling_beta_b: float = 0.0
     ```

  c) 实验：基于 E4 + `t_sampling_beta_a=3, t_sampling_beta_b=1`

- **Test Requirements**:
  - 默认参数时采样分布与 uniform 一致
  - Beta(3,1) 下 70%+ 样本 t > 0.7

---

## Task 6: E10 — Endpoint 方差注入 (#6)

- **Priority**: P1
- **Depends On**: None
- **Description**:

  a) 在 `src/model620.py` 的 endpoint_lowhigh 分支中新增 sigma 支路：
     ```python
     if self.endpoint_variance_injection:
         sigma_style = self.endpoint_sigma_head(h)  # [B, C, 1, 1]
         sigma_style = torch.exp(sigma_style.clamp(-3, 3))
         endpoint = endpoint * sigma_style + delta_style
     ```

  b) 新增 config 参数：`endpoint_variance_injection: bool = False`

  c) 实验：基于 E4 + `endpoint_variance_injection = true`

- **Test Requirements**:
  - 高频能量(hf_energy)指标提升
  - 特征图 channel_std 增加

---

## Task 7: E11 — Residual-First 门控 (#9)

- **Priority**: P2
- **Depends On**: None
- **Description**:

  a) 在 `src/blocks620.py` 中新增门控模式 `residual_first`：
     ```python
     if self.style_gate_mode == "residual_first":
         gate_val = torch.sigmoid(self.style_gate + 4.0)  # init ≈ 0.98
         out = gate_val[..., None, None] * x + attn_output  # 内容被门控
     ```

  b) 新增 config 参数支持（复用现有 `style_gate_mode`）

  c) 实验：基于 E4 + `style_gate_mode = "residual_first"`

- **Test Requirements**:
  - 初始 gate 值 ≈ 0.98（而非 tanh(0.05)≈0.05）
  - 训练过程中 gate 不坍缩到接近 0

---

## Task 8: E12 — CFG 条件丢弃训练 (#10)

- **Priority**: P2
- **Depends On**: None
- **Description**:

  a) 在 `src/trainer.py` 或 `src/losses620.py` compute() 中添加条件丢弃：
     ```python
     use_cfg = random.random() < self.cfg_dropout_prob
     if use_cfg:
         style_tokens = null_tokens  # 替换为空向量
         # 同时计算 uncond loss
     ```

  b) 新增 config 参数：`cfg_dropout_prob: float = 0.0`

  c) 实验：基于 E4/E7 最佳 + `cfg_dropout_prob = 0.15`

  d) 推理脚本支持 CFG 外推：`v_final = v_uncond + s*(v_cond - v_uncond)`

- **Test Requirements**:
  - cfg_dropout_prob=0 时与 baseline 一致
  - 不同 style_scale 下输出有显著差异

---

## Task 9: 全量评估汇总 + 目视诊断

- **Priority**: P0
- **Depends On**: Task 1, Task 2, Task 3（至少 E5/E6/E7 完成）
- **Description**:

  a) 汇总 E1-E7（或更多）全部实验的五大核心指标对比表

  b) 对每个实验的 summary_grid.png 进行目视诊断：
     - 风格强度（笔触、色彩饱和度）
     - 内容保真度（结构保持）
     - 白化程度（灰雾感）

  c) 确定最优配置和下一步方向

  d) 如果 E7 组合成功（clip_style>0.70, LPIPS<0.35），考虑更长训练（10 epoch）

- **Test Requirements**:
  - 完整对比表格（含所有核心指标）
  - 每个实验至少有一张目视检查记录
  - 明确最优配置及理由

---

## Task Dependencies

```
Task 1 (VP-Flow) ──┬──→ Task 3 (E7 组合 1+7+5) ──┐
Task 2 (Top-K)   ──┘                            │
                                                    ├──→ Task 9 (全量汇总)
Task 4 (Cosine Loss) ─────────────────────────────┤
Task 5 (Late Sampling) ────────────────────────────┤
Task 6 (Variance Inj) ─────────────────────────────┤
Task 7 (Residual Gate) ────────────────────────────┤
Task 8 (CFG Dropout) ──────────────────────────────┘
```

注意：Task 1, 2, 4, 5, 6, 7, 8 彼此独立，可并行开发代码。但远程 GPU 只能串行训练。
