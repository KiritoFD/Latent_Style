# Tasks: Phase 3 深度突破平凡解 — 理论驱动的根因修复

## 前置条件（Phase 1-2 继承的最优基线）

```
最优训练配置 (R4-D1, Phase 2):
  style_condition_source = "latent"
  endpoint_film_use_norm = false
  style_gate_mode = "fixed_one"
  style_film_enabled = true, endpoint_film_enabled = true
  training_objective_mode = "velocity"
  w_velocity_magnitude = 0.5
  w_pixel_color_match = 1.0
  w_contrast_preserve = 1.0, w_hf_energy = 1.0, w_channel_variance = 0.05
  num_epochs = 3

推理时: inference_adain = true

基线指标: clip_style ≈ 0.683, LPIPS ≈ 0.47, 雾化(无AdaIN) ≈ 5.5/10, 有AdaIN ≈ 3/10
目标: clip_style > 0.72, LPIPS < 0.40, 雾化 < 3/10
```

## Task 1: 降低 FM 权重实验（打破 FM 主导条件）
- **Priority**: P0
- **Depends On**: None
- **Description**:
  理论核心：FM loss 是强凸二次函数，主导优化方向。降低其权重可以让 SWD/style loss 相对更强。

  a) 在 `losses620.py` 中新增 `w_flow_scale` 参数：
     ```python
     flow_loss = self.w_flow * self.w_flow_scale * F.mse_loss(v_pred, v_target)
     ```
  
  b) 新增 config 参数：`w_flow_scale: float = 1.0`
  
  c) 实验矩阵（各 3 epoch，串行）：

     | 实验 | w_flow_scale | 其他 |
     |------|-------------|------|
     P3-A | 0.5 | 基线其余不变 |
     P3-B | 0.3 | 基线其余不变 |
     P3-C | 0.5 + w_swd_boost=2.0 | 同时增强 SWD |

  d) 每个 eval 都必须生成 summary_grid.png 并查看
  
  e) 重点观察：
     - velocity_ratio 是否进一步提升？
     - 不同 style 的输出是否开始分化（不再看起来一样）？
     - clip_style 是否提升？

- **Test Requirements**:
  - `programmatic` TR-1.1: 训练稳定，无 NaN
  - `programmatic` TR-1.2: style_cosine_between_diff_styles 指标（不同 style 输出的余弦相似度）应**下降**
  - `human-judgement` TR-1.3: 图片中不同 target 列的风格区分度是否提升

## Task 2: Style Contrastive Loss（强制风格分化）
- **Priority**: P0
- **Depends On**: None（可与 Task 1 并行）
- **Description**:
  理论核心：平凡解的特征是"不同 style 输出几乎相同"。直接惩罚这种相同性。

  a) 在 `losses620.py` 中新增 Style Contrastive Loss：
     ```python
     # 同一 batch 中，对每个 source，收集所有 target style 的 z_1_hat
     # 计算 pairwise cosine similarity
     # 惩罚高相似度: loss = mean(max(0, cos_sim(z_i, z_j) - margin))
     # 只对不同 style 的 pair 计算（同 style 不罚）
     ```

  b) 新增 config 参数：
     ```python
     w_style_contrastive: float = 0.0   # 对比损失权重
     contrastive_margin: float = 0.1    # 允许的最大相似度
     ```

  c) 实验（基于基线 + 最佳 w_flow_scale）：
     | 实验 | w_contrastive | margin | epochs |
     |------|--------------|--------|--------|
     P3-D | 0.1 | 0.1 | 3 |
     P3-E | 0.5 | 0.05 | 3 |

  d) 关键指标：cross-style cosine similarity 应显著下降

- **Test Requirements**:
  - `programmatic` TR-2.1: 计算并输出 `style_cross_sim_mean` metric
  - `programmatic` TR-2.2: 该指标应从 ~0.99（几乎相同）下降到 < 0.9
  - `human-judgement` TR-2.3: summary_grid.png 中不同列的视觉差异增大

## Task 3: FiLM 大初始化（非保守起点）
- **Priority**: P1
- **Depends On**: Task 1（选择最佳 flow_scale）
- **Description**:
  理论核心：L4（零初始化）让模型从 identity 开始，容易卡在保守盆地。大初始化让初始点在盆地外。

  a) 修改 FiLM 层初始化逻辑：
     ```python
     # 当前: nn.init.normal_(self.gamma, mean=1.0, std=0.0)
     # 改为: nn.init.normal_(self.gamma, mean=1.0, std=film_init_std)
     # 当前: nn.init.zeros_(self.beta)
     # 改为: nn.init.normal_(self.beta, mean=0.0, std=film_init_std)
     ```

  b) 新增 config 参数：`film_init_std: float = 0.0`（默认 0 向后兼容）
  
  c) 实验：
     | 实验 | film_init_std | epochs |
     |------|--------------|--------|
     P3-F | 0.05 | 3 |
     P3-G | 0.10 | 3 |

  d) 观察 film_gamma_abs 初始值和收敛值

- **Test Requirements**:
  - `programmatic` TR-3.1: 初始 epoch 的 film_gamma_abs 应 > 0（而非 0）
  - `programmatic` TR-3.2: 训练稳定，不因大初始化而爆炸
  - `programmatic` TR-3.3: 最终 clip_style vs 基线（std=0 时）

## Task 4: 最优组合 + AdaIN + 完整评估
- **Priority**: P0
- **Depends On**: Task 1, Task 2, Task 3
- **Description**:
  组合 Task 1-3 的有效方案，加上 AdaIN 后处理，做最终完整评估。

  a) 候选组合（选效果最好的 2-3 个跑 3 epoch）：
     | 实验 | w_flow_scale | w_contrastive | film_init_std | AdaIN |
     |------|-------------|---------------|---------------|-------|
     Final-A | 0.3 | 0.1 | 0.0 | ON |
     | Final-B | 0.5 | 0.5 | 0.05 | ON |
     | Final-C | 0.3 | 0.0 | 0.10 | ON |

  b) **关键：完整量化评估**
     - 必须用标准 run_evaluation.py 跑完整 metrics（clip_style, LPIPS, FID 等）
     - 不能只看 summary_grid.png，需要数字指标
     - 对比 Phase 2 最优基线的数字差异

  c) 目标检查：
     - clip_style > 0.72 ? （Phase 2 基线 0.683，需 +5.5%+）
     - LPIPS < 0.40 ? （Phase 2 基线 0.47，需 -15%+）
     - 雾化评分 < 3/10 ?

- **Test Requirements**:
  - `programmatic` TR-4.1: 输出完整的 metrics 表格（含 clip_style, LPIPS, FID 等）
  - `programmatic` TR-4.2: 至少一个组合的 clip_style > 0.70
  - `human-judgement` TR-4.3: 最终图片目视确认优于 Phase 2
  - `programmatic` TR-4.4: 所有实验的 summary_grid.png 都生成并查看

## Task Dependencies
```
Task 1 (FM权重) ──┬──→ Task 4 (最终组合)
Task 2 (Contrastive) ─┤
Task 3 (FiLM Init) ───┘
```

注意：Task 1 和 Task 2 可并行（不互相依赖）
