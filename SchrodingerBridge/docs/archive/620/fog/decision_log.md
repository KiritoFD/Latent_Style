# 620 白化问题诊断决策台账

**日期**: 2026-06-21  
**范围**: Round A → Round E3 关键决策记录  
**目的**: 记录每个关键决策的提出假设、证据、结论与下一步动作，避免重复试错

---

## 决策总览表

| 时间 | 决策点 | 提出的假设 | 证据 | 结论 | 下一步 |
|---|---|---|---|---|---|
| 2026-06-21 | attention 改造（gated_raw / relu2 / style_select）能否解决白化 | 改变 cross-attention 的 token 聚合方式（移除 softmax 重归一化、ReLU²、top-k 选择）能增强 style 信号特异性，从而降低 WFI | gated: WFI=0.4902；gated_raw: WFI=0.6435；relu2: WFI=0.5340；style_select: WFI=0.5005；四者 clip_style 均在 0.696–0.699 窄幅波动 | **否证**。attention 内部改动未降低 WFI，反而多数恶化；style 信号在 cross-attention 后仍被平均化 | 不再在 attention 模式上继续扫参；将 style 注入点从 cross-attention 后移到 endpoint head，绕过平均化瓶颈 |
| 2026-06-21 | endpoint_film 是否有效 | 将 FiLM 直接放入 endpoint head，让 style 全局嵌入调制 endpoint 特征图，能打破 endpoint shrinkage | Endpoint-FiLM hd128: WFI=0.4283（↓12.6%），clip_style=0.7066（↑），content_lpips=0.3226（↓）；style_sensitivity 保持 | **支持**。Endpoint-FiLM Head 同时改善 WFI、clip_style 与 content_lpips，是当前最有希望的方向 | 进一步验证如何增强 endpoint 的 style 调制能力，尝试容量提升与初始化调整 |
| 2026-06-21 | 更多 epoch 是否有益 | Endpoint-FiLM Head 在更多 epoch 后 WFI 会继续下降，clip_style 继续上升 | 3 epoch 训练：WFI=0.4271→0.4532→0.4680（单调上升），content_lpips=0.3236→0.3768 | **否证**。在当前 lr=2e-4 下，更多 epoch 反而加剧白化，优化器滑向统计更安全但更白化的 basin | 采用 1 epoch 最优 checkpoint；Round E4 尝试降低学习率或加入 early stopping |
| 2026-06-21 | endpoint_film_init_std=0.02 是否足够 | 非零初始化 FiLM 投影层能让 style 调制从训练早期生效，从而压低 WFI | H1 (init_std=0.02): WFI=0.4022，接近但未低于 0.40 放行门；clip_style=0.7044 | **部分支持**。非零初始化确实有效（0.4283→0.4022），但单独使用仍不足够 | 可与 H2（hd512）组合测试，看是否能进一步压低 WFI |
| 2026-06-21 | endpoint_style_hidden_dim=512 是否过门 | 增大 style→FiLM 映射的隐藏层容量能让 per-style 调制不被压缩，从而通过 WFI 放行门 | H2 (hd512): WFI=0.3906（<0.40），clip_style=0.7015（≥0.695），content_lpips=0.3382（<0.36） | **支持**。容量提升是直接瓶颈，hd512 成为新的当前最优 | 以 hd512 为基线进入 Round E4；后续可尝试组合 init_std、移除 GroupNorm、调整 lowpass kernel |
| 2026-06-21 | 高频残差（HF Residual）是否有效 | 在 velocity head 输出层显式保留 source latent 的高频成分可防止高频被洗掉，降低 WFI | HF Residual: WFI=0.4746（略好于基线 0.4902，远差于 Endpoint-FiLM 0.4283）；c_high 与基线几乎相同（0.081 vs 0.078） | **否证**。简单保留 source 高频不能解决风格高频迁移不足的问题；网络倾向于弱化该残差 | 不将 HF Residual 作为独立修复；若需保留高频细节，应结合 endpoint 低/高分支的 style 调制 |

---

## 详细说明

### D1. attention 改造能否解决白化

- **决策时间**: 2026-06-21（Round E1 基线审计）
- **背景**: 在 `gated` attention 基础上，尝试三种变体以验证 attention 平均化瓶颈。
- **实验设置**: `620_film_v5_gated_local_smoke`、`620_film_v5_gated_raw_local_smoke`、`620_film_v5_relu2_local_smoke`、`620_film_v5_style_select_local_smoke`；唯一变量为 `model.style_attn_mode`。
- **关键证据**:
  - `gated_raw` 移除 softmax 重归一化后 WFI 升至 0.6435，亮度被显著拉高到 0.719，饱和度最低。
  - 四种变体 clip_style 几乎相同，说明 attention 模式对高层风格对齐影响极小。
- **结论**: attention 改造不是根因，应把精力转向 endpoint 路径的 style 调制。

### D2. endpoint_film 是否有效

- **决策时间**: 2026-06-21（Round E2）
- **背景**: 基于 Round E1/E2 理论，提出将 FiLM 直接放入 endpoint head，绕过 block 级 shrinkage。
- **实验设置**: `620_film_v5_endpoint_film_local_smoke`；`endpoint_head_mode=endpoint_lowhigh`，`endpoint_film_enabled=true`，`endpoint_style_hidden_dim=128`。
- **关键证据**:
  - WFI 从基线 0.4902 降至 0.4283。
  - clip_style 从 0.6987 升至 0.7066。
  - content_lpips 从 0.3300 降至 0.3226。
  - 三指标同时改善，说明方向正确。
- **结论**: Endpoint-FiLM 是有效修复方向，但容量不足导致仍未过门。

### D3. 更多 epoch 是否有益

- **决策时间**: 2026-06-21（Round E3）
- **背景**: 验证 Endpoint-FiLM Head 是否随训练继续改善。
- **实验设置**: `620_film_v5_endpoint_film_3ep_local`；配置同 hd128，训练 3 epoch。
- **关键证据**:
  - epoch 1: WFI=0.4271
  - epoch 2: WFI=0.4532
  - epoch 3: WFI=0.4680
  - content_lpips 也单调上升：0.3236→0.3505→0.3768
- **结论**: 当前 lr=2e-4 下存在过训练风险，后期优化滑向白化 basin。

### D4. endpoint_film_init_std=0.02 是否足够

- **决策时间**: 2026-06-21（Round E3 H1）
- **背景**: 测试非零初始化 FiLM 投影层是否能让 style 调制早期生效。
- **实验设置**: `620_film_v5_endpoint_film_init02_local_smoke`；`endpoint_film_init_std=0.02`。
- **关键证据**:
  - WFI=0.4022（vs hd128 的 0.4283）
  - clip_style=0.7044，content_lpips=0.3217
- **结论**: 有效果但不足以单独过门；可与容量提升组合。

### D5. endpoint_style_hidden_dim=512 是否过门

- **决策时间**: 2026-06-21（Round E3 H2）
- **背景**: 测试增大 FiLM 映射容量是否能解决 modulation 信号压缩。
- **实验设置**: `620_film_v5_endpoint_film_hd512_local_smoke`；`endpoint_style_hidden_dim=512`。
- **关键证据**:
  - WFI=0.3906（<0.40）
  - clip_style=0.7015（≥0.695）
  - content_lpips=0.3382（<0.36）
- **结论**: 容量提升是直接瓶颈，hd512 成为最终最优。

### D6. 高频残差是否有效

- **决策时间**: 2026-06-21（Round E2 P1）
- **背景**: 假设在 velocity head 输出层显式保留 source 高频可防止白化。
- **实验设置**: `620_film_v5_hf_residual_local_smoke`；`velocity_hf_residual_enabled=true`，初始权重 0.1。
- **关键证据**:
  - WFI=0.4746，仅略好于基线 0.4902。
  - `c_high` 与基线几乎相同（0.081 vs 0.078）。
  - 学到的残差权重从 0.1 降至 0.089，网络主动弱化该残差。
- **结论**: 简单 source 高频保留不能解决风格高频迁移不足。

---

## 后续决策建议

| 候选决策 | 优先级 | 预期验证方式 |
|---|---|---|
| 组合 H1 + H2（init_std=0.02 + hd512） | P1 | 跑 1 epoch smoke，对比 hd512 单独效果 |
| 移除 FiLMEndpointHead 内的 GroupNorm | P1 | 跑 1 epoch smoke，监控 WFI 与 clip_style |
| 调低学习率（lr=1e-4）配合 hd512 | P2 | 跑 1–3 epoch，验证是否能延缓过训练 |
| text 条件引入 | P3（白化通过后可启动） | 在 hd512 基础上开启 text，必须先过 WFI 检查 |
| DINO 去留对照 | P3 | 无 DINO vs DINO 同成本对比 |

---

## 参考文档

- `docs/620/fog/baseline_audit/local_audit_2026-06-21.md`
- `docs/620/fog/baseline_audit/static_diagnosis_2026-06-21.md`
- `docs/620/fog/round_e2/experiment_report_2026-06-21.md`
- `docs/620/fog/round_e3/acceptance_report_2026-06-21.md`
- `docs/620/fog/gradient_probe/theory_update_round_e3.md`
- `docs/620/fog/final_summary.md`
