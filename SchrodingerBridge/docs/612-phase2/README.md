# Phase 2 — 基于 612 回顾的结构优先计划

## 权威说明

- 本文是当前 Distinct5 Phase 2 的唯一执行权威。
- `docs/612-lookback/*.md` 保留为回顾诊断，不再直接决定正式实验排队。
- 本轮最关键的解释收紧是:
  - `LPIPS >= 0.70` 不是“高 style 的高风险候选”，而是完全失败。
  - `0.40 <= LPIPS < 0.70` 不是 compromise，而是 archival-only 证据。

## 核心判断

- 真正的问题不再是“如何把 style 硬推高”，而是如何在 `content_lpips < 0.40` 的结构安全带内持续抬高 Distinct5 的 style。
- Velocity 线已经证明可以稳定守住结构，但 style 仍停在 `0.70x / 0.38x` 一带。
- Endpoint / exact-I2SB 线已经反复证明能把 style 拉到 `0.72-0.73`，但也会把 LPIPS 推到 `0.57-0.71+`。
- 因而 Distinct5 的 paper-facing 主问题不是“如何接受高损伤换高风格”，而是“如何在安全带内做出真正 breakout”。
- 本轮执行上，`LPIPS >= 0.70` 也不再只是 run-level 停止条件，而是 family-level 失格证据: 该家族退出 Distinct5 正式晋升路径，除非换安全父本重做。

## 硬门槛

- `content_lpips >= 0.70`
  - 完全失败。
  - 立即停掉正式远程训练。
  - family 退出 Distinct5 正式晋升队列，除非后续从安全父本重新设计。
- `0.40 <= content_lpips < 0.70`
  - 仅可归档，不可晋升。
  - 可以保留为理论 / 实现 / 负例证据，但不能继续占用唯一正式远程训练 lane。
  - 也不能再用来定义下一个正式 packet 的优先级。
- `content_lpips < 0.40`
  - 才有资格继续占用正式训练资源。
- `style >= 0.72`
  - 仍然是长期目标，但不是充分条件。
  - 只要 LPIPS 出带，这个点就不算成功。

## 成功阶梯

- Stage A
  - 在 `LPIPS < 0.40` 内稳定超过当前安全 shelf `all-pairs 0.701666 / 0.381724`。
- Stage B
  - 把 in-band best 推到 `all-pairs style >= 0.705` 且 `LPIPS <= 0.380`。
- Stage C
  - 把 in-band best 推到 `all-pairs style >= 0.710` 且 `LPIPS <= 0.370`。
- Long target
  - `style >= 0.72` 且 `LPIPS <= 0.35`。

## 612 回顾后的计划变更

### 退出主线

| 项 | 当前决策 | 原因 |
|---|---|---|
| Endpoint / exact-I2SB 作为 Distinct5 主线 | 退出正式主队列 | `0.57-0.71+` LPIPS 已经证明它们不在 paper gate 内 |
| DINO 依赖 | 继续退休 | 纯潜空间路线已经足够成立，DINO 工程复杂度不值得继续堆到主线 |
| 旧 heuristic structure 补丁 | 只保留对照地位 | 这类补丁大多是在给 endpoint 擦屁股，不解决主矛盾 |

### 保留并推进

| 项 | 当前决策 | 说明 |
|---|---|---|
| true tokenizer | 正式主线核心 | 继续在 `velocity + pure latent` 家族内做安全带扫描 |
| training-side structure control | 延后到第二顺位 | 只能建立在更强的 in-band 父本之上 |
| true I2SB 代码 | 保留实现能力 | 仅允许 diagnostic-only，不再直接抢正式 lane |

## Phase 2 执行顺序

1. `vel_tok32_safe_rescan_r2`
   - 继续压榨 `velocity + true tokenizer + safe-band` 家族。
   - 目标不是做 style-first rescue，而是验证 safe-family 是否还能产出新的 in-band 非支配点。
2. `vel_structure_control_reentry`
   - 只有当 safe-family 明确用尽，或先拿到更强的 in-band 父本，才允许进入。
   - 结构控制只能作为 training-side 保结构工具，不能再被当成 endpoint 风格放大后的补救。
3. `i2sb_diagnostic_only`
   - 只保留理论和实现价值。
   - 仅允许廉价 diagnostic / smoke / NFE 对照，不允许挤占唯一正式远程训练 lane。
4. `DINO`
   - 保持退休，不进入当前主线计划。

## 当前板面

- 当前安全 shelf:
  - `vel_pattn_enhanced_tok` best `epoch_0002`
  - transfer `0.673934 / 0.384340`
  - all-pairs `0.701666 / 0.381724`
- `vel_tok32_pos_refresh`
  - best `epoch_0004`
  - transfer `0.673399 / 0.376463`
  - all-pairs `0.701161 / 0.374695`
  - 结论: 全程在安全带内，但没有突破旧 shelf，按 plateau 关闭。
- `vel_tok32_safe_rescan_r1`
  - `epoch_0002`
  - transfer `0.676378 / 0.400694`
  - all-pairs `0.702543 / 0.397891`
  - 结论: style 抬升是真实的，但一旦越过 `0.40`，它就只能归档，不能晋升。
- `vel_pattn_topology_anchor`
  - `epoch_0002`
  - transfer `0.680803 / 0.417910`
  - all-pairs `0.706132 / 0.413976`
  - 结论: 结构补丁类候选暂时不能排在 tokenizer-safe sweep 之前。
- `eval_only_pc_solver`
  - transfer `0.729014 / 0.621056`
  - all-pairs `0.735295 / 0.611310`
  - 结论: inference-time corrector 无法把 style-strong 父本拉回安全带。
- `true I2SB` ladder
  - corrected `rtfix epoch_0001`: all-pairs `0.724472 / 0.707551`
  - residual endpoint `epoch_0001`: all-pairs `0.697686 / 0.569086`
  - 结论: exact-I2SB 在 Distinct5 上只保留实现与理论价值，不再占用正式训练 lane。

## 当前正式实验计划

### 队列1: `vel_tok32_safe_rescan_r2`

- 地位:
  - 当前唯一允许优先尝试的正式候选。
  - 但它不是“默认晋升线”，而是一次严格短筛。
- 原则:
  - 继续留在 `velocity + true tokenizer + safe-band` 家族内。
  - 只扫安全旋钮，不重开 endpoint / I2SB。
- 允许的改动:
  - `tokenizer_structured_temperature`
  - `tokenizer_global_gate_scale`
  - `w_kinetic`
  - 必要时只做很小幅的 tokenizer-safe rollback
- 当前 packet:
  - [phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r2_seed42_b20a1.json)
  - [2026-06-13-phase2-vel-tok32-safe-rescan-r2.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r2.md)
- 停止规则:
  - 第一批 settled authority 点只要进入 `0.40+`，立即 archival stop。
  - 只要读到 `0.70+`，直接判 complete failure。
  - 一旦出现 `0.70+`，结论同时回写到 family 级别: 该路数不再作为 Distinct5 正式主线候选。
  - 如果它仍然不能打破 `0.701666 / 0.381724`，则 safe-family sweep 视为用尽。

### 队列2: `vel_structure_control_reentry`

- 只有在满足以下任一条件后才进入:
  - `safe_rescan_r2` 明确失败并证明 safe-family sweep 已用尽；
  - 或者先得到新的、更强的 in-band 父本。
- 仍坚持:
  - `velocity`，不回到 endpoint。
  - structure control 必须是 training-side 工具，不是 style 放大器。
- 允许的低阶校正:
  - tokenizer-guided output appearance alignment
  - but only as a conservative same-family probe after a clean in-band structure point
  - not as a replacement for tokenizer or solver quality
- structure-side queue policy:
  - `topogate -> appalign -> pnp -> topo_anchor`
  - `queued_reference` rows are documentation-only and must not become automatic launch targets
  - active structure rows now carry their own watch fields and close into the next same-lane successor after plateau / LPIPS stop
- 允许的结构工具:
  - latent lowpass / edge-aware content correction
  - adaptive skip / PnP self-inject
  - `semantic_self_topology_gate / semantic_self_topology_blend` on `legacy_semantic_crossattn`
  - 轻量 kinetic / topology 约束
- queued reference:
  - [phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_seed42_b20a1.json)
  - [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085.md)
  - [phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_semantic_topogate_k085_appalign_seed42_b16a1.json)
  - [2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085-appalign.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-semantic-topogate-k085-appalign.md)
  - [phase2_vel_tok32_safe_pnp_selfinject_seed42_b16a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_pnp_selfinject_seed42_b16a1.json)
  - [2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-pnp-selfinject.md)
  - [phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_semantic_topogate_k085_seed42_b20a1.json)
  - [2026-06-13-phase2-vel-tok32-semantic-topogate-k085.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-semantic-topogate-k085.md)
  - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
  - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)

### 队列3: `i2sb_diagnostic_only`

- `true I2SB` 只保留为实现 / 理论验证。
- current preferred packet:
  - [phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
  - [phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- 允许做:
  - NFE 对照
  - noise schedule 对照
  - endpoint parameterization 对照
  - refreshed-tokenizer diagnostic packets such as:
    - [phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
    - [phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_tok32_semantic_topogate_sigma0p02_residual_seed42_b20a1.json)
- 不允许做:
  - 未经廉价预读就直接占用正式远程训练 lane
  - 任何 `LPIPS >= 0.40` 结果回流正式晋升队列

## 远程执行规则

- 3060 上同一时间只保留一条正式训练 lane。
- 训练中的远程 `CLIP-S + LPIPS` 是收敛与停训依据，不接受 post-train 重新解释。
- 每个 retained checkpoint 都要在远程完成 `CLIP-S + LPIPS` 评估。
- 第一批 settled checkpoint 就是 authority:
  - `>= 0.70` 直接 fail-stop；
  - `0.40-0.70` 直接 archival stop。
- `>= 0.70` 的点不仅停当前包，还要被视为该 family 失去 Distinct5 正式晋升资格的直接证据。
- structure-side packet 的训练日志必须保留 `content_lowpass_anchor` 与 `content_edge_anchor`。
- exact-I2SB diagnostic packet 的训练日志必须保留 `bridge_noise_schedule_exact`，避免把历史 heuristic 噪声误记成 true I2SB。
- true-tokenizer packets should now also be read through tokenizer observability, not only board metrics.
  - minimum structured-tokenizer reads to keep:
    - `structured_style_tokenizer_attn_entropy`
    - `structured_style_tokenizer_attn_effective_count`
    - `structured_style_tokenizer_attn_max`
    - `structured_style_tokenizer_gate_mean`
    - `structured_style_tokenizer_mask_mean`
    - `structured_style_tokenizer_spatial_map_abs`
    - `structured_style_tokenizer_global_gate_abs`
- 不允许再用“后面也许会掉下来”来继续烧正式训练资源。
- 首轮健康检查在 `30s` 内完成。
- 目标显存带保持在 `9.0-10.8 GiB`，硬上限按 `< 11.0 GiB` 执行。
- 远程训练不得依赖本地 GPU 或网络回传后再补 eval。

## 当前状态

- 当前 formal lane 已关闭:
  - `vel_tok32_safe_rescan_r2`
  - latest settled authority point is now `epoch_0008`
  - transfer `0.672774 / 0.389067`
  - all-pairs `0.700669 / 0.384913`
  - closure reason:
    - `in_band_style_plateau`
  - interpretation:
    - safe-family 证明了 true tokenizer 可以稳定维持 in-band
    - 但它没有做出 promotable safe-shelf break
- 当前 active next step:
  - `vel_tok32_safe_semantic_topogate_k085`
  - first `b20a1` launch hit the runtime guard at `11093 MiB`
  - the preferred relaunch packet is now `b16a1`
  - current live read:
    - `epoch_0002` is now the latest settled point
    - transfer `0.671915 / 0.361009`
    - all-pairs `0.700605 / 0.357866`
  - interpretation:
    - `epoch_0001` was the first real in-band structure-side recovery candidate after the safe-family plateau close
    - but the latest settled point has slipped back below the formal safe shelf while staying very clean on LPIPS
    - so the lane remains scientifically alive, but it is not yet a promoted shelf break
  - next low-intrusion follow-on if qualitative review still points to tone mismatch:
    - `vel_tok32_safe_semantic_topogate_k085_appalign`
    - same parent and same topology-gate family
    - only adds tokenizer-guided latent appearance alignment before trying a different attention family
  - structure-side close rule:
    - after `4` settled checkpoints, if the active packet is still below the formal safe shelf and no new best lands in the newest-2 window, hand off to the next same-lane successor
- 所有 round2 endpoint / I2SB 文档都应按以下标准重读:
  - `0.40-0.70` = archival only
  - `0.70+` = complete failure
