# Phase 2 — 基于 612 回顾的结构优先重构计划

## 核心判断

**真正的问题不再是“如何把 style 硬推高”**，而是:
- 如何在 `content_lpips < 0.40` 的结构安全带内，持续抬高 Distinct5 的 style，而不是复读 `0.70x / 0.60+`
- Velocity 线目前大致停在 `0.70 / 0.32`
- Endpoint / I2SB 线虽然能到 `0.72-0.73` style，但已经反复落在 `0.60-0.70+` LPIPS
- 因而这些高 style 点不能再被当成 paper-facing frontier，连“高风险可追”都不算

**关键发现**:
- WikiArt512 上 LBM 已经证明模型能力够强，可到 `0.79 / 0.31`
- Distinct5 的难点不是模型完全不会生成，而是更难在结构不崩的前提下把五类风格拉开

## 晋升门槛

- `content_lpips >= 0.70`
  - 完全失败
  - 立刻停掉远程正式训练
  - 结项状态记为 `stopped_lpips_fail`
  - 对应 family 退出 Distinct5 正式晋升队列，除非后续从安全父本重新设计
- `0.40 <= content_lpips < 0.70`
  - 仅可归档，不可晋升
  - 可以保留为理论 / 实现 / 对照证据，但必须退出唯一正式远程训练 lane
  - 不再允许用“style 够高”来包装成 compromise、frontier 或 next-step candidate
- `content_lpips < 0.40`
  - 才有资格继续占用正式远程训练资源
- `style >= 0.72`
  - 是必要条件，但不是充分条件
  - 只要 LPIPS 出带，这个点就不算成功

## Phase 2 成功阶梯

- Stage A
  - 先在 `LPIPS < 0.40` 内稳定超过当前安全 shelf `all-pairs 0.701666 / 0.381724`
- Stage B
  - 把 in-band best 推到 `all-pairs style >= 0.705` 且 `LPIPS <= 0.380`
- Stage C
  - 再把 in-band best 推到 `all-pairs style >= 0.710` 且 `LPIPS <= 0.370`
- Paper-facing long target
  - `style >= 0.72` 且 `LPIPS <= 0.35`
- 解释
  - Phase 2 不再接受“先到 0.72，再想办法救结构”的执行逻辑
  - 所有正式 lane 都必须沿着上述安全阶梯前进

## 当前结论

- `true I2SB + pure_latent_spatial` 代码路径已经被修正成真正的随机桥 runtime
- 但修正后的 `rtfix epoch_0001` 是:
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- residual endpoint 重参数化后的首个 settled 点是:
  - transfer `0.688376 / 0.571735`
  - all-pairs `0.697686 / 0.569086`
- 结论非常明确:
  - style 强
  - exact-I2SB 的结构仍然长期处在失败或 archival-only 区间
  - 因此 endpoint / I2SB 不再是 Distinct5 的远程主训练线
- 进一步收紧解释:
  - 旧 round2 中所有 `LPIPS 0.40-0.70` 的点，也不再叫“可推进 compromise”
  - 它们只是历史诊断证据

## 远程主线执行规则

- 3060 上同一时间只保留一条正式训练 lane
- 训练中的远程 `CLIP-S + LPIPS` 仍然是收敛与停训依据
- 第一批 settled checkpoint 就是硬闸门:
  - 若 `LPIPS >= 0.70`，立即 fail-stop
  - 若 `0.40 <= LPIPS < 0.70`，归档并让出主线，除非该 run 明确只是 infra / theory validation
- 不允许再用“后面也许会掉下来”作为继续烧正式训练资源的理由
- `eval_only` 型 solver 实验可以做，但不能阻塞唯一正式训练 lane

## 当前 Phase 节点（2026-06-13 07:30）

- `vel_pattn_enhanced_tok` 已经在 `epoch_0006` 关闭
  - best `epoch_0002`
    - transfer `0.673934 / 0.384340`
    - all-pairs `0.701666 / 0.381724`
  - latest `epoch_0006`
    - transfer `0.668831 / 0.370651`
    - all-pairs `0.698086 / 0.367844`
- 解释:
  - 这条线始终留在 `LPIPS < 0.40` 安全带内
  - 但 `epoch_0002 -> epoch_0006` 没有继续抬 style，已经进入平台震荡
  - 因此它不是失败线，但也不是值得继续占用正式训练 lane 的突破线
- `eval_only_pc_solver` 读取已经完成
  - transfer `0.729014 / 0.621056`
  - all-pairs `0.735295 / 0.611310`
  - 这说明 `solver_pc` 没能把 style-strong `xpred + pattn` 父本拉回结构安全带
  - 因而 queue2 只保留为 archival evidence，下一条正式候选重新回到 training-side 结构改造
- `vel_pattn_topology_anchor` 已关闭
  - closure point:
    - `epoch_0002`
    - transfer `0.680803 / 0.417910`
    - all-pairs `0.706132 / 0.413976`
  - 解释:
    - style 有抬升
    - 但 LPIPS 越过 `0.40`
    - 因而只能归档，不能继续占 formal lane
    - 这也意味着 topology-anchor 类补丁暂时不能排在 tokenizer-safe sweep 之前
- `true I2SB` fallback ladder 已全部闭环并退出主线
  - `sigma=0.25`
    - all-pairs `0.719743 / 0.725755`
  - `sigma=0.10 warm_vel2`
    - all-pairs `0.702178 / 0.711280`
  - `sigma=0.10 + pattn + topo anchor`
    - all-pairs `0.713362 / 0.684586`
  - `sigma=0.02 + pattn + topo anchor`
    - all-pairs `0.709801 / 0.675418`
  - `sigma=0.02 + residual endpoint`
    - transfer `0.688376 / 0.571735`
    - all-pairs `0.697686 / 0.569086`
  - 结论:
    - residual reparameterization 只把 LPIPS 从 `~0.67` 拉到 `~0.57`
    - 仍然远高于 `LPIPS < 0.40` 的 paper gate
    - 因而 exact-I2SB 在 Distinct5 上只保留实现与理论价值，不再占用正式训练 lane
- 当前动作:
  - 远程正式训练 lane 已切到:
    - `aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1`
    - launch time `2026-06-13 07:49`
  - 首轮健康检查:
    - `30s health = 10073 MiB`
    - later runtime guard mis杀 occurred during epoch-end eval offload:
      - `RUNTIME_UNDER_BAND_STOP used=2101MiB floor=9216MiB`
    - 当前状态:
      - `epoch_0001.pt` 与 `epoch_0002.pt` 已保存
      - 当前第一个有效 settled authority 点是 `epoch_0002`
        - transfer `0.673024 / 0.390256`
        - all-pairs `0.700342 / 0.387609`
      - 第二个有效 settled authority 点是 `epoch_0003`
        - transfer `0.668702 / 0.364875`
        - all-pairs `0.698072 / 0.361798`
      - 第三个有效 settled authority 点是 `epoch_0004`
        - transfer `0.673399 / 0.376463`
        - all-pairs `0.701161 / 0.374695`
      - 第四个有效 settled authority 点是 `epoch_0005`
        - transfer `0.670604 / 0.375912`
        - all-pairs `0.699187 / 0.373331`
      - launcher guard 已修复
      - 同一 run 已从本地 `epoch_0001` 续跑
      - relaunch 30s health `10151 MiB`
      - 两个 authority 点都还在 continuation band 内，因此 formal lane 继续保留
      - 现在三个 authority 点都还在 continuation band 内，因此 formal lane 继续保留
      - 它仍然没有超过上一条安全父本 `0.701666 / 0.381724`
      - 但 `epoch_0004` 已经做到:
        - style 只略低于旧 shelf
        - LPIPS 明显优于旧 shelf
      - 同时它也严格优于本 packet 的 `epoch_0003` 点
      - `epoch_0005` 则继续往更低 LPIPS 方向推了一步，但 style 没有继续抬升
      - 它目前仍在安全带内，所以不该因为追求过高远期目标而误杀
      - 但 Phase 2 现在只认安全带内 breakout:
        - 如果后续继续只是做这种“小幅降 LPIPS、不抬 style”的点，就进入 plateau 审核
        - 如果 `epoch_0006+` 仍不能越过 `0.701666 / 0.381724`，则下一步优先做 safe-family rescan，而不是立刻上更激进结构补丁
      - `epoch_0001` 现在只记为 `stale_pending`，不再把 live state 错报成 eval-pending
    - 本地 watcher 已挂起:
      - `watch_phase2_velocity_handoff.py --run-name aaai2027_phase2_vel_tok32_pos_refresh_seed42_b20a1 --wait --execute --handoff-mode stop_only`
  - 当前正式候选仍然只允许来自 `velocity + true tokenizer + training-side structure control`

## 三刀手术

### 第一刀：退出主线的东西

| 项 | 处理 | 原因 |
|---|---|---|
| Endpoint / I2SB 作为 Distinct5 主线 | 退出主队列 | `LPIPS 0.7+` 是 complete failure，`0.4-0.7` 也只剩 archival evidence |
| DINO 依赖 | 保留归档，不在主线继续投入 | 纯潜空间路线已足够成立，且 DINO 工程复杂度不值得 |
| 旧 heuristic structure 补丁 | 只做对照，不做主路径 | 大多是给 endpoint 擦屁股，不能解决主矛盾 |

### 第二刀：Tokenizer 升级

| 改动 | 文件 | 内容 |
|---|---|---|
| query_extractor 加深 | `semantic_tokenizer.py` | 2层 Conv → 4 个残差块 |
| 位置编码 | `semantic_tokenizer.py` | query 加 2D sinusoidal positional encoding |
| 扩大聚类 | `semantic_tokenizer.py` | `num_clusters: 16 -> 32` |
| Global-Spatial 关联 | `semantic_tokenizer.py` | `global_code` 从 `spatial_map` 聚合得到偏置关联 |

### 第三刀：Structure-First Solver

| 改动 | 文件 | 内容 |
|---|---|---|
| Latent Content Correction | `model.py` | 用潜空间低频内容校正替代 DINO gate |
| PC / Corrector 路径 | `model.py` | 先拿 style，再把结果往内容原点拉回 |
| 延迟或约束式加噪 | `losses.py` | 只在结构可控的时间段尝试噪声，不再把 SDE 当主训练假设 |

## 实验队列

### 队列1: `vel_pattn_enhanced_tok`（已于 `epoch_0006` 关闭）

- 组合:
  - velocity
  - enhanced `PureLatentSpatialTokenizer`
  - `manifold_adaptive_split`
  - `crossattn_texture`
- 目标:
  - 在 `LPIPS < 0.40` 内把 style 顶到 `0.72+`
- 继续条件:
  - 新 settled 点仍然留在 `< 0.40`
  - 且 style 或 Pareto 面仍然在变好
- 退出条件:
  - 第一批或后续 settled 点落到 `0.40+`
  - 或 style 在安全带内明显停滞

### 队列2: `eval_only_pc_solver`（已完成，negative readout）

- 复用已有 style 强 ckpt
- 不重新训练，只测 `solver_pc` / content correction 能否把结构拉回
- 目标:
  - 验证 “training for style, inference for structure” 是否成立
- 注意:
  - 这是辅助判断，不替代主训练队列
  - 当前结果:
    - transfer `0.729014 / 0.621056`
    - all-pairs `0.735295 / 0.611310`
  - 结论:
    - style 仍高，但结构仍处于 archival only 区间
    - inference-time corrector 不足以拯救 style-strong endpoint 父本

### 队列3: `vel_tok32_pos_refresh`（已于 `epoch_0006` 关闭）

- 父本:
  - `vel_pattn_enhanced_tok` 的安全带 best 点 `epoch_0002`
  - config anchor:
    - [inmortal_k_manifold_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_k_manifold_seed42_b16.json)
    - [inmortal_xpred_kmanifold_pattn_seed42_b16.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/inmortal_xpred_kmanifold_pattn_seed42_b16.json)
- packet note:
  - [2026-06-13-phase2-vel-tok32-pos-refresh.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-pos-refresh.md)
- formal config:
  - [phase2_vel_tok32_pos_refresh_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_pos_refresh_seed42_b20a1.json)
- 改动:
  - deeper `query_extractor`
  - 2D positional encoding
  - `num_clusters = 32`
  - stronger global-spatial coupling
- 目标:
  - 在不越过 `LPIPS 0.40` 的前提下突破 `all-pairs 0.701666 / 0.381724`
  - 若能做到 `style >= 0.705` 且 `LPIPS <= 0.380`，才算进入下一阶段
- 退出条件:
  - 第一批 settled 点进入 `0.40+`
  - 或仍停在 `0.70x / 0.38x` 平台
- 当前结论:
  - best `epoch_0004`
    - transfer `0.673399 / 0.376463`
    - all-pairs `0.701161 / 0.374695`
  - closure `epoch_0006`
    - transfer `0.671522 / 0.385051`
    - all-pairs `0.699725 / 0.381878`
  - 解释:
    - 线始终留在 `LPIPS < 0.40` 带内
    - 但没有突破旧 shelf `0.701666 / 0.381724`
    - 因而正式 lane 交给 safe-family rescan

### 队列4: `vel_safe_family_rescan`（tok32 若平盘时的第一后续）

- 原则:
  - 不离开 `velocity + tokenizer + in-band` 主族
  - 不重新打开 endpoint / I2SB，不先上 topology-anchor
- 允许的第一批扫描:
  - tokenizer temperature / structured temperature
  - global-spatial coupling scale
  - `w_kinetic` 的安全带内小范围 rescan
  - 必要时补一个更高 cluster 数，但仍从安全父本 warm-start
- 启动条件:
  - `vel_tok32_pos_refresh` 收敛但未突破 Stage B
  - 且所有 settled 点都保持在 `LPIPS < 0.40`
- 当前具体 packet:
  - [2026-06-13-phase2-vel-tok32-safe-rescan-r1.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-safe-rescan-r1.md)
  - [phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_safe_rescan_r1_seed42_b20a1.json)
  - 当前状态:
    - 已启动为正式远程 lane
    - 30s health `10142 MiB`
    - 首个 settled 点:
      - `epoch_0001`
      - transfer `0.672934 / 0.384740`
      - all-pairs `0.700686 / 0.383351`
    - 解释:
      - 仍在 `LPIPS < 0.40` 带内
      - 但暂未超过旧 shelf
      - 因而继续跑，但暂不视为 breakout

### 队列5: `vel_structure_control_reentry`（降级为第三顺位）

- 仍坚持 `velocity`，不回到 endpoint
- 前提:
  - 必须先有更强的 in-band 父本
  - structure patch 不能再直接拿当前 shelf 当跳板去冒 `0.40+` 风险
- 允许的结构工具:
  - latent lowpass / edge content correction
  - adaptive skip / PnP self-inject 仅作为结构工具，而非 style 放大器
  - 更轻的 kinetic / topology 约束，但只作为后续候选而非默认下一步
- queued reference packet:
  - [2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/experiments/2026-06-13-phase2-vel-tok32-topo-anchor-reentry.md)
  - [phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_tok32_topo_anchor_k075_seed42_b20a1.json)
- 当前判断:
  - `topology_anchor` 已经证明这类补丁很容易把 line 推出安全带
  - 因此只在 tokenizer-safe sweep 用尽后再进入

### 队列6: `i2sb_diagnostic_only`（非正式 lane）

- `true I2SB` 只保留为实现/理论验证
- 允许做 NFE / noise schedule / endpoint parameterization 对照
- 但任何 `LPIPS >= 0.40` 的 I2SB 结果都不回正式训练队列

## 代码与文档策略

- DINO 继续退休，除非后续出现压倒性 board 优势
- true I2SB 代码保留，作为实现能力和理论资产
- true I2SB 只允许 `diagnostic-only` 运行；除非先出现廉价读数证明 `LPIPS < 0.40`，否则不再进入 Distinct5 formal queue
- 但 paper-facing Distinct5 主计划已经切换到:
  - `velocity + tokenizer-safe sweep + deferred training-side structure control`
- 所有 round2 endpoint / I2SB 文档都应按这个门槛重读:
  - `0.40-0.70` 是 archival only
  - `0.70+` 是 complete failure
