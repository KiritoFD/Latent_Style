# Phase 2 — 基于 612 回顾的结构优先重构计划

## 核心判断

**真正的问题不再是“如何把 style 硬推高”**，而是:
- 如何在 `content_lpips < 0.40` 的结构安全带内，把 Distinct5 的 style 从 `0.70` 推到 `0.72+`
- Velocity 线目前大致停在 `0.70 / 0.32`
- Endpoint / I2SB 线虽然能到 `0.72-0.73` style，但已经反复落在 `0.60-0.70+` LPIPS
- 因而这些高 style 点不能再被当成 paper-facing frontier

**关键发现**:
- WikiArt512 上 LBM 已经证明模型能力够强，可到 `0.79 / 0.31`
- Distinct5 的难点不是模型完全不会生成，而是更难在结构不崩的前提下把五类风格拉开

## 晋升门槛

- `content_lpips >= 0.70`
  - 完全失败
  - 立刻停掉远程正式训练
  - 结项状态记为 `stopped_lpips_fail`
- `0.40 <= content_lpips < 0.70`
  - 仅可归档，不可晋升
  - 可以保留为理论 / 实现 / 对照证据，但必须退出唯一正式远程训练 lane
- `content_lpips < 0.40`
  - 才有资格继续占用正式远程训练资源
- `style >= 0.72`
  - 是必要条件，但不是充分条件
  - 只要 LPIPS 出带，这个点就不算成功

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
      - `epoch_0001.pt` 已保存
      - `first settled eval` 仍 pending
      - launcher guard 已修复
      - 同一 run 已从本地 `epoch_0001` 续跑
      - relaunch 30s health `10151 MiB`
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

### 队列3: `vel_tok32_pos_refresh`（下一条正式候选）

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
- 退出条件:
  - 第一批 settled 点进入 `0.40+`
  - 或仍停在 `0.70x / 0.38x` 平台

### 队列4: `vel_structure_control_reentry`（仅在队列3拿到更强安全带父本后启动）

- 仍坚持 `velocity`，不回到 endpoint
- 在训练侧引入结构控制，而不是 solver-only 或 I2SB rescue:
  - lighter kinetic + topology anchor
  - latent lowpass / edge content correction
  - adaptive skip / PnP self-inject 仅作为结构工具，而非 style 放大器
- 目标:
  - 把结构约束直接作用在安全带 velocity 父本上
  - 验证真正有价值的是 training-side structure control，而不是 stochastic endpoint

### 队列5: `i2sb_diagnostic_only`（非正式 lane）

- `true I2SB` 只保留为实现/理论验证
- 允许做 NFE / noise schedule / endpoint parameterization 对照
- 但任何 `LPIPS >= 0.40` 的 I2SB 结果都不回正式训练队列

## 代码与文档策略

- DINO 继续退休，除非后续出现压倒性 board 优势
- true I2SB 代码保留，作为实现能力和理论资产
- true I2SB 只允许 `diagnostic-only` 运行；除非先出现廉价读数证明 `LPIPS < 0.40`，否则不再进入 Distinct5 formal queue
- 但 paper-facing Distinct5 主计划已经切换到:
  - `velocity + tokenizer enhancement + structure-first solver`
- 所有 round2 endpoint / I2SB 文档都应按这个门槛重读:
  - `0.40-0.70` 是 archival only
  - `0.70+` 是 complete failure
