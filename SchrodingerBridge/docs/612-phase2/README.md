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
- 结论非常明确:
  - style 强
  - 结构完全失败
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

## 当前 Phase 节点（2026-06-13 04:10）

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
- 当前动作:
  - 正式训练 lane 已释放
- `eval_only_pc_solver` 读取已经完成
    - transfer `0.729014 / 0.621056`
    - all-pairs `0.735295 / 0.611310`
  - 这说明 `solver_pc` 没能把 style-strong `xpred + pattn` 父本拉回结构安全带
  - 因而 queue2 只保留为 archival evidence，下一条正式候选重新回到 training-side 结构改造
  - 当前新的训练侧假设:
    - 不再只靠 `w_kinetic` 压住结构
    - 改为 `lighter kinetic + latent topology anchor`
    - 直接约束 endpoint 的低频拓扑和边缘骨架
  - 当前运行态:
    - `aaai2027_phase2_vel_pattn_topo_anchor_k075_seed42_b22a1` 已关闭
    - closure point:
      - `epoch_0002`
      - transfer `0.680803 / 0.417910`
      - all-pairs `0.706132 / 0.413976`
    - 解释:
      - style 有抬升
      - 但 LPIPS 越过 `0.40`
      - 因而只能归档，不能继续占 formal lane
    - 当前 active lane:
      - `aaai2027_phase2_i2sb_pattn_topo_anchor_sigma0p10_warm_vel2_seed42_b22a1`
      - `true I2SB + pure_latent_spatial + topology anchor + internal proximal rescue`

## 三刀手术

### 第一刀：退出主线的东西

| 项 | 处理 | 原因 |
|---|---|---|
| Endpoint / I2SB 作为 Distinct5 主线 | 退出主队列 | `LPIPS 0.7+` 已经证明是完全失败 |
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

### 队列3: `vel_pattn_topology_anchor`（当前下一正式候选）

- 以前一条 velocity packet 为父本
- 第一条候选 config:
  - [phase2_vel_pattn_topo_anchor_k075_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_vel_pattn_topo_anchor_k075_seed42_b22a1.json)
- 目标:
  - 用更轻的 kinetic 换取 style 抬升
  - 用 topology anchor 而不是 solver-only 修补来守住结构
- 前提:
  - 队列2 已经给出否定答案:
    - 仅靠 inference corrector 不足以把结构拉回安全带
  - 因而若继续推进，就必须回到训练时结构约束，而不是继续复用同一 style-heavy 父本做 solver-only 修补

### 队列4: `true_i2sb_topology_anchor`（若队列3仍平则切换）

- config:
  - [phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase2_i2sb_topo_anchor_sigma0p25_seed42_b22a1.json)
- 目标:
  - 保留 `true I2SB` 的 style headroom
  - 用训练期 endpoint topology anchor 直接救结构
- 解释:
  - 如果 velocity 仍然无法突破 `0.70x / 0.39x` 这个棚顶，那么更合理的下一步不是更复杂的 velocity 补丁
  - 而是让 `true I2SB` 重新上场，但只带最干净的一层结构锚

## 代码与文档策略

- DINO 继续退休，除非后续出现压倒性 board 优势
- true I2SB 代码保留，作为实现能力和理论资产
- 但 paper-facing Distinct5 主计划已经切换到:
  - `velocity + tokenizer enhancement + structure-first solver`
- 所有 round2 endpoint / I2SB 文档都应按这个门槛重读:
  - `0.40-0.70` 是 archival only
  - `0.70+` 是 complete failure
