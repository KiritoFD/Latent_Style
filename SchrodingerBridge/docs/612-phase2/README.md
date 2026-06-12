# Phase 2 — 基于 612 回顾的手术级重构计划

## 诊断总结

**核心矛盾**: Style vs Structure Trade-off
- Velocity 模式: LPIPS 好 (0.32) 但 style 天花板低 (0.70)
- Endpoint 模式: style 可达 0.73 但 LPIPS 崩溃 (0.60+)
- 根因: Endpoint 预测 x_1 是"重绘"，Velocity 预测 delta 才是"编辑"

**关键发现**: WikiArt512 上 LBM 可达 0.79/0.31，说明模型能力足够，瓶颈在 Distinct5 更难区分

## 硬门槛

- `content_lpips >= 0.70`:
  - 直接判定为失败
  - 立即停掉远程训练，不再等待“后续也许会掉下来”
- `0.40 <= content_lpips < 0.70`:
  - 不可晋升
  - 仅保留为风格上限或结构崩溃对照
- 只有 `content_lpips < 0.40` 的线才有资格继续占用正式远程训练资源

## 当前结论

- `true I2SB + pure_latent_spatial` 代码路径现在已经被修正成真正的随机桥 runtime
- 但修正后的 `rtfix epoch_0001` 结果是:
  - transfer `0.724444 / 0.712723`
  - all-pairs `0.724472 / 0.707551`
- 这说明:
  - style 很强
  - 但 `LPIPS 0.7+` 仍然是完全失败
  - 因此 I2SB / endpoint 路线当前不再作为远程主训练线推进
- 处理原则:
  - I2SB 代码保留, 作为理论和实现分支
  - 但 Distinct5 训练计划切到结构优先的 Phase 2 路线

## 三刀手术

### 第一刀：删除 (Day 1)

| 删除项 | 文件 | 原因 |
|--------|------|------|
| Endpoint 作为主线 | configs, losses.py 模式选择 | 预测 x_1 = 重绘，不可接受；当前 Distinct5 上已被 `LPIPS >= 0.70` 判死 |
| DINO 依赖 | tok_a/b/c/d tokenizer 入口 | 已有纯潜空间能力证明 |
| Heuristic Losses | anisotropic, stokes 等 | Endpoint 模式下保结构的缝补，换 Velocity 后有害 |

### 第二刀：Tokenizer 升级 (Day 1-2)

| 改动 | 文件 | 内容 |
|------|------|------|
| query_extractor 加深 | semantic_tokenizer.py | 2层Conv → 4 ResBlock |
| 位置编码 | semantic_tokenizer.py | 2D Sinusoidal PE on queries |
| 扩大聚类 | semantic_tokenizer.py | clusters: 16 → 32 |
| Global-Spatial 关联 | semantic_tokenizer.py | global_code = GAP(spatial_map) + bias |

### 第三刀：PC Solver 破局 (Day 1-2)

| 改动 | 文件 | 内容 |
|------|------|------|
| Latent Content Correction | model.py | 低频 MSE 校正替代 DINO Gate |
| SDE 延迟加噪 | losses.py | t 在 [0.2,0.8] 才加噪声 |

## 实验队列

### 队列1: vel_pattn_enhanced_tok (训练)
- Velocity + enhanced PureLatentSpatialTokenizer (32 clusters, ResBlock, PE)
- k-manifold kinetic + pattn proximal
- 目标: 把 style 天花板从 0.70 推到 0.72+ (Distinct5)
- promotion gate:
  - 必须保持 `LPIPS < 0.40`
- historical config anchors:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\inmortal_k_manifold_seed42_b16.json`
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\inmortal_xpred_kmanifold_pattn_seed42_b16.json`

### 队列2: eval_only_pc_solver (仅评估)
- 使用现有 xpred+pattn ckpt (style=0.73)
- solver=solver_pc + Latent Content Correction
- 不重新训练
- 目标: 证明 "Training for Style, Inference for Structure"
- reference config anchor:
  - `G:\GitHub\Latent_Style\SchrodingerBridge\configs\aaai2027\round1_full_sweep\aaai2027_round1_solver_pc_seed42_b8a2.json`

### 队列3: vel_kman_pattn_kin_sweep (训练)
- 基于队列1
- 扫描 `w_kinetic = 0.5 / 1.0 / 2.0`
- 目标:
  - 找到 style 提升与 LPIPS 保持之间的最佳点

## 资源规则

- 当前远程主线:
  - 不再给 `LPIPS >= 0.70` 的 endpoint / I2SB 线继续训练时间
- 当前代码策略:
  - DINO 继续退休
  - I2SB 保留为实现能力
  - 远程实验优先级切到 `velocity + tokenizer enhancement + solver_pc`
