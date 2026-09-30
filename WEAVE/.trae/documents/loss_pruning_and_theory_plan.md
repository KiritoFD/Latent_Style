# Loss/模块砍除 + 代码精简 + 理论文档建立计划

## 摘要

基于 628/629 历史 ablation 数据深度分析 + 当前代码结构探索，提出 **6 个候选砍除方向**（4 主 + 2 辅）。每个方向有历史证据支撑，按风险/收益排序。同步建立 `docs/theory/SpectralODE_Bridge.md` 数学理论文档。最终交付：性能不下降 + 简洁干净的 codebase + 完善优雅的理论。

## 当前状态分析

### 已确立的 baseline（不可退化）
- **allpairs clip_style = 0.7293**（远程 0.7299，Δ=0.0006 噪声）
- **allpairs content_lpips = 0.3203**（远程 0.3420，本地略好）
- 验收阈值：clip ≥ 0.7243, lpips ≤ 0.3453

### core_keep 4 项（绝对不可砍）
| 模块 | 配置键 | 禁用 Δclip |
|------|--------|-----------|
| spectral_ode 架构 | `model.contract_family="620_spectral_ode"` | -0.0167 |
| endpoint AdaIN | `model.endpoint_adain_scale > 0` | -0.0142 |
| style 外推 | `model.style_extrap_alpha > 0` | -0.0016 |
| LL 谱 loss | `bridge.spectral_w_ll > 0` | -0.0042 |

### 当前 SpectralODEObjective620 实际使用的 loss（仅 4 项）
| Loss | 权重 | 628 证据 | 性质 |
|------|------|---------|------|
| `loss_ll` (w_ll) | 0.0（注释"≈0 锁低频"）但 core_keep | L7: -0.0042 | CORE |
| `loss_lh` (w_lh) | 1.0 | L9: +0.0010 移除反升 | HARMFUL |
| `loss_hl` (w_hl) | 1.0 | L9: +0.0010 移除反升 | HARMFUL |
| `loss_hh` (w_hh) | 2.0 | L8: ±0.0001 | DEAD |

### 关键矛盾（必须解决）
- **628 单项**：砍 lh/hl 反升 clip +0.0010（harmful）
- **629 S2 组合**：砍 lh/hl 后 clip 0.7292（边缘通过，但未反升）
- **结论**：需定向实验单独验证 lh/hl 砍除效果

### 代码结构发现（Phase 1 新发现）

#### 1. spectral_losses620.py 有 60+ 个占位 metric
- 行 113-201 全部 `zero = content.new_tensor(0.0)` 占位
- trainer.py 行 1523-1524 有 `metrics.setdefault("terminal_swd", 0.0)` 兜底
- **可安全删除**

#### 2. StyleConditioner620 有 3 个死分支
- `adapter_enabled=False`（clean_base_v2）→ dino_adapter 仍构造但不执行 `_adapt_dino`
- `local_cnn_enabled=False` → local_cnn/local_pool 不构造
- `text_enabled=False` → text_proj/null_text/null_image 不构造，`_apply_modality_dropout` 直接返回

#### 3. SpatialBridgeBlock620 有 5 种死 attn_mode + 3 个死机制
- clean_base_v2 用 "relu2"，其余 4 种（gated/gated_raw/style_select/sparsemax）是死代码
- `film_enabled=False` → film_proj/film_q_proj/style_bias_proj = None
- `style_moe_enabled=False` → 走非 MoE 路径
- `style_query_source="concat"` → 不走 content_dino/sa_out_only 分支

#### 4. integrate_transport 有 4 个死 ablation hooks
- `endpoint_adain_mode='wct'/'wct_diag'`（默认 'full'）→ WCT 分支死代码（~30 行）
- `multiband_adain_mode='two_level'`（默认 'single'）→ two_level 分支死代码（~45 行）
- `patch_adain_kernel > 0`（默认 0）→ patch 分支死代码（~25 行）
- `style_extrap_levels > 1`（默认 1）→ 多级 DWT 分支死代码（~10 行）

---

## 提议变更：6 个探索方向

### 方向 1: 砍除 spectral_w_hh (DEAD loss) + head_hh ⭐高优先

**证据**：
- 628 L8 单项禁用：Δclip=+0.0000, Δlpips=+0.0018 → "HH 频带几乎无影响"
- SpectralODEObjective620 第 31/109/110 行实际读取 w_hh
- head_hh 占用参数：dim→latent_channels 的 conv（约 1.7K 参数）

**修改**：
- `src/spectral_losses620.py`: 删除 `loss_hh` 计算 + `w_hh` 读取
- `src/spectral_bridge620.py`: 删除 `head_hh` 模块 + forward 返回值
- 推理 `integrate_transport`: 删除 hh 子带积分
- `configs/clean_base_v2*.json`: 删除 `spectral_w_hh` 字段

**风险**：低。628 已验证单项 dead。
**验证**：本地训练 10 epoch + eval，clip ≥ 0.7243

### 方向 2: 定向验证 spectral_w_lh + spectral_w_hl 砍除 ⭐高优先

**证据**：
- 628 L9 单项禁用：Δclip=+0.0010（反升，harmful）
- 628 Round 1 报告："LH/HL 频带是噪声项"
- 629 S2 组合测试：砍 lh/hl 后 clip 0.7292（未反升，矛盾未解）
- **矛盾原因猜测**：629 S2 同时砍了其他 13 项 dead loss，产生负面交互掩盖了 lh/hl 的反升效应

**实验设计**（定向 ablate，只砍 lh/hl，不砍其他）：
- 配置 A: clean_base_v2 baseline（w_lh=1.0, w_hl=1.0）→ 已有结果 clip=0.7293
- 配置 B: clean_base_v2 + w_lh=0, w_hl=0（只砍 lh/hl）
- 训练 10 epoch + eval，对比 clip/lpips
- 若 clip ≥ 0.7293（反升或持平）→ 砍除 lh/hl + head_lh + head_hl
- 若 clip < 0.7243 → 保留，记录矛盾未解

**修改**（若验证通过）：
- 同方向 1，针对 lh/hl

**风险**：中。需定向实验，但历史证据支持反升假设。
**验证**：本地训练 10 epoch + eval

### 方向 3: 清理 SpectralODEObjective620 的 60+ 占位 metric

**证据**：
- spectral_losses620.py 行 113-201 全为 `zero = content.new_tensor(0.0)` 占位
- trainer.py 行 1523-1524 有 `metrics.setdefault("terminal_swd", 0.0)` 兜底
- 占位 metric 不参与 loss，仅为兼容历史 logging

**修改**：
- `src/spectral_losses620.py`: 删除 60+ 个占位 key，只保留实际计算的 4 个 spectral loss + t_mean + noise_scale
- 验证 trainer.py 的 logging 是否因缺失 key 报错（setdefault 应兜底）

**风险**：低。trainer.py 有 setdefault。
**验证**：smoke test + 训练 1 epoch 确认 logging 正常

### 方向 4: 清理 StyleConditioner620 的 3 个死分支

**证据**：
- clean_base_v2_local.json: `style_dino_adapter_enabled=false`, `style_local_cnn_enabled=false`, `style_text_enabled=false`
- 628/629 从未在 SpectralODE 路径启用这三个
- 代码分支完全死代码

**修改**：
- `src/style_encoder620.py`:
  - 删除 `adapter_enabled` 分支 + `dino_adapter` 模块 + `_adapt_dino` 方法
  - 删除 `local_cnn_enabled` 分支 + `local_cnn`/`local_pool` 模块
  - 删除 `text_enabled` 分支 + `text_proj`/`null_text_tokens`/`null_image_tokens`/`null_image_cls` + `_apply_modality_dropout`
- `src/spectral_bridge620.py`: 清理 StyleConditioner620 构造参数传递
- `src/config_schema.py`: 保留字段但标记 deprecated（不删，避免破坏遗留配置）

**风险**：低。clean_base_v2 不使用这些分支。
**验证**：smoke test

### 方向 5: 清理 SpatialBridgeBlock620 的死 attn_mode + 死机制分支

**证据**：
- clean_base_v2 用 `attn_mode="relu2"`，其余 4 种（gated/gated_raw/style_select/sparsemax）是死代码
- `film_enabled=false`, `style_moe_enabled=false`, `style_query_source="concat"` → 死分支
- 每个死 attn_mode 约 15-30 行代码

**修改**（保守，只删完全死的）：
- `src/blocks620.py`:
  - 删除 `attn_mode == "gated"` / `"gated_raw"` / `"style_select"` / `"sparsemax"` 四个分支
  - 保留 `"relu2"`（clean_base_v2 用）和默认 softmax 路径（安全 fallback）
  - 删除 `film_enabled` 分支 + `film_proj`/`film_q_proj`/`style_bias_proj` 模块
  - 删除 `style_moe_enabled` 分支 + `style_moe_router`/`k_proj_experts`/`v_proj_experts`
  - 删除 `style_query_source == "content_dino"` / `"sa_out_only"` 分支

**风险**：中。需确认无其他配置使用这些分支。
**验证**：smoke test + grep 确认无其他消费者

### 方向 6: 清理 integrate_transport 的死 ablation hooks

**证据**：
- `endpoint_adain_mode='wct'/'wct_diag'`（默认 'full'）→ WCT 分支死代码
- `multiband_adain_mode='two_level'`（默认 'single'）→ two_level 分支死代码
- `patch_adain_kernel > 0`（默认 0）→ patch 分支死代码
- `style_extrap_levels > 1`（默认 1）→ 多级 DWT 分支死代码
- core_keep 保留 endpoint_adain_scale 和 style_extrap_alpha，但只走 'full' 和 levels=1 路径

**修改**：
- `src/spectral_bridge620.py` `integrate_transport`:
  - 删除 WCT/wct_diag 分支（行 302-332，约 30 行）
  - 删除 multiband_adain 'two_level' 分支（行 333-377，约 45 行）
  - 删除 patch_adain 分支（行 378-411，约 25 行）
  - 删除 style_extrap_levels > 1 分支（行 290-298，约 10 行）
  - 保留默认 'full' + levels=1 路径

**风险**：低。clean_base_v2 不走这些分支。
**验证**：smoke test + eval

---

## 理论文档建立

### `docs/theory/SpectralODE_Bridge.md` 大纲

1. **引言**
   - 风格迁移的 Schrödinger Bridge 定位
   - 与 GAN/Diffusion 的对比优势

2. **数学基础**
   - Schrödinger Bridge 公式（SDE 形式）
   - Epislon-Schrodinger Bridge 的 OT 耦合
   - Flow Matching 框架

3. **FC-SB 理论：频域条件化**
   - 动机：低频保内容，高频传风格
   - Haar DWT 的正交性保证
   - 4 子带解耦的数学解释

4. **SpectralODEBridge620 架构**
   - 输入：latent → DWT → 4 子带 stack
   - 共享 backbone：SpatialBridgeBlock620
   - 4 独立速度头：v_LL/v_LH/v_HL/v_HH
   - Style 编码：DINO patches → StyleConditioner620

5. **训练目标**
   - 4 子带 FM loss 公式
   - w_ll≈0 的理论解释（锁低频保 LPIPS）
   - w_hh 的 dead 性质（628 证据）

6. **推理：谱域 Euler 积分**
   - 4 路独立积分 + iDWT 合成
   - Endpoint AdaIN：fiber 统计匹配
   - Style 外推：fiber 高频放大

7. **代码映射**
   - 理论概念 → 代码文件/类/函数映射表
   - core_keep 4 项的数学必要性

---

## 假设与决策

1. **决策**：方向 1（砍 hh）和方向 2（定向验证 lh/hl）是高优先，可能提升性能或简化模型
2. **决策**：方向 3-6 是 codebase 卫生，不影响性能但大幅减少代码量
3. **假设**：trainer.py 的 setdefault 能兜底占位 metric 删除（需验证）
4. **假设**：clean_base_v2 不使用任何死分支（Phase 1 grep 确认）
5. **决策**：理论文档独立成文，可独立阅读，含代码映射
6. **验证标准**：每个方向完成后 clip ≥ 0.7243, lpips ≤ 0.3453

## 验证步骤

1. 方向 1: 砍 hh 后 smoke + train 10 epoch + eval
2. 方向 2: 定向实验 lh/hl 砍除，train 10 epoch + eval
3. 方向 3: 删占位 metric 后 smoke + train 1 epoch 确认 logging
4. 方向 4: 清理 StyleConditioner620 后 smoke
5. 方向 5: 清理 Block620 后 smoke + grep 确认无其他消费者
6. 方向 6: 清理 integrate_transport 后 smoke + eval
7. 理论文档：同行评审（用户审阅）
8. 最终 git commit

## 执行顺序

1. **方向 1**（砍 hh）+ 验证
2. **方向 2**（定向实验 lh/hl）+ 验证 → 决定是否砍除
3. **方向 3**（删占位 metric）+ 验证
4. **方向 4**（清理 StyleConditioner620）+ 验证
5. **方向 5**（清理 Block620 死分支）+ 验证
6. **方向 6**（清理 integrate_transport 死 hooks）+ 验证
7. **理论文档**撰写（贯穿，但最后定稿）
8. **git commit**

## 风险与回退

- **方向 1 风险**：低，628 已验证 dead。回退：git checkout
- **方向 2 风险**：中，可能不反升。回退：保留 lh/hl，记录矛盾
- **方向 3-6 风险**：低，死代码清理。回退：git checkout
- **整体回退**：每个方向独立 commit，可逐项 bisect

## 预期成果

- 砍除 1-2 个 spectral loss（hh 必砍，lh/hl 视实验）
- 清理 200+ 行死代码（占位 metric + 死分支）
- 简化 StyleConditioner620 / Block620 / integrate_transport
- 建立 docs/theory/SpectralODE_Bridge.md 完整理论文档
- 性能不下降（clip ≥ 0.7243）
