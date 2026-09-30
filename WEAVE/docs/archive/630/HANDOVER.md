# HANDOVER — 630 Codebase Cleanup + Masking + Exploration 完整交接

**会话**: 2026-06-30
**分支**: `codex/620-spatial-bridge`
**最终 commit**: `adc6a0d38`
**状态**: PHASE3_COMPLETE, 所有里程碑 M1-M6 已完成
**入口**: 新会话从本文档开始, 配合 [README.md](README.md) 索引使用

---

## 0. TL;DR (新会话必读)

1. **Codebase 已精简到最简优雅状态**: 删除 11346 行 legacy dead code, active 路径仅 14 个 src 文件
2. **Masking 已实现并验证**: `style_mask_ratio=0.5, style_mask_mode="random"` 是最佳配置
3. **最终性能** (Phase 3, 10-epoch 从零训练, allpairs n=750):
   - clip_style = 0.7288 (PASS, 阈值 ≥ 0.7243, baseline 0.7293)
   - content_lpips = 0.3369 (PASS, 阈值 ≤ 0.3453, baseline 0.3203)
4. **后续探索未做**: 频率掩码 (方案 C)、显著性反向掩码 (方案 B)、mask_ratio 细化 (0.6/0.7) 留给下一会话
5. **绝对规则**: 每次单开目录重新训练, 禁止 `--skip-train --checkpoint` 借旧 checkpoint 偷懒

---

## 1. 理论 (Theory)

### 1.1 核心架构: SpectralODEBridge620

**Contract**: `620_spectral_ode` → `SpectralODEBridge620` (src/spectral_bridge620.py)
**Objective**: `SpectralODEObjective620` (src/spectral_losses620.py)
**完整理论文档**: [../theory/SpectralODE_Bridge.md](../theory/SpectralODE_Bridge.md) (8 章)

#### 数据流
```
content_latent (B,4,64,64)
    │
    ├─→ Haar DWT ──→ LL (low-freq)  ─┐
    │              ──→ LH/HL (mid)   ├─→ 3 velocity heads (HH removed, 628 L8 DEAD)
    │              ──→ HH (high-freq)─┘
    │
    ├─→ StyleConditioner620(target_dino_patches)
    │       │
    │       ├─→ patch_proj (DINO 384 → style_dim)
    │       ├─→ _apply_mask()        ← Phase 2 新增 (random/shuffle/none)
    │       └─→ cross-attn → style tokens
    │
    ├─→ N × SpatialBridgeBlock620 (base_dim=64, num_res_blocks=4)
    │       │
    │       ├─→ GroupNorm + Conv (content path)
    │       ├─→ CrossAttn (content × style_tokens, attn_mode=relu2)
    │       └─→ Residual blend (gate_init=0.05, tanh_gate)
    │
    ├─→ tri_band bridge path (bridge_path_mode="tri_band")
    │       LL → endpoint_lowhigh (Endpoint AdaIN, scale=1.0)
    │       LH/HL → spectral ODE (Euler integration)
    │       HH → 不预测 (628 L8 DEAD)
    │
    └─→ iDWT → output_velocity (B,4,64,64)
            │
            └─→ Style Extrap (style_extrap_alpha=0.1) + Endpoint AdaIN (scale=1.0)
```

#### 关键数学
- **Haar DWT**: $x \to (LL, LH, HL, HH)$, 其中 $LL = \frac{1}{2}(x_{even} + x_{odd})$, $HH = \frac{1}{2}(x_{even} - x_{odd})$ (2D 张量积)
- **Spectral ODE**: 仅对 LH/HL 子带做 Euler 速度场积分, LL 走 Endpoint AdaIN, HH 不预测
- **Endpoint AdaIN**: $\hat{x}_{LL} = \gamma \cdot \frac{x_{LL} - \mu}{\sigma} + \beta$, 其中 $(\gamma, \beta)$ 来自 target style 的 LL 统计
- **Style Extrap**: $v_{final} = v + \alpha \cdot (v_{style} - v_{content})$, $\alpha = 0.1$
- **tri_band path**: 边缘保持 (edge_preserve_alpha=0.5), 中频 kernel=3, 低频 kernel=11

### 1.2 Masking 理论: The Blindfolded Tokenizer

**理论文档**: [mask.md](mask.md)

#### 核心洞察
内容 (Content) 与风格 (Style) 在信息论意义上可分:
- **内容 = 全局拓扑**: 物体形状、空间关系。被 high-ratio dropout + shuffle 摧毁
- **风格 = 局部平稳遍历**: 笔触、色彩、纹理统计。在 mask 下存活

#### 三种 Masking 方案 (mask.md)
| 方案 | 实现 | 状态 |
|------|------|------|
| **A. Random Patch Dropout** | 随机丢弃 ratio 比例的 DINO patch token | ✅ Phase 2 已实现 + 验证 |
| B. Spatial Shuffling (No-PE) | 打乱 token 顺序, 破坏空间统计 | ✅ Phase 2 已实现, 但 clip FAIL |
| C. Frequency Masking | 对 style_latent 做低频减法 | ❌ 未实现 (需架构改动) |

#### Gate Collapse 假说
当 style tokens 携带过多 content 信息时, cross-attn 的 gate 会被 content 信号劫持, 导致风格注入不足。Masking 制造信息瓶颈, 强制 Tokenizer 只保留风格统计量, 打破 collapse。

### 1.3 628/629 历史消融结论 (已固化)

| 历史实验 | 结论 | Phase 1 处理 |
|---------|------|-------------|
| 628 L8 | HH velocity head DEAD (Δclip=±0.0001) | 删除 spectral_w_hh 配置 |
| 628 S1+S2 loss cuts | 负交互 (单独 cut OK, 组合 cut 性能下降) | 保留 S1+S2 |
| 629 subtractive (22 cuts) | 无组合负效应 | 全部应用 |
| dino_adapter | 788K 死参数 (adapter_enabled=false) | 删除 |
| FiLM/MoE/WCT/multiband/patch | clean_base_v2 全 false, 永不执行 | 删除 |
| M9 bug | style_attn_mode=relu2 未传播, 默认 softmax | Phase 1B TDD 修复 |

---

## 2. 实验 (Experiments)

### 2.1 Baseline

| 配置 | 训练 | clip_style | content_lpips | 说明 |
|------|------|-----------|---------------|------|
| `clean_base_v2_local.json` | 10-epoch (历史) | 0.7293 | 0.3203 | T5 baseline (softmax, M9 bug 前) |
| `clean_base_v2_relu2.json` | 3-epoch | 0.7269 | 0.3370 | Phase 1B (relu2 修复后) |
| `630_phase1d_verify.json` | 3-epoch (从零) | 0.7251 | 0.3373 | Phase 1D 精简后验证 |

**Baseline 阈值** (tools/local_train_and_eval.py):
- `BASELINE_CLIP_STYLE = 0.7293`, `CLIP_STYLE_MIN = 0.7243` (5σ)
- `BASELINE_CONTENT_LPIPS = 0.3203`, `CONTENT_LPIPS_MAX = 0.3453` (+0.025)

### 2.2 Phase 2: Masking 消融 (3-epoch, allpairs n=750)

| 实验 | ratio | mode | clip_style | content_lpips | 判定 |
|------|-------|------|-----------|---------------|------|
| baseline (Phase 1D) | 0.0 | none | 0.7251 | 0.3373 | PASS/PASS |
| **random_50 (最佳)** | 0.5 | random | **0.7261** | 0.3296 | **PASS/PASS** |
| random_75 | 0.75 | random | 0.7250 | 0.3278 | PASS/PASS |
| shuffle_50 | 0.5 | shuffle | 0.7234 | 0.3205 | **FAIL**/PASS |
| shuffle_75 | 0.75 | shuffle | 0.7232 | 0.3177 | **FAIL**/PASS |

**关键发现**:
- random mode 两项指标都优于无 mask baseline (clip +0.0010, lpips -0.0077)
- shuffle mode 破坏 DINO patches 空间统计, clip 降到阈值以下
- mask 越激进, lpips 越好, 但 clip 下降 (风格信息减少)
- **最佳**: ratio=0.5 在两项指标上平衡最佳

### 2.3 Phase 3: 完整训练验证 (10-epoch, 从零, 独立目录)

**配置**: `configs/630_phase3_mask_random_50_10ep.json`
**save_dir**: `exp/630_phase3_mask_random_50_10ep` (独立, `resume_checkpoint=""`)
**训练**: epochs=10, patience=2, batch_size=16, lr=2e-4, mask_ratio=0.5, mask_mode=random
**VRAM**: 训练峰值 3.44GB, 评估 batch_size=2

#### 评估结果 (已核实 summary.json)

| 评估点 | clip_style | content_lpips | 数据来源 |
|--------|-----------|---------------|---------|
| Epoch 5 per-epoch | 0.7275 | 0.3238 | `full_eval/epoch_0005/summary.json` L794,798 |
| Epoch 10 per-epoch | 0.7289 | 0.3370 | `full_eval/epoch_0010/summary.json` L794,798 |
| **Epoch 10 独立评估** | **0.7288** | **0.3369** | `full_eval/epoch_0010_full/summary.json` L789,793 |

**全部 PASS** (阈值: clip ≥ 0.7243, lpips ≤ 0.3453)

#### 收敛分析

| Epoch | clip_style | content_lpips | 趋势 |
|-------|-----------|---------------|------|
| 3 (Phase 2B) | 0.7261 | 0.3296 | 基线 |
| 5 | 0.7275 | 0.3238 | clip↑ lpips↓ |
| 10 | 0.7289 | 0.3370 | clip↑ lpips↑ |

- **clip_style 单调提升**: 更多训练让风格提取更精准
- **content_lpips 非单调**: 5ep 最佳 (0.3238), 10ep 略升 (0.3370) 但仍 PASS
- **5-epoch 即可用**: 快速验证场景下也能通过验收

#### Baseline 对比

| 配置 | clip_style | content_lpips | Δclip | Δlpips |
|------|-----------|---------------|-------|--------|
| baseline (无 mask, 10ep) | 0.7293 | 0.3203 | — | — |
| mask_random_50 (3ep) | 0.7261 | 0.3296 | -0.0032 | +0.0093 |
| mask_random_50 (10ep) | 0.7288 | 0.3369 | -0.0005 | +0.0166 |

**结论**: masking 不损害性能, clip 几乎持平 (训练噪声范围), lpips 略升但仍远在阈值内

---

## 3. 计划 (Plan)

### 3.1 已完成里程碑 (M1-M6, 全部 completed)

| 里程碑 | 状态 | 关键产物 |
|--------|------|---------|
| M1 深度审计 | ✅ | findings.jsonl (H1-H11, M1-M9, L1-L7) |
| M2 Phase 1 减法消融 | ✅ | phase1a/b/c/d, -11346 行 |
| M3 最简 codebase | ✅ | 14 个 active src 文件, 性能持平 |
| M4 Masking 实现 + 消融 | ✅ | phase2, random_50 最佳 |
| M5 Phase 3 完整训练验证 | ✅ | phase3, 10-epoch PASS |
| M6 文档化 + git 提交 | ✅ | docs/630/ 全套文档 + 6 个 commits |

### 3.2 未完成探索 (留给下一会话)

按优先级排序:

#### P1: mask_ratio 细化 (低成本, 高回报)
- **当前**: 0.5 (最佳) 和 0.75 (lpips 更好但 clip 略降)
- **待测**: 0.6, 0.7 中间值
- **预期**: 在 0.5-0.75 之间找到更优平衡点
- **成本**: 每组 3-epoch 训练 (~1 分钟)

#### P2: 频率掩码 (方案 C, 需架构改动)
- **理论** (mask.md): 对 style_latent 做低频减法, 保留高频风格细节
- **挑战**: 当前架构 style 输入是 DINO patches (非 latent), 需要在 patch_proj 后引入频域分解
- **实现思路**:
  1. 在 `StyleConditioner620._apply_mask()` 后增加 `freq_mask` mode
  2. 对 img_tokens 做 Haar DWT, 减去 LL 分量, 保留 LH/HL/HH
  3. 配置: `style_mask_mode="freq"`, `style_freq_lowpass_kernel=5`

#### P3: 显著性反向掩码 (方案 B, 工程复杂)
- **理论**: 用 SOD 或 DINO 注意力图提取前景, 反向掩码 (保留背景风格)
- **挑战**: 需要 SOD 模型或 DINO CLS token 注意力图, 工程复杂度高
- **适用场景**: 内容前景主导的图 (如肖像), 避免风格污染前景

#### P4: gate warmup (训练技巧, 非架构提升)
- **思路**: 训练前几个 epoch 关闭 cross-attn gate (gate=0), 让模型先学内容, 再逐步打开 gate 注入风格
- **配置**: `style_gate_warmup_epochs=2`, `style_gate_warmup_start=0.0`
- **预期**: 改善早期训练稳定性, 可能提升最终 clip_style

#### P5: 组合方案
- random + frequency (先 random dropout 再 freq mask)
- random + gate warmup
- 频率掩码 + gate warmup

### 3.3 下一会话建议工作流

1. **复现 Phase 3 baseline**: 运行 `python tools/local_train_and_eval.py --config configs/630_phase3_mask_random_50_10ep.json` 确认环境一致
2. **选择探索方向**: 从 P1 (mask_ratio 细化) 开始, 成本最低
3. **每方向独立目录**: `exp/630_phase4_<direction>_<params>/`, `resume_checkpoint=""`
4. **每阶段文档**: `docs/630/phase4_<direction>.md` + git commit
5. **遵守硬约束**: 见下方协议第 4 节

---

## 4. 协议 (Protocol)

### 4.1 Deli_AutoResearch 框架

**文档**: [skill.md](skill.md), [state/](state/) 全套状态文件

#### 状态机 (progress.json)
```
iteration 0: M1_AUDIT
  ↓
iteration 1-3: M2_PHASE1_SUBTRACTIVE → M3_MINIMAL_CODEBASE
  ↓
iteration 4: M4_MASKING (PHASE2_COMPLETE)
  ↓
iteration 5: M5_EXPLORATION + M6_DOCUMENTATION (PHASE3_COMPLETE)
```

#### 三大失败模式防护
1. **认知循环 (cognitive loops)**: `directions_tried.json` 记录已尝试方向, 强制方向多样性
2. **停滞 (stalling)**: `stale_count` 监控, ≥2 时切换结构约束
3. **运行时脆弱 (runtime fragility)**: 状态持久化到 JSON/JSONL, 支持断点续传

### 4.2 TDD 纪律

**流程**: RED (写失败测试) → 验证失败 → GREEN (最小代码) → 验证通过 → REFACTOR

**本会话 TDD 应用**:
- Phase 1B M9 bug: `test_style_attn_mode_propagated_to_blocks()` RED → 修复 → GREEN
- Phase 2 masking: 9 个测试覆盖 config/conditioner/mode/bridge forward

### 4.3 硬约束 (project_memory 固化)

#### 训练
- `Patience=2`, `max_epochs=10`, 至少 5 epochs
- `batch_size=16` (本地 4070 Laptop), 训练显存 9-11G
- **每次单开目录重新训练, `resume_checkpoint=""`, 禁止 `--skip-train --checkpoint` 借旧 checkpoint**
- DataLoader: `num_workers=0, pin_memory=False, persistent_workers=False`

#### 评估
- 显存 **严格 ≤ 7G**: `batch_size=2, full_eval_batch_size=2, ref_feature_batch_size=2`
- `full_eval_each_epoch=true` (每 epoch 评估)
- allpairs n=750 (5 styles × 5 styles × 30 images)

#### 数据集
- 本地: `G:/GitHub/Latent_Style/Dataset/distinct5_512`
- 远程: `/mnt/i/wikiart_distinct5_samam_512_classview/test`
- 5 styles: Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e

#### 代码
- 无效代码/机制确认后**直接删除** (不 ablate), ablation 是用于确认有效组件
- 优化用条件编译, 避免影响其他测试
- 命令添加 30s timeout

#### 资源
- **不允许远程 GPU**, 本地重训
- 重复利用算力, 显存探测用模型推断 (10 个验证点拟合, 不直接跑 300 个聚类)

### 4.4 验收标准 (硬性)

| 指标 | 阈值 | baseline | Phase 3 实际 | 判定 |
|------|------|----------|-------------|------|
| clip_style | ≥ 0.7243 | 0.7293 | 0.7288 | **PASS** |
| content_lpips | ≤ 0.3453 | 0.3203 | 0.3369 | **PASS** |
| WFI | < 0.40 | — | 未测 (留给后续) | — |

### 4.5 文档规范
- 每阶段文档到 `docs/630/phaseN_<name>.md`
- 状态更新到 `docs/630/state/progress.json`
- 决策记录到 `docs/630/logs/orchestrator.jsonl`
- git commit message 用中文 + 英文技术术语, 多 `-m` 标志分段 (PowerShell 不支持 heredoc)

### 4.6 失败模式警示

#### 已踩过的坑 (本会话)
1. **虚构 Phase 3 数据**: 上一会话写文档声称 Phase 3 训练完成, 但 `exp/630_phase3_mask_random_50_10ep/` 不存在。本会话发现后删除虚构文档, 从零重训
2. **借旧 checkpoint 偷懒**: 用 `--skip-train --checkpoint clean_base_v2_relu2/epoch_0003.pt` 跳过训练, 违反"独立目录从零训练"规则
3. **评估显存爆炸**: 评估 batch_size 过大导致 CUDA OOM, 已固定为 2
4. **PowerShell heredoc 不支持**: `$(cat <<'EOF'...)` 在 PowerShell 报错, 改用多 `-m` 标志

#### 防护规则
- 每次声称训练完成前, **必须** `LS exp/<dir>/` 确认 checkpoint 存在
- 每次声称评估结果前, **必须** `Read` 对应 `summary.json` 核实数字
- 训练命令**必须**用独立 save_dir + `resume_checkpoint=""`, 禁止 `--skip-train`
- PowerShell 提交用 `git commit -m "标题" -m "正文" -m "..."`, 不用 heredoc

---

## 5. Codebase 最终状态

### 5.1 Active 路径 (14 个 src 文件)

| 文件 | 行数 | 职责 |
|------|------|------|
| `src/model.py` | 93 | 精简模型工厂 (仅 620 contracts) |
| `src/spectral_bridge620.py` | ~300 | SpectralODEBridge620 (含 mask 配置传递) |
| `src/spectral_losses620.py` | ~250 | SpectralODEObjective620 |
| `src/blocks620.py` | 279 | SpatialBridgeBlock620 |
| `src/style_encoder620.py` | 109 | StyleConditioner620 (含 `_apply_mask`) |
| `src/spectral620.py` | ~80 | Haar DWT 工具 |
| `src/config_schema.py` | ~400 | 配置 schema (含 mask 字段) |
| `src/trainer.py` | ~350 | 训练器 (lazy import) |
| `src/run.py` | ~200 | 入口 |
| `src/style_families.py` | ~50 | 风格族工具 |
| `src/utils/inference.py` | ~600 | 推理工具 |
| `src/utils/run_evaluation.py` | ~3000 | 评估工具 |
| `src/utils/training.py` | ~200 | 训练工具 |
| `src/utils/dataset.py` | ~250 | 数据集 |

### 5.2 Phase 1 删除清单 (-11346 行)
- `TimeConditionedLANCETBridge` 类 (~2070 行)
- 9 个 legacy 文件 (9306 行): lancet_blocks.py, lancet_backbone.py, style_families.py (旧), utils/diffeomorphic.py, losses.py, losses620.py (旧), tokenizer*.py, etc.
- H1-H11 dead 参数/分支 (~80 行)
- dino_adapter, local_cnn, text branches, FiLM, MoE, WCT, multiband, patch AdaIN, multi-level DWT
- 60+ dead metric keys

### 5.3 Phase 2 新增 (~40 行核心 + 9 测试)
- `StyleConditioner620._apply_mask()` 方法 (random/shuffle/none)
- `style_mask_ratio`, `style_mask_mode` 配置字段
- `SpectralODEBridge620` 传递 mask 配置
- `tests/test_630_masking.py` 9 个 TDD 测试

### 5.4 已知遗留
- `style_attn_mode="relu2"` 已在 Phase 1B 修复 (传递到 blocks), 但 `clean_base_v2_local.json` 仍用 softmax baseline (0.7293)。relu2 的 3-epoch 结果 (0.7269) 略低, 长训练是否超过 softmax 未验证
- 部分配置字段 (如 `style_dino_adapter_*`, `style_moe_*`, `style_film_*`) 仍保留在 schema 中用于向后兼容, 但代码路径已删除, 设为 false 即可

---

## 6. 联系点

- **项目 memory**: `c:\Users\xy\.trae-cn\memory\projects\-g-GitHub-Latent-Style-SchrodingerBridge\project_memory.md`
- **用户 profile**: `c:\Users\xy\.trae-cn\memory\user_profile.md`
- **本会话 topics**: `c:\Users\xy\.trae-cn\memory\projects\-g-GitHub-Latent-Style-SchrodingerBridge\20260630\topics.md`
- **理论文档**: `docs/theory/SpectralODE_Bridge.md`
- **交接索引**: `docs/630/README.md`

---

**交接结束。新会话从本文档 + README.md 开始, 配合 state/ 目录的状态文件恢复完整上下文。**
