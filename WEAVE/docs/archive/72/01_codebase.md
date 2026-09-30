# 01 — 代码实现总览

> 源码位于 [src/](../../src/)，14 个 active 文件。本文档按"数据流 → 模块 → 关键算法"组织，并标注每个文件的角色与历史清理记录。

---

## 1. 顶层架构

```
┌────────────────────────────────────────────────────────────────────┐
│  src/run.py                                                        │
│  ├─ load_experiment_config (config_schema.py)                      │
│  ├─ AdaCUTLatentDataset (utils/dataset.py)                         │
│  ├─ build_model_from_config (model.py)                             │
│  │     └─ SpectralODEBridge620 (spectral_bridge620.py)             │
│  ├─ SpectralODEObjective620 (spectral_losses620.py)                │
│  └─ SBTrainer (trainer.py)                                         │
│        └─ full_eval → utils/run_evaluation.py + utils/inference.py │
└────────────────────────────────────────────────────────────────────┘
```

### 入口流程

1. **`python run.py --config configs/<exp>.json`**
2. `load_experiment_config` 解析 JSON，叠加 `_base` 链式继承，应用 `INFERENCE_DEFAULTS`。
3. `_set_seed(42)` + `_set_cpu_threads`。
4. 构造 `AdaCUTLatentDataset`（packed latent cache，路径 `G:/GitHub/Latent_Style/Dataset/distinct5_512_latents_ema`）。
5. `build_model_from_config` 根据 `contract_family` 分发：
   - `620_spectral_ode` → `SpectralODEBridge620`（active）
   - `620_spatial_bridge` → `build_spatial_bridge620_from_config`（legacy，保留）
   - 其它 → `ValueError`（630 Phase 1C 已删除 LANCET legacy）
6. `SpectralODEObjective620(config)` 构造 FM loss。
7. `SBTrainer` 训练循环：每 epoch 后可选 `full_eval_each_epoch`，调用 `run_evaluation.py` 子进程。

---

## 2. 模块清单

### 2.1 核心模型

#### [src/spectral_bridge620.py](../../src/spectral_bridge620.py) — `SpectralODEBridge620`

整个 FC-SB 模型的核心。结构：

```python
class SpectralODEBridge620(nn.Module):
    # 输入: x [B, 4, 32, 32] (SDXL VAE latent), t, style_id
    # 输出: {"ll": v_ll, "lh": v_lh, "hl": v_hl}  # HH removed (628 L8 DEAD)

    def forward(self, x, t, style_id, ...):
        ll, lh, hl, hh = dwt2_haar(x)                # 4 子带 [B,4,16,16]
        stacked = torch.cat([ll,lh,hl,hh], dim=1)    # [B,16,16,16]
        style_tokens, style_global = self.style_conditioner(style_id, ...)
        time_emb = self.time_proj(sinusoidal_time_embedding(t))
        h = self.input_proj(stacked)                 # [B, dim, 16, 16]
        for block in self.blocks:                    # 4 × SpatialBridgeBlock620
            h = block(h, time_emb, style_tokens, style_global)
        return {"ll": self.head_ll(h), "lh": ..., "hl": ...}

    @torch.no_grad()
    def integrate_transport(self, x, style_id, num_steps=8, ...):
        # 推理: 多步 ODE 积分 + 末步 Endpoint AdaIN
        # 支持 Euler / Heun / RK4 + linear/cosine/rquad/warp_cos schedule
        # 支持 per_subband / per_subband_wct / spatial_fiber AdaIN 模式
        # 支持 EOTA (only_last_step) / Progressive Alpha Scheduling
```

**关键子模块**：
- `StyleConditioner620` — 学习式 style_memory（256 tokens × 384 dim × 5 styles）
- `input_proj` — `Conv2d(4C=16 → dim=64, 3×3)`
- `time_proj` — `Linear(time_dim → dim) → SiLU → Linear(dim → dim)`
- `blocks` — `4 × SpatialBridgeBlock620`（depth 可配置，T19a 测试 depth=6 失败）
- `head_ll/lh/hl` — 3 个 `SpectralVelocityHead`（zero-init conv，HH 已删除）

**关键工具函数**：
- `_adain_match_subband(content, style)` — 单子带 AdaIN (mean+std)
- `_wct_match_fiber(content, style)` — 单子带 WCT (mean + 完整协方差)，含对角线正则化 + AdaIN 回退（T19a 数值稳定性修复）

#### [src/blocks620.py](../../src/blocks620.py) — `SpatialBridgeBlock620`

每个 block 的前向：
```
x → norm1 → AdaLN(time) → Self-Attention → +x
  → norm2 → Cross-Attention(content × style) → tanh_gate → +α·x
  → norm3 → FFN(Conv1×1 → SiLU → Conv1×1) → +x
```

**DWT Route Cross-Attention**（4J.1 核心）：
- `training` + `dwt_route_train_prob > 0`：以概率 `p` 对特征图做 DWT，LL bypass，仅 LH/HL/HH tokens query style_memory。
- `training` + `eval_only_dwt_route=True`：训练全空间 query（T5，已废弃）。
- `training` + `dwt_route=True`：训练+推理都用 DWT route（4J.1 原版）。
- `eval`：始终使用 DWT route（如果启用）。

**LL Global Style Injection**（T13-T16，已确认无效方向）：
- `ll_style_inject_source="style_mem"` (T13 LLGSI)：style_tokens 池化统计量 → LL AdaIN
- `ll_style_inject_source="ca_output"` (T14 CASI)：高频 cross-attn 输出统计量 → LL AdaIN
- `ll_style_inject_source="global_query"` (T15 LLGQCA)：LL 全局向量 query style_mem，输出空间均匀注入
- T16 gate sweep (0.2/0.3/0.5) 全部失败：style_mem 高频偏向是根本限制。

**Attention Mode**：
- `relu2`（active, clean_base_v2）：`logits = q·k^T · scale/temp; gates = relu(logits)^2; attended = gates·v`
- `softmax`（fallback）：标准 SDPA

#### [src/style_encoder620.py](../../src/style_encoder620.py) — `StyleConditioner620`

```python
self.style_memory = nn.Parameter(randn(num_styles=5, 256, 384) * 0.02)
self.patch_proj = LayerNorm(384) → Linear(384→64) → SiLU → Linear(64→64)
self.cls_proj   = LayerNorm(384) → Linear(384→64) → SiLU → Linear(64→64)
```

**Masking（Phase 2, Blindfolded Tokenizer）**：
- `mask_mode="random"`：随机 dropout `mask_ratio` 比例的 tokens
- `mask_mode="shuffle"`：空间打乱 tokens（破坏位置信息）

**Frequency Masking（Phase 4B）**：
- `freq_mode="avg_pool"` (4B-1)：box 低通，减去低频成分
- `freq_mode="haar_dwt"` (4B-3)：正交 Haar DWT，LL 子带 ×(1-α)，IDWT 重建

> **630 Phase 6 (DINO 退役)**：外部 DINO 输入路径已全部删除。`style_memory` 是唯一的 style token 源。`dino_dim` 字段名保留仅为 checkpoint 兼容。

### 2.2 频域工具

#### [src/spectral620.py](../../src/spectral620.py)

精确 Haar DWT 实现（正交变换，IDWT(DWT(x))=x）：
- `dwt2_haar(x) → (LL, LH, HL, HH)` — 单级 2D Haar DWT
- `idwt2_haar(ll, lh, hl, hh) → x` — 单级 2D Haar IDWT
- `dwt2_haar_lowpass(x, levels) → x_low` — N 级低通（仅保留 LL_N）
- `dwt2_haar_multi_decompose(x, levels) → [LL_N, ...]` — 多级分解
- `idwt2_haar_multi_reconstruct(...) → x` — 多级重建
- `dwt2_lowpass(x, levels, basis)` — 支持 Haar / Daubechies db2 基

### 2.3 训练目标

#### [src/spectral_losses620.py](../../src/spectral_losses620.py) — `SpectralODEObjective620`

```python
class SpectralODEObjective620:
    # 3 个独立 FM loss (per-subband), 权重 w_ll/w_lh/w_hl
    # w_ll ≈ 0 锁死低频保 LPIPS (T11 确认: w_ll=0.0 是 clip 最佳)
    # w_lh / w_hl 传中频风格

    def compute(self, model, content, target_style, ...):
        t = sample_t(content)
        x_t = (1-t)·content + t·target_style      # FM 插值
        target_v = target_style - content          # FM 速度目标
        if wct_aligned_target:
            target_style = _wct_align_target(...)  # 4J.2 预对齐
        v_pred = model(x_t, t, style_id)
        return {
            "loss_flow_ll": w_ll · fm_loss(v_pred["ll"], target_v_ll),
            "loss_flow_lh": w_lh · fm_loss(v_pred["lh"], target_v_lh),
            "loss_flow_hl": w_hl · fm_loss(v_pred["hl"], target_v_hl),
            "loss": sum(...),
        }
```

**辅助 loss**（默认关闭，仅 4J.6 few-shot 启用过）：
- `spectral_w_endpoint_style_lh/hl` — endpoint style loss，监督 `x_1_pred` 子带接近 target。在 4J.6 v3 验证无效（梯度通路太弱）。

### 2.4 配置与入口

#### [src/config_schema.py](../../src/config_schema.py)

`ExperimentConfig` dataclass，主要字段：
- `model: ModelConfig` — `base_dim`, `num_res_blocks`, `style_attn_num_heads`, `style_cross_attn_gate_init`, `style_gate_mode`, `style_attn_mode`, `cross_attn_dwt_route`, `dwt_route_train_prob`, `endpoint_adain_mode`, `endpoint_adain_scale`, `endpoint_adain_only_last_step`, `solver_type`, `time_schedule`, `style_extrap_alpha`, ...
- `bridge: BridgeConfig` — `spectral_w_ll`, `spectral_w_lh`, `spectral_w_hl`, `loss_type`, `wct_aligned_target`, `wct_aligned_alpha`, ...
- `training: TrainingConfig` — `batch_size`, `num_epochs`, `patience`, `full_eval_each_epoch`, `cpu_threads`, ...
- `data: DataConfig` — `pairing_cache_active_topk`, ...
- `inference / full_eval: InferenceConfig` — `num_steps`, `step_size`, `batch_size`, `vae_decode_batch_size`, ...

`load_experiment_config` 支持 `_base` 链式继承（递归 merge）。

#### [src/run.py](../../src/run.py)

入口脚本：
1. 解析 `--config` + `--config_override`（key=value pairs）
2. 构造 dataset / model / loss / trainer
3. 训练循环：每 epoch 后可选触发 `run_evaluation.py` 子进程做 full_eval
4. 训练完成后刷新 `clip_lpips_curve.csv` + `round2_convergence.json`

#### [src/model.py](../../src/model.py)

模型工厂。`build_model_from_config` 按 `contract_family` 分发：
- `620_spectral_ode` → `SpectralODEBridge620`（active）
- `620_spatial_bridge` → `build_spatial_bridge620_from_config`（保留）
- 其它 → `ValueError`（630 Phase 1C 已删除 LANCET legacy ~2070 行）

#### [src/trainer.py](../../src/trainer.py)

`SBTrainer`：
- `fit()` — 训练主循环，含 patience 早停
- `_train_epoch()` — 单 epoch，含 gradient accumulation（如配置）
- `_eval_epoch()` — 训练中 quick eval（轻量）
- `_full_eval()` — 调用 `utils/run_evaluation.py` 子进程做完整 5×5 矩阵评估

### 2.5 评估工具

#### [src/utils/run_evaluation.py](../../src/utils/run_evaluation.py)

完整评估流程：
1. 加载 VAE decoder（`ema` / `ema_fp16`）
2. 加载 CLIP (HF `openai-clip-vit-base-patch32`) + LPIPS (alex)
3. 对 5×5 风格矩阵（150 source × 5 target style = 750 generated）：
   - `lancet_generation`: 8 步 ODE 积分 + VAE decode → 生成图
   - `metric_clip`: 计算 CLIP-S / CLIP-dir / CLIP-content / CLIP-T
   - `metric_lpips`: 计算 content LPIPS
4. 输出 `summary.json`（含 `analysis.all_pairs_overview` 关键指标）

#### [src/utils/inference.py](../../src/utils/inference.py)

`integrate_transport` 的 numpy/torch wrapper，处理 VAE encode/decode、batch 调度、source latent cache。

#### [src/utils/dataset.py](../../src/utils/dataset.py)

`AdaCUTLatentDataset`：从 packed latent cache 加载，支持 pairing cache（top-k 风格配对）。

### 2.6 其它

- [src/style_families.py](../../src/style_families.py) — 契约验证函数（`validate_i2sb_contract`, `validate_pure_latent_contract`, `validate_phase616_clean_contract`）
- [src/utils/wfi.py](../../src/utils/wfi.py) — Whitening/Flattening Indicator（白化验收，< 0.40 PASS）
- [src/utils/artfid_metric.py](../../src/utils/artfid_metric.py) — ArtFID 学术指标
- [src/utils/introstyle_eval.py](../../src/utils/introstyle_eval.py) — IntroStyle 评估（默认关闭）
- [src/utils/style_classifier.py](../../src/utils/style_classifier.py) — 风格分类器（辅助指标）

---

## 3. 关键算法详解

### 3.1 Haar DWT 与正交性

```python
# 单级 2D Haar DWT (spectral620.py)
inv_sqrt2 = 0.7071067811865476
coef = inv_sqrt2 * inv_sqrt2  # = 0.5
LL = (a + b + c + d) * coef   # 低频（平均）
LH = (a + b - c - d) * coef   # 垂直高频
HL = (a - b + c - d) * coef   # 水平高频
HH = (a - b - c + d) * coef   # 对角高频
```

**正交性**：Haar 矩阵 `[1,1;1,-1]/√2` 是正交矩阵，因此 `IDWT(DWT(x)) = x`，无信息损失。这是 per-subband AdaIN 统计隔离的理论基础。

### 3.2 DWT Route Cross-Attention（4J.1 核心）

```python
# blocks620.py, SpatialBridgeBlock620.forward()
if use_dwt:
    ll_f, lh_f, hl_f, hh_f = dwt2_haar(x_f)
    # LL bypass: 不参与 cross-attention query
    ca_in = torch.cat([lh_tokens, hl_tokens, hh_tokens], dim=1)  # 仅高频
else:
    ca_in = x.flatten()  # 全空间 query

q = self.q_proj(ca_in)
k = self.k_proj(style_tokens)  # style_memory 投影
v = self.v_proj(style_tokens)
attended = attention(q, k, v)

if use_dwt:
    # IDWT 重建: LL 保持原值, 高频被 cross-attn 输出替换
    attended_2d = idwt2_haar(ll_f, lh_out, hl_out, hh_out)
else:
    attended_2d = attended.reshape(b, c, h, w)
```

**理论**：LL（低频结构）不 query style_mem → style_memory 100% 容量表达笔触/色彩，不被迫学"维持结构"。

### 3.3 Stochastic DWT Route（T11 核心）

```python
# 训练时
if self.training and self.dwt_route_train_prob > 0.0:
    use_dwt = self.dwt_route and (torch.rand(1).item() < self.dwt_route_train_prob)
# 推理时
else:
    use_dwt = self.dwt_route  # 始终 DWT route
```

**`p=0.8` 的含义**：训练时 80% 步用 DWT route，20% 用全空间 query。
- 80% DWT → q_proj 精通 DWT 系数分布 → 推理时 DWT route 有效
- 20% 全空间 → style_memory 学到更完整风格表达（不被高频偏向完全主导）

### 3.4 Endpoint AdaIN（per_subband_wct 模式）

```python
# spectral_bridge620.py, integrate_transport()
if adain_mode == "per_subband_wct":
    ll, lh, hl, hh = dwt2_haar(h)
    s_ll, s_lh, s_hl, s_hh = dwt2_haar(style_latent)
    # LL 不动 (内容锚), 高频做 WCT
    lh_new = (1-α)·lh + α·_wct_match_fiber(lh, s_lh)
    hl_new = (1-α)·hl + α·_wct_match_fiber(hl, s_hl)
    hh_new = (1-α)·hh + α·_wct_match_fiber(hh, s_hh)
    h = idwt2_haar(ll, lh_new, hl_new, hh_new)
```

**WCT vs AdaIN**：
- AdaIN：仅匹配 mean + std（对角协方差），丢失通道相关性
- WCT：匹配完整协方差矩阵 `Σ_s^{1/2} · Σ_c^{-1/2} · (f - μ_c) + μ_s`，捕获通道相关结构

**EOTA (End-of-Trajectory AdaIN, 4H.1)**：`only_last_step=True`，仅在第 8 步应用 AdaIN。解耦 ODE 求解与风格注入，恢复 α 作为有效 trade-off 旋钮。

### 3.5 ODE 求解器

| 求解器 | 阶数 | 截断误差 | 前向调用/步 | 备注 |
|--------|------|----------|-------------|------|
| Euler | 1 | O(h²) | 1 | baseline |
| Heun | 2 | O(h³) | 2 | 4I.2 结构性突破 |
| RK4 | 4 | O(h⁴) | 4 | 4I.6 饱和（无额外收益） |

**4I.2 发现**：Euler→Heun 是结构性 DOF（打破 1D Pareto 前沿），Heun→RK4 饱和（其他误差源主导）。

### 3.6 Time Schedule

```python
def _schedule(s):
    if time_schedule == "cosine":
        return (1 - cos(π·s)) / 2          # S 形，两端慢中间快
    elif time_schedule == "warp_cos":
        return (1 - cos(π·s^p)) / 2        # 参数化 cosine
    elif time_schedule == "quad":
        return s·s                           # 内容偏置
    elif time_schedule == "rquad":
        return 1 - (1-s)·(1-s)              # 风格偏置
    return s                                 # linear (T11 使用)
```

**4I.5/4I.8 分类**：schedule shape 是 Pareto-mapping knob（沿前沿移动），**不是**结构性 DOF。只有 solver order 是结构性 DOF。

---

## 4. 历史清理记录

### 628/629 清理
- 删除 9 项辅助 loss + `spectral_w_hh`（L8 确认 DEAD，Δclip=±0.0001）
- 删除 `attn_modes`: gated / gated_raw / style_select / sparsemax
- 删除 FiLM modulation, style MoE, learnable shortcut, skip_coarse, top-k truncation, style_bias
- 删除多级 DWT 分支 + Brownian 噪声分支（active config 永不启用）

### 630 Phase 1 清理
- **Phase 1A** (`925b6bea7`): H1-H11 零风险 dead code 删除
- **Phase 1B** (`69da87cb0`): M9 attn_mode bug TDD 修复（`relu2` 未传播）
- **Phase 1C** (`bcea0a41b`): Legacy 文件批量删除（-11346 行，含 `TimeConditionedLANCETBridge` ~2070 行）
- **Phase 1D** (`9de1e9e03`): 最简 codebase 性能验证（3-epoch PASS）

### 630 Phase 6 (DINO 退役)
- 13 文件修改，所有功能性 DINO 引用从 `src/` 移除
- `style_memory` 成为唯一 style token 路径
- `dino_dim` 字段名保留仅为 checkpoint 兼容

### T19a WCT 数值稳定性修复（本轮）
- `spectral_bridge620.py::_wct_match_fiber`: 对角线正则化 + try-except 回退 AdaIN
- 原因：depth=6 导致协方差矩阵病态，eigh 分解失败

---

## 5. 清理与重构建议（未执行）

以下为文档撰写过程中识别的潜在清理点，**未执行**（需用户确认）：

### 5.1 可删除的无效代码
1. **T13/T14/T15 LLGSI/CASI/LLGQCA 代码** (`blocks620.py` L264-L318)：
   - T13-T16 系统性证明无效，方向已关闭
   - 但 `ll_global_style_inject` / `ll_style_inject_source` / `ll_style_gate` 参数仍在 config schema
   - **建议**：直接删除（用户偏好"无效代码确认后直接删除"）

2. **T5 eval_only_dwt_route 代码** (`blocks620.py` L189-L191)：
   - T5 失败，方向关闭
   - **建议**：删除 `eval_only_dwt_route` 分支与 config 字段

3. **Phase 4J.6 endpoint style loss** (`spectral_losses620.py` L79-L84)：
   - 4J.6 v3 验证无效（梯度通路太弱）
   - **建议**：删除 `spectral_w_endpoint_style_lh/hl` 字段

4. **`wct_aligned_target` 4J.2** (`spectral_losses620.py` L77-L78, L96-L110)：
   - 4J.2 方向未在 progress.json 记录为成功
   - **建议**：需确认是否仍用于任何 active config，若无则删除

### 5.2 可重构的复杂逻辑
1. **`integrate_transport` 函数过长** (~250 行)：
   - 含 Euler/Heun/RK4 + 4 种 schedule + 3 种 AdaIN 模式 + Progressive Alpha
   - **建议**：拆分为 `_euler_step` / `_heun_step` / `_rk4_step` + `_apply_adain`

2. **`SpatialBridgeBlock620.forward` 的 DWT route 分支**：
   - 3 种 mode (dwt_route / eval_only_dwt_route / dwt_route_train_prob) 嵌套
   - **建议**：抽取 `_compute_use_dwt()` 方法

> 上述清理**未执行**，等待用户确认。详见 [06_cleanup_notes.md](06_cleanup_notes.md)。
