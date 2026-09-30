# 620 白化/消融项目 Git 历史调研摘要

> 调研时间：2026-06-21  
> 当前工作分支：`codex/620-spatial-bridge`（HEAD = `e267e4fac`）  
> 调研范围：所有本地/远程分支、commit message、docs/620/ 文档、configs/620*.json、核心源码

---

## 1. 分支地图

### 1.1 全部相关分支列表

| 分支 | 类型 | 与 620 关系 | 关键 HEAD / 说明 |
|---|---|---|---|
| `codex/620-spatial-bridge` | 本地，当前检出 | **620 主开发分支** | `e267e4fac`，含 H4-H7 实验与 fog 修复 |
| `codex/tokenizer-clean-c3058eab` | 本地 | 620 之前的 tokenizer 清理基线 | `c3058eab`，无 620 代码，仅保留 docs/620/{OT,bridge,math,tokenizer} |
| `codex/backup-pre-clean-20260608-185941` | 本地 | 清理前备份 | 含大量旧实验产物与独立快照目录 |
| `codex/pre-cleanup-stash-20260527` | 本地 | 2026-05-27 清理 stash | 记录清理前脏工作区 |
| `main` | 本地 | **已移除 620 代码的清理分支** | `fa1e9fc65` "remove legacy style code path"；无 620 源码 |
| `pushfix-clean` / `pushfix-ff` / `replay-ordered` | 本地 | 辅助/修复分支 | 与 620 实验关系较弱 |
| `remotes/origin/codex/620-spatial-bridge` | 远程 | 620 远端镜像 | 同本地 codex/620-spatial-bridge |
| `remotes/origin/codex/tokenizer-clean-c3058eab` | 远程 | tokenizer 清理镜像 | 同本地 tokenizer-clean |
| `origin/exp/style-injection-priority-proto-sep` | 远程 | 早期 style-injection 实验 | 主要探索 proto-separation / overfit50 |
| `origin/attn` | 远程 | 早期 attention 实验 | 曾达到 `clip_style=0.72, 1-lpips=0.5` |
| `origin/SWD` | 远程 | SWD 损失探索 | 含 moment 实现、分类器评估 |
| `origin/Style8_Moment+SWD` | 远程 | Style8 + Moment + SWD | 单独蒸馏 tokenizer 有指标提升 |
| `origin/re-SWD` | 远程 | SWD 重做 | 修复 FP32/BF16 数值问题 |
| `origin/multistep-texture` | 远程 | 多步纹理生成 | 针对 3060 的 infra 优化 |
| `origin/Classify` | 远程 | 分类器引导实验 | SWD/GRAM 消融结论 |
| `origin/Cycle-upscale` / `origin/Diff-Gram` / `origin/Gram-Moment` / `origin/Thermal` | 远程 | 周边/历史损失实验 | 探索 Gram/Diff-Gram/Thermal 等 |
| `origin/rebuild-clean-20260319` / `origin/sdxl-fp16` | 远程 | 早期重构/SDXL | 与 620 关系较远 |

### 1.2 分支关系拓扑（简化）

```
codex/tokenizer-clean-c3058eab  ───────────────────────┐
                                                       │
main  ──(remove legacy style code path)───────────────┤  清理/去 620 方向
                                                       │
codex/620-spatial-bridge  ────────────────────────────┘  620 主开发
    ↑ 由 d94b5d4f6 "Add 620 spatial bridge mainline" 引入
    │
    ├── H4: FiLM endpoint heads (5ff9e8faa)
    ├── H5: dim=128 (af4807621)
    ├── H6: intrinsic cross-attention (af4807621)
    └── H7: SWD weight 8→2 (2c51bb3b2)

origin/attn / origin/SWD / origin/Style8_Moment+SWD 等：
    └── 620 之前的早期探索，部分结论被 620 继承（如 cross-attention、SWD）
```

### 1.3 关键发现：main 与 620 已分离

- `main` 当前 HEAD（`fa1e9fc65`）已删除 620 源码（`remove legacy style code path`）。
- 620 全部工作发生在 `codex/620-spatial-bridge`，尚未合并回 `main`。
- 因此 `git diff <branch>..main` 对 620 源码文件均为空/删除；有意义的设计演进比较应看 `codex/tokenizer-clean-c3058eab..codex/620-spatial-bridge`。

---

## 2. 实验时间线

### 2.1 619 → 620 转折期（2026-06 中上旬）

| 分支 | Commit | 实验/决策 | 关键改动/结论 |
|---|---|---|---|
| codex/620-spatial-bridge | `d94b5d4f6` | Add 620 spatial bridge mainline | 引入 620 核心架构：DINO patch 风格条件、真实 cross-attention、单步 SWD、I2SB 推理 |
| codex/620-spatial-bridge | `93cdbea14` | Route 620 eval through target DINO patches | 评估也走 target DINO patches |
| codex/620-spatial-bridge | `a4f8eb8d3` | Add 620 eval sweep remote launcher | 远程 3060 评估管线 |
| codex/620-spatial-bridge | `de1dc3727` | Wire 620 evals into live dashboard | 实时看板 |

### 2.2 Round 1 诊断实验（8 epoch 远程 3060）

基于 `docs/620/round1_diagnosis.md`：

| 分支 | 实验名 | 关键改动 | clip_style | LPIPS | 结论 |
|---|---|---:|---:|---|---|
| codex/620-spatial-bridge | base_swd8 | SWD weight=8，dim=64，4 blocks | 0.6720 | 0.2900 | LPIPS 极好，style 卡在 0.67 |
| codex/620-spatial-bridge | swd4 | SWD weight=4 | 0.6706 | **0.2794** | 最低 LPIPS |
| codex/620-spatial-bridge | swd12 | SWD weight=12 | **0.6725** | 0.2968 | 最高 style，但仍在 0.67 平台 |
| codex/620-spatial-bridge | adapter | DINO adapter + cross-attn | 0.6715 | 0.2916 | 无提升 |
| codex/620-spatial-bridge | moe | Style K/V MoE | 0.6711 | 0.2906 | 无提升 |
| codex/620-spatial-bridge | gate12 | gate init=1.2 | 0.6714 | 0.2918 | 无提升 |
| codex/620-spatial-bridge | lowmix05 | low_anchor=0.5 | 0.6765 | 0.3492 | style 微升，LPIPS 崩（水平泄漏） |

**Round 1 结论**：
- 所有变体 clip_style 集中在 0.668–0.677，远低于目标 0.72+。
- 增加 adapter/MoE/gate 容量无法提升 style，瓶颈在 Q 侧维度（dim=64）。
- 自注意力被完全移除，空间一致性不足。
- 建议：dim=64→128，num_res_blocks 4→6，恢复 self-attention。

### 2.3 Phase 1 SWD Weight Scan（突破 0.70 天花板）

基于 `docs/620/experiment_progress.md`、`docs/620/final_summary.md`：

| 分支 | 实验 | SWD | vlen | best epoch | clip_style | lpips | 结论 |
|---|---|---:|---|---:|---:|---:|---|
| codex/620-spatial-bridge | swd12 | 12 | 1.0 | e8 | 0.6725 | 0.2968 | Round 1 基线 |
| codex/620-spatial-bridge | swd16 | 16 | 1.0 | e1 | 0.7053 | 0.2901 | **首次突破 0.70** |
| codex/620-spatial-bridge | swd16_vlen0.2 | 16 | 0.2 | e9 | 0.7038 | 0.3064 | 有效 |
| codex/620-spatial-bridge | **swd16_vlen0.04** | **16** | **0.04** | **e5** | **0.7051** | **0.2935** | **当前最优（突破 0.67→0.705）** |
| codex/620-spatial-bridge | swd20_vlen0.04 | 20 | 0.04 | e1 | 0.7006 | 0.2750 | 边际收益下降 |

**突破原因**（docs/620/final_summary.md）：
1. OT 在线不稳定 → DINO 离线 top-K 固定配对。
2. 伪 CrossAttention 1D 瓶颈 → DINOv2 256×384 空间特征作为 K/V。
3. ODE 展开梯度截断 → 单步 SWD `SWD(ẑ₁, z_s)`。

### 2.4 H4-H7 近期实验（fog 修复 + 架构探索）

| 分支 | Commit | 实验名 | 关键改动 | clip_style | lpips/wfi | 结论 |
|---|---|---|---|---:|---:|---|
| codex/620-spatial-bridge | `5ff9e8faa` | H4 | 引入 FiLM endpoint heads (`endpoint_head_mode=endpoint_lowhigh`, `endpoint_film_enabled=true`) | — | — | 代码实现 |
| codex/620-spatial-bridge | `af4807621` | H5 | dim=128，num_res_blocks=4 | — | — | 容量提升 |
| codex/620-spatial-bridge | `af4807621` | H6 | intrinsic cross-attention（style_condition_source=latent，DINO 去掉） | — | — | 端到端风格表征 |
| codex/620-spatial-bridge | `8fbb39148` | H6 results | intrinsic cross-attention 结果 | **0.6717** | **lpips=0.3678** | style 未提升，lpips 变差 |
| codex/620-spatial-bridge | `739535e90` | WFI metric | 增加 fog/whiteness 指标与图像保存 | — | — | 评估基础设施 |
| codex/620-spatial-bridge | `2c51bb3b2` | H7 | SWD weight 8→2 解决梯度冲突 | — | — | 修复梯度冲突 |

### 2.5 Fog / Whitening 诊断实验（Round E1-E3，2026-06-21）

基于 `docs/620/fog/` 系列报告：

| 阶段 | 实验 | 关键改动 | clip_style | content_lpips | wfi_score | 结论 |
|---|---|---:|---:|---:|---:|---|
| E1 基线 | 620_film_v5_gated_local_smoke | gated attn + style_film | 0.6987 | 0.3300 | 0.4902 | 白化严重 |
| E2 P0 | endpoint_film (hd128) | endpoint_head_mode=endpoint_lowhigh + endpoint_film_enabled | 0.7066 | 0.3226 | 0.4283 | WFI 显著下降但未过门 |
| E2 P1 | hf_residual | velocity 高频残差 | 0.7020 | 0.3263 | 0.4746 | 几乎无效 |
| E3 | endpoint_film 3ep | 同 hd128 训练 3 epoch | 0.7099 (e3) | 0.3768 (e3) | 0.4680 (e3) | **更多 epoch 加剧白化** |
| E3 H1 | endpoint_film_init02 | `endpoint_film_init_std=0.02` | 0.7044 | 0.3217 | 0.4022 | 接近但未低于 0.40 |
| **E3 H2** | **endpoint_film_hd512** | **`endpoint_style_hidden_dim=512`** | **0.7015** | **0.3382** | **0.3906** | **通过 WFI < 0.40 放行门** |

**关键观察**：
- WFI 放行门：wfi_score < 0.40，clip_style ≥ 0.695，content_lpips < 0.36。
- H2 hd512 是当前最优，但距离 Seedream IDT（wfi≈0.158）仍有 +0.233 差距。
- 更多 epoch 反而恶化 WFI（0.4271→0.4532→0.4680），说明优化 landscape 存在“安全但白化”的 basin。

### 2.6 早期历史分支实验（origin/*）

| 分支 | 关键 commit / 结论 |
|---|---|
| `origin/attn` | `bbfbb7138` style_oa_5 达到 `clip_style=0.72`；`984d01415` Style_Clip=0.72 同时 1-lpips=0.5 |
| `origin/SWD` | `df70db070` 修正 moment 实现；`c194dd580` 推理 16.93 速度 / 6.12 显存 |
| `origin/Style8_Moment+SWD` | `4a6302813` 单独蒸馏 tokenizer、优化 style_embedding 有明显提升；`c0df84205` Domain 1x1 (512 proj) Ratio 5.77x 为最高 |
| `origin/re-SWD` | `7f05d8577` loss 异常直接跳过 batch；`08a98dcd0` Style-8 SWD FP32 处理 |
| `origin/multistep-texture` | `3a624d566` compile 鲁棒性；`36e10a7ad` 优化器状态从头加载；`a4c9ddb4c` 3060 infra 优化 |
| `origin/exp/style-injection-priority-proto-sep` | `0a0c55fb5` style-first discipline + proto-separation E7/E8；`228662d13` 高频多尺度风格监督；`81102b6d6` skip-gated style pathway |
| `origin/Classify` | `84b525f36` SWD 某些情况微弱作用，GRAM 完全没用；`33bb00efc` 分类很漂亮但可能过拟合风格 |
| `origin/Diff-Gram` | `9467e84b0` 微分 Gram 终于正了；`025b77e7b` diff-gram 在 SDXL-fp32 表现极差 |
| `origin/Gram-Moment` | `c505d3d68` gram 白化；`8123db153` semigroup 占用 +5507.2MB |

---

## 3. 设计演进树

### 3.1 StyleFiLM / AdaLN

| 阶段 | 位置/形态 | 来源 | 结论 |
|---|---|---|---|
| 早期 | AdaGN in encoder/decoder | `origin/attn`, `c04376700` | 风格强但需调权重 |
| 620 v1 | `style_film_enabled` 在 block 内：post-cross-attention FiLM + pre-cross-attention Q-FiLM + style_bias | `src/blocks620.py` | 绕过 attention 平均化瓶颈 |
| 620 H4 | `FiLMEndpointHead`：style 全局嵌入调制 endpoint head 特征图 | `5ff9e8faa` | **被证明有效**，hd512 解决白化 |
| 620 H1 | `endpoint_film_init_std=0.02` | `docs/620/fog/decision_log.md` | 部分有效但不足 |
| 620 H2 | `endpoint_style_hidden_dim=512` | `docs/620/fog/decision_log.md` | **关键容量突破** |

**结论**：FiLM 是有效风格注入路径；瓶颈在映射容量（128→512）和初始化；block 内 FiLM 对白化改善不足，endpoint FiLM 直接且有效。

### 3.2 Cross-Attention gated/softmax/sparsemax

| 形态 | 来源 | 结论 |
|---|---|---|
| softmax (默认) | `src/blocks620.py` | 基础，attention entropy ~5.53 |
| gated (sigmoid + renormalize) | `model.style_attn_mode=gated` | 用于 E1 基线 |
| gated_raw (sigmoid 不归一化) | `gated_raw` | WFI 升至 0.6435，**恶化白化** |
| relu2 | `relu2` | WFI=0.5340，无效 |
| style_select (top-k) | `style_select` | WFI=0.5005，无效 |
| sparsemax | `sparsemax` | 实现但无实验结果 |
| top-k | `style_attn_topk` | 配置存在 |

**结论**：attention 内部模式改造（gated_raw/relu2/style_select）均**未降低 WFI**，clip_style 在 0.696–0.699 窄幅波动。attention 平均化是 style 弱化起点，但修改 attention 核函数不是根因；应绕过 attention 直接送 style 到 endpoint。

### 3.3 Endpoint head velocity / endpoint_lowhigh

| 形态 | 来源 | 说明 |
|---|---|---|
| velocity head (3 层 conv，无 GN) | `src/model620.py` `endpoint_head_mode=velocity` | 620 早期默认，避免动态范围压缩 |
| endpoint_lowhigh | `src/model620.py` `endpoint_head_mode=endpoint_lowhigh` | 低频/高频分支 + style_to_low/high 调制 |
| + FiLM | `FiLMEndpointHead` | H4/H2 核心，通过 style 调制 head 特征图 |
| + HF Residual | `velocity_hf_residual_enabled` | E2 P1，几乎无效 |

**结论**：endpoint_lowhigh + FiLM 是当前最优组合；单纯 velocity head 或 HF residual 不足以解决白化。

### 3.4 Endpoint FiLM

- 起源：`5ff9e8faa` feat(620): add FiLM endpoint heads for style modulation inside head trunk (H4)
- 关键参数：`endpoint_film_enabled`, `endpoint_style_hidden_dim`, `endpoint_film_init_std`
- 最优：`endpoint_style_hidden_dim=512`，`endpoint_film_init_std=0.0`（zero-init）
- 效果：WFI 0.4902→0.3906，clip_style 0.6987→0.7015
- 理论：style→endpoint 的 FiLM 映射容量不足导致 modulation 信号被压缩到零；增大 hidden_dim 直接提升表达能力。

### 3.5 SWD / NSWD

| 形态 | 来源 | 结论 |
|---|---|---|
| 全频段 SWD | 早期 `origin/SWD` | 推到边缘，NCE/rep 导致爆炸 |
| 频率拆分 SWD | 历史 | 已移除 |
| 单步 SWD (620) | `src/losses620.py` | 替代 integrate()，梯度链长=1，稳定 |
| NSWD (noise σ=0.02) | `swd_noise_sigma=0.02` | 打破 SWD 梯度正交性 |
| 多尺度 SWD | `swd_scale_mode=2-scale/3-scale` | Phase 4 待实验 |
| attention-weighted SWD | `swd_scale_mode=attention-weighted` | 代码实现，待实验 |

**历史结论**：
- SWD weight=16、vlen=0.04 时达到 clip_style=0.7051，突破 0.67 天花板。
- SWD 梯度非零但 cos(∇SWD, v_target)≈-0.024（几乎正交），所以引入 noise（NSWD）。
- H7 将 SWD weight 8→2 以解决梯度冲突。

### 3.6 DINO patch conditioning

| 阶段 | 来源 | 说明 |
|---|---|---|
| 619 之前 | learned tokens + 1D bias | 信息量 KB 级，被批判为伪 CrossAttn |
| 620 | DINOv2 256×384 spatial tokens → K/V | 信息量 ×100，真实 cross-attention |
| adapter | `style_dino_adapter_enabled` | Round 1 实验，**无提升** |
| multi-scale DINO | Phase 4 A2 计划 | 待实验 |
| intrinsic (latent CNN) | H6 | clip_style=0.6717, lpips=0.3678，**未超越 DINO** |

**结论**：DINO patch conditioning 是 620 突破的关键基础设施；adapter/MoE 在 dim=64 下无收益；H6  intrinsic 路径目前弱于 DINO。

### 3.7 Style spatial prior / tokenizer

| 阶段 | 来源 | 结论 |
|---|---|---|
| PureLatentSpatial tokenizer | 616 | `8f5e7cfce` 诊断确认 **zero ROI** |
| Legacy tokenizer | 旧架构 | `ee763c5f8` Retire legacy style spatial priors |
| Style8_Moment+SWD tokenizer | `origin/Style8_Moment+SWD` | 单独蒸馏 tokenizer 有提升 |
| 620 | DINO frozen + optional adapter/MoE | 当前主路径 |

**结论**：legacy spatial prior/tokenizer 已被证无效并移除；620 风格表征完全依赖 DINO。

### 3.8 MoE / adapter

- `style_moe_enabled` + `style_moe_num_experts` + `style_moe_router_hidden_dim`
- `style_dino_adapter_enabled` + `style_dino_adapter_hidden_dim=1024`
- `style_kv_moe_content_routed`（content-aware K/V router）
- **Round 1 结论**：adapter / moe / gate12 均没有比 base_swd8 更好的 style 分数。
- 原因：Q 侧仍只有 dim=64，cross-attention 输出维度受限于 Q 维度。

### 3.9 Self-attention

- 当前 `src/blocks620.py` 已包含 self-attention（AdaLN(time) → Self-Attn → Cross-Attn → FFN）。
- Round 1 诊断指出早期 620 无 self-attention 导致空间位置无法通信、笔触模式无法传播。
- 已在当前分支恢复。

### 3.10 Capacity (dim, num_res_blocks)

| 配置 | 来源 | 参数 | 结论 |
|---|---|---|---|
| 620 base | `configs/620_spatial_bridge_base.json` | dim=64, num_res_blocks=4 | 183K block 参数，style 卡 0.67 |
| H5 / 620 v2 | `configs/620_spatial_bridge_dim128.json`, `620_spatial_bridge_v2.json` | dim=128, num_res_blocks=4/6 | 计划验证 |
| Round 1 建议 | `docs/620/round1_diagnosis.md` | dim=128, num_res_blocks=6, heads=8 | 理论可提升，待正式验证 |

---

## 4. 历史教训

### 4.1 被证明有效的设计

| 设计 | 证据 | 建议 |
|---|---|---|
| DINO 离线 top-K 配对 | 620 突破 0.70 | 继承，无需重复验证 |
| DINOv2 256 tokens 作为 K/V | 信息量 ×100 | 继承 |
| 单步 SWD 替代 ODE 展开 | 梯度稳定，突破 0.70 | 继承 |
| Vertical FM / target_linear low mode | LPIPS 0.29 水平 | 继承 |
| Self-attention in block | 笔触一致性 | 保留 |
| Endpoint-FiLM Head | WFI 0.4902→0.3906 | **核心修复，必须保留** |
| `endpoint_style_hidden_dim=512` | 通过 WFI 放行门 | 当前最优配置 |
| gate_init=0.3（从 0.05 提升） | velocity magnitude +16% | 比 0.05 更有效 |
| SWD weight=16 + vlen=0.04 | clip_style=0.7051 | 当前最优调参 |

### 4.2 被证明无效或有害的设计

| 设计 | 证据 | 建议 |
|---|---|---|
| Legacy spatial prior / PureLatentSpatial tokenizer | zero ROI | 已移除，不要恢复 |
| Adapter / MoE / gate12 在 dim=64 下 | Round 1 无提升 | 在 dim=64 下无效；dim 提升后可重新验证 |
| Gated_raw attention | WFI 0.6435（恶化） | 不要用于白化修复 |
| ReLU2 / style_select attention | WFI 0.5340/0.5005 | 对白化无效 |
| HF Residual | WFI 0.4746，网络主动弱化 | 不作为独立修复 |
| 更多 epoch（lr=2e-4） | WFI 单调上升 | 对 hd512 用 1 epoch 或 early stopping |
| Structure loss / Gram / Diff-Gram（早期） | `84b525f36` GRAM 完全没用；`025b77e7b` SDXL 上极差 | 不优先恢复 |
| OT 在线 minibatch Sinkhorn | 619 诊断均值坍缩 | 已替换为离线 DINO 配对 |

### 4.3 有争议或需要重新验证的设计

| 设计 | 争议点 | 建议 |
|---|---|---|
| DINO adapter | Round 1 在 dim=64 无效；dim=128 后可能有效 | dim=128/H5 后复测 |
| Style MoE | 同样受限于 Q 维度 | dim=128 后复测 |
| Cross-attention Q source | `content_dino` / `sa_out_only` / `concat` | Phase 4 D block 计划 |
| Multi-scale DINO | 浅层纹理+深层语义 | Phase 4 A2 待做 |
| Per-region / multi-scale SWD | 可能提升区域匹配精度 | Phase 4 B block 待做 |
| Top-k / sparse attention | 代码已实现，但未带来指标提升 | 与 dim/encoder 升级结合重测 |
| Text conditioning | 在白化通过后可恢复，但需先过 WFI 检查 | Phase 4 后 |
| `endpoint_film_init_std=0.02` | H1 单独有效但不过门；与 hd512 组合可能更优 | 可作为 E4 候选 |
| 移除 FiLMEndpointHead 内 GroupNorm | 可能防止动态范围压缩 | E4 P1 候选 |

---

## 5. 对当前消融实验的建议

### 5.1 最值得重新测试的维度

基于历史数据，以下维度在 **白化门控通过** 后最值得重新测试：

1. **容量升级（H5）**：dim=64→128 / num_res_blocks 4→6。Round 1 指出这是最关键瓶颈，adapter/MoE 在 dim=64 下无效部分原因正是 Q 维度受限。应在 `endpoint_film_hd512` 基础上验证。
2. **DINO adapter / MoE 在 dim=128 下复测**：之前失败可能因为 dim=64 瓶颈，而非设计本身无效。
3. **Cross-attention Q source**：`content_dino` / `sa_out_only` / `concat`，看是否能进一步提升 clip_style 而不破坏 WFI。
4. **Multi-scale DINO（A2）**：浅层纹理+深层语义，可能提升笔触锐利度。
5. **Per-region / multi-scale SWD（B2/B3/B4）**：全局 SWD 已接近天花板，区域化可能带来下一跳。
6. **组合 H1+H2**：`endpoint_film_init_std=0.02` + `endpoint_style_hidden_dim=512`，看能否进一步压低 WFI 到接近 Seedream 水平。
7. **学习率 / early stopping**：当前 lr=2e-4 导致更多 epoch 白化；测试 lr=1e-4 或更短 early stopping。

### 5.2 可以直接继承、无需重复的实验

| 结果 | 依据 |
|---|---|
| DINO 离线配对 vs 在线 OT | 619 诊断 + 620 突破 |
| DINO 256 tokens vs learned tokens | 620 突破 |
| 单步 SWD vs ODE 展开 | 620 突破 + 训练稳定 |
| endpoint_lowhigh + FiLM hd512 通过 WFI 门 | Round E3 验收报告 |
| gated_raw/relu2/style_select 对白化无效 | Round E1/E2 decision_log |
| HF Residual 独立无效 | Round E2 experiment_report |
| legacy spatial prior/tokenizer 无效 | 616 诊断 + `ee763c5f8` |
| dim=64 下 adapter/MoE/gate12 无收益 | Round 1 diagnosis |

### 5.3 应该恢复或删除的设计

**建议保留/作为新基线**：
- `endpoint_head_mode=endpoint_lowhigh`
- `endpoint_film_enabled=true`
- `endpoint_style_hidden_dim=512`
- `style_attn_mode=gated`
- `style_cross_attn_gate_init=0.3`
- `style_film_enabled=true`
- `training_target_projection_low_mode=target_linear`
- `bridge.swd_noise_sigma=0.02`
- `bridge.single_step_swd_weight=8.0`（或基于 H7 调整为 2.0，需验证）
- `bridge.single_step_edge_weight=0.1`

**建议删除/不再投入**：
- `velocity_hf_residual_enabled` 独立使用
- `style_attn_mode=gated_raw`（白化恶化）
- legacy style spatial prior / tokenizer 相关代码
- `ablation_disable_spatial_prior` 等已退休 key

**建议重新评估后决定是否保留**：
- `style_dino_adapter_enabled`（dim=128 后复测）
- `style_moe_enabled`（dim=128 后复测）
- `style_attn_topk` / `style_attn_mode=sparsemax`（结合新容量重测）

### 5.4 推荐的下一组实验配置

基于 `620_film_v5_endpoint_film_hd512_local_smoke` 为基线，建议创建：

1. `620_film_v5_hd512_dim128_local_smoke`：容量升级 + hd512。
2. `620_film_v5_hd512_dim128_adapter_smoke`：容量升级 + adapter。
3. `620_film_v5_hd512_dim128_moe_smoke`：容量升级 + MoE。
4. `620_film_v5_hd512_init02_local_smoke`：H1+H2 组合。
5. `620_film_v5_hd512_noGN_local_smoke`：移除 FiLMEndpointHead 内 GroupNorm。
6. `620_film_v5_hd512_lr1e4_3ep`：低学习率 + early stopping。

所有实验必须先过 WFI < 0.40 门，再比较 clip_style/lpips。

---

## 6. 附录：关键文件与产物

### 6.1 核心源码（当前分支）

| 文件 | 内容 |
|---|---|
| `src/blocks620.py` | SpatialBridgeBlock620：self-attn、cross-attn、多种 attn_mode、FiLM、MoE |
| `src/model620.py` | SpatialBridge620、FiLMEndpointHead、endpoint_lowhigh/velocity |
| `src/losses620.py` | SpatialBridgeObjective620：vertical FM、单步 SWD/edge、source endpoint aux |
| `src/style_encoder620.py` | StyleConditioner620：DINO patch/CLS 投影、adapter、text、local CNN、intrinsic |
| `src/config_schema.py` | 全部 620 配置字段与默认值 |

### 6.2 关键文档

| 文档 | 内容 |
|---|---|
| `docs/620/round1_diagnosis.md` | Round 1 7 变体 × 8 epoch 结果与根因诊断 |
| `docs/620/fog/final_summary.md` | Round E3 白化修复最终总结 |
| `docs/620/fog/decision_log.md` | 关键决策台账（attention/endpoint_film/HF residual/hd512） |
| `docs/620/fog/round_e2/experiment_report_2026-06-21.md` | E2 最小修复实验报告 |
| `docs/620/fog/round_e3/acceptance_report_2026-06-21.md` | E3 验收报告 |
| `docs/620/experiment_progress.md` | 当前最优 swd16_vlen0.04 e5 clip_style=0.7051 |
| `docs/620/phase4_plan.md` | 23 个架构实验、7 blocks、~14h |
| `docs/620/info_flow_analysis.md` | OT/Attn/SWD/Encoder 四问题理论分析 |
| `docs/620/final_summary.md` | 620 突破 0.67 天花板的三个结构性修复 |

### 6.3 生成的 diff patch（位于本目录）

| 文件 | 说明 |
|---|---|
| `diff_tokenizer_clean_to_620_spatial_bridge.patch` | tokenizer-clean → 620-spatial-bridge 的核心文件差异 |
| `diff_main_to_620_spatial_bridge.patch` | main → 620-spatial-bridge（main 已删 620 代码，差异主要为新增） |
| `diff_main_to_style_injection.patch` | main → origin/exp/style-injection... |
| `diff_main_to_attn.patch` | main → origin/attn |
| `diff_main_to_swd.patch` | main → origin/SWD |
| `diff_main_to_multistep_texture.patch` | main → origin/multistep-texture |

> 注：由于 `main` 已移除 620 源码，针对 `src/*620*.py` 的 branch-vs-main diff 基本为空；完整设计演进请重点参考 `diff_tokenizer_clean_to_620_spatial_bridge.patch`。

---

## 7. 总结

620 项目经历了一个清晰的演进弧线：

1. **619 诊断** 指出 OT 在线不稳定、伪 CrossAttention、ODE 展开梯度截断三大致命缺陷。
2. **620 主架构**（`codex/620-spatial-bridge`）引入 DINO 离线配对、真实 CrossAttention、单步 SWD，一举突破 0.67 天花板到 0.705。
3. **Round 1** 发现容量不足（dim=64）和缺少 self-attention 是 style 卡在 0.67 的主因。
4. **Fog 诊断** 发现白化根因是 endpoint shrinkage / style→endpoint 调制容量不足；Endpoint-FiLM hd512 通过 WFI 门。
5. **当前状态**：白化门已通，但距离 Seedream IDT 仍有差距；下一步应在 hd512 基线上做容量升级（dim=128）与 Phase 4 架构实验。
