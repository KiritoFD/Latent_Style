# 620 消融审计：Git 历史结论 vs 当前消融结果

> 生成时间：2026-06-21  
> 对照文件：`docs/620/fog/ablation_audit/git_history_digest.md`  
> 当前消融结果：`docs/620/fog/ablation_audit/results_summary.md`

---

## 1. 总体判断

本次 Phase 2/3 消融在 `endpoint_film_hd512` 基线上运行，大量历史结论被**复现、修正或推翻**。最显著的反转发生在 **DINO patch 条件源**、**容量升级必要性** 和 **endpoint_lowhigh + FiLM 的唯一性** 三个领域。以下按维度逐项对照。

---

## 2. 一致点（历史结论被当前消融复现）

### 2.1 DINO 离线 top-K 配对、真实 cross-attention、单步 SWD 是 620 突破根基

- **历史结论**：619 诊断指出 OT 在线不稳定、伪 CrossAttention、ODE 展开梯度截断三大缺陷；620 引入 DINO 离线配对、真实 CrossAttention、单步 SWD 后突破 0.67 天花板。
- **当前消融**：所有通过 WFI 门的变体均建立在这三项基础设施之上，没有挑战其有效性。
- **评级**：✅ 一致，无需重复验证。

### 2.2 NSWD 噪声（`swd_noise_sigma=0.02`）抑制白化

- **历史结论**：SWD 梯度几乎正交于目标方向（cos≈-0.024），引入 noise 可改善梯度方向。
- **当前消融**：关闭噪声 (`loss_nosigma`) WFI 从 0.3959 升至 0.4105；在 edge=0 设置下关闭噪声也导致 WFI 从 0.3786 升至 0.4077。
- **评级**：✅ 一致，应保留。

### 2.3 更多 epoch 在当前 lr 下会加剧白化

- **历史结论**：E3 `endpoint_film_hd512` 3 epoch WFI 从 e1 的 0.4271 升至 e3 的 0.4680。
- **当前消融**：所有 smoke 实验均为 1 epoch；尚未直接复测 3 epoch，但历史数据已被接受为约束。
- **评级**：✅ 一致，当前默认保持 1 epoch，多 epoch 需配合 early stopping / 低 lr。

### 2.4 Gated_raw / ReLU2 / style_select attention 在历史基线上恶化白化

- **历史结论**：`gated_raw` WFI=0.6435、`relu2` WFI=0.5340、`style_select` WFI=0.5005，均无法通过 WFI 门。
- **当前消融**：在 `endpoint_film_hd512` + `latent` 基线上，`gated_raw`/`relu2`/`style_select` WFI 分别为 0.3850/0.3856/0.3751，**均通过门**。
- **评级**：⚠️ **条件性一致**。历史结论在其自身基线上成立；当前基线已具备足够鲁棒性，这些 attention 核不再成为白化主因。

### 2.5 Legacy spatial prior / tokenizer 无效

- **历史结论**：PureLatentSpatial tokenizer zero ROI，legacy spatial prior 已移除。
- **当前消融**：未涉及此维度，但无证据要求恢复。
- **评级**：✅ 一致，维持移除。

---

## 3. 冲突点（当前消融推翻或修正历史结论）

### 3.1 dim=64 → 128 不会显著提升 clip_style

- **历史结论**：Round 1 诊断认为 dim=64 是 clip_style 0.67 平台的天花板，建议升级到 128 并恢复 self-attention。
- **当前消融**：
  - `capacity_128x4`：CLIP-S 0.7026（相对 64×4 的 0.7021 提升 0.0005）
  - `capacity_128x6`：CLIP-S 0.7019（低于 64×4）
  - WFI 在 128×4/128×6 上均未改善，反而 64×6 的 WFI 最优（0.3828）。
- **解读**：在 `endpoint_film_hd512` + `latent` 基线上，**style 瓶颈已从 Q 侧维度转移到其他因素**（可能是条件源、SWD/edge 权重、耦合方式）。单纯加宽模型无法复制 Round 1 预期中的收益。
- **评级**：🔴 **冲突**。历史建议的 dim=128 在白化门框架下不成立，至少在当前基线上不成立。

### 3.2 DINO patches 不再是风格注入的必要条件

- **历史结论**：620 突破依赖 DINOv2 256 tokens 作为 K/V；H6 intrinsic cross-attention 表现差（CLIP-S=0.6717，LPIPS=0.3678），未超越 DINO。
- **当前消融**：
  - `intrinsic_latent`：WFI=0.3842，CLIP-S=0.7020，LPIPS=0.3417，**通过 WFI 门**。
  - `target_dino_patches`：WFI=0.6407，CLIP-S=0.7097，LPIPS=0.2773，**严重白化**。
- **解读**：`endpoint_film_hd512` 的大容量 FiLM 映射补偿了 intrinsic latent 风格信号的不足，使得 latent 条件源也能达到此前 DINO 才能达到的风格强度。**DINO 在当前配置下从“必要”变为“有害”**。
- **评级**：🔴 **强烈冲突**。这是本次消融最重大的反转。

### 3.3 DINO adapter 无法修复 DINO 带来的白化

- **历史结论**：adapter 在 dim=64 下无效可能是因为 Q 维度受限，dim=128 后可能有效。
- **当前消融**：
  - `dino_adapter`：WFI=0.6076，相对 `dino_baseline` 的 0.6407 仅微降，仍严重超标。
  - adapter 同时未能在 latent 基线上测试；在 DINO 路径上已证伪。
- **评级**：🔴 **冲突**。adapter 不是 DINO 白化的解法。

### 3.4 endpoint_lowhigh + FiLM 并非白化修复的唯一路径

- **历史结论**：E2/E3 认为 endpoint_lowhigh + FiLM（hd512）是当前最优，WFI 从 0.4902 降至 0.3906。
- **当前消融**：
  - `endpoint_velocity`（无 FiLM）：WFI=0.3769，CLIP-S=0.7020，LPIPS=0.3315，**优于 hd512 基线**。
  - `endpoint_lowhigh_hd128`：WFI=0.3801，优于 hd256/hd512。
  - `endpoint_lowhigh_nofilm`：WFI=0.3957，与 hd512 接近。
- **解读**：历史“hd512 是关键容量突破”的结论在当前基线上被稀释。**真正的白化修复主要来自 `style_condition_source=latent` 和 `swd_noise_sigma=0.02`**，endpoint 结构只是次要的稳定器。
- **评级**：🟡 **部分冲突**。hd512 仍然有效，但不是唯一/最优解。

### 3.5 style_film_enabled 可关闭

- **历史结论**：`style_film_enabled=true` 被列为建议保留的默认配置。
- **当前消融**：`stylefilm_on` 与 `stylefilm_off` 的 WFI 差仅 0.0003，LPIPS 差 0.0001，CLIP-S 几乎相同。
- **评级**：🟡 **部分冲突**。保留无害，但关闭可简化模型，不影响指标。

### 3.6 H7 建议 SWD weight 8→2 不再适用

- **历史结论**：H7 将 SWD weight 从 8 降到 2 以解决梯度冲突。
- **当前消融**：
  - `loss_swd2`：WFI=0.4001（略超门），CLIP-S=0.7013，LPIPS=0.3304。
  - `loss_swd8`：WFI=0.3959，CLIP-S=0.7018，LPIPS=0.3369。
  - `loss_swd16_edge0`：WFI=0.3885，CLIP-S=0.7030，LPIPS=0.3396。
- **解读**：在白化优先框架下，SWD=2 不是最优；SWD=8 更平衡，而高 SWD 配合 edge=0 可进一步提升风格。
- **评级**：🟡 **条件性冲突**。H7 的动机（梯度冲突）可能真实存在，但其推荐的权重在当前评估体系下不优。

---

## 4. 需要重新框定的历史假设

### 4.1 “Q 侧维度不足”假设

- 历史认为 adapter/MoE/gate12 在 dim=64 下无效是因为 Q 维度受限。
- 当前消融显示 dim=128 也不提升 style，说明瓶颈可能不在 Q 维度，而在于：
  1. 条件源过强（DINO patches 导致均值解）；
  2. SWD/edge 权重与噪声的交互；
  3. endpoint/head 结构中的动态范围压缩。
- ** implication**：adapter/MoE 在 dim=128 下复测的优先级应降低，或至少改变测试基线（latent + edge=0）。

### 4.2 “DINO 是风格表征唯一来源”假设

- 历史认为 620 风格表征完全依赖 DINO。
- 当前消融显示 latent intrinsic cross-attention 在 endpoint-FiLM 辅助下可达到同等甚至更好的 WFI 表现。
- **implication**：应重新审视 intrinsic latent 路径的设计空间，例如更深的 intrinsic encoder、多尺度 latent 条件等。

### 4.3 “endpoint-FiLM 是解决白化的核心修复”假设

- 历史认为 Endpoint-FiLM hd512 是当前最优，WFI 0.4902→0.3906。
- 当前消融显示 velocity head  alone 即可达到 WFI=0.3769，说明白化修复的关键在于 `latent` 条件源和 NSWD 噪声，而非 endpoint-FiLM 本身。
- **implication**：endpoint-FiLM 可降级为可选增强，而非核心依赖。

---

## 5. 一致/冲突汇总表

| 历史结论 | 当前消融结果 | 一致/冲突 | 说明 |
|---|---|:---:|---|
| DINO 离线配对 / 真实 cross-attn / 单步 SWD 有效 | 未挑战，所有通过门变体依赖之 | ✅ 一致 | 基础设施 |
| NSWD noise 必要 | 关闭 σ 显著抬高 WFI | ✅ 一致 | 保留 |
| 更多 epoch 加剧白化 | 历史数据被接受 | ✅ 一致 | 1 epoch 默认 |
| gated_raw/relu2/style_select 有害 | 在当前基线上均通过门 | ⚠️ 条件一致 | 基线鲁棒性提升 |
| dim=64→128 突破 style 天花板 | 128 不提升 clip_style | 🔴 冲突 | 瓶颈转移 |
| DINO patches 必要 | latent 通过门，DINO 严重白化 | 🔴 冲突 | 最重大反转 |
| DINO adapter 在 dim=128 可能有效 | adapter 无法修复 DINO 白化 | 🔴 冲突 | 不默认开启 |
| endpoint_lowhigh+FiLM hd512 最优 | velocity/hd128 更优 | 🟡 部分冲突 | 非唯一路径 |
| style_film_enabled 保留 | 开关无差异 | 🟡 部分冲突 | 可关闭 |
| H7 SWD=2 更优 | SWD=2 WFI 超门 | 🟡 条件冲突 | 当前框架下不优 |
| legacy spatial prior/tokenizer 无效 | 未涉及 | ✅ 一致 | 维持移除 |

---

## 6. 对后续实验方向的影响

1. **不要默认恢复 DINO patches / adapter**：当前证据强烈表明它们在 `endpoint_film_hd512` 基线上导致白化，即使 CLIP-S 更高也不可接受。
2. **不要优先做 dim=128 升级**：在当前基线上收益极小，应把算力留给条件源细化、SWD/edge 组合扫描、多 epoch 稳定性。
3. **重新评估“必须 endpoint_lowhigh + FiLM hd512”**：velocity head 和 hd128 已展示更简洁更优的路径。
4. **保持 NSWD noise=0.02 作为硬性约束**：任何新实验若关闭 noise 必须先证伪现有证据。
5. **edge loss 默认关闭**：这是当前唯一的“三赢”开关，应立即作为新默认。

---

## 7. 原始数据索引

| 文件 | 内容 |
|---|---|
| `docs/620/fog/ablation_audit/git_history_digest.md` | 完整 Git 历史调研 |
| `docs/620/fog/ablation_audit/results_summary.md` | 当前消融统一汇总 |
| `docs/620/fog/ablation_audit/phase2_results.md` | Phase 2 核心维度 |
| `docs/620/fog/ablation_audit/phase3_capacity_results.md` | Phase 3.1 容量 |
| `docs/620/fog/ablation_audit/phase3_loss_results.md` | Phase 3.2 loss |
| `docs/620/fog/ablation_audit/phase3_dino_results.md` | Phase 3.3 DINO |
