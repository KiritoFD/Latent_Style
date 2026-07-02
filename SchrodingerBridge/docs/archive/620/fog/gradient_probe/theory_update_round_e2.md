# Round E2 理论更新：高频塌缩叙事细化

**日期**: 2026-06-21  
**依据**:
- `docs/620/fog/round_e2/experiment_report_2026-06-21.md`
- `docs/620/fog/gradient_probe/internal_probe_endpoint_film_epoch_0001.json`
- `docs/620/fog/gradient_probe/internal_probe_hf_residual_epoch_0001.json`
- `docs/620/fog/gradient_probe/probe_report_2026-06-21.md`

---

## 1. Round E1 核心结论回顾

Round E1 判定 620 白化主因是 **高频选择性塌缩**：

$$
c = \frac{\text{endpoint_alpha}}{1-t} \approx 0.62, \quad
 c_\text{high} = \frac{\text{endpoint_high_alpha}}{1-t} \approx 0.08
$$

基于该结论，Round E2 验证了两个最小修复：

1. **P0 Endpoint-FiLM Head**：让 style 信号直接调制 endpoint 的低频/高频分支。
2. **P1 High-Frequency Residual**：在 velocity head 输出上加入 source latent 的高通残差。

---

## 2. Round E2 实验结果对理论的修正

### 2.1 关键发现

| 实验 | WFI ↓ | clip_style ↑ | content_lpips ↓ | $c$ | $c_\text{high}$ |
|---|---:|---:|---:|---:|---:|
| Baseline (E1) | 0.4902 | 0.6987 | 0.3300 | 0.617 | 0.078 |
| P0 Endpoint-FiLM | **0.4283** | **0.7066** | **0.3226** | **0.290** | **0.053** |
| P1 HF Residual | 0.4746 | 0.7020 | 0.3263 | 0.596 | 0.081 |

### 2.2 理论修正

#### （1）"高频塌缩"不等于"高频 RMS 位移不足"

P0 Endpoint-FiLM **显著降低了 WFI**，但 $c$ 和 $c_\text{high}$ 都 **下降** 了（$c$ 从 0.62→0.29，$c_\text{high}$ 从 0.08→0.05）。这说明：

> **WFI/白化与 RMS 位移大小不是单调关系**。即使整体 RMS 位移变小，只要位移方向更准确地指向目标风格的低频/高频统计分布，图像质量仍可改善。

因此，"高频选择性塌缩"的叙事需要从 **"高频 RMS 位移不足"** 细化为 **"style 信号未有效调制高频/细节方向"**。

#### （2）Endpoint head 是 style 调制的关键瓶颈

P0 通过让 style global embedding 直接驱动 endpoint 的 low/high delta，绕过了 trunk block 中 attention 平均化和 FiLM 后 GN 的削弱。结果：

- clip_style 提升（0.6987→0.7066）
- content_lpips 下降（0.3300→0.3226）
- WFI 下降（0.4902→0.4283）

这支持 Round M/E1 中 **Endpoint-driven** 机制的主导地位，但具体机制不是"增大位移幅度"，而是"让 style 直接决定 endpoint 的方向"。

#### （3）简单的 source 高频保留不能解决风格迁移高频缺失

P1 HF Residual 加入 $w \cdot (x - \text{avg_pool}(x))$ 到 velocity，希望保留 source 高频。结果：

- $c$ 和 $c_\text{high}$ 与基线几乎相同
- WFI 仅微降（0.4902→0.4746）
- 学到的 $w$ 从 0.1 降至 0.089

说明 **保留 source 高频不是风格高频迁移的充分条件**。风格化需要目标风格的高频，而不是源内容的高频。

#### （4）Attention 平均化仍是背景约束

三个实验的 `cross_attn_entropy` 均接近 $\ln(256) \approx 5.54$，没有改善。FiLM/Endpoint-FiLM 是 **绕过 attention 平均化** 的有效手段，但 attention 本身仍是 style-specific 信号弱化的起点。

---

## 3. Round M 预测状态更新

| 编号/预测 | 原判定 | E2 更新 | 理由 |
|---|---|---|---|
| P-E3: endpoint + style-FiLM 恢复 sensitivity | 待补证/高度优先 | **部分支持** | WFI↓、clip_style↑，但 $c_\text{high}$ 未达 ≥0.3；改善来自方向而非幅度 |
| "高频塌缩 = 高频 RMS 不足" | 隐含 | **否证** | Endpoint-FiLM 降 WFI 但 $c_\text{high}$↓ |
| HF Residual 保留 source 高频可防白化 | 新假设 | **不支持** | WFI 微降，$c_\text{high}$ 不变，网络弱化该残差 |
| Attention 平均化是 style 弱化起点 | 支持 | **支持** | entropy 仍接近上限 |
| Endpoint-driven 机制 | 高 | **高（方向调制 > 幅度）** | P0 验证 endpoint head 是决定性位置 |

---

## 4. 下一轮允许验证的候选方案

基于 E2 结果，以下方案与证据链直接对应：

### H1: 增大 Endpoint-FiLM 高频分支的幅度

- **改动**: `endpoint_high_scale=2.0` 或 `endpoint_velocity_floor=0.01`
- **理论预测**: 直接提升 $c_\text{high}$，可能进一步降低 WFI
- **成功门槛**: WFI < 0.40，$c_\text{high}$ ≥ 0.15
- **失败信号**: WFI 不变或恶化，clip_style 下降

### H2: 调整 low/high 分解带宽

- **改动**: `endpoint_lowpass_kernel=3` 或 `7`
- **理论预测**: 当前 kernel=5 的高频带过窄，调整可能让 style-FiLM 覆盖更合适的频段
- **成功门槛**: WFI 下降
- **失败信号**: WFI 不变

### H3: 去掉 Endpoint-FiLM Head 内的 GroupNorm

- **改动**: 修改 `FiLMEndpointHead`，移除 `self.norm`
- **理论预测**: GN 压缩动态范围，移除后 high_delta 幅度增大
- **成功门槛**: $c_\text{high}$ 提升，WFI 下降
- **失败信号**: 训练不稳定或 WFI 恶化

### H4: 在 Endpoint-low/high 模式下加入 source 高频残差

- **改动**: 对 `high_delta` 加入 `w * (x - lowpass(x))`
- **理论预测**: 同时获得 style-FiLM 调制与 source 细节保留
- **成功门槛**: WFI < 0.40
- **失败信号**: 与 P0 相比无额外收益

**禁止并行堆叠**：每次只改动一个参数。

---

## 5. 总结

Round E2 的核心理论修正是：

> 620 白化的本质不是 endpoint **幅度** shrinkage，而是 **style 信号未能有效调制 endpoint 的低频/高频方向**。Endpoint-FiLM Head 通过让 style 直接驱动 endpoint delta，在 RMS 位移减小的情况下同时改善了风格迁移质量与图像动态范围。

下一步应继续沿 Endpoint-FiLM 方向做最小参数扫描（high_scale、kernel、GN 去留），直至 WFI 压到 <0.40。
