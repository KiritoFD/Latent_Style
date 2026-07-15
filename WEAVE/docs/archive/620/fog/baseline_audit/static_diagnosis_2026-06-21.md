# Round E1.1 静态诊断：四个本地 smoke 实验对比

**日期**: 2026-06-21  
**范围**: `exp/620_spatial_bridge/620_film_v5_*_local_smoke/`（4 个本地 smoke 实验）  
**数据集**: 训练 `F:/wikiart_distinct5_samam_512_latents_ema/train`，测试 `F:/wikiart_distinct5_samam_512_classview_real/test`  
**基础模型**: SD 1.5，配置族 `620_spatial_bridge`，solver `solver_i2sb`  
**训练设置**: batch=4，accum=16，lr=2e-4，1 epoch，唯一变量为 `model.style_attn_mode`

---

## 1. 四个实验的完整指标对比

| 实验 | style_attn_mode | clip_style | content_lpips | transfer_clip_style | wfi_score | ΔWFI | clip_s_delta_idt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `620_film_v5_gated_local_smoke` | `gated` | **0.6987** | 0.3300 | **0.6646** | **0.4902** | **+0.1685** | 0.0588 |
| `620_film_v5_gated_raw_local_smoke` | `gated_raw` | 0.6987 | **0.2973** | 0.6634 | 0.6435 | +0.3218 | 0.0588 |
| `620_film_v5_relu2_local_smoke` | `relu2` | 0.6964 | 0.3102 | 0.6619 | 0.5340 | +0.2123 | 0.0564 |
| `620_film_v5_style_select_local_smoke` | `style_select` | 0.6982 | 0.3331 | 0.6642 | 0.5005 | +0.1788 | 0.0582 |

> 注：`gated` 实验缺少 `wfi_eval_report.json`，其指标来自 `full_eval_wfi/epoch_0001/summary.json` 与 `wfi_benchmark.json`；其余三个实验的 `wfi_eval_report.json` 均存在且与 benchmark 一致。

### 1.1 WFI 分量对比

| 实验 | contrast_ratio | dynamic_range | saturation | brightness | entropy |
|---|---:|---:|---:|---:|---:|
| source（共享） | 11.57 | 51.35 | 0.316 | 0.527 | 6.876 |
| `gated` | **2.40** | **36.95** | **0.139** | 0.569 | **6.713** |
| `gated_raw` | 1.62 | 26.87 | 0.102 | **0.719** | 6.056 |
| `relu2` | 2.10 | 33.62 | 0.124 | 0.600 | 6.568 |
| `style_select` | 2.30 | 35.19 | 0.141 | 0.564 | 6.631 |

**解读**：

- `gated` 在四个变体中 WFI 最低（0.4902），主要优势来自更高的对比度比（2.40）、更大的动态范围（36.95）和更高的饱和度（0.139）。
- `gated_raw` 白化最严重：对比度比仅 1.62，亮度被显著拉高到 0.719，饱和度最低（0.102）。这说明移除 softmax 重归一化后，cross-attention 输出幅度不稳定，被后续 norm 拉回高亮、低饱和状态。
- `relu2` 与 `style_select` 介于两者之间，但均未解决低对比度/低饱和度问题。

### 1.2 source vs generated 统计差异

以 `gated`（当前最优基线）为例，生成图相对 source 的关键变化：

| 指标 | source | generated | 变化 | 含义 |
|---|---:|---:|---|---|
| contrast_ratio | 11.57 | 2.40 | ↓ 79.3% | 生成图对比度显著压缩 |
| dynamic_range | 51.35 | 36.95 | ↓ 28.0% | 动态范围缩小 |
| saturation | 0.316 | 0.139 | ↓ 56.0% | 饱和度大幅下降 |
| brightness | 0.527 | 0.569 | ↑ 8.0% | 亮度轻微上抬 |
| entropy | 6.876 | 6.713 | ↓ 2.4% | 直方图熵略降 |
| wfi_score | 0.322 | 0.490 | ↑ +0.168 | 白化分数显著上升 |

其余三个变体呈现相同模式，但 `gated_raw` 的亮度上升更剧烈（+36.5%），`style_select` 与 `gated` 最接近。

### 1.3 identity 与 transfer 子集

`gated` benchmark 显示：

- **identity_wfi**（source→同风格）：0.5056，高于 source 均值 0.3217，说明即使是身份重建也会引入轻微白化。
- **transfer_wfi**（source→异风格）：0.4881，略低于 identity，说明风格迁移本身不是白化加剧的主因；白化是模型输出的系统性特征。
- 25 个 style-pair 中，WFI 最低的为 `Ukiyo→Early_Renaissance`（0.4094）和 `Ukiyo→Rococo`（0.4117），最高的为 `Minimalism→Impressionism`（0.5796）和 `Minimalism→Ukiyo_e`（0.5579）。Ukio-e 作为 source 时白化较轻，可能与其本身高对比度、高动态范围有关；Minimalism 作为 source 时方差大、白化重，说明低饱和 source 更难被模型恢复动态范围。

---

## 2. CLIP-S 与 LPIPS 的权衡

| 实验 | clip_style | content_lpips | 权衡关系 |
|---|---:|---:|---|
| `gated` | 0.6987 | 0.3300 | 白化最少，但内容失真最大 |
| `gated_raw` | 0.6987 | 0.2973 | 内容保留最好，但白化最严重 |
| `relu2` | 0.6964 | 0.3102 | 中间状态 |
| `style_select` | 0.6982 | 0.3331 | 与 gated 类似 |

**关键观察**：

- 四个变体的 `clip_style` 几乎相同（0.696–0.699），说明 attention 模式对高层风格相似度影响极小。
- 但 `content_lpips` 与 `wfi_score` 呈**反向关系**：`gated_raw` 内容保留最好（LPIPS 最低）但白化最重；`gated`/`style_select` 白化较轻但内容失真更大。
- 这一冲突表明：当前架构下，attention 机制的改动只能在“保留原图细节”和“抑制白化”之间做零和交换，无法同时优化两者。真正的瓶颈不在 attention 内部，而在 **style 信号是否足够强地转化为 endpoint 高频位移**。

---

## 3. 当前最优基线确认

**当前最优基线**: `620_film_v5_gated_local_smoke`

理由：

1. WFI 最低（0.4902），ΔWFI 最小（+0.1685）。
2. `transfer_clip_style` 最高（0.6646），说明跨风格迁移能力略优。
3. `clip_style` 持平（0.6987），未牺牲高层风格相似度。
4. 图像空间统计相对最接近健康参考：对比度比 2.40、饱和度 0.139。

**遗留代价**：`content_lpips=0.3300` 是四者最高，说明“减少白化”当前仍靠牺牲部分内容保留实现。

---

## 4. 静态诊断结论

1. **Attention 模式不是根因**：四种 attention 变体在 `clip_style` 上差异极小，WFI 差异主要来自图像空间统计的漂移，而非风格对齐失败。
2. **`gated` 是当前最优**：其 softmax 重归一化虽然仍接近均匀分布，但至少能避免 `gated_raw` 的亮度失控和 `relu2`/`style_select` 的过度稀疏。
3. **白化的主要统计特征是高频缺失**：低对比度、低饱和度、轻微亮度上抬的组合，指向 latent/图像高频分量未被充分迁移，而非整体端点 shrinkage。
4. **下一步应聚焦 endpoint head 高频调制**：单纯的 attention/gate/FiLM 在 block 内已不足以恢复高频；需要让 style 信号直接、强有力地调制 endpoint/velocity 的高频分量。

---

## 附录：被审计实验文件清单

| 实验目录 | config | train.log | epoch_0001.pt | wfi_eval_report.json | summary.json | wfi_benchmark.json |
|---|---|---|---|---|---|---|
| `620_film_v5_gated_local_smoke` | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| `620_film_v5_gated_raw_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `620_film_v5_relu2_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `620_film_v5_style_select_local_smoke` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
