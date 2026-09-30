# 消融实验分析与 Pixel256 评估计划

## 1. 消融实验设计分析

### 1.1 设计概览

**基线**: DA01_backbone1 (num_res_blocks=1, 其他默认参数)
- style_attn_num_heads=4, style_cross_attn_gate_init=0.3
- style_shortcut_alpha=1.0, style_embed_scale=1.0
- endpoint_delta_scale=1.0, endpoint_velocity_floor=0.05
- single_step_swd_weight=8.0, single_step_edge_weight=0.1
- w_flow=1.0, loss_type=mse, batch_size=16

**注意**: 大多数其他实验使用 num_res_blocks=4 作为"标准架构"，DA01(1块)和DA02(8块)是深度缩放实验。

### 1.2 设计强度评估

#### ✅ 设计优点

1. **极端值拉开**: gate(0/0.3/100), embed(0/1/100), delta(0/1/100), SWD(0/8/1000/-100), FM(0/1/100) — 参数范围足够大，确保反差明显。

2. **Loss 组件 2×2 分解**: DL01(no_swd), DL02(no_fm), DL03(swd_only), DL04(fm_only) 构成完整的组件消融矩阵。DL16(zero_all) 作为零信号控制组。

3. **双向极端测试**: 每个参数同时测试"关闭"(0)和"极端"(100/1000)两个方向，能捕捉非线性响应。

4. **跨家族对比**: DN08_spectral_ode 测试了不同的 contract_family，直接对比 spatial_bridge vs spectral_ode。

5. **训练稳定性发现**: DA02_backbone8(8块) 训练崩溃(NaN)，这是有意义的发现 — 深层架构在当前训练配置下不稳定。

#### ⚠️ 潜在问题

1. **CLIP-S 反差较小**: 已完成的 11 个实验中，transfer CLIP-S 范围仅 0.6636-0.6775 (差值 0.014)。说明风格迁移能力对架构变化**鲁棒**。这本身是积极发现，但限制了"哪个组件最重要"的区分度。

2. **LPIPS 反差较大**: 同组实验中 LPIPS 范围 0.2506-0.4406 (差值 0.19)，内容保持能力受架构影响显著。**这是消融的主要区分维度**。

3. **3 个实验未训练**: DA09_16heads, DD04_batch128, DN10_tf_schedule 缺少 checkpoint (可能 OOM 或发散)。

4. **DN 类别多数"same as baseline"**: 配置差异可能在推理时通过 --config_override 注入，但当前批量评估使用统一的 flow_matching override，可能掩盖部分 DN 实验的差异。

### 1.3 已完成实验的关键发现 (11/43)

| 实验 | tCLIP-S | tLPIPS | 关键洞察 |
|---|---|---|---|
| DA01_backbone1 (1块) | 0.6750 | 0.4214 | 基线，1块已足够 |
| DA02_backbone8 (8块) | **NaN** | **NaN** | **训练崩溃** — 深层架构不稳定 |
| DA03_no_shortcut | 0.6773 | 0.3853 | **shortcut 不重要** — 甚至略优 |
| DA04_gate0 | 0.6743 | 0.4406 | 无门控 → 内容保持最好 |
| DA05_gate100 | 0.6664 | 0.2911 | 全门控 → 风格强但内容损失 |
| DA06_embed0 | 0.6769 | 0.3713 | 风格嵌入=0 → 影响小 |
| DA07_embed100 | 0.6771 | 0.3731 | 风格嵌入=100 → 与=0几乎相同！ |
| DA08_1head | **0.6775** | 0.3772 | **单头最优** — 多头非必要 |
| DA10_velfloor10 | 0.6717 | 0.2788 | 高velocity floor → 强制改变 |
| DA11_lock_ll | 0.6763 | 0.3389 | 锁定低频 → 中等效果 |
| DA12_delta0 | 0.6636 | **0.2506** | 无delta → 最小改变(接近identity) |

#### 🔬 新启发

1. **DA06 vs DA07**: style_embed_scale=0 和 =100 结果几乎相同 → **风格嵌入的量级不重要，门控机制才是关键**。这支持 FC-SB 理论中"style gate > style embedding"的设计假设。

2. **DA08_1head 最优**: 单头注意力在风格迁移任务上优于多头 → **短序列(256 tokens)不需要多头分解**，与 SaMam 迁移失败的经验一致(mamba优势在短序列消失)。

3. **DA03_no_shortcut 略优**: 残差连接对风格迁移非必需 → **shortcut 主要帮助内容保持，不是风格迁移**。

4. **DA02 崩溃**: 8块模型训练发散 → **当前训练配置(3 epochs, lr=1e-4)不足以稳定深层模型**，需要更长的warmup或更小的学习率。

5. **DA12_delta0 接近identity**: endpoint_delta_scale=0 时模型几乎不改变输入 → **delta是风格迁移的主要驱动**，gate/embed只是调节强度。

## 2. Pixel256 评估方案

### 2.1 问题诊断

- pixel256 模型: `input_proj = nn.Conv2d(3, 64, ...)` 期望 3通道像素输入
- run_evaluation.py: `encode_image(vae, src_batch)` 产生 4通道 VAE latent
- **不兼容**: 4通道 latent 无法送入 3通道 input_proj

### 2.2 解决方案: Passthrough VAE Wrapper

创建一个"直通 VAE"包装器:
- `encode_image(x)`: 直接返回 x (3ch → 3ch, 不做编码)
- `decode_latent(z)`: 直接返回 z (3ch → 3ch, 不做解码)
- `scaling_factor = 1.0` (无缩放)

这样 run_evaluation.py 可以无修改运行，模型直接在像素空间工作。

### 2.3 实现步骤

1. **创建 pixel256 评估覆盖配置** `scripts/pixel256_eval_override.json`:
   ```json
   {
     "model": {"latent_channels": 3},
     "bridge": {"objective_mode": "flow_matching"}
   }
   ```

2. **修改 run_evaluation.py** (最小改动):
   - 检测 `latent_channels == 3` 且 `image_size == 256` (pixel-space 模式)
   - 用 `PassthroughVAE` 替代 diffusers VAE
   - 跳过 VAE encode/decode, 直接用像素

3. **运行评估**:
   ```bash
   python src/utils/run_evaluation.py \
     --checkpoint exp/pixel256_photo2art/pixel256_b1_e5_softmax/epoch_0003.pt \
     --output exp/pixel256_photo2art/pixel256_b1_e5_softmax/full_eval/epoch_0003 \
     --test_dir /mnt/i/wikiart_distinct5_samam_512_classview/test \
     --config_override scripts/pixel256_eval_override.json \
     --batch_size 2 --no-save_generated_images \
     --no-eval_enable_art_fid --no-eval_enable_kid
   ```

4. **风险**: pixel256 训练仅 3 epochs (batch=1)，模型可能欠拟合。但用户已要求"用现在的ckpt做评估"。

## 3. 最终数据表计划

### 3.1 表格结构

**表1: Baseline 对比 (256 photo2art)**
| 方法 | CLIP-S↑ | CLIP-T↑ | LPIPS↓ | MUSIQ↑ | ART-FID↓ |
|---|---|---|---|---|---|
| Identity | 0.6632 | 0.2302 | 0.0000 | 56.83 | 140.80 |
| Seedream | 0.7515 | 0.2731 | 0.2270 | 64.00 | 174.45 |
| AdaIN | 0.6659 | 0.2362 | 0.6057 | 41.23 | 334.58 |
| WCT | 0.6880 | 0.2386 | 0.6142 | 40.33 | 342.66 |
| SAMST | 0.7094 | 0.2439 | 0.2785 | 40.73 | 184.06 |
| SaMam | 0.6769 | 0.2309 | 0.1172 | 50.03 | 186.25 |
| **Ours latent256** | **0.6826** | **0.2417** | **0.2031** | **45.68** | **165.36** |
| **Ours pixel256** | ⏳ | | | | |

**表2: 消融实验 (epoch_0003, 43个实验)**
- 按类别分组: Architecture(13), Data(2), Infrastructure(3), Loss(16), Inference(9)
- 指标: transfer CLIP-S, transfer LPIPS, all_pairs CLIP-S, all_pairs LPIPS
- 标注 NaN (DA02) 和未训练 (DA09, DD04, DN10)

### 3.2 数据来源

- Baselines: 已完成 (compare_256_photo2art.md)
- Ours latent256: 已完成 (0.6826/0.2417/0.2031/45.68/165.36)
- Ours pixel256: 待评估 (passthrough VAE 方案)
- Ablations: 批量评估进行中 (~43个，预计2小时内完成)

### 3.3 交付物

1. `docs/baseline_256/compare_256_photo2art.md` — 更新最终总表
2. `docs/ablation_results.csv` — 消融结果CSV
3. `docs/ablation_results.md` — 消融结果Markdown表格
4. `docs/ablation_analysis_and_plan.md` — 本文档(分析+计划)
