# 给 KiritoFD: 突破 style 天花板的具体行动

> 当前状态: topogate_appalign e3 收敛 (style 0.672/0.703, LPIPS 0.315/0.313)
> **结构防线已坚固 (LPIPS 0.31)，唯一瓶颈是 style 推不到 0.72+**
> 以下三条路径，按优先级排列。

## 对用户三个提案的评估

### 提案1: SDE / PC Solver 推理 — ✅ 完全合理，立即执行
- 代码已就绪 (`solver_pc` 的 `latent_lowpass` 校正, `solver_unsb_cycle` 的 SDE-EM 噪声)
- 零风险: 不重新训练，只改 eval config
- 已有 topogate 锁结构 + PC 校正双保险，LPIPS 大概率不会超过 0.38
- **立即执行，不需要等任何东西**

### 提案2: I2SB Endpoint + TopoGate 训练 — ⚠️ 方向对，细节需修正
- "Endpoint 不会崩因为 topogate 锁空间" — 这个判断有道理但未验证
- "I2SB公式 c1*xt 锚定源图" — 正确，这是 I2SB 的核心保证
- **但不要用 σ=0.5！** topogate 保护下用 σ=0.02-0.05 就够了
- 队列中已有 `i2sb_sigma0p02_residual_tfloor005`，直接启动即可

### 提案3: Tokenizer 增强 — ⚠️ 方向对但大半已实现
- "加 Positional Encoding" → **已经有了** (`pe_temperature=1.0` 默认开启 sinusodial PE)
- "扩大 spatial_dim 128→256" → 可行，但需重新训练，优先级放最后
- "32 clusters 扩大" → 当前已用32，在配置中可调
- 这些是边际改进，等 SDE/PC 路线验证完再考虑

## 执行计划 (按优先级)

---

## 路径 A: PC Solver 推理 (推荐首选)

**原理**: Predictor 正常走 ODE 风格化，Corrector 用 latent 低频 MSE 把宏观结构拉回源图。风格笔触（高频）完全不受影响。

**配置**: 已就绪 `configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json`

**参数扫描建议** (创建 3 个 override config):
```
step_size=0.04  (轻校正, style可能保持)
step_size=0.06  (中校正, 推荐起点)
step_size=0.10  (重校正, 可能LPIPS更好但style被压制)
```

**执行命令** (在远程 WSL):
```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
# 轻校正
python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.04
# 中校正
python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.06
# 重校正
python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.10
```

**预期**: LPIPS 微升到 0.33-0.35，style 推到 0.69-0.71。如果 style 没有提升，说明低频 MSE 不贡献 style——需要走 B 路线。

**代码已就绪**: `model.py:591-623` (`_correct_transport_state`), `model.py:1100-1103` (`solver_pc`)

---

## 路径 B: SDE-EM 推理 (直接注入随机性)

**原理**: 用 solver_unsb_cycle，每一步先走 ODE + 内容校正，再注入微量布朗噪声：
$$x_{t+1} = x_t + v_\theta(x_t, t)\Delta t + \sigma\sqrt{\Delta t}\epsilon$$

噪声能打破确定性 style 轨迹的 mode collapse，逼出更多目标笔触。

**配置**: 已就绪 `configs/aaai2027/phase2_eval_sde_em_topogate_e2.json`

**参数扫描建议**:
```
noise_scale=0.005  (极轻噪声)
noise_scale=0.010  (轻噪声, 推荐起点)  
noise_scale=0.020  (中噪声)
noise_scale=0.030  (重噪声, 可能 LPIPS 开始崩)
```

**执行命令**:
```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python src/run.py --config configs/aaai2027/phase2_eval_sde_em_topogate_e2.json \
  --override model.solver_stochastic_noise_scale=0.015
```

**预期**: style 明显突破 (0.70-0.72)，LPIPS 微升到 0.33-0.37。noise_scale 是关键超参——太大 LPIPS 崩，太小没有 style 提升。

**代码已就绪**: `model.py:1104-1114` (`solver_unsb_cycle`), noise_scale 在 model config 中.

---

## 路径 C: I2SB Endpoint + TopoGate 训练 (如果 A/B 都不够)

**前提**: 仅当路径 A 和 B 都无法把 style 推到 0.72 时才走这条路。

**原理**: 回归 endpoint 模式，但用 topogate 锁结构 + 极小 σ 的 I2SB 训练。
之前的 endpoint 崩是因为 SemanticCrossAttn 满天飞 + σ 太大(0.5)。
现在 topogate 锁死空间路由，I2SB 公式 $\mu = c_1 x_t + c_2 \hat{x}_1$ 第一项强烈锚定源图。

**需要做的事**:
1. 创建训练 config: `transport=endpoint, objective=i2sb_endpoint, bridge_sigma=0.02, tokenizer_family=pure_latent_spatial, solver_family=solver_i2sb, semantic_self_topology_gate=true`
2. 用 topogate e1 ckpt 做 warmstart
3. 训练 8-12 epochs，观察 style 是否突破 0.72

**⚠️ 注意**: 
- 不要用 σ=0.5！用 σ=0.02-0.05。topogate 已经解决了结构问题，只需要微布朗噪声突破 style
- 观察 LPIPS 是否会突破 0.40。如果会→立即降低 σ
- 已经在队列中的 `i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005` 可以直接启动

**代码已就绪**: `model.py:516-542` (`_i2sb_transport_step`), `model.py:886-894` (solver 选择)

---

## 关于 Tokenizer 增强

当前 tokenizer **已经具备**:
- PE (pe_temperature=1.0 默认开启的 sinusodial positional encoding)
- 32 clusters
- 4+ ResBlock query_extractor
- global_code = GAP(spatial_map) + gate + embedding

如果需要可以直接在 config 中调整:
- `query_dim`: 64 → 96 (需要重新训练)
- `num_clusters`: 32 → 48 (需要重新训练)
- `pe_temperature`: 1.0 → 0.75 (可 eval 时改)
- `spatial_dim`: 128 → 256 (需要重新训练，显著增加参数量)

但这些需要重新训练，优先级放在 SDE/PC 推理之后。

---

## 推荐执行顺序

```
GPU 空闲后:
  1. 先跑路径A (PC solver eval, 3 个 step_size) — 最快, 零风险
  2. 同时跑路径B (SDE-EM eval, 2 个 noise_scale) — 并行
  3. 看 A/B 结果:
     - style > 0.72? → 成功了, 提交论文
     - style 0.70-0.72 但 LPIPS < 0.35? → 组合 A+B 试试
     - style < 0.70? → 路径C (I2SB 训练)
  4. 路径C 如果也失败 → 考虑减弱 topogate blend 或换成 PnP self-inject
```

## 预期时间

| 路径 | 操作 | 时间 |
|------|------|------|
| A | PC eval (3 个 step_size) | ~12min (3×4min eval) |
| B | SDE-EM eval (2 个 noise_scale) | ~8min (2×4min eval) |
| C | I2SB 训练 | ~20min/epoch × 8 epoch = ~3h |

---

# 第二部分: 训练与模型设计的探索方向

> 以下 8 个方向覆盖训练策略和模型架构两个层面。
> 每个方向标注了难度、预期收益和风险。
> 建议按推荐顺序执行——前面的失败再试后面的。

---

## 方向 1: 自适应 Kinetic 调度 (训练策略)

**难度**: 低 | **需重训**: 是 | **预期收益**: style +0.02~0.04 | **风险**: 低

**假设**: 当前固定 `w_kinetic=0.85-0.95` 从头用到尾，对 style 的压制是均匀的。但模型在早期需要 kinetic 来学结构，后期结构已学会，kinetic 就成了 style 的枷锁。

**方案**: 在 bridge config 中实现 kinetic 衰减调度：
- epoch 1-3: `w_kinetic=1.0` (学结构)
- epoch 4-6: `w_kinetic=0.7`
- epoch 7-12: `w_kinetic=0.4`
- epoch 13+: `w_kinetic=0.2` (释放 style)

**配置**: 当前代码可能不支持 epoch 级调度。需要新增 `kinetic_warmup_epochs` + `kinetic_decay_end` 参数。
也可以在 `trainer.py` 中硬编码一个线性衰减的 lambda。

**预期**: style 从 0.70 推到 0.71-0.72，LPIPS 可能在后期微升到 0.33-0.35。

**风险**: 如果后期 kinetic 过低导致 LPIPS 崩溃→回退到上一个 kinetic 级别。

---

## 方向 2: 渐进式 Topogate 解锁 (训练策略)

**难度**: 低 | **需重训**: 是 | **预期收益**: style +0.03~0.06 | **风险**: 中

**假设**: topogate 的 `blend=1.0` 直接把结构锁死了——LPIPS 极好但 style 上不去。就像开车把手刹拉到最紧然后踩油门。

**方案**: 两阶段训练：
- **Stage 1 (epochs 1-4)**: topogate blend=1.0, kinetic=1.0 → 学结构基础
- **Stage 2 (epochs 5-16)**: topogate blend 逐步降到 0.5-0.7, kinetic 降到 0.3-0.5 → 在保有一定结构约束的同时释放 style

**配置**: 需要修改 `semantic_self_topology_blend` 支持 epoch 调度。
或者更简单：用 stage2 的 config 从 stage1 的 ckpt warmstart。

**执行**: 
```bash
# Stage 1: 已经跑完了 (topogate_k085 e1-e3)
# Stage 2: 用新 config, resume 从 topogate e1
config: "semantic_self_topology_blend": 0.55,
        "semantic_self_topology_gate": true,
        "w_kinetic": 0.45,
        "kinetic_penalty_mode": "manifold_adaptive_split"
```

**预期**: style 突破 0.71-0.73，LPIPS 回升到 0.34-0.38 但仍可接受。

**风险**: blend 降太多可能导致 LPIPS > 0.40。需要扫描 `blend ∈ {0.8, 0.65, 0.50, 0.35}`。

---

## 方向 3: Dual-Path Content/Style 训练 (模型设计)

**难度**: 中 | **需重训**: 是 | **预期收益**: style +0.03~0.05 | **风险**: 中(显存翻倍)

**假设**: MasaCtrl/StyleAligned 的核心发现——同一个网络运行两次，一次输入 content style，一次输入 target style，但在 Attention 层共享 content path 的 QK 矩阵。这样 target path 的 V(风格信息) 被 content path 的 A(空间拓扑) 约束。

**方案**: 在 `model.py` 的 forward 中实现 dual-path：
```python
# Path A (结构锚点): style=source
feat_A = run_body(x, style_id=source_style_id)
# Path B (风格渲染): style=target, 但 Attention QK 从 feat_A 注入
feat_B = run_body(x, style_id=target_style_id, attn_inject_from=feat_A)
# delta = feat_B 的高频部分 + feat_A 的低频部分
```

**配置**: 新增 `model.backend_attention_family = "attn_dual_path"` (需要在 `lancet_blocks.py` 实现)。

**预期**: 显存翻倍，每 epoch 约 40min (batch 要减半)。style 上限可能推到 0.72+, LPIPS 保持 < 0.35。

**风险**: 实现复杂度高。如果 topogate 已经足够好 (LPIPS 0.31)，dual-path 的边际收益可能不大。先试方向 2 再决定。

---

## 方向 4: 多尺度 Topogate (模型设计)

**难度**: 中 | **需重训**: 是 | **预期收益**: LPIPS 更低或 style 更高 | **风险**: 低

**假设**: 不同空间尺度的结构约束需求不同。粗尺度 (low-res feature maps) 需要强拓扑锁定来保大局结构；细尺度 (high-res) 可以放松来允许笔触细节。

**方案**: 在 UNet 的多个分辨率层施加不同 blend 的 topogate：
```json
"semantic_self_topology_blend_per_scale": {
    "8": 1.0,    // 最粗尺度: 完全锁定 (大局结构)
    "16": 0.8,   // 中间尺度: 较强约束
    "32": 0.5,   // 较细尺度: 部分放松
    "64": 0.3    // 最细尺度: 大幅放松 (允许笔触)
}
```

**配置**: 需要修改 `lancet_blocks.py` 中的 topogate 逻辑支持 per-scale blend。

**预期**: 比单尺度 blend 更精细的风格-结构 tradeoff。在保持 LPIPS < 0.35 的同时 style 能到 0.71+。

---

## 方向 5: Velocity-Endpoint 混合模式 (训练策略)

**难度**: 中 | **需重训**: 是 | **预期收益**: style +0.04~0.07 | **风险**: 中高

**假设**: Velocity 在近源端 (t≈0) 做"微小编辑"保结构，Endpoint 在近目标端 (t≈1) 做"大胆重绘"提风格。当前纯 velocity 模式到了 t≈1 也不敢"下重手"。

**方案**: 混合预测模式——训练时同时预测 velocity 和 endpoint，用时间门控加权：
```
t < 0.5:  loss = velocity_loss * 1.0 + endpoint_loss * 0.1
t >= 0.5: loss = velocity_loss * 0.3 + endpoint_loss * 0.7
```
推理时：t<0.5 用 velocity Euler，t≥0.5 用 endpoint I2SB。

**配置**: 新增 `transport_prediction_mode = "mixed_velocity_endpoint"` 和 `endpoint_blend_threshold = 0.5`。

**预期**: 综合了 velocity 的早期结构保护和 endpoint 的后期风格爆发。可能推到 0.73+。

**风险**: 实现复杂度高。如果 endpoint 在 t>0.5 阶段把 source 结构洗掉了，LPIPS 还是会崩。需要配合 topogate。

---

## 方向 6: Style-Specific Contrastive Terminal SWD (训练策略)

**难度**: 低 | **需重训**: 是 | **预期收益**: style +0.01~0.03 | **风险**: 极低

**假设**: 当前 Terminal SWD 是"把 z_1 的分布推向任意目标风格图片的分布"，但没有区分"这是 Early_Renaissance 的分布 vs 这是 Impressionism 的分布"之间的 fine-grained 差异。

**方案**: 对每个 target style，只用该 style 的样本算 SWD，附加一个 style-contrastive 项：
```python
# 正样本: z_1 与相同 style 的目标分布的距离
loss_pos = SWD(z_1, target_images_of_same_style)
# 负样本: z_1 与不同 style 的目标分布的距离 (推远)
loss_neg = -SWD(z_1, target_images_of_different_style)
# 总 SWD = w_pos * loss_pos + w_neg * loss_neg
```

**配置**: 可以在 `losses.py` 中修改 Terminal SWD 的计算逻辑，在 batch 内按 style_id 分组。

**预期**: 小幅提升 (0.01-0.03)。主要是增加"风格鉴别力"——对 Impressionism 画出来的真的是 Impressionism 而不是别的。

---

## 方向 7: Tokenizer 温度退火 (训练策略)

**难度**: 极低 | **需重训**: 是 | **预期收益**: style +0.01~0.02 | **风险**: 极低

**假设**: Tokenizer 的 attention temperature 决定了 cluster 路由的"软硬"程度。高温→软路由→多个 cluster 混合→平滑但缺乏特异性。低温→硬路由→单一 cluster 主导→锐利但可能丢失多样性。

**方案**: 训练过程中退火 temperature：
- epoch 1-4: temperature=0.10 (软路由，探索)
- epoch 5-8: temperature=0.07
- epoch 9-12: temperature=0.05
- epoch 13+: temperature=0.03 (硬路由，锐利笔触)

**配置**: 新增 `tokenizer_temperature_schedule` 参数或在 trainer 中动态修改 tokenizer 的 temperature。

**预期**: 后期更锐利的 cluster 路由产生更明确的笔触风格，小幅推高 style。

---

## 方向 8: One-Cycle Style Push (训练策略 + 推理)

**难度**: 低 | **需重训**: 否 (eval only) | **预期收益**: style +0.01~0.02 | **风险**: 极低

**假设**: 推理时可以多跑几个不同的 step_size / style_strength，取 Pareto 前端的点。如果某组参数 style 更高而 LPIPS 可接受→直接用。

**方案**: 对 topogate e2 ckpt，做 sweep：
```bash
for strength in 1.0 1.1 1.2 1.3; do
    for steps in 8 12 16; do
        python eval.py --ckpt e2 --style_strength $strength --num_steps $steps
    done
done
```

**配置**: `full_eval_num_steps` 和 `full_eval_style_strength` 已在 config 中可配。

**预期**: 在某个 (strength=1.15, steps=14) 组合下，style 可能微升 0.01-0.02 而 LPIPS 不变。这是"免费午餐"——改参数不改变网络。

---

## 方向优先级总览

| 优先级 | 方向 | 类别 | 时间 | 预期 style 收益 |
|--------|------|------|------|:---:|
| 1 | PC solver eval (路径A) | 推理 | 12min | +0.01~0.03 |
| 2 | SDE-EM eval (路径B) | 推理 | 8min | +0.02~0.04 |
| 3 | One-Cycle Style Push (方向8) | 推理 | 20min | +0.01~0.02 |
| 4 | 自适应 Kinetic 调度 (方向1) | 训练 | ~3h | +0.02~0.04 |
| 5 | Style-Specific SWD (方向6) | 训练 | ~3h | +0.01~0.03 |
| 6 | 渐进 Topogate 解锁 (方向2) | 训练 | ~3h | +0.03~0.06 |
| 7 | Tokenizer 温度退火 (方向7) | 训练 | ~3h | +0.01~0.02 |
| 8 | 多尺度 Topogate (方向4) | 模型 | ~4h | +0.02~0.04 |
| 9 | Velocity-Endpoint 混合 (方向5) | 训练 | ~4h | +0.04~0.07 |
| 10 | Dual-Path 训练 (方向3) | 模型 | ~6h | +0.03~0.05 |

**推荐执行策略**: 先做完 1→2→3（推理，不做重训）。如果三者加起来 style 从 0.70 推到 0.71-0.72 → 成功了。
如果不到 0.72 → 选方向 1+6+2 组合训练（kinetic 衰减 + topogate 解锁 + I2SB σ=0.02）。
如果还不够 → 方向 4+5（多尺度 topogate + mixed prediction）。

---

# 第三部分: 从第一性原理出发的革命性方向

> 以下方向不从"调参"出发，而是从风格迁移的数学本质、
> 最优传输的几何结构、薛定谔桥的随机过程含义、以及潜空间的微分几何性质出发。
> 每个方向附有理论论证和具体的代码落地路径。

---

## 革命方向 A: 潜空间黎曼流匹配 (Riemannian Latent Flow Matching)

### 理论

当前框架的核心假设——VAE 潜空间是平坦欧氏空间——是错的。

SDXL VAE 将 512×512×3 的图像压缩到 4×64×64 的潜变量。这个压缩映射不是等距的：
两个在像素空间距离为 d 的图像，在潜空间的距离不是 d 的线性函数。
潜空间本身是一个弯曲的流形 $\mathcal{M} \subset \mathbb{R}^{4 \times 64 \times 64}$，带有诱导度量 $g_{ij}$。

**当前做法**: 在欧氏空间中沿直线 $z_t = (1-t)z_0 + t z_1$ 插值。这条直线大概率**穿出数据流形**——
中间状态对应的图像不是"半风格化"的照片，而是无意义的噪声。这就是为什么中间 state 的预测质量差，也是为什么 Euler 多步积分有时比单步还差。

**正确做法**: 在流形上沿测地线 (geodesic) 传输。测地线同样是 $z_t = (1-t)z_0 + t z_1$，但是**速度场被投影到切空间**：
$$v_{\text{geo}}(z, t) = \Pi_{\mathcal{T}_z\mathcal{M}} \left[ v_{\text{raw}}(z, t) \right]$$

其中 $\Pi$ 是到流形切空间的投影算子。

### 为什么这能推高 style

当前 style 卡在 0.70 的根本原因可能是：模型在 t≈0.8-1.0 的关键区间，由于预测的中间状态已经 off-manifold，
无法产生高质量的"收尾"笔触。测地线传输保证每一步都在流形上，每一步的预测都是"有效的图像状态"，
收尾阶段自然更锐利。

### 实现路径

**不需要精确计算黎曼度量**。可以用廉价代理：

```python
# 在 lancet_runtime.py 的 transport_step 中:
# 计算 VAE decoder 对当前状态的 Jacobian 的近似
# 用这个近似构造切空间投影
def _riemannian_projection(self, v_raw, z):
    # 用 VAE decoder 的前几层作为"流形探测器"
    # 如果 v_raw 会导致 decoder 输出"不自然"的图像 → 抑制该分量
    z_next = z + v_raw * dt
    # 用 decoder 浅层特征的一致性作为 on-manifold 判断
    feat_z = self.vae_encoder_shallow(z)
    feat_next = self.vae_encoder_shallow(z_next)
    # 特征差异大的方向 = off-manifold 方向 → 抑制
    gate = torch.sigmoid(-10.0 * (feat_next - feat_z).abs().mean(dim=1, keepdim=True))
    return v_raw * gate
```

**更简单的方式**: 用 `diffeomorphic_stroke` 作为 Riemannian retraction。
每次 transport step 后，用微小的 diffeomorphic warping 把状态拉回流形（已有代码，只需改调用位置）。

### 收益与风险

- **预期**: style +0.03~0.05, LPIPS 不变或略好
- **难度**: 中高（需要理解流形投影的数学）
- **风险**: 如果流形投影太强→风格被压制；太弱→无效果

---

## 革命方向 B: Gromov-Wasserstein 风格匹配 (拓扑同构传输)

### 理论

当前的最优传输（OT）匹配是通过 SWD/Sinkhorn 在潜空间做**逐点匹配**：
$$C_{ij} = \|z_i^{\text{content}} - z_j^{\text{style}}\|_2^2$$

这个成本矩阵只比较了单个潜变量的值，没有比较**内部结构**。
如果内容图的一棵树在潜空间由一组特定空间关系的向量编码，
那么 OT 应该保证树在目标风格的潜变量中仍然是一棵树——即保持内部的 pairwise distance 结构。

**Gromov-Wasserstein 距离**不比较点与点，而是比较"距离的距离"：
$$C_{ijkl}^{\text{GW}} = \left| d_{\text{content}}(z_i, z_j) - d_{\text{style}}(z_k', z_l') \right|^2$$

在最优传输计划 $\Pi$ 下：
$$\text{GW}(\mu, \nu) = \min_{\Pi} \sum_{i,j,k,l} C_{ijkl}^{\text{GW}} \Pi_{ik} \Pi_{jl}$$

### 为什么这能解决 style-structure tradeoff

GW-OT 的内在性质是：如果找到一个使 GW 距离最小的传输计划，
那么该计划**自然地保持度量空间的拓扑结构**。
不需要额外加 kinetic loss、anisotropic penalty 或 topogate——GW 匹配本身就是结构保持的。

**当前状态**: 代码中已有 `coupling_solver="sinkhorn"` 和 `coupling_feature_mode="lowfreq_edge"`。
这些都在用欧氏空间的 Sinkhorn。GW-OT 需要一个不同的 solver。

### 实现路径

用 `geomloss` 库（已安装在远程 WSL？需确认）或者手写轻量版 GW solver：

```python
# 在 losses.py 的 _solve_group_coupling 中添加 GW 模式
if self.coupling_solver == "gromov_wasserstein":
    # 计算 content 内部距离矩阵
    D_content = pairwise_euclidean(content_group.flatten(1), content_group.flatten(1))
    # 计算 target 内部距离矩阵
    D_target = pairwise_euclidean(target_group.flatten(1), target_group.flatten(1))
    # 用 Sinkhorn-like 迭代求解 GW plan
    plan = gromov_wasserstein_sinkhorn(D_content, D_target, epsilon=self.sinkhorn_epsilon)
    matched = plan @ target_group
```

如果 `geomloss` 不可用，可以用 **Fused Gromov-Wasserstein** 的简化版：
$$C_{\text{fused}} = \alpha \cdot C_{\text{Euclidean}} + (1-\alpha) \cdot C_{\text{GW}}$$

即把欧氏距离和 GW 距离按比例混合——$\alpha=0.7$ 时既有点对点的语义匹配，又有结构保持。

### 收益与风险

- **预期**: 结构保持天然优于当前 OT，可能不再需要 topogate 或只需要弱 topogate
- **难度**: 高（GW solver 实现复杂，计算开销大 $O(n^2 m^2)$）
- **风险**: 显存和时间开销可能难以承受 (batch 从 12 降到 4)

---

## 革命方向 C: 分布条件薛定谔桥 (Distribution-Conditioned SB)

### 理论

当前 I2SB 公式：
$$x_t = (1-t)x_0 + t x_1 + \sigma \sqrt{t(1-t)} \epsilon$$

这条桥连接的是**两个具体的点** $(x_0, x_1)$。但在无配对风格迁移中，我们不知道 $x_1$ 应该是哪个具体的风格图像——
我们只知道 $x_1$ 应该来自目标风格分布 $\nu_{\text{style}}$。

真正的 Schrödinger Bridge 问题（静态形式）是在已知边缘分布 $\mu_0$ (内容) 和 $\mu_1=\nu_{\text{style}}$ (目标风格) 的条件下，
寻找一个满足布朗运动先验的随机过程，使得其边缘分布恰好在 $t=0$ 为 $\mu_0$，在 $t=1$ 为 $\mu_1$。

**我们目前在做的**: 用 OT 采样一个"伪配对" $(x_0, x_1^{\text{OT}})$，然后在这对之间建桥。
这不是真正的 SB——这是"用 OT 近似 SB"。

**真正的 SB** 要求在训练时，模型不仅学会"从 $x_0$ 到 $x_1^{\text{OT}}$"，还学会
"从 $x_0$ 出发，最终落在风格分布中的**任何合理位置**"。
这意味着训练目标应该包含**分布级别的匹配**，而非单点匹配。

### 为什么这能突破

当前瓶颈是每个 content image 只有一个 OT 配对的目标 $x_1^{\text{OT}}$。
如果 OT 配对选得不够"风格化"（比如把一个噪点很多的 Impressionism 图像匹配给了 Minimalism 的内容图），
模型学到的目标就是不正确的。分布条件 SB 允许模型从多个可能的目标中选择最合适的。

### 实现路径

**简化版（可立即实验）**: 对每个 content image，采样 k 个目标候选（来自同 style 的 k 张不同图像），
计算 k 个 loss，取 min：
```python
# 在 losses.py 的 compute 中:
targets = [matched_target_1, matched_target_2, matched_target_3]  # 3 个不同的 OT 结果
losses = [compute_bridge_loss(content, tgt) for tgt in targets]
loss = min(losses)  # 或者 softmin
```

这等价于说"模型可以选择最容易风格化的那个目标"——降低了"坏配对"的惩罚。

**完整版**: 实现真正的 Neural SB 训练循环：
1. Forward: 从 content 出发，用当前模型积分到 t=1，得到生成的"伪目标"
2. 计算生成的分布与真实目标风格的分布之间的 SWD
3. Backward: 梯度同时更新模型参数
4. 这需要**多步积分在训练循环中**——计算开销极大

### 收益与风险

- **简化版预期**: style +0.01~0.03，实现成本低
- **完整版预期**: 理论上的最优解，但计算开销可能增长 5-10 倍
- **风险**: 简化版的 min-loss 可能导致 mode collapse（模型只学一种风格的"捷径"）

---

## 革命方向 D: 潜空间风格纤维丛 (Style Foliation / Fiber Bundle)

### 理论

考虑潜空间 $\mathcal{Z} \subset \mathbb{R}^{4 \times 64 \times 64}$。
定义等价关系：$z \sim z'$ 当且仅当它们对应"相同内容但不同风格"的图像。
每个等价类是一条"风格轨道"（style orbit）。

**风格迁移的几何本质**: 给定 $z_0$（内容），找到它所在轨道的方向，
沿着该方向移动到目标风格在轨道上的交点。

在纤维丛的语言中：
- **底空间 (Base)**: 内容结构的不变特征（边缘、形状、空间布局）
- **纤维 (Fiber)**: 给定内容结构下的所有可能风格外观
- **联络 (Connection)**: 一个向量场，告诉你"在不改变内容结构的情况下，如何改变风格"

当前 topogate 的做法是在 Attention 层锁定了底空间的方向（$A_{\text{self-content}}$），
允许纤维方向自由变化（$V_{\text{style}}$）。这实际上是一个**离散的、手动的联络**。

**改进方向**: 让网络**学习这个联络**，而不是用 topogate 强制锁定。

### 实现路径

```python
# 在 model.py 中新增 StyleConnection 模块
class StyleConnection(nn.Module):
    """Learns a content-dependent but style-directional vector field."""
    def __init__(self):
        # 用于提取"内容不变特征"的网络
        self.content_invariant = nn.Sequential(...)
        # 用于生成"风格方向"的网络
        self.style_direction = nn.Sequential(...)

    def forward(self, z, content_feat, style_code):
        # 提取内容不变特征（底空间）
        base_feat = self.content_invariant(content_feat)
        # 生成风格方向（纤维上的向量）
        style_vec = self.style_direction(torch.cat([base_feat, style_code], dim=1))
        # 强制 style_vec 与 base_feat 正交（在纤维上移动，不在底空间移动）
        style_vec = style_vec - project_onto(style_vec, base_feat)
        return style_vec
```

训练时：底空间特征通过 reconstruction loss 约束（z_0 → z_1 → back to z_0 必须一致）。
风格方向通过 SWD 约束（沿风格方向的终点必须匹配目标风格分布）。

### 为什么这能突破

这个方向从根本上**重新定义了问题**。不再把风格迁移视为"两个分布间的传输"，
而是视为"在内容不变流形上的风格方向移动"。结构保持不是通过 loss 或 topogate 强制实现的，
而是**网络架构的几何性质**保证的——因为风格方向被显式约束为与内容方向正交。

### 收益与风险

- **预期**: 结构保持理论上完美，style 可自由调节
- **难度**: 极高（需要重新设计整个 transport 模块）
- **风险**: 要在当前 codebase 中实现这个，等于重写半个 model.py。但论文价值极大。

---

## 革命方向 E: 随机插值器的自适应噪声调度 (Learned Stochastic Interpolants)

### 理论

I2SB 使用固定的布朗桥：$\sigma_t = \sigma \sqrt{t(1-t)}$。但"最佳噪声水平"可能是**内容相关、时间相关、空间相关**的。

考虑更一般的随机插值器：
$$dx = v_\theta(x, t) dt + g_\phi(x, t) dW$$

其中 $g_\phi$ 是**可学习的扩散系数**，由一个轻量网络参数化。

**核心洞察**: 
- 在图像的平坦区域（天空、水面），确定性传输就够了
- 在纹理丰富区域（树叶、草丛、笔触），需要更多随机性来打破模式坍缩
- 在边缘区域（建筑轮廓），需要零随机性来保证结构锐利

当前 topogate 已经锁定了结构（解决了边缘区域的问题）。现在需要的是**在纹理区域注入更强的随机性**。

### 实现路径

用一个轻量的"噪声门控网络"，输入当前状态 $x_t$ 的内容特征和 tokenizer 的 spatial map，
输出一个空间可变的噪声尺度：

```python
# g_phi: [B, 1, H, W] -> [B, 1, H, W] 噪声门控
noise_gate = self.noise_scheduler(x_t, spatial_map, t)
# 在纹理区域门控接近 1（加噪声），在边缘区域门控接近 0
effective_sigma = self.bridge_sigma * noise_gate * math.sqrt(t * (1-t))
```

训练：$g_\phi$ 的损失是"加噪声后对 LPIPS 的影响"和"加噪声后对 style 的提升"之间的 tradeoff。
可以用 RL 风格的 reward 或者直接端到端训练。

### 收益与风险

- **预期**: 在 topogate 保护下，专注在纹理区域注入噪声，style 提升可能很大
- **难度**: 中
- **风险**: 学习 $g_\phi$ 需要小心平衡——太激进会破坏结构

---

## 革命方向总览

| # | 方向 | 理论来源 | 解决的核心问题 | 难度 | 论文价值 |
|---|------|----------|----------------|------|:---:|
| A | 黎曼潜流匹配 | Riemannian Geometry / RFM | 中间状态 off-manifold | 中高 | ★★★★ |
| B | GW 风格匹配 | Gromov-Wasserstein OT | 结构保持的数学保证 | 高 | ★★★★★ |
| C | 分布条件 SB | Schrödinger Bridge 原义 | 配对质量/分布匹配 | 中 | ★★★★ |
| D | 风格纤维丛 | Fiber Bundle / Connection | 结构-风格解耦的几何保证 | 极高 | ★★★★★ |
| E | 自适应噪声调度 | Learned Stochastic Interpolants | 区域自适应的随机性 | 中 | ★★★ |

### 给 KiritoFD 的建议

**如果追求快速出结果**: 方向 E (自适应噪声) + 方向 C 简化版 (多目标 min-loss) 可以在 1-2 天内实现。

**如果追求顶级论文**: 方向 B (GW-OT) 或方向 D (纤维丛)。方向 B 在学术界有明确的相关工作可以对比 (GW-OT for image matching)；方向 D 则是全新的几何框架，如果做出来就是开山之作。

**最推荐**: 方向 A (黎曼投影) 作为"增量改进"——在现有 transport_step 中加一个 Riemannian projection 层，不改架构，风险最小，但概念上有足够深度支撑论文的 Method 章节。
