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
