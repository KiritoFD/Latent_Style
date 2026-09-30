# �?KiritoFD: 突破 style 天花板的具体行动

> 当前状�? topogate_appalign e3 收敛 (style 0.672/0.703, LPIPS 0.315/0.313)
> **结构防线已坚�?(LPIPS 0.31)，唯一瓶颈�?style 推不�?0.72+**
> 以下三条路径，按优先级排列�?
## 对用户三个提案的评估

### 提案1: SDE / PC Solver 推理 �?�?完全合理，立即执�?- 代码已就�?(`solver_pc` �?`latent_lowpass` 校正, `solver_unsb_cycle` �?SDE-EM 噪声)
- 零风�? 不重新训练，只改 eval config
- 已有 topogate 锁结�?+ PC 校正双保险，LPIPS 大概率不会超�?0.38
- **立即执行，不需要等任何东西**

### 提案2: I2SB Endpoint + TopoGate 训练 �?⚠️ 方向对，细节需修正
- "Endpoint 不会崩因�?topogate 锁空�? �?这个判断有道理但未验�?- "I2SB公式 c1*xt 锚定源图" �?正确，这�?I2SB 的核心保�?- **但不要用 σ=0.5�?* topogate 保护下用 σ=0.02-0.05 就够�?- 队列中已�?`i2sb_sigma0p02_residual_tfloor005`，直接启动即�?
### 提案3: Tokenizer 增强 �?⚠️ 方向对但大半已实�?- "�?Positional Encoding" �?**已经有了** (`pe_temperature=1.0` 默认开�?sinusodial PE)
- "扩大 spatial_dim 128�?56" �?可行，但需重新训练，优先级放最�?- "32 clusters 扩大" �?当前已用32，在配置中可�?- 这些是边际改进，�?SDE/PC 路线验证完再考虑

## 执行计划 (按优先级)

---

## 路径 A: PC Solver 推理 (推荐首�?

**原理**: Predictor 正常�?ODE 风格化，Corrector �?latent 低频 MSE 把宏观结构拉回源图。风格笔触（高频）完全不受影响�?
**配置**: 已就�?`configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json`

**参数扫描建议** (创建 3 �?override config):
```
step_size=0.04  (轻校�? style可能保持)
step_size=0.06  (中校�? 推荐起点)
step_size=0.10  (重校�? 可能LPIPS更好但style被压�?
```

**执行命令** (在远�?WSL):
```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
# 轻校�?python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.04
# 中校�?python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.06
# 重校�?python src/run.py --config configs/aaai2027/phase2_eval_pc_lowpass_topogate_e2.json \
  --override model.solver_corrector_step_size=0.10
```

**预期**: LPIPS 微升�?0.33-0.35，style 推到 0.69-0.71。如�?style 没有提升，说明低�?MSE 不贡�?style——需要走 B 路线�?
**代码已就�?*: `model.py:591-623` (`_correct_transport_state`), `model.py:1100-1103` (`solver_pc`)

---

## 路径 B: SDE-EM 推理 (直接注入随机�?

**原理**: �?solver_unsb_cycle，每一步先�?ODE + 内容校正，再注入微量布朗噪声�?$$x_{t+1} = x_t + v_\theta(x_t, t)\Delta t + \sigma\sqrt{\Delta t}\epsilon$$

噪声能打破确定�?style 轨迹�?mode collapse，逼出更多目标笔触�?
**配置**: 已就�?`configs/aaai2027/phase2_eval_sde_em_topogate_e2.json`

**参数扫描建议**:
```
noise_scale=0.005  (极轻噪声)
noise_scale=0.010  (轻噪�? 推荐起点)  
noise_scale=0.020  (中噪�?
noise_scale=0.030  (重噪�? 可能 LPIPS 开始崩)
```

**执行命令**:
```bash
cd /mnt/i/Github/Latent_Style/SchrodingerBridge
python src/run.py --config configs/aaai2027/phase2_eval_sde_em_topogate_e2.json \
  --override model.solver_stochastic_noise_scale=0.015
```

**预期**: style 明显突破 (0.70-0.72)，LPIPS 微升�?0.33-0.37。noise_scale 是关键超参——太�?LPIPS 崩，太小没有 style 提升�?
**代码已就�?*: `model.py:1104-1114` (`solver_unsb_cycle`), noise_scale �?model config �?

---

## 路径 C: I2SB Endpoint + TopoGate 训练 (如果 A/B 都不�?

**前提**: 仅当路径 A �?B 都无法把 style 推到 0.72 时才走这条路�?
**原理**: 回归 endpoint 模式，但�?topogate 锁结�?+ 极小 σ �?I2SB 训练�?之前�?endpoint 崩是因为 SemanticCrossAttn 满天�?+ σ 太大(0.5)�?现在 topogate 锁死空间路由，I2SB 公式 $\mu = c_1 x_t + c_2 \hat{x}_1$ 第一项强烈锚定源图�?
**需要做的事**:
1. 创建训练 config: `transport=endpoint, objective=i2sb_endpoint, bridge_sigma=0.02, tokenizer_family=pure_latent_spatial, solver_family=solver_i2sb, semantic_self_topology_gate=true`
2. �?topogate e1 ckpt �?warmstart
3. 训练 8-12 epochs，观�?style 是否突破 0.72

**⚠️ 注意**: 
- 不要�?σ=0.5！用 σ=0.02-0.05。topogate 已经解决了结构问题，只需要微布朗噪声突破 style
- 观察 LPIPS 是否会突�?0.40。如果会→立即降�?σ
- 已经在队列中�?`i2sb_tok32_safe_semantic_topogate_sigma0p02_residual_tfloor005` 可以直接启动

**代码已就�?*: `model.py:516-542` (`_i2sb_transport_step`), `model.py:886-894` (solver 选择)

---

## 关于 Tokenizer 增强

当前 tokenizer **已经具备**:
- PE (pe_temperature=1.0 默认开启的 sinusodial positional encoding)
- 32 clusters
- 4+ ResBlock query_extractor
- global_code = GAP(spatial_map) + gate + embedding

如果需要可以直接在 config 中调�?
- `query_dim`: 64 �?96 (需要重新训�?
- `num_clusters`: 32 �?48 (需要重新训�?
- `pe_temperature`: 1.0 �?0.75 (�?eval 时改)
- `spatial_dim`: 128 �?256 (需要重新训练，显著增加参数�?

但这些需要重新训练，优先级放�?SDE/PC 推理之后�?
---

## 推荐执行顺序

```
GPU 空闲�?
  1. 先跑路径A (PC solver eval, 3 �?step_size) �?最�? 零风�?  2. 同时跑路径B (SDE-EM eval, 2 �?noise_scale) �?并行
  3. �?A/B 结果:
     - style > 0.72? �?成功�? 提交论文
     - style 0.70-0.72 �?LPIPS < 0.35? �?组合 A+B 试试
     - style < 0.70? �?路径C (I2SB 训练)
  4. 路径C 如果也失�?�?考虑减弱 topogate blend 或换�?PnP self-inject
```

## 预期时间

| 路径 | 操作 | 时间 |
|------|------|------|
| A | PC eval (3 �?step_size) | ~12min (3×4min eval) |
| B | SDE-EM eval (2 �?noise_scale) | ~8min (2×4min eval) |
| C | I2SB 训练 | ~20min/epoch × 8 epoch = ~3h |

---

# 第二部分: 训练与模型设计的探索方向

> 以下 8 个方向覆盖训练策略和模型架构两个层面�?> 每个方向标注了难度、预期收益和风险�?> 建议按推荐顺序执行——前面的失败再试后面的�?
---

## 方向 1: 自适应 Kinetic 调度 (训练策略)

**难度**: �?| **需重训**: �?| **预期收益**: style +0.02~0.04 | **风险**: �?
**假设**: 当前固定 `w_kinetic=0.85-0.95` 从头用到尾，�?style 的压制是均匀的。但模型在早期需�?kinetic 来学结构，后期结构已学会，kinetic 就成�?style 的枷锁�?
**方案**: �?bridge config 中实�?kinetic 衰减调度�?- epoch 1-3: `w_kinetic=1.0` (学结�?
- epoch 4-6: `w_kinetic=0.7`
- epoch 7-12: `w_kinetic=0.4`
- epoch 13+: `w_kinetic=0.2` (释放 style)

**配置**: 当前代码可能不支�?epoch 级调度。需要新�?`kinetic_warmup_epochs` + `kinetic_decay_end` 参数�?也可以在 `trainer.py` 中硬编码一个线性衰减的 lambda�?
**预期**: style �?0.70 推到 0.71-0.72，LPIPS 可能在后期微升到 0.33-0.35�?
**风险**: 如果后期 kinetic 过低导致 LPIPS 崩溃→回退到上一�?kinetic 级别�?
---

## 方向 2: 渐进�?Topogate 解锁 (训练策略)

---

## ������۷��������

��ά������(����): [FIBER_BUNDLE_DESIGN.md](FIBER_BUNDLE_DESIGN.md)
ȫ����̽��: �ѹ鵵����Ҫʱ�� git log �ָ�
