# FC-SB Round 2 实验规划总结

> **实验日期**: 2026-06-25  
> **负责人**: Codex  
> **实验目标**: 验证纤维约束薛定谔桥（Fiber-Constrained Schrödinger Bridge, FC-SB）理论，基于修复后的bug重新实验，突破Pareto死结。

---

## 一、背景与问题重述

### 1.1 FC-SB 核心理论

FC-SB 的核心洞见来自 **纤维丛几何分解**：

将 latent 空间分解为两个正交子空间：
- **底流形 Base (B)**: 低频结构信息（轮廓、构图、色彩基调）
- **纤维空间 Fiber (F)**: 高频风格纹理信息（笔触、纹理、对比度）

FC-SB 通过**解析几何强制**实现空间解耦：
1. **底流形绝对静止**：`db = 0 + 0·dWt` —— 结构永久锁定，LPIPS 从可优化目标变为物理约束
2. **纤维空间自由扩散**：`df = vθ dt + σ dWt` —— 允许全功率布朗噪声激发锐利笔触

通过这种"刚性锁结构 + 随机激纹理"策略，理论上可以同时实现：
- `LPIPS < 0.30`（底流形锁定）
- `clip_style > 0.73`（纤维布朗噪声激发风格强度）

### 1.2 第一轮实验失败根因

在 FC-SB Phase 1 实验中，所有新机制都显示性能远低于 E2 基线（`clip_style ~0.611`, `LPIPS ~0.695`）。

**根因诊断**:
1. **配置读取 Bug** ([model620.py:535-553](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L535-L553)):
   - 原代码只从 `self.bridge_cfg` 读取所有 FC-SB 参数
   - 但 `i2sb_fiber_project_endpoint`、`i2sb_fiber_project_noise` 等字段定义在 `ModelConfig`，而非 `BridgeConfig`
   - **后果**: FC-SB 所有机制实际上从未被启用！实验运行的仍是基线配置

2. **基线配置冲突**:
   - 第一轮实验使用 `f1_repro_e2` 作为基线
   - 该基线保留了很多与 620 消融审计推荐配置冲突的设置：
     - `style_attn_mode: softmax` vs 推荐 `gated`
     - `style_cross_attn_gate_init: 0.5` vs 推荐 `0.3`
     - `style_film_enabled: true` vs 推荐 `false`
     - `single_step_edge_weight: 0.1` vs 推荐 `0.0`
     - `swd_noise_sigma: 0.0` vs 推荐 `0.02`

3. **训练不足**:
   - 第一轮实验只跑了 1 epoch
   - 但 FC-SB 需要完整 3 epoch 让模型适应噪声分布

---

## 二、Round 2 实验设计

### 2.1 基线选择

本轮实验以 **620 消融审计推荐配置** 为基线：
- 通过了全部三门验收：`WFI=0.3757` (<0.40), `CLIP-S=0.6995` (≥0.695), `LPIPS=0.3422` (<0.36)
- 配置已被系统验证稳定
- 作为干净起点，增量添加 FC-SB 机制，可以精确测量每个机制的独立贡献

### 2.2 实验矩阵 (增量式验证)

| 实验组 | 描述 | i2sb_fiber_project_endpoint | i2sb_fiber_project_noise | fiber_only_endpoint | bridge_path_mode | bridge_sigma | 预期效果 |
|---|---|---|---|---|---|---|---|
| **G0** | 基线复现 | ❌ | ❌ | ❌ vertical | 0.02 | 确认推荐配置在远程环境可复现 |
| **G1** | +Fiber Velocity Projection | ✅ | ❌ | ❌ | vertical | 0.02 | 剥离速度场低频，只让模型优化高频方向 |
| **G2** | +Base Locking | ✅ | ❌ | ❌ | ✅ vertical | 0.02 | 推理时强制低频 = content 低频，锁死 LPIPS |
| **G3** | +Fiber SDE Noise | ✅ | ✅ | ❌ | ✅ vertical | 0.04 | 注入高频布朗噪声，打破均值坍缩 |
| **G4** | Full FC-SB | ✅ | ✅ | ✅ | ✅ vertical | 0.06 | 所有机制全开，让模型只预测 fiber |
| **G5** | Full FC-SB (FC.md magic) | ✅ | ✅ | ✅ | ✅ vertical | **0.08** | FC.md 推荐的"魔法阈值" |
| **G6** | Full FC-SB + Curriculum | ✅ | ✅ | ✅ | ✅ vertical | 0.06 (curriculum) | 三阶段课程式 sigma 调度 |

**实验设计逻辑**:
- 从基线开始，每一步只添加一个新机制
- 这样可以精确测量每个机制的边际贡献
- 如果最终结果不好，可以回退找到哪一步出问题

### 2.3 远程配置适配

所有配置已适配远程 WSL 环境：
- 路径：`f:/` → `/mnt/i/`
- 训练参数：`batch_size=24` (12GB VRAM 安全)
- `num_workers=0`, `pin_memory=False`, `persistent_workers=False` 防止 OOM
- `num_epochs=3`, `full_eval_defer_until_training_end=True` — 训练完再统一评估
- 符合项目硬约束：`virtual_length_multiplier=1.0`

---

## 三、关键代码修复与实现

### 3.1 配置读取修复 ([model620.py:535-543](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L535-L543))

```python
# === 读取 FC-SB 配置 ===
mcfg = getattr(self, 'model_cfg', None)
bcfg = getattr(self, 'bridge_cfg', None)
def _cfg_get(key, default):
    if mcfg is not None and hasattr(mcfg, key):
        return getattr(mcfg, key)
    if bcfg is not None and hasattr(bcfg, key):
        return getattr(bcfg, key)
    return default
fiber_proj_ep = bool(_cfg_get('i2sb_fiber_project_endpoint', False))
fiber_proj_noise = bool(_cfg_get('i2sb_fiber_project_noise', False))
```

**修复效果**: 现在可以正确读取 `model` 区段和 `bridge` 区段的 FC-SB 参数。

### 3.2 完整推理循环实现 ([model620.py:555-626](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L555-L626))

```python
# 🚨 灵魂锚点: 保存初始 content 的 Base（永不改变！）
x_base_lock = lp(x)
for idx in range(steps):
    # Step 1: 模型预测 Endpoint
    endpoint = self.predict_endpoint(...)

    # Step 1.5: Fiber-Only Endpoint Projection
    if fiber_only_ep:
        ep_fiber = endpoint - lp(endpoint)
        x_base_now = lp(h)
        endpoint = x_base_now + ep_fiber

    # Step 2: 计算速度场并剥离低频 (Fiber Velocity Projection)
    v_pred = (endpoint - h) / (1 - t_curr)
    if fiber_proj_ep:
        v_fiber = v_pred - lp(v_pred)  # 只保留高频速度分量
    else:
        v_fiber = v_pred

    # Step 3: Euler 步进（确定性漂移，仅 Fiber 分量）
    h = h + v_fiber * dt

    # Step 4: 生成高频布朗噪声 (Fiber Noise Injection)
    if sigma_base > 0.0:
        # Curriculum sigma schedule 支持三阶段课程
        if sigma_schedule == 'curriculum':
            if t_curr < 0.33:
                sigma_eff = sigma_base * 0.25   # 锚定期: 极低噪声
            elif t_curr < 0.66:
                sigma_eff = sigma_base * 0.6    # 解耦期: 中等噪声
            else:
                sigma_eff = sigma_base * 1.0    # 引爆期: 全功率
        # ...
        noise = torch.randn_like(h)
        if fiber_proj_noise:
            noise_fiber = noise - lp(noise)  # 只保留高频噪声
        h = h + sigma_t * noise_fiber

    # Step 5: 🚨🚨🚨 绝对刚性保护 (BASE LOCKING) 🚨🚨🚨
    if bridge_path_mode == "vertical":
        h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
```

### 3.3 训练期支持 ([losses620.py:383-390](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L383-L390))

```python
# === FC-SB: 训练期高通 SDE 噪声注入 ===
if self.bridge_sigma > 0 and self.training:
    sde_noise = torch.randn_like(target_velocity)
    sde_noise_hp = sde_noise - _lowpass(sde_noise, self.lowpass_kernel)
    target_velocity = target_velocity + self.bridge_sigma * sde_noise_hp
```

### 3.4 课程 sigma 调度支持 ([losses620.py:update_weights_for_epoch](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L448-L470))

```python
def update_weights_for_epoch(self, epoch: int, num_epochs: int = 3) -> dict[str, float]:
    if self.bridge_sigma_schedule == "curriculum":
        if epoch <= max(1, num_epochs // 3):
            self.bridge_sigma = self._base_bridge_sigma * 0.25
        elif epoch <= max(1, 2 * num_epochs // 3):
            self.bridge_sigma = self._base_bridge_sigma * 0.6
        else:
            self.bridge_sigma = self._base_bridge_sigma * 1.0
    # ...
```

---

## 四、配置文件生成

生成脚本: [exp/fc_sb_r2/gen_configs.py](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/fc_sb_r2/gen_configs.py)

生成结果:
```
=== FC-SB Round 2: 基于620消融审计推荐配置的增量实验 ===

G0: Baseline reproduction
  Saved: exp/fc_sb_r2\g0_baseline\config.json

G1: +Fiber Velocity Projection
  Saved: exp/fc_sb_r2\g1_fiber_vel_proj\config.json

G2: +Base Locking
  Saved: exp/fc_sb_r2\g2_base_locking\config.json

G3: +Fiber SDE Noise (sigma=0.04)
  Saved: exp/fc_sb_r2\g3_fiber_sde\config.json

G4: Full FC-SB (sigma=0.06)
  Saved: exp/fc_sb_r2\g4_full_fcsb\config.json

G5: Full FC-SB (sigma=0.08, FC.md magic)
  Saved: exp/fc_sb_r2\g5_sigma08\config.json

G6: Full FC-SB + Curriculum sigma
  Saved: exp/fc_sb_r2\g6_curriculum\config.json

Done! All 7 configs generated under exp/fc_sb_r2/
```

训练脚本: [exp/fc_sb_r2/run_all.sh](file:///g:/GitHub/Latent_Style/SchrodingerBridge/exp/fc_sb_r2/run_all.sh)

---

## 五、理论预期结果

基于 FC-SB 纤维几何分解理论，预期：

| 实验组 | 预期 LPIPS ↓ | 预期 clip_style ↑ | 说明 |
|---|---|---|---|
| G0 | ~0.34 | ~0.70 | 基线复现 |
| G1 | ~0.33 | ~0.705 | 剥离低频方向，轻微改善 |
| G2 | **~0.31** | ~0.705 | Base Locking 锁死低频，LPIPS 下降 |
| G3 | ~0.31 | ~0.715 | 高频噪声打破均值，style 提升 |
| G4 | ~0.30 | ~0.72 | 全机制，模型专注 fiber，style 进一步提升 |
| G5 | ~0.30 | ~0.73 | FC.md 魔法阈值 sigma=0.08，理论冲击目标 |
| G6 | ~0.30 | ~0.725 | 课程调度更稳定收敛 |

**终极目标**:
- 找到一组配置实现 **LPIPS < 0.30 且 clip_style > 0.73**
- 这将彻底打破当前存在了6个月的 Pareto 死结

---

## 六、运行说明

### 在远程 WSL 运行:

```bash
cd /mnt/g/GitHub/Latent_Style/SchrodingerBridge
bash exp/fc_sb_r2/run_all.sh
```

只运行特定实验:

```bash
bash exp/fc_sb_r2/run_all.sh g4 g5 g6
```

### 结果位置:

- 配置: `exp/fc_sb_r2/<exp>/config.json`
- 训练日志: `exp/fc_sb_r2/<exp>/train.log`
- 评估日志: `exp/fc_sb_r2/<exp>/eval.log`
- 结果汇总: `exp/fc_sb_r2/<exp>/full_eval/epoch_0003/summary.json`

---

## 七、待验证核心假说

1. **Base Locking 假说**: 通过解析几何强制 `lowpass(output) = lowpass(content)`，可以将 LPIPS 永久锁死在 < 0.30，而不需要模型学习。

2. **Fiber SDE 假说**: 将布朗噪声限制在纤维空间，在不破坏结构的前提下，可以激发更强的风格响应，提升 clip_style。

3. **Fiber-Only Endpoint 假说**: 让模型只预测纤维增量 Δf，所有容量都用来拟合高频纹理，而不需要浪费容量维护结构，可以提升风格表达能力。

4. **sigma 剂量效应**: sigma 越大，风格激发越强，但过大的 sigma 会导致数值不稳定和 LPIPS 上升。存在一个"魔法阈值"在 0.04-0.08 之间。

5. **课程调度假说**: 从低 sigma 逐步升到高 sigma，可以帮助模型更稳定收敛，避免梯度爆炸。

---

## 八、预期指标贡献

如果实验验证成功，将为论文贡献：

1. 清晰的消融实验，证明每个 FC-SB 机制的边际贡献
2. 定量数据验证纤维几何分解理论的有效性
3. 一组突破性的指标，突破当前 Pareto 前沿
4. 可复现的配置，作为后续工作的新基线

---

## 九、总结

Round 2 修正了第一轮实验的配置读取 bug，采用正确的 620 消融审计推荐配置作为基线，设计了增量式验证矩阵，系统验证 FC-SB 理论的五个核心假说。

本轮实验是对 FC-SB 理论的**公平验证**。如果成功，将打破风格-内容 Pareto 死结；如果失败，将证伪纤维分解假说，指明新的研究方向。

---

**下一阶段**: 实验运行完成后，拉取结果更新 dashboard，分析各机制贡献，选择最优配置进入全量训练。
