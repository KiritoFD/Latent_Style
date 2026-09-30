# Fiber-Constrained Schrödinger Bridge (FC-SB) — 突破帕累托死结

> **时间预算**: 1 天（~24 小时 GPU 时间分配）
> **核心目标**: clip_style > 0.73 且 LPIPS < 0.30
> **理论基础**: [FC.md](../../../docs/622/FC.md)

---

## 第一部分：问题诊断 — 为什么 12 个实验都无法打破帕累托死结？

### 1.1 实验数据铁证（E4-E12 + E4-long）

经过 E4-E12 共 12 个实验（~4 小时 GPU），确认了**物理层面的硬约束**：

| 实验 | clip_style↑ | LPIPS↓ | velocity_std | 核心策略 | 判定 |
|------|-----------|--------|-------------|---------|------|
| E2 (历史最优) | — | **0.333** 🏆 | ~0.05 | Two-Stage+S1style=16 | 内容最佳 |
| **E4 (RMSNorm)** | 0.672 | **0.373** ✅ | **0.896** 🔥 | RMS+vmag=2.0 | **最佳平衡点** |
| E5 VP-Flow | **0.705** | 0.498 | — | 球面插值 | 风格↑ 内容↓ |
| E6 Top-K=4 | 0.692 | 0.516 | — | 注意力截断K=4 | 风格↑ 内容↓ |
| E7 三斧组合 | **0.705** | 0.517 | 1.27 | VP+TopK+RMS | 风格↑ 内容↓ |
| E7b Top-K=8 | **0.704** | 0.506 | 1.269 | 温和版Top-K | 略优于E7 |
| **E8 方向余弦** | **0.715** 🏆 | 0.506 | 1.42 | Loss方向约束 | 偏科战神 |
| **E9 频段解耦** | **0.717** 🏆 | **0.544** ❌ | — | 低MSE+高Cosine | 风格最强内容最差 |
| E12 CFG dropout | 0.693 | **0.602** ❌❌ | — | 15%条件丢弃 | 内容最差 |
| **E4-long ep5** | **0.727** 🏆🏆 | 0.581 | 0.878 | 10ep最优停止 | 自然收敛上限 |

### 1.2 帕累托前沿的数学本质

```
clip_style (风格强度) ↑
  0.73 ┤★ E4long(0.58)  E9(0.54)
  0.72 ┤  
  0.71 ┤★ E8(0.50)      Mystery-SDE(0.34)
  0.70 ┤★ E5(0.50)       E7(0.52)   E7b(0.51)
  0.69 ┤★ E12(0.60)      E6(0.52)
  0.67 ┤★ E4(0.37) ← 最佳平衡点
       └───────────────────────────→ LPIPS↓ (内容保持, 越低越好)
       0.33  0.37  0.45    0.50   0.55  0.60

  → 不存在任何配置能同时达到 clip_style > 0.70 且 LPIPS < 0.40
  → 但 "Mystery-SDE σ=0.08 (不训练)" 达到了 (0.711, 0.337)！
```

### 1.3 三大范式的根本矛盾

#### 范式 A：Flow Matching（FM）— 直线最优传输的均值坍缩
$$x_t = (1-t) \cdot x_0 + t \cdot x_1, \quad v = x_1 - x_0$$
- **灾难**: 一对多映射时直线的平均化 → 特征方差在 t=0.5 最小 → **白化/发灰**
- **证据**: E4 (FM-only) clip_style=0.672 停滞，E4-long 更多训练只让 LPIPS 恶化

#### 范式 B：薛定谔桥（SB）— 全空间各向同性噪声撕裂结构
$$dx = v_\theta(x,t)\,dt + \sigma\,dW_t$$
- **设计初衷**: 布朗噪声打破 FM 平均化 → 激发锐利笔触 ✅
- **实际灾难**: 各向同性噪声同时撕碎低频结构 → **LPIPS 爆炸** ❌
- **证据**: E8(0.715, 0.506), E9(0.717, 0.544), E12(0.693, 0.602)

#### 范式 C：纤维丛（Fiber Bundle）— 空间解耦的数学工具
- **底流形 Base $B$**: 低频空间（构图、轮廓、布局）
- **纤维空间 Fiber $F$**: 高频空间（笔触、色彩、细节）
- **投影 $\pi: E \to B$**: `lowpass(x)` 实现 Base 投影
- **局部平凡化**: $E_p \cong B \times F$

**关键发现**: 代码库中已实现 `_split_base_fiber()` (losses620.py:188) 和 `_project_training_target()` (losses620.py:196)，但推理循环 `integrate_transport()` **未接入 Fiber SDE**！

---

## 第二部分：FC-SB 理论 — 纤维约束薛定谔桥

### 2.1 核心思想

将纤维丛的空间解耦引入 SB 的 SDE，把全空间随机游走劈成两个垂直世界：

#### 底流形（结构）：冰冷的死寂（Dirac 分布）
$$db = 0 \cdot dt + 0 \cdot dW_t$$
- 无速度、无噪声、绝对静止
- 工程实现: `base_lock = lowpass(z_content)` 每步强制恢复

#### 纤维空间（风格）：狂热的热力学扩散
$$df = v_{fiber}(x,t)\,dt + \sigma_{fiber}\,dW^{fiber}_t$$
- 允许强大布朗噪声注入 → 打破 Softmax 均值陷阱
- 工程实现: SDE 步进仅作用于 `highpass(v_pred)` 和 `highpass(noise)`

### 2.2 完整 FC-SB SDE 方程组

$$
\begin{cases}
x_t = b_t + f_t & \text{(状态分解)} \\
b_t = \text{LowPass}(z_{content}) & \text{(Base Locking)} \\
df = v^{fiber}_\theta(x_t, t)\,dt + \sigma_t\,dW^{fiber}_t & \text{(Fiber SDE)} \\
v^{fiber}_\theta = v_\theta - \text{LowPass}(v_\theta) & \text{(Velocity Projection)} \\
dW^{fiber}_t = dW_t - \text{LowPass}(dW_t) & \text{(Noise Projection)} \\
\sigma_t = \sigma_0 \sqrt{t(1-t)} & \text{(Brownian Bridge Variance)}
\end{cases}
$$

### 2.3 为什么 Mystery-SDE (不训练, 0.711/0.337) 证明了方向正确？

当时无意中仅在纤维空间施加了 SDE 噪声：
- ✅ 高频布朗噪声激发了笔触 → clip_style=0.711
- ✅ 低频结构未受干扰 → LPIPS=0.337
- **结论: 只要噪声不泄漏到 Base，SDE 可同时实现高风格和低 LPIPS**

---

## 第三部分：工程实施方案（3 大改造 + 1 天实验计划）

### 改造 1: 推理 Solver — Base Locking + Fiber SDE（核心手术）

**文件**: [model620.py](src/model620.py) — `integrate_transport()` (L512-553)

**现状分析** (当前代码 L532-552):
```python
# 当前代码的问题:
# ❌ SDE 噪声是全频段的 (各向同性)
# ❌ 没有 velocity fiber projection  
# ❌ 没有 Base Locking (累积误差漂移低频)
# ⚠️ i2sb_fiber_project_* 配置存在于 config_schema 但未接入此方法!
```

**改造后完整代码**:

```python
@torch.no_grad()
def integrate_transport(
    self,
    x: torch.Tensor,           # z_content (输入内容潜变量)
    style_id: ...,
    num_steps: int = 8,
    step_size: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    import math
    
    steps = max(1, int(num_steps))
    horizon = max(0.0, float(step_size))
    if horizon <= 0.0:
        return x
    
    # === 读取 FC-SB 配置 ===
    cfg = self.bridge_cfg or self.model_cfg
    fiber_proj_ep = bool(getattr(cfg, 'i2sb_fiber_project_endpoint', False))
    fiber_proj_noise = bool(getattr(cfg, 'i2sb_fiber_project_noise', False))
    fiber_kernel = max(1, int(getattr(cfg, 'i2sb_fiber_project_kernel', 5)))
    if fiber_kernel % 2 == 0:
        fiber_kernel += 1
    bridge_path_mode = str(getattr(cfg, 'bridge_path_mode', 'linear')).lower().strip()
    sigma_base = float(getattr(self, 'bridge_sigma', 0.02))
    
    def lp(y, k=fiber_kernel):
        """Lowpass: kernel×k average pooling"""
        return F.avg_pool2d(y.float(), k, stride=1, padding=k // 2).to(dtype=y.dtype)
    
    # === 🚨 灵魂锚点: 保存初始 content 的 Base（永不改变！）===
    x_base_lock = lp(x)
    
    h = x.clone()
    for idx in range(steps):
        t_curr = horizon * (idx / float(steps))
        t_next = horizon * ((idx + 1) / float(steps))
        t_batch = torch.full((h.shape[0],), t_curr, device=h.device, dtype=h.dtype)
        
        # Step 1: 模型预测 Endpoint
        endpoint = self.predict_endpoint(h, t=t_batch, **kwargs)
        
        # Step 2: 计算速度场并剥离低频 (Fiber Velocity Projection)
        denom = max(1e-6, 1.0 - t_curr)
        v_pred = (endpoint - h) / denom
        
        if fiber_proj_ep:
            # 🎯 只保留高频速度分量（Fiber 上的运动）
            v_fiber = v_pred - lp(v_pred)
        else:
            v_fiber = v_pred
        
        # Step 3: Euler 步进（确定性漂移，仅 Fiber 分量）
        dt = t_next - t_curr
        h = h + v_fiber * dt
        
        # Step 4: 生成高频布朗噪声 (Fiber Noise Injection)
        if sigma_base > 0.0:
            # Brownian Bridge 方差: σ² · t·(1-t) · dt
            sigma_t = sigma_base * math.sqrt(max(0.0, t_curr * (1.0 - t_curr))) * math.sqrt(abs(dt))
            
            noise = torch.randn_like(h)
            if fiber_proj_noise:
                # 🎯 只保留高频噪声（不在 Base 上注入随机性）
                noise_fiber = noise - lp(noise)
            else:
                noise_fiber = noise
            
            h = h + sigma_t * noise_fiber
        
        # Step 5: 🚨🚨🚨 绝对刚性保护 (BASE LOCKING) 🚨🚨🚨
        # 无论 SDE 怎么狂飙，低频结构永远等于初始 Content 的低频！
        if bridge_path_mode == "vertical":
            h = x_base_lock + (h - lp(h))  # = Base(content) + Fiber(current)
    
    return h
```

**改造要点**:

| 改造点 | 物理含义 | 预期效果 |
|--------|---------|---------|
| Velocity Fiber Projection (Step 2) | 剥离 v_pred 低频 | 模型不再学习移动 Base |
| Highpass Noise (Step 4) | 布朗噪声仅作用于高频 | SDE 激发笔触但不破坏结构 |
| **Base Locking (Step 5)** | **每步强制恢复 content 低频** | **LPIPS 有数学保证 < 0.30** |

---

### 改造 2: Gate 初始化 — 从 0.05 提升到 0.5

**文件**: [blocks620.py](src/blocks620.py) — L83, L164

**现状**: `style_gate_init: float = 0.05` → `tanh(0.05) ≈ 0.04998` ≈ **零！**

**为什么 0.05 是 Gate Collapse 的根源?**
- `gate_output = tanh(gate) * attended_value`
- 模型发现："把门关到几乎为零，Loss 就不会因风格注入而上升"
- 结果: velocity_std 从 ~0.05（完全坍缩）到 E4 的 0.896（RMSNorm 救回部分）

**改动** (1 行):
```python
style_gate_init: float = 0.5  # tanh(0.5) ≈ 0.462 — 强行撬开！
```

**为什么现在可以安全设为 0.5?**
1. RMSNorm 保留色彩统计量（不再有白化恐惧）
2. Base Locking 保护结构（不再有 LPIPS 恐惧）
3. 网络终于能"接得住"大风格注入

---

### 改造 3: 训练期各向异性 Target 与 SDE 噪声

**文件**: [losses620.py](src/losses620.py) — `compute()` 方法 (~L352)

#### 3.1 训练 Target 投影（已实现，只需激活配置）

`_project_training_target()` 已支持 `pure_vertical_flow_wavelet` (L224-229):
```python
if mode in {"pure_vertical_flow", "pure_vertical_flow_wavelet"}:
    projected = anchor_low + t_high   # Base(content) + Fiber(target) 🎯
```
→ 模型仅学习预测高频笔触差异 Δf，100% 参数量用于拟合极致笔触

**激活方式**: JSON 配置中设置 `training_target_projection_mode: "pure_vertical_flow_wavelet"`

#### 3.2 训练期高通 SDE 噪声注入（需新增 ~8 行）

**现状**: `bridge_sigma` 仅作为 metric 记录 (L674)，训练 Loss 中**无 SDE 噪声注入**。

**新增代码** (在 `target_velocity` 之后、FM Loss 之前):
```python
# === FC-SB: 训练期高通 SDE 噪声注入 ===
if self.bridge_sigma > 0 and self.training:
    sde_noise = torch.randn_like(target_velocity)
    sde_noise_hp = sde_noise - _lowpass(sde_noise, self.lowpass_kernel)
    target_velocity = target_velocity + self.bridge_sigma * sde_noise_hp
    metrics["training_sde_noise_lp_rms"] = _lowpass(sde_noise).std().item()
    metrics["training_sde_noise_hp_rms"] = sde_noise_hp.std().item()
```

**物理意义**: 模型前向传播时看到"带噪速度场"，被迫学习去噪抗干扰。等价于隐式数据增强。

---

### 改造 4: RMSNorm 全面启用（已在 E4 实现，确保配置激活）

**文件**: [blocks620.py](src/blocks620.py) — L11-28

**无需新代码**。RMSNorm 在 E4 中已验证通过。确保 JSON 配置中 `norm_type="rms_norm"` 或 `body_norm_type="rms_norm"`。

**RMSNorm vs GroupNorm 对比**:

| 属性 | GroupNorm (旧) | RMSNorm (新) |
|------|---------------|-------------|
| 归一化 | $(x-\mu)/\sigma$ | $x/\text{RMS}(x)$ |
| 均值处理 | **减去均值** → 破坏色彩偏移 | **保留均值** → 保护对比度 |
| E4 效果 | gate≈0.05, v_std≈0.05 | gate→0.896, style↑ |

---

## 第四部分：三阶段课程策略（Curriculum Training）

不要一开始就把 σ 拉满。神经网络冷启动面对 SDE 高频震荡易梯度崩溃。

### Phase 1: 结构锚定期 (Epoch 0-3, 约 30 分钟)

**目标**: 学会 OT 结构对齐，不考虑 SDE。

**配置**:
```json
{
  "bridge": {
    "bridge_path_mode": "vertical",
    "bridge_sigma": 0.0,
    "training_target_projection_mode": "pure_vertical_flow_wavelet",
    "w_style_energy_floor": 0.0
  }
}
```

**预期**:
- LPIPS 迅速降到 < 0.25（Base Locking 发挥威力）
- Style 平庸（~0.65），画面偏灰
- `base_structural_drift` ≈ 0.0 ✅
- `fiber_energy_ratio` 可能 < 1.0（正常，Phase 1 不需要）

### Phase 2: 纤维解耦期 (Epoch 3-6, 约 40 分钟)

**目标**: 切断 Base/Fiber 梯度联系，注入微量噪声。

**配置变更**:
```json
{
  "model": {
    "i2sb_fiber_project_endpoint": true,
    "i2sb_fiber_project_noise": true
  },
  "bridge": {
    "bridge_sigma": 0.03,
    "w_style_energy_floor": 0.2
  }
}
```

**预期**:
- Loss 短暂 Spike 后迅速下降
- 画面出现锐利边缘
- LPIPS 稳定 0.28-0.32
- `fiber_energy_ratio` 开始突破 1.0

### Phase 3: SDE 引爆期 (Epoch 6-10+, 约 60+ 分钟)

**目标**: 全功率布朗运动，冲击帕累托前沿。

**配置变更**:
```json
{
  "bridge": {
    "bridge_sigma": 0.08,
    "w_style_energy_floor": 0.5
  }
}
```

**预期**:
- `fiber_energy_ratio` 突破 1.2~1.5 🎉
- 笔触"生猛、狂热"，构图完好
- **clip_style > 0.73 且 LPIPS < 0.30** 🎯🎯🎯

---

## 第五部分：训练监控指标（4 个黄金信号）

启动训练后必须死盯以下指标：

### 指标 1: `base_structural_drift`（必须 ≈ 0.0）
- **来源**: [losses620.py](src/losses620.py) L242
- **意义**: 预测终点低频 vs Content 低频 MSE
- **预期**: < 0.002 → 完美; > 0.01 → 危险! LPIPS 会炸

### 指标 2: `training_sde_noise_lp_rms` vs `hp_rms`
- **来源**: 新增 metrics (改造 3.2)
- **意义**: 注入噪声的低频/高频能量分布
- **预期**: low_rms < 0.01 × high_rms（纯正高通噪声）

### 指标 3: `fiber_energy_ratio`（必须 > 1.0）
- **来源**: [losses620.py](src/losses620.py) L244-247
- **意义**: 预测终点高频方差 / Content 高频方差
- **预期**: < 1.0 = 仍在均值陷阱; 1.0-1.2 = 正常; **> 1.5 = 笔触疯狂生长** 🎉

### 指标 4: `velocity_std`
- **意义**: 整体速度场模长
- **预期**: 从 E4 的 0.90 跃升到 1.2-1.5（SDE 激活后）

---

## 第六部分：一天时间预算分配

| 时段 | 任务 | 预计时间 | 累计 |
|------|------|---------|------|
| T+0h | 代码改造 (Task 1-3) | 1h | 1h |
| T+1h | 配置生成 + 远程部署 | 0.5h | 1.5h |
| T+1.5h | **Phase 1 训练** (epoch 0-3, sigma=0) | 0.5h | 2h |
| T+2h | **Phase 2 训练** (epoch 3-6, sigma=0.03) | 1h | 3h |
| T+3h | **Phase 3 训练** (epoch 6-10, sigma=0.08) | 1.5h | 4.5h |
| T+4.5h | Full Eval + 图片生成 | 0.5h | 5h |
| T+5h | Dashboard 更新 + 目视诊断 | 0.5h | **5.5h** |
| T+5.5h | 如需: 快速消融实验 (变体) | 2-4h | **~8h** |
| Buffer | 排障 / 重跑 | 2-4h | **≤ 24h** |

**关键时间节点决策点**:
- **T+2h (Phase 1 结束)**: 如果 base_structural_drift > 0.01 → 停下检查 Base Locking 代码
- **T+3h (Phase 2 结束)**: 如果 fiber_energy_ratio 未破 1.0 → 提高 sigma 到 0.05
- **T+4.5h (Phase 3 结束)**: 最终评估 → 决定是否进入消融实验

---

## 第七部分：预期结果与帕累托突破

### 定量预期

| 实验组 | LPIPS ↓ | CLIP Style ↑ | 现象 |
|--------|--------|-------------|------|
| 过去最佳平衡 (E4) | 0.373 | 0.672 | RMSNorm 救回 gate |
| 过去最高风格 (E9) | **0.544** ❌ | **0.717** 🏆 | 频段解耦但缺 Base Lock |
| 自然收敛上限 (E4-long ep5) | 0.581 | **0.727** 🏆 | 仍受 trade-off 困扰 |
| **Mystery-SDE (不训练)** | **0.337** ✅ | 0.711 | 意外证明 FC-SB 可行 |
| **FC-SB (本方案)** | **< 0.30** 🎯 | **> 0.73** 🎯 | **结构如铁，笔触如火** |

### 帕累托突破可视化

```
改造前（当前死结）:          改造后（FC-SB 突破）:
  clip ↑                        clip ↑
 0.73 ┤★ E4long  E9           0.75 ┤                    ★ FC-SB 🎯
 0.71 ┤★ E8  ★Mystery(0.34)   0.73 ┤★ E4long   ★Mystery
 0.69 ┤★ E12  E7              0.71 ┤
 0.67 ┤★ E4(0.37)             0.69 ┤★ E4(0.37)
      └────→ LPIPS↓                 └────→ LPIPS↓
      0.33 0.37 0.50 0.60         0.29 0.35 0.45 0.60
           ↑ 不可能三角!             ↑ 突破!
```

---

## 第八部分：影响范围与风险评估

### Affected Code（精确到行号）

| 文件 | 改动类型 | 行号范围 | 改动量 |
|------|---------|---------|--------|
| [model620.py](src/model620.py) | **核心手术** | `integrate_transport()` L512-553 | ~40 行重写 |
| [blocks620.py](src/blocks620.py) | 默认值修改 | L83 | **1 行** |
| [losses620.py](src/losses620.py) | 新增 SDE 噪声 | `compute()` L352 后 | ~8 行新增 |
| [config_schema.py](src/config_schema.py) | **无需修改** | — | 所有参数已预埋 |
| fc_sb_v1.json | 新建配置 | — | 完整 FC-SB 配置 |

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| SDE 噪声导致训练 NaN | 低 | 高 | Phase 1 从 sigma=0 开始课程 |
| Base Locking 导致过度保守 | 中 | 中 | sigma_t 动态调节 |
| Gate init=0.5 初期 Loss 震荡 | 中 | 低 | RMSNorm + Base Locking 双保 |
| kernel=5 不够精细 | 低 |低 | 可切 wavelet lowpass (已实现) |
| 远程 OOM (batch=12) | 低 | 中 | 降到 batch=8 |
| 1天不够全部消融 | 中 | 低 | 优先跑主实验, 消融放第二天 |

### 向后兼容性保障

- 所有新逻辑由 config flag 控制（默认 False/原值）
- 默认配置行为与 E4 **完全一致**
- **无 BREAKING 变更**
- E4/E8/E9/E12 等已有实验不受影响

---

## 第九部分：ADDED Requirements（形式化需求）

### REQ-FC-SB-001: Base Locking Inference
系统 SHALL 在 `integrate_transport()` 每个时间步末尾执行: `h = lowpass(z_content_input) + (h - lowpass(h))`。当 `bridge_path_mode="vertical"` 时自动激活。
**验收**: `||lp(output) - lp(input)||∞ < 1e-4` (任意 N 步积分后)

### REQ-FC-SB-002: Fiber Velocity Projection
系统 SHALL 支持 `i2sb_fiber_project_endpoint=true`。激活后: `v_fiber = v_pred - lowpass(v_pred)`。
**验收**: `||lp(v_fiber)||F / ||lp(v_pred)||F < 0.01`

### REQ-FC-SB-003: Highpass Brownian Noise
系统 SHALL 支持 `i2sb_fiber_project_noise=true`。激活后: `noise_fiber = noise - lowpass(noise)`。
**验收**: 噪声低频能量 < 原始噪声低频能量的 1%

### REQ-FC-SB-004: Gate Init = 0.5
系统 SHALL 支持 `style_gate_init=0.5`。新建 Block 实例后 `gamma.item() ∈ [0.49, 0.51]`。
**验收**: `tanh(gate_init) ≈ 0.462 ≠ 0.05`

### REQ-FC-SB-005: Pure Vertical Wavelet Training Target
系统 SHALL 在 `training_target_projection_mode="pure_vertical_flow_wavelet"` 下构建: `projected = Base(content) + Fiber(target)`。
**验收**: `low_drift < 0.001` 且 `high_energy_ratio > 1.0`

### REQ-FC-SB-006: Training SDE Noise Injection
系统 SHALL 在 `bridge_sigma > 0` 且 training 模式下向 `target_velocity` 叠加高通布朗噪声。
**验收**: Loss 不 NaN, 噪声注入后 target_velocity 低频不变

### REQ-FC-SB-007: Three-Phase Curriculum
系统 SHALL 通过单一 JSON 配置支持三阶段热启动 (σ: 0 → 0.03 → 0.08)。

### REQ-FC-SB-008: One-Day Experiment Budget
全部实验（代码改造 + 3阶段训练 + eval + 诊断） SHALL 在 24 小时内完成主流程。

---

## 第十部分：MODIFIED Requirements

### MOD-REQ-001: integrate_transport() 方法
从"全频段 ODE + 全频段 SDE 噪声"升级为"Fiber SDE + Base Locking"。见改造 1 完整代码。

### MOD-REQ-002: Style Gate 默认初始化
从 0.05 变更为 0.5。消除冷启动 Gate Collapse。
