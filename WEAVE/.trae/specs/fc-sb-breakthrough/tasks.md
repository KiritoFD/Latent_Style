# FC-SB Breakthrough — Implementation Tasks (1-Day Budget)

## Task 1: 推理 Solver 核心手术 — Base Locking + Fiber SDE

**文件**: `src/model620.py` — `integrate_transport()` (L512-553)
**优先级**: 🔴 最高（核心改造）
**预计时间**: 30 分钟

### 1.1 现状
当前推理循环 (L532-552):
- 已有 bridge_sigma SDE 噪声 → 但**全频段各向同性**
- 无 velocity fiber projection
- **无 Base Locking**
- `i2sb_fiber_project_*` 配置标志存在于 config_schema.py 但**未接入此方法**

### 1.2 改造步骤

1.2.1 在方法开头读取 FC-SB 配置:
```python
cfg = self.bridge_cfg or self.model_cfg
fiber_proj_ep = bool(getattr(cfg, 'i2sb_fiber_project_endpoint', False))
fiber_proj_noise = bool(getattr(cfg, 'i2sb_fiber_project_noise', False))
fiber_kernel = max(1, int(getattr(cfg, 'i2sb_fiber_project_kernel', 5)))
if fiber_kernel % 2 == 0: fiber_kernel += 1
bridge_path_mode = str(getattr(cfg, 'bridge_path_mode', 'linear')).lower().strip()
sigma_base = float(getattr(self, 'bridge_sigma', 0.02))
```

1.2.2 定义局部 lowpass 函数 + Base 锚点:
```python
def lp(y, k=fiber_kernel):
    return F.avg_pool2d(y.float(), k, stride=1, padding=k // 2).to(dtype=y.dtype)
x_base_lock = lp(x)  # 🚨 灵魂锚点
```

1.2.3 循环内 Step 2: Velocity Fiber Projection
```python
v_pred = (endpoint - h) / denom
if fiber_proj_ep:
    v_fiber = v_pred - lp(v_pred)
else:
    v_fiber = v_pred
h = h + v_fiber * dt  # 仅 Fiber 分量步进
```

1.2.4 循环内 Step 4: Highpass Brownian Noise
```python
if sigma_base > 0.0:
    sigma_t = sigma_base * math.sqrt(max(0.0, t_curr*(1-t_curr))) * math.sqrt(abs(dt))
    noise = torch.randn_like(h)
    noise_fiber = (noise - lp(noise)) if fiber_proj_noise else noise
    h = h + sigma_t * noise_fiber
```

1.2.5 **Step 5: Base Locking**（每步末尾）
```python
if bridge_path_mode == "vertical":
    h = x_base_lock + (h - lp(h))
```

### 验证标准
- [ ] 单元测试: 随机输入 x 经 N 步后 `||lp(output) - lp(input)||∞ < 1e-4`
- [ ] `fiber_proj_ep=False` 时行为与原代码一致（向后兼容）

---

## Task 2: Gate 初始化提升到 0.5

**文件**: `src/blocks620.py` — L83, L164
**优先级**: 🔴 高
**预计时间**: 5 分钟

### 步骤
2.1 将 L83 和 L164 的 `style_gate_init: float = 0.05` 改为 `style_gate_init: float = 0.5`
2.2 同步检查 config_schema.py 中默认值（如有）

### 验证标准
- [ ] 新建 Block 实例后 `block.style_gate.item() ≈ 0.5`
- [ ] `tanh(0.5) ≈ 0.462`，首步 gate 输出不再接近零

---

## Task 3: 训练期高通 SDE 噪声注入

**文件**: `src/losses620.py` — `compute()` (~L352 之后)
**优先级**: 🟡 中（增强项）
**预计时间**: 15 分钟

### 步骤
在 `target_velocity` 赋值之后、FM Loss 计算之前插入:
```python
if self.bridge_sigma > 0 and self.training:
    sde_noise = torch.randn_like(target_velocity)
    sde_noise_hp = sde_noise - _lowpass(sde_noise, self.lowpass_kernel)
    target_velocity = target_velocity + self.bridge_sigma * sde_noise_hp
    metrics["training_sde_noise_lp_rms"] = _lowpass(sde_noise).std().item()
    metrics["training_sde_noise_hp_rms"] = sde_noise_hp.std().item()
```

### 验证标准
- [ ] 训练不 NaN/梯度爆炸
- [ ] `noise_lp_rms < 0.01 * noise_hp_rms`（确认高通有效）

---

## Task 4: FC-SB 配置生成 + 远程部署 + 三阶段训练

**优先级**: 🔴 高（主实验）
**预计时间**: ~4 小时（含训练等待）

### 4.1 生成 fc_sb_v1.json
基于 E4 最佳配置 (`exp/p3_remote_10h/e4_anti_degeneration/config.json`)：

| 参数 | E4 值 | FC-SB 值 | 说明 |
|------|-------|---------|------|
| model.transport_prediction_mode | "velocity" | **"endpoint"** | 直接预测终点 |
| model.i2sb_fiber_project_endpoint | false | **true** | 推理 velocity 纤维投影 |
| model.i2sb_fiber_project_noise | false | **true** | 推理噪声高通滤波 |
| model.solver_stochastic_noise_scale | 0.01 | **0.08** | SDE 魔法阈值 |
| bridge.bridge_path_mode | "vertical" | "vertical" | 保持（Base Locking 需要）|
| bridge.bridge_sigma | 0.05 | **0.08** | Phase 3 全功率 |
| bridge.training_target_projection_mode | "legacy" | **"pure_vertical_flow_wavelet"** | Base(c)+Fiber(t) |
| bridge.w_style_energy_floor | 0.0 | **0.5** | 高频能量保护 |
| model.style_gate_init | 0.05 | **0.5** | Gate 全开 |
| model.body_norm_type / norm_type | "rms_norm" | "rms_norm" | RMSNorm |
| training.batch_size | 24 | **12** | 保守防 OOM |
| training.num_epochs | 3 | **10** | 三阶段课程 |

保存到: `exp/p3_remote_10h/fc_sb_v1/config.json`

### 4.2 部署到远程
4.2.1 SCP 源文件: model620.py, blocks620.py, losses620.py, config.json
4.2.2 部署脚本验证: grep base_lock/fiber_project/gate_init/sde_noise_hp

### 4.3 启动训练 (blocking=false, long_running_process)

### 4.4 三阶段监控决策点

| 时间 | 决策点 | 通过条件 | 失败动作 |
|------|--------|---------|---------|
| T+~30min | Phase 1 结束 (ep3) | drift<0.01, LPIPS<0.25 | 检查 Base Locking |
| T+~1.5h | Phase 2 结束 (ep6) | energy_ratio>1.0 | 提高 sigma 到 0.05 |
| T+~3h | Phase 3 结束 (ep10) | clip>0.70, LPIPS<0.40 | 进入消融 |

### 验证标准
- [ ] 无 OOM / NaN
- [ ] base_structural_drift < 0.01
- [ ] fiber_energy_ratio > 1.0 (Phase 3)
- [ ] velocity_std > 1.0 (Phase 3)

---

## Task 5: Full Eval + Dashboard 更新 + 目视诊断

**优先级**: 🔴 高
**预计时间**: 1 小时

### 5.1 评估
- full_eval 强制保存图片 (`--save_generated_images --save_summary_grid --style_subdirs Hayao,cezanne,monet,photo,vangogh`)
- 提取: clip_style, LPIPS, velocity_std, fiber_energy_ratio, base_drift

### 5.2 Dashboard 更新
- phase616_live_dashboard.html 新增 FC-SB 数据点
- 更新帕累托前沿线
- 更新实验汇总表

### 5.3 目视诊断
- 下载 summary_grid.png → Read 工具查看
- 对比: 风格强度 vs E4-long ep5, 内容保持 vs E4

### 成功标准
- [ ] **最低**: clip_style > 0.70 且 LPIPS < 0.40
- [ ] **理想**: clip_style > 0.73 且 LPIPS < 0.30 （帕累托突破！）

---

## Task 6 (可选): 快速消融实验

如果时间允许（T+5.5h 后），按优先级执行:

| # | 变体 | 改动 | 目的 |
|---|------|------|------|
| 6a | fc_sb_nolock | Base Locking 关闭 | 验证 Locking 的 LPIPS 贡献 |
| 6b | fc_sb_sigma04 | sigma=0.04 (非 0.08) | 找 sigma 甜点 |
| 6c | fc_sb_kernel7 | lowpass kernel=7 (更温和切割) | 验证频率边界敏感性 |

每个变体: batch=12, epoch=5, 约 40 分钟。

---

## Task Dependencies & 并行策略

```
Task 1 (Solver 手术) ──────┐
Task 2 (Gate init) ────────┼──→ Task 4 (配置+部署+三阶段训练) ──→ Task 5 (Eval+Dashboard)
Task 3 (训练SDE噪声) ─────┘          ↓ (如果时间允许)
                              Task 6 (消融实验)
```

- Task 1, 2, 3 **完全独立，可并行实现**
- Task 4 依赖 1+2+3 完成
- Task 5 依赖 Task 4 训练完成
- Task 6 可选，依赖 Task 5 结果决定方向
