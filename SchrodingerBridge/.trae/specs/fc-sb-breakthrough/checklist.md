# FC-SB Breakthrough — Verification Checklist

## Task 1: 推理 Solver Base Locking + Fiber SDE
- [ ] `integrate_transport()` 方法开头读取了 6 个 FC-SB 配置标志 (fiber_proj_ep, fiber_proj_noise, fiber_kernel, bridge_path_mode, sigma_base)
- [ ] 定义了局部 `lp()` lowpass 函数 (avg_pool2d kernel=k)
- [ ] 循环前保存了 `x_base_lock = lp(x)` 灵魂锚点
- [ ] **Velocity Fiber Projection**: 当 `fiber_proj_ep=True` 时 `v_fiber = v_pred - lp(v_pred)`，否则 `v_fiber = v_pred`
- [ ] 步进使用 `h = h + v_fiber * dt`（仅 Fiber 分量）
- [ ] **Highpass Noise**: 当 `fiber_proj_noise=True` 时 `noise_fiber = noise - lp(noise)`
- [ ] SDE 噪声方差公式: `sigma_t = sigma_base * sqrt(t*(1-t)) * sqrt(dt)` (Brownian Bridge)
- [ ] **🚨 Base Locking** 在每步末尾: `if vertical: h = x_base_lock + (h - lp(h))`
- [ ] 所有新逻辑由 config flag 守卫（默认行为不变）
- [ ] 向后兼容: `fiber_proj_ep=False` + `bridge_path_mode="linear"` → 行为与原代码一致

## Task 2: Gate 初始化 0.5
- [ ] blocks620.py L83: `style_gate_init: float = 0.5`（原值 0.05）
- [ ] blocks620.py L164: 同步更新（如有第二处定义）
- [ ] 新建 SpatialBridgeBlock620 后 `self.gamma.item() ∈ [0.49, 0.51]`
- [ ] `tanh(0.5) ≈ 0.462`，首步 style gate 输出显著 > 0

## Task 3: 训练期 SDE 噪声注入
- [ ] losses620.py compute() 中 bridge_sigma>0 且 training 时注入高通噪声
- [ ] 噪声通过 `_lowpass(sde_noise, self.lowpass_kernel)` 做高通滤波
- [ ] 噝声叠加到 `target_velocity`（非直接叠加到 h 或 endpoint）
- [ ] metrics 字典新增 `training_sde_noise_lp_rms` 和 `training_sde_noise_hp_rms`
- [ ] 训练不 NaN / 不梯度爆炸
- [ ] CFG dropout 分支不受影响

## Task 4: 配置 + 部署 + 训练
- [ ] fc_sb_v1.json 包含全部 FC-SB 关键参数:
  - [ ] transport_prediction_mode = "endpoint"
  - [ ] i2sb_fiber_project_endpoint = true
  - [ ] i2sb_fiber_project_noise = true
  - [ ] solver_stochastic_noise_scale = 0.08
  - [ ] bridge_path_mode = "vertical"
  - [ ] bridge_sigma = 0.08
  - [ ] training_target_projection_mode = "pure_vertical_flow_wavelet"
  - [ ] w_style_energy_floor = 0.5
  - [ ] style_gate_init = 0.5
  - [ ] body_norm_type / norm_type = "rms_norm"
  - [ ] batch_size = 12, num_epochs = 10
- [ ] 远程部署成功:
  - [ ] grep model620.py 确认 `x_base_lock` 存在
  - [ ] grep model620.py 确认 `v_fiber` 存在
  - [ ] grep model620.py 确认 `noise_fiber` 存在
  - [ ] grep blocks620.py 确认 `gate_init.*0\.5` 存在
  - [ ] grep losses620.py 确认 `sde_noise_hp` 存在
- [ ] 训练启动无报错
- [ ] 训练过程监控:
  - [ ] Phase 1 (ep0-3): base_structural_drift < 0.01 ✅
  - [ ] Phase 2 (ep3-6): fiber_energy_ratio 开始上升
  - [ ] Phase 3 (ep6-10): fiber_energy_ratio > 1.0, velocity_std > 1.0
- [ ] 无 OOM / 无 NaN / 无崩溃

## Task 5: Eval + Dashboard + 目视诊断
- [ ] full_eval 完成，summary.json 包含所有指标
- [ ] clip_style 提取成功
- [ ] content_lpips 提取成功
- [ ] summary_grid.png 生成并下载到 visual_diag/
- [ ] 目视确认: 风格强度 ≥ E4-long ep5
- [ ] 目视确认: 内容保持 ≥ E4 baseline
- [ ] Dashboard HTML 更新:
  - [ ] 新增 FC-SB 数据点（正确坐标: x=1-LPIPS, y=clip_style-0.6399）
  - [ ] 新增帕累托前沿线连接到 FC-SB 点
  - [ ] 图例新增 FC-SB 条目
  - [ ] 说明面板新增 FC-SB 行和结论
- [ ] 成功标准判定:
  - [ ] **最低通过**: clip_style > 0.70 且 LPIPS < 0.40
  - [ ] **理想突破**: clip_style > 0.73 且 LPIPS < 0.30

## Task 6 (可选): 消融实验
- [ ] 如执行: 每个消融变体有独立配置和结果
- [ ] 如执行: 消融结果记录到 dashboard

---

## 全局向后兼容性保障
- [ ] 所有新逻辑由 config flag 控制（默认 False/原值）
- [ ] 默认配置下（fiber_project=false, gate_init=0.05, sigma=0.02）行为与 E4 一致
- [ ] E4/E8/E9/E12 已有实验 checkpoint 可正常加载和推理
- [ ] **无 BREAKING 变更**

## 时间预算检查
- [ ] 代码改造 (Task 1-3) 在 1 小时内完成
- [ ] 部署 + 训练在 4.5 小时内完成
- [ ] Eval + 诊断在 1 小时内完成
- [ ] 总主流程 ≤ 7 小时（留 17 小时 buffer 给排障和消融）
