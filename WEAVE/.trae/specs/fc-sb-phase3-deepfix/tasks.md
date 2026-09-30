# FC-SB Phase 3 深入排查与开关修复 — Tasks

## 阶段 0: 根因深度排查（理论 + 实现）

### Task 0.1: 排查训练时 style_latent 传递链 ✅
**结论**: 训练时 style_latent 是 tensor（[losses620.py:416](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L416) `target_style_for_model`），但 **N1 块根本不在训练路径上**。N1 块位于 `integrate_transport()` 方法内（[model620.py:553](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/model620.py#L553)，`@torch.no_grad()` 推理专用），训练只走 `forward()` 路径。T/U/V 是推理期后处理，训练时未执行是设计如此，非 bug。

### Task 0.2: 理论分析 N1 的 style_latent 应该是什么 ✅
**结论**: style_latent 语义是**目标风格参考图的 VAE latent**，形状 `(B, 4, H, W)`（[losses620.py:400](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/losses620.py#L400) `B,C,H,W = target_style.shape`，C=latent_channels=4）。
- 训练时来源：数据加载器提供目标风格参考图的 VAE latent
- 推理时缺失：[run_evaluation.py:3190](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/utils/run_evaluation.py#L3190) 只构造 dict（含 DINO patches），**没有构造 VAE latent tensor**
- StyleConditioner620 无法从 DINO 投影出 style_latent
- I7.json 中无 `endpoint_adain_scale` 键（默认 0.0）——I7 本身 N1 关闭，T/U/V 变体才开启

### Task 0.3: 排查 W loss 在 train.log 中无输出的原因 ✅
**结论**: W loss **代码执行了**，加入总 loss 并参与反向传播，但 **loss 值恒为 0**！
- 根因：`anti_input_margin=0.3` 远小于 `dist_input`（C·H·W 维向量 L2 范数，通常 O(10)）
- `F.relu(0.3 - dist_input)` 几乎全为 0 → loss=0 → 梯度=0
- train.log 不打印 W loss 是因为 [trainer.py:1443-1451](file:///g:/GitHub/Latent_Style/SchrodingerBridge/src/trainer.py#L1443) postfix 硬编码只输出 loss/flow/kin/curv/ot/tswd
- **真正需要调的是 `anti_input_margin`**（使其大于典型 dist_input），而非权重 w

## 阶段 1: 修复 T/U/V 推理 style_latent

### Task 1.1: 确定修复方案 ✅
**依赖**: Task 0.1, Task 0.2 完成
**结论**: 选定**方案 B-refined**（3 个协调改动）
**根因链（比阶段 0 更深）**:
- `inference.py:541-549` 当 `target_style_latent` 是 dict 时，只提取 DINO 字段，**完全丢弃 `target_style_latent` kwarg**
- → `integrate_transport` 收到 `target_style_latent=None` → L567 守卫永不触发 → style_latent 永远 None → L676 N1 块跳过
**改动清单**:
1. `src/utils/run_evaluation.py` L3190: 构造 `style_latent_tensor`（VAE encode 目标风格参考图），加入 dict
2. `src/utils/inference.py` L541-549: 当 dict 时，提取 `style_latent_tensor` 作为 `target_style_latent` kwarg 传递
3. `src/model620.py` L567: **无需改动**（guard 已正确处理 tensor）
**W loss 尺度分析**:
- `dist_input` 是 16384 维向量 L2 范数，典型量级 O(10-50)
- `anti_input_margin=0.3` 远小于 dist_input → `F.relu(0.3 - 40)` 恒为 0
- 重训需设 `anti_input_margin=20.0`（保守值），并加 debug print 测量实际分布

### Task 1.2: 实现修复 ✅
**依赖**: Task 1.1 完成
**修改文件**:
1. `src/utils/inference.py` L548-551: dict 分支中提取 `style_latent_tensor` 作为 `target_style_latent` kwarg 传递
2. `src/utils/run_evaluation.py` L3174-3248: 构造 `style_latent_tensor`（VAE encode 目标风格参考图），按 style ID 缓存，注入 dict
3. `src/model620.py` L567: **无需改动**（guard 已正确处理 tensor）
**验证**: ast.parse 通过，向后兼容（无 style_latent_tensor 时行为不变）

### Task 1.3: Smoke test 验证开关生效 ✅
**依赖**: Task 1.2 完成
**目标**: 单样本推理，确认 N1 块执行
**关键发现**: `model_endpoint_style_high_abs` 测量的是 `forward()` L518-520 的 endpoint head 投影层，**不是 N1 块**。N1 块在 `integrate_transport()` L676-847 完全没有 last_debug 写入。spec 中把它作为 N1 gate 是错误的。
**修复**: 在 model620.py N1 块内添加可观测性：
- L677: `self.last_debug["n1_adain_executed"] = 1.0`（条件满足时）
- L849-857: else 分支写入 `n1_adain_executed=0.0` + `n1_skip_reason`（scale_zero/style_latent_none/style_latent_not_tensor）
- 关键操作后: `self.last_debug["n1_ep_fiber_abs"] = ep_fiber_matched.detach().float().abs().mean().item()`
**验证结果**:
- T1 smoke test: `model_n1_adain_executed = 1.0`（3 次出现）✅
- `model_n1_ep_fiber_abs ≈ 0.3564`（非零，AdaIN 匹配健康）✅
- **style_latent_tensor 修复完全生效，N1 块真正执行**
**工程教训**: 远程 SSH shell 是 cmd.exe，路径需用 Windows 风格 `I:\...`，`/mnt/i/...` 会被误解析为 `C:\mnt\i\`（残留副本待清理）

## 阶段 2: 修复 W 训练配置

### Task 2.1: 清理 W2b 旧结果 + W loss 测量改动 ✅
**修改文件**:
1. `src/losses620.py` L636-676: 加 W loss debug print（每 50 步打印 dist_input/pairwise_dist 分布），位于 `if w_xxx > 0:` 块内，w=0 时不受影响
2. `exp/625_fc_sb/from_scratch_win/w_configs/W2b.json` (新建): `w_anti_input_style=3.0, anti_input_margin=20.0`（从 0.3 增大），基于 I7 base config
**待执行**: 删除远程 I 盘 `exp/625_fc_sb/from_scratch_win/w_W2b/` 旧 checkpoint（需远程操作）

### Task 2.2: 重新训练 W2b（新配置 3.0）
**依赖**: Task 2.1 完成
**目标**: 用正确配置训练 W2b
**步骤**:
- [ ] 启动前 cat W2b.json | grep w_anti_input_style，确认 3.0
- [ ] 训练 2 epoch
- [ ] 检查 train.log 首行包含 `w_anti_input_style=3.0`
- [ ] 检查 train.log 中出现 W loss 分量（anti_input_loss）
- [ ] checkpoint config 确认 `w_anti_input_style=3.0`
**输出**: W2b 新训练完成，probe 通过

### Task 2.3: 评估 W2b 新结果 ✅
**依赖**: Task 2.2 完成
**验证结果**:
- W2b epoch_0002: clip_style=0.6947, content_lpips=0.4645
- I7 baseline epoch_0002: clip_style=0.7017, content_lpips=0.3625
- Δclip_style = -0.0070, Δlpips = +0.1020
- **W 方向明显生效**（Δ 远超 0.001 阈值，lpips 差异是阈值的 100 倍）
**影响性质**: margin=20.0 让 anti_input_style loss 生效，但 content_lpips 大幅恶化（+0.10），说明过强 margin 对内容重构施加了过强推力，模型为满足 margin 约束牺牲了内容保真度。
**调参建议**: 后续可降低 margin（5.0~10.0）或降低 w_anti_input_style 权重，在"生效"与"保真"之间找折中。
**注意**: 任务描述中 I7 baseline 数值（clip=0.7031, lpips=0.3399）实际来自 epoch_0001，非 epoch_0002。

## 阶段 3: 重新评估 T/U/V（修复后）

### Task 3.1: 批量评估 T/U/V（修复后）✅
**依赖**: Task 1.3 完成
**验证结果**: 10 个变体（4T+3U+3V）全部评估成功，n1_adain_executed=1.0

| 变体 | 参数 | clip_style | lpips | n1_ep_fiber_abs | Δclip vs I7 | Δlpips vs I7 |
|------|------|-----------|-------|-----------------|------------|--------------|
| I7 | baseline | 0.7017 | 0.3625 | — | 0 | 0 |
| T1 | mid03,hh03 | 0.6518 | 0.6650 | 0.3508 | -0.0499 | +0.3025 |
| T2 | mid05,hh03 | 0.6574 | 0.6684 | 0.4096 | -0.0443 | +0.3059 |
| T3 | mid03,hh05 | 0.6587 | 0.6641 | 0.3762 | -0.0430 | +0.3016 |
| T4 | mid05,hh05 | 0.6609 | 0.6685 | 0.4290 | -0.0408 | +0.3060 |
| U1 | α0.2 | 0.7164 | 0.3735 | 0.4622 | +0.0147 | +0.0110 |
| U2 | α0.5 | 0.6959 | 0.4307 | 0.5786 | -0.0058 | +0.0682 |
| U3 | α1.0 | 0.6736 | 0.5218 | 0.7718 | -0.0281 | +0.1593 |
| V1 | k4 | 0.7242 | 0.5196 | 0.3863 | +0.0225 | +0.1571 |
| V2 | k8 | 0.7290 | 0.4497 | 0.3881 | +0.0273 | +0.0872 |
| V3 | k16 | 0.7295 | 0.3963 | 0.3862 | +0.0278 | +0.0338 |

**关键发现**: 修复前 9 个变体（T2-T4, U1-U3, V1-V3）LPIPS 全部 ≈ 0.4180（N1 死路径），修复后分布在 0.3735~0.6685（全部激活）。T1 修复前后一致（0.6650），因 T1 的 N1 修复前已执行。

### Task 3.2: 分析 T/U/V 修复后结果 ✅
**依赖**: Task 3.1 完成
**结论**:
- **U 方向完全生效**: α0.2→0.5→1.0, lpips 0.3735→0.4307→0.5218 单调递增，span=0.1483。n1_ep_fiber_abs 也单调递增（0.46→0.58→0.77）
- **V 方向完全生效**: k4→8→16, lpips 0.5196→0.4497→0.3963 单调递减，span=0.1233。大 kernel=更强空间平滑=更好内容保留
- **T 方向部分生效**: mid 参数对 lpips 生效（Δ=0.0034/0.0044），hh 参数对 lpips 不生效（Δ<0.001）但对 n1_ep_fiber_abs 生效（hh03→hh05 增加 ~0.02），说明 hh 的 N1 内部效应未传递到最终 lpips
- **最佳点**: V3(k16) clip=0.7295 lpips=0.3963（clip 高于 I7 且 lpips 可接受）；U1(α0.2) clip=0.7164 lpips=0.3735（最接近 I7 baseline）
- **T 方向所有变体 lpips 都很高（0.66+）**，说明 multiband_adain 对内容损害大，T 方向不适合直接使用

## 阶段 4: 反思与流程改进

### Task 4.1: 在 run_rtuv_eval.py 中加 probe gate ✅
**修改**: 新增 4 个 helper 函数（_find_n1_adain_executed, _find_endpoint_adain_scale, _load_ckpt_endpoint_adain_scale, check_probe_gate）
**gate 规则**: n1==1.0 → VALID；n1==0.0/missing 且 endpoint_adain_scale>0 → INVALID
**集成**: eval_variant 返回结果增加 n1_adain_executed/valid 字段；CSV 表头增加 n1_adain_executed,valid 列；结尾打印 VALID/INVALID 计数

### Task 4.2: 在 run_w_batch.py 加 config 启动前校验 ✅
**修改**: 新增 W_LOSS_MAP（权重→loss 分量名映射）、W_PARAM_KEYS、W_EXPECTED；3 个 helper 函数（_extract_w_params, active_w_losses, validate_w_config）
**校验**: 启动前打印 6 个关键 W 参数 + 激活 loss 列表；train.log 首行写入 [W-CFG-HEADER]；柔性校验（不符仅警告不退出）

### Task 4.3: 反思文档 ✅
**project_memory.md 更新**: 追加 12 条 lessons learned
**核心教训**:
1. "代码已写" ≠ "功能已生效" —— 必须有运行时 probe 验证（n1_adain_executed=1.0）
2. `model_endpoint_style_high_abs` 测量的是 endpoint head 投影层，不是 N1 块 —— observability 指标的语义必须明确
3. inference.py dict 分支会丢弃 target_style_latent kwarg —— 必须显式提取 style_latent_tensor
4. W loss 的 margin 必须匹配 dist_input 量级（O(10-50)），否则 F.relu 恒为 0
5. 远程 SSH shell 是 cmd.exe，路径用 Windows 风格 I:\...

## Task Dependencies

```
阶段 0 (Task 0.1-0.3) ── 根因排查（理论 + 实现）
    ↓
阶段 1 (Task 1.1-1.3) ── 修复 T/U/V + smoke test
    ↓                     ↘
阶段 2 (Task 2.1-2.3) ── 修复 W 训练    阶段 3 (Task 3.1-3.2) ── T/U/V 重新评估
    ↓                     ↗
阶段 4 (Task 4.1-4.3) ── 流程改进 + 反思
```

阶段 1 和阶段 2 可并行（T/U/V 修复与 W 重训互不依赖）。

## 显存预算

| 阶段 | 类型 | 显存 | 策略 |
|------|------|------|------|
| 阶段 0 | 排查（无训练） | ~0 | 只读代码 |
| 阶段 1.3 | smoke test 推理 | ~6-8G | 单样本 |
| 阶段 2.2 | W2b 重训 | ~9-10G | batch=24, 2 epoch |
| 阶段 2.3 | W2b 评估 | ~9-11G | batch=16, num_steps=12 |
| 阶段 3.1 | T/U/V 评估 | ~9-11G | batch=16, num_steps=12 |
