# FC-SB Phase 3 参数搜索与 hh 排查 Spec

## Why

`fc-sb-phase3-deepfix` 已修复 T/U/V/W 四方向开关，验证：
- U/V 完全生效，但当前采样点稀疏（U: α=0.2/0.5/1.0；V: k=4/8/16），最佳点 V3(k16) 的 lpips=0.3963 仍高于 I7 baseline 0.3625
- T 方向部分生效：hh 参数对 n1_ep_fiber_abs 生效（+0.02）但不传递到 lpips（Δ<0.001），存在传递链路断点
- W 方向生效但 margin=20 过强（lpips 恶化 +0.10），未找到生效与保真的折中点

目标：在修复后的开关基础上，找到优于 I7 baseline (clip=0.7017, lpips=0.3625) 的参数点，并定位 T 方向 hh 断点。

## What Changes

### 工程约束（硬性）
- 显存控制：训练/评估 ≤ 11GB（RTX 3060 12GB）
- U/V 复用 checkpoint：U/V 是推理期参数（N1 块在 integrate_transport），只需修改 checkpoint config 字段，不需重新训练
- W 需重新训练：W 是训练侧 loss，每个 margin 值需独立训练
- probe-first：每个新变体评估后检查 n1_adain_executed=1.0
- 远程 SSH shell 是 cmd.exe，路径用 I:\ 风格

### Change 1: T 方向 hh 传递链路排查（诊断优先）
**现象**: T1→T3（hh03→hh05）n1_ep_fiber_abs 0.3508→0.3762（+0.0254 生效），但 lpips 0.6650→0.6641（Δ=-0.0009 不生效）
**排查目标**: 定位 hh_adain_scale 在 model620.py N1 块内的代码路径，找出"作用于中间变量但未影响最终输出"的断点
**方法**: 只读代码 + smoke test，不需训练

### Change 2: U/V 参数细化搜索
**U 方向（style_extrap_alpha）**: 当前 α=0.2 最佳（clip=0.7164, lpips=0.3735）。搜索 α=0.1/0.15/0.25/0.3，找比 I7 更好的点
**V 方向（patch_adain_kernel）**: 当前 k=16 最佳（clip=0.7295, lpips=0.3963）。搜索 k=20/24/32，找 lpips 降至 I7 以下的点
**方法**: 用 gen 脚本生成新 config，从 I7 checkpoint 生成新变体 checkpoint（改 config 字段），评估

### Change 3: W 方向 margin 调参
**当前**: W2b margin=20, lpips 恶化 +0.10
**搜索**: margin=5/10/15，找 lpips 恶化 <0.02 的折中点
**方法**: 每个 margin 值独立训练 2 epoch + 评估

## Impact

- **Affected specs**: `fc-sb-phase3-deepfix`（前置，本 spec 在其修复基础上探索）
- **Affected code**:
  - `src/model620.py` — T hh 排查（只读诊断，可能新增可观测性）
  - `exp/625_fc_sb/gen_i7_direction_configs.py` — 扩展 U/V 参数点
  - `exp/625_fc_sb/run_rtuv_eval.py` — 评估新变体（已有 probe gate）
  - `exp/625_fc_sb/run_w_batch.py` — W 调参训练（已有 config 校验）
- **Affected docs**: 无

## ADDED Requirements

### Requirement: T 方向 hh 传递链路断点定位
系统 SHALL 定位 hh_adain_scale 在 N1 块内的代码路径断点。
- **WHEN** hh_adain_scale 从 0.3 增大到 0.5
- **THEN** n1_ep_fiber_abs 增加（已验证）
- **BUT** lpips 不变化（断点存在）
- **验收**: 找到断点位置（哪个中间变量未传递到输出），或确认 hh 通过其他路径不影响 lpips

### Requirement: U/V 参数搜索找到优于 I7 的点
系统 SHALL 搜索 U/V 参数空间，找到 clip_style > 0.7017 且 lpips < 0.3963 的点。
- **WHEN** U/V 参数细化搜索完成
- **THEN** 至少一个变体满足 clip_style > I7 且 lpips ≤ I7+0.02
- **验收**: 新变体评估结果表，标注是否优于 I7

### Requirement: W 方向找到 margin 折中点
系统 SHALL 搜索 W margin 参数，找到 lpips 恶化 <0.02 的点。
- **WHEN** margin=5/10/15 训练评估完成
- **THEN** 至少一个 margin 值满足 Δlpips < 0.02 且 W loss 生效（dist_input 非零）
- **验收**: W 调参结果表，标注折中点

## MODIFIED Requirements
无

## REMOVED Requirements
无
