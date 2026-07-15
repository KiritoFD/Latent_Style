# FC-SB Phase 3 深入排查与开关修复 — Verification Checklist

## 阶段 0: 根因排查验证
- [x] Task 0.1: 训练时 style_latent 来源已定位（tensor，losses620.py:416 target_style_for_model；N1 块不在训练路径）
- [x] Task 0.2: N1 的 style_latent tensor 语义已明确（目标风格参考图 VAE latent, (B,4,H,W)）
- [x] Task 0.3: W loss 是否执行已有结论（计算了但 loss 恒为 0，因 anti_input_margin=0.3 远小于 dist_input O(10-50)）

## 阶段 1: T/U/V 修复验证
- [x] Task 1.1: 修复方案已选定（方案 B-refined），有排查数据支撑
- [x] Task 1.2: 代码修改完成，语法检查通过，已同步到远程 I 盘
- [x] Task 1.3: Smoke test 通过
  - [x] T1 推理后 `model_n1_adain_executed = 1.0`（N1 块执行）✅
  - [x] T1 推理后 `model_n1_ep_fiber_abs ≈ 0.3564`（AdaIN 匹配健康）✅
  - [x] 关键纠正：`endpoint_style_high_abs` 测量的是 endpoint head 投影层，非 N1 块；新增 `n1_adain_executed` 才是 N1 gate

## 阶段 2: W 训练修复验证
- [x] Task 2.1: W2b 旧目录已清空，W2b.json 确认 `w_anti_input_style=3.0, anti_input_margin=20.0`
- [x] Task 2.2: W2b 新训练完成
  - [x] train.log 首行确认 `w_anti_input_style=3.0`
  - [x] train.log 中出现 [W2-debug] dist_input 输出（mean=41-57, loss step1=3.90 非零）
  - [x] checkpoint config 确认 `w_anti_input_style=3.0`
- [x] Task 2.3: W2b 评估完成
  - [x] W2b clip_style=0.6947 / lpips=0.4645，与 I7 ep2 (clip=0.7017, lpips=0.3625) 有显著差异（Δclip=-0.0070, Δlpips=+0.1020）
  - [x] W 方向生效，但 margin=20 过强导致 lpips 恶化，后续需调参

## 阶段 3: T/U/V 重新评估验证
- [x] Task 3.1: 所有 T/U/V 变体评估完成，每个变体 probe 通过（n1_adain_executed=1.0，10/10 全部 VALID）
- [x] Task 3.2: T/U/V 各方向有效性判定
  - [x] T 方向：mid 参数生效（Δ=0.0034），hh 参数对 lpips 不生效（Δ<0.001）但对 n1_ep_fiber_abs 生效 → 部分生效
  - [x] U 方向：α0.2→0.5→1.0, lpips 0.3735→0.4307→0.5218 单调递增 → 完全生效
  - [x] V 方向：k4→8→16, lpips 0.5196→0.4497→0.3963 单调递减 → 完全生效
  - [x] 修复前 9 变体 LPIPS=0.4180（死路径），修复后 0.3735~0.6685（全激活）

## 阶段 4: 流程改进验证
- [x] Task 4.1: run_rtuv_eval.py 增加 probe gate
  - [x] 评估后自动读取 runtime_observability（_find_n1_adain_executed）
  - [x] n1_adain_executed=0 且 endpoint_adain_scale>0 时标记 INVALID
- [x] Task 4.2: run_w_batch.py 增加 config 启动前校验
  - [x] 启动前打印 w_fiber_repulsion/w_anti_input_style/w_style_disc（validate_w_config）
  - [x] 训练日志首行打印 loss 分量列表（[W-CFG-HEADER]）
  - [x] json 值与预期不符时柔性警告（不退出）
- [x] Task 4.3: 反思文档完成
  - [x] project_memory.md 追加 12 条 lessons learned
  - [x] 总结"代码已写 ≠ 功能已生效"教训

## 工程约束验证
- [x] 所有 smoke test / 评估显存 ≤ 11GB（W2b 训练峰值 5.72GB，T1 smoke 显存未超预算）
- [x] 修复时复用现有 checkpoint（T/U/V 无需重新训练）
- [x] probe-first 原则落实（修复后先 smoke test 验证 n1_adain_executed=1.0）
- [x] 代码改动已同步到远程 I 盘（注意：远程 SSH shell 是 cmd.exe，路径用 I:\ 风格）

## 最终结论

### T/U/V/W 四方向修复后生效状态

| 方向 | 修复内容 | 生效状态 | 关键证据 |
|------|---------|---------|---------|
| **T (multiband)** | style_latent_tensor 传递 | ⚠️ 部分生效 | mid 参数对 lpips 生效（Δ=0.0034），hh 参数对 n1_ep_fiber_abs 生效但不传递到 lpips |
| **U (extrap)** | style_latent_tensor 传递 | ✅ 完全生效 | α0.2→0.5→1.0, lpips 0.3735→0.4307→0.5218 单调递增，span=0.1483 |
| **V (patch)** | style_latent_tensor 传递 | ✅ 完全生效 | k4→8→16, lpips 0.5196→0.4497→0.3963 单调递减，span=0.1233 |
| **W (anti_input)** | margin 0.3→20.0 | ✅ 生效（过强） | Δclip=-0.0070, Δlpips=+0.1020；但 margin=20 导致 lpips 恶化，后续降至 5-10 |

### 修复前 vs 修复后对比
- 修复前：9 个变体（T2-T4, U1-U3, V1-V3）LPIPS 全部 0.4180（N1 死路径），参数完全无效果
- 修复后：10 个变体 LPIPS 分布在 0.3735~0.6685（全部激活），参数全部生效

### 最佳点
- **V3(k16)**: clip_style=0.7295, lpips=0.3963（clip 高于 I7 0.7017 且 lpips 可接受）
- **U1(α0.2)**: clip_style=0.7164, lpips=0.3735（最接近 I7 baseline，Δlpips=+0.011）

### 可进入参数搜索阶段
- U/V 方向已验证完全生效，可进入精细参数搜索
- T 方向需排查 hh→lpips 传递链路（hh 对 n1_ep_fiber_abs 生效但不传递到最终 lpips）
- W 方向需调参（margin 从 20 降至 5-10，在"生效"与"保真"之间找折中）
