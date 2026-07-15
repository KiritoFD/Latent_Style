# FC-SB Phase 3 参数搜索与 hh 排查 — Verification Checklist

## 阶段 1: T 方向 hh 排查验证
- [x] Task 1.1: hh_adain_scale 代码路径已定位（hh_final → ep_fiber_matched → endpoint → x，无断点）
- [x] Task 1.2: hh 断点已确认（无断点，是设计如此）
  - [x] hh 相关可观测性已添加（9 个 key: n1_hh_input_abs 等）
  - [x] T1/T3 smoke test 对比 hh 可观测性（hh_final_abs +38.8%, contribution_ratio +22.4%）
  - [x] 断点定位结论：hh 生效在 clip 维度(+0.007)而非 lpips(-0.001)，BASE LOCKING 保护 lpips

## 阶段 2: U/V 参数搜索验证
- [x] Task 2.1: 新变体 checkpoint 已生成
  - [x] U: α=0.1/0.15/0.25/0.3（4 个新变体）
  - [x] V: k=20/24/32（3 个新变体）
  - [x] checkpoint config 验证正确（endpoint_adain_scale=1.0）
- [x] Task 2.2: 新变体评估完成
  - [x] 所有变体 n1_adain_executed=1.0（7/7 VALID）
  - [x] clip_style, lpips 已记录
- [x] Task 2.3: U/V 最佳点已确定
  - [x] U 方向 α-clip/lpips 曲线：α 越小越好，U4(α0.1) 最佳
  - [x] V 方向 k-clip/lpips 曲线：仅 2 幂次 kernel 工作，V6(k32) 最佳低 lpips
  - [x] 找到 5 个击败 I7 的点（U4/U5/U1/U6/V6）

## 阶段 3: W 调参验证
- [x] Task 3.1: W 调参配置已生成
  - [x] W2c (margin=5), W2d (margin=10), W2e (margin=15)
  - [x] config 校验通过
- [x] Task 3.2: W 调参训练完成
  - [x] 3 个变体训练成功（VRAM 5.71GB）
  - [x] train.log 中 [W2-debug] dist_input 非零（step=1 loss: 0.42/1.28/2.48）
  - [x] checkpoint config 确认 margin 值
- [x] Task 3.3: W 调参评估完成
  - [x] 3 个变体评估完成
  - [x] margin-lpips 曲线已绘制（5→0.358, 10→0.427, 15→0.465, 20→0.465）
  - [x] 未找到有效折中点（margin=5 是平凡解，margin≥10 lpips 恶化）

## 工程约束验证
- [x] 所有训练/评估显存 ≤ 11GB（W 训练 5.71GB，评估 ~4GB）
- [x] U/V 复用 checkpoint（不重新训练）
- [x] probe-first 原则落实（所有新变体 n1_adain_executed=1.0）
- [x] 代码改动已同步到远程 I 盘

## 最终结论

### T hh 断点定位
**无断点，是设计如此**。hh_adain_scale 作用于 hh_final → ep_fiber_matched → endpoint → x，路径完整。hh 生效在 clip_style 维度（+0.007），不在 content_lpips 维度（-0.001），因 BASE LOCKING 锁死 content lowpass 保 lpips。hh 的作用是"提 clip 不损 lpips"，与 mid/endpoint_adain_scale 职责正交。

### U/V 最佳点
**找到 5 个击败 I7 的点**：
1. **U4(α0.1)**: clip=0.7225(+2.97%), lpips=0.3660(+0.97%) — **综合最佳**
2. U5(α0.15): clip=0.7195(+2.54%), lpips=0.3683(+1.60%)
3. U1(α0.2): clip=0.7164(+2.10%), lpips=0.3735(+3.03%)
4. U6(α0.25): clip=0.7131(+1.63%), lpips=0.3807(+5.02%)
5. **V6(k32)**: clip=0.7262(+3.49%), lpips=0.3722(+2.67%) — **clip 增益最大**

帕累托前沿: I7 → U4(α0.1) → V6(k32) → V3(k16)

### W 折中点
**未找到有效折中点**。hinge loss 仅 step=1 生效（模型一步推过 margin），margin=5 是平凡解，margin≥10 lpips 恶化 +0.06~0.10。需改 soft hinge / KL / 动态 margin。

### 是否找到优于 I7 的配置
**是**。U4(α0.1) 是最佳综合点：clip +2.97%, lpips +0.97%，几乎无副作用的风格增强。V6(k32) clip 增益更大（+3.49%）但 lpips 稍高（+2.67%）。推荐后续探索 U4+V6 联合（α=0.1 + k=32）是否产生协同效应。
