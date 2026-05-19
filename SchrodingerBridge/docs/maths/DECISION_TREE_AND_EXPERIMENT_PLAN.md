# 提升 clip_style 的决策树实验方案

更新日期：`2026-05-19`。

目标：在 `S-add__K-1_C-0_W-20_Col-0` 谱系上，提高 `clip_style_all`，同时不让 `content_lpips_all` 弱于参考太多。速度和文件规模是 bonus，质量和代码一致性优先。

参考指标：

```text
baseline:
  clip_style_all    = 0.7167235834
  content_lpips_all = 0.4615265376
  clip_content_all  = 0.7977139172

current batch64 refactor:
  epoch 8 clip_style_all    = 0.7128111729
  epoch 8 content_lpips_all = 0.4613536712
```

## 1. 核心假设

当前模型不是风格容量不足，而是 baseline 附近偏保守。`clip_style` 的主要控制量是：

```text
style_gain ~= terminal_swd_pressure
           * delivered_residual_amplitude
           / kinetic_pressure
```

同时内容保持大致受：

```text
content_preservation ~= kinetic_pressure
                     * skip/content retention
                     * routing smoothness
```

所以第一轮实验不加新 loss，不改数据，不改 evaluation，只在当前有效目标上做受控搜索。

## 2. 决策树

```text
Start
|
|-- A. batch=64 的参考谱系是否可复现？
|      |
|      |-- 否：
|      |     停止风格搜索，先修训练/eval/config 对齐。
|      |
|      |-- 是：
|            进入运动预算搜索。
|
|-- B. 降低 kinetic 后 style 是否上升？
|      |
|      |-- 是，且 LPIPS 没坏：
|      |     继续提高 terminal_swd_weight 或 residual_gain。
|      |
|      |-- 是，但 LPIPS 明显坏：
|      |     回退 kinetic，保留较温和 SWD；必要时测试 Sinkhorn 修复。
|      |
|      |-- 否：
|            说明不是 motion budget 主瓶颈，转向 endpoint pressure。
|
|-- C. 提高 terminal_swd_weight 后 style 是否上升？
|      |
|      |-- 是，且 LPIPS 稳：
|      |     锁定该区域，补评中间 epoch。
|      |
|      |-- 是，但 LPIPS 坏：
|      |     降低 residual_gain 或提高 kinetic。
|      |
|      |-- 否：
|            SWD 已饱和，转向 residual_gain / routing 分支。
|
|-- D. residual_gain 提高后是否带来更好 Pareto？
|      |
|      |-- 是：
|      |     记录为推理/训练幅度候选。
|      |
|      |-- 否：
|            不继续放大残差，避免进入 overshoot。
|
|-- E. 如果 style 高但 LPIPS 滑坡：
|      |
|      |-- 测试 Sinkhorn / partial Sinkhorn。
|      |-- 若 style 被压回 baseline 以下，放弃该修复路线。
|
|-- F. 如果质量达标：
       |
       |-- 测试 projection=32 速度分支。
       |-- 只有质量基本不掉，才把它作为最终候选。
```

## 3. 实验预算

用户允许 `36-48` 组量级。统一训练 `8 epoch`，batch 固定为 `64`。因为最佳 epoch 可能出现在中间，evaluation 分两层：

### 第一层：所有实验必评

评估：

```text
epoch_0004
epoch_0006
epoch_0008
```

理由：

- epoch 4 能捕捉早期 style 峰值；
- epoch 6 是中期折中；
- epoch 8 对齐参考配置；
- 对 48 组来说是 144 次 eval，仍比 48 组全 epoch eval 更可控。

### 第二层：Pareto 前列补评

对第一层里 Pareto 前 `12` 个 run，再补评：

```text
epoch_0005
epoch_0007
```

理由：如果最佳点卡在 5/7，补评会抓到；如果某组在 4/6/8 都不行，就不浪费补评预算。

## 4. 36-48 组矩阵

实验矩阵分四个 block，总计默认 `48` 组。

### Block A：主运动预算 x 端点压力

```text
w_kinetic in [1.00, 0.85, 0.70, 0.55, 0.40]
terminal_swd_weight in [20, 24, 28, 32]
```

共 `20` 组。

验证：

- 如果 kinetic 降低带来 style 增益且 LPIPS 稳，motion budget 假设成立；
- 如果 terminal SWD 提高带来 style 增益，endpoint pressure 假设成立；
- 如果二者都不动，说明主要瓶颈不在 active objective 的权重比例。

### Block B：残差幅度小范围

在较可能的区域上测试：

```text
w_kinetic in [0.85, 0.70, 0.55]
terminal_swd_weight in [20, 24, 28]
residual_gain in [1.10, 1.20]
```

共 `18` 组。

验证：

- 如果 `1.10/1.20` 提升 style 且 LPIPS 可控，说明 baseline 交付幅度偏保守；
- 如果 LPIPS 恶化快，说明应回到权重比而不是继续放大 residual。

### Block C：routing 修复

只在高风格风险区测试：

```text
(w_kinetic, terminal_swd_weight) in [(0.70, 28), (0.55, 28), (0.55, 32), (0.40, 24)]
semantic_attn_routing_mode = sinkhorn
semantic_sinkhorn_iters in [2, 3]
```

共 `8` 组。

验证：

- 如果 Sinkhorn 保住 LPIPS 且 style 不掉太多，它是修复器；
- 如果 style 被明显压低，则不作为主路线。

### Block D：速度分支

在两组主候选上测试：

```text
semantic_swd_num_projections = 32
swd_num_projections = 32
```

共 `2` 组。

验证：只在质量接近时保留；否则宁可保持 64 projection。

## 5. 判定规则

脚本使用以下分层判定：

```text
target:
  clip_style_all >= baseline_style
  content_lpips_all <= baseline_lpips + 0.010

promising:
  clip_style_all >= current_batch64_style + 0.002
  content_lpips_all <= baseline_lpips + 0.020

collapse:
  content_lpips_all >= 0.530
  or clip_content_all <= 0.740
```

最终不是只选最高 style，而是先过滤 collapse，再按：

```text
score = 100 * (clip_style - baseline_style)
      - 25  * max(0, content_lpips - baseline_lpips)
      - 5   * max(0, epoch_time_sec - 70) / 70
```

排序。

## 6. 执行脚本

大脚本：

```text
SchrodingerBridge/run_clip_style_decision_tree.py
```

常用命令：

```powershell
cd G:\GitHub\Latent_Style\SchrodingerBridge
python run_clip_style_decision_tree.py --dry-run
python run_clip_style_decision_tree.py --train --eval-main --eval-topk
```

脚本职责：

- 生成 48 组 config；
- 每组训练 8 epoch；
- 跳过已有 checkpoint 和已有 summary，支持断点续跑；
- 第一层评估 epoch 4/6/8；
- 按 Pareto 选前 12，补评 epoch 5/7；
- 汇总 CSV/JSON；
- 给每条结果标注 `target/promising/collapse/weak`；
- 输出下一轮应该增加/回退的方向。
