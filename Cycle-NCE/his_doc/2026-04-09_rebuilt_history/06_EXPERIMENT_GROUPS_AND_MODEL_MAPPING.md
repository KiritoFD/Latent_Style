# 实验分组与模型阶段映射

数据主来源：`Y:\experiments\EXPERIMENT_RECORD_FULL_DATA.csv`

## 1. 为什么要做这个映射

目前 `Y:\experiments` 下目录很多，但如果不和模型时代绑定，就会出现两个问题：

1. 只知道某实验分数高，不知道它验证了哪条结构假设。
2. 只知道某模块后来存在，不知道它是被哪些实验实际支持过。

## 2. 分组总表

### 2.1 主线实验与对照组（EXP）

代表实验：

- `exp_1_control`
- `exp_2_zero_id`
- `exp_3_macro_strokes`
- `exp_4_zero_tv`
- `exp_5_signal_overdrive`
- `exp_S1_zero_id`
- `exp_S2_color_blind`
- `final_demodulation`

对应模型时代：

- Era B 到 Era C 的主线配方试验

主要验证：

- identity 是否太强
- TV 是否需要
- macro strokes 是否真的改善观感
- 信号强度上调会不会直接把内容推坏

关键结论：

- `exp_S1_zero_id`
  - style = `0.720806...`
  - lpips / fid / art_fid 明显恶化
  - 说明去掉 identity 能冲高 style，但代价巨大
- `final_demodulation`
  - style 尚可，但 lpips / fid / art_fid 非常差
  - 说明单靠 demodulation 类思路不够

### 2.2 A 系列参数消融组

代表实验：

- `ablate_A0_base_p5_id045_tv005`
- `ablate_A1_p7_id045_tv005`
- `ablate_A2_p11_id045_tv005`
- `ablate_A3_p5_id030_tv005`
- `ablate_A4_p5_id070_tv005`
- `ablate_A5_p5_id045_tv003`

对应模型时代：

- Era B 主结构相对稳定后的单因素扫描

主要验证：

- patch size
- identity weight
- TV weight

这一组很重要，因为它提供了最干净的“其余不变，仅改单轴”的证据。

### 2.3 注入路径消融组

代表实验：

- `inject_I0_all_open`
- `inject_I1_body_only`
- `inject_I2_hires_decoder_only`
- `inject_I3_progressive_1_05_01`
- `inject_I4_body_hires`
- `inject_I5_body_decoder`

对应模型时代：

- Era B 到 Era C 之间

主要验证：

- 风格注入放在哪条路径最有效

从旧总结看，I 系列之间差距不算特别夸张，这暗示：

- 注入位置固然重要
- 但真正的瓶颈更像是“怎么调制”和“怎么约束”，而不是“放哪”

### 2.4 decoder 结构与配方组

代表目录：

- `decoder-A-anchor-nohf`
- `decoder-B-hf-strict-id`
- `decoder-C-relaxed-id-nohf`
- `decoder-D-sweetspot`
- `decoder-H-MSCTM`
- `Decoder_D0_baseline` 到 `Decoder_D7_user_proj720`

对应模型时代：

- Era C

主要验证：

- decoder 是否应该 no-norm
- HF-SWD 是否必要
- identity 和 color 权重如何和 decoder 配方耦合
- 重 decoder 是否真正带来收益

旧分析里最关键的提醒是：

- `heavy_decoder` 这类实验不能简单归因给“decoder 更深”
- 因为常常同时打包了 rank、HF-SWD、trainer 逻辑等多个变化

### 2.5 NCE / SWD 路线组

代表目录：

- `nce`
- `nce-gate_content`
- `nce-gate_norm`
- `nce-swd_0.25-cl_0.01`
- `nce_A*`

对应模型时代：

- Era C

主要验证：

- NCE 是否提升内容结构可分性
- SWD 与 NCE 的配比
- gate / norm 对 NCE 表现的影响

### 2.6 空间调制与颜色策略组

代表目录：

- `spatial-adagn*`
- `coord-spade-50e`
- `clocor1_*`
- `color_*`

对应模型时代：

- Era C

主要验证：

- style-conditioned normalization 的具体形式
- 空间调制是否带来更强风格定位
- color loss 不同定义的利弊

### 2.7 快速扫描与微型回归组

代表目录：

- `scan*`
- `micro01_hf2_lr1`
- `micro02_macro_patch`
- `micro03_gate75`
- `micro04_hf1p5_macro`
- `micro05_id_anchor`

对应模型时代：

- Era C 到 Era D

主要验证：

- 用更低成本训练快速判断方向对不对

`micro05_id_anchor` 在记录里有完整 `full_eval/logs/summary_history`，但 strict 指标列为空，说明它更像“被反复观察的候选配方”，而不是统一主表里的一个最终定格点。

### 2.8 架构搜索 / CGW / 参数联合优化组

代表目录：

- `arch_ablate_*`
- `cgw/*`
- `ca_pram/*`
- `style_oa/*`
- `optuna_hpo/*`

对应模型时代：

- Era D

主要验证：

- feature block 类型组合
- global / window attention 配比
- SWD / identity / color / lr 联合最优点

旧分析里对 `style_oa` 的结论很强：

- `w_swd = 60` 通常优于 `90`
- `w_identity = 3.0` 通常优于 `1.5`
- `w_color = 2.0` 比 `5.0` 更均衡
- `lr = 5e-4` 能推高 style，但更容易把 LPIPS 推坏

## 3. 重要实验与模型阶段的直接挂钩

### 3.1 `exp_S1_zero_id`

对应：

- Era B / C 的 identity 去除试验

说明：

- 一旦 identity 约束拿掉，模型会非常愿意“冲风格分数”
- 但内容和分布会被明显破坏

### 3.2 `ablate_A0_base_p5_id045_tv005`

对应：

- Era B 的经典单因素锚点

说明：

- patch / identity / TV 的 sweet spot 搜索，是这时期最有解释力的实验套路

### 3.3 `final_demodulation`

对应：

- Era C 中一条更激进的调制配方试验

说明：

- 风格并不算差
- 但整体画面代价过大

### 3.4 `style_oa_*`

对应：

- Era D 参数联合优化

说明：

- 这是从“结构是否对”转向“在结构基本对的前提下，损失权重怎么配”

## 4. 我对 `Y:\experiments` 的总体判断

这批实验不是一堆无序目录，而是已经明显分成三层：

1. 主线锚点实验
2. 单因素 / 路径 / 模块消融
3. 架构搜索与联合优化

这对后续写正式文档很有价值，因为可以按这个层级组织，而不是按目录名平铺。

