# Task Spec: DINO-style 主指标研究

## Goal

以 DINO-style 为主指标，通过"消融减法 → 架构加法 → 参数调整"三阶段提升模型性能。
当前最优 hp_simple_swd12_15ep: CLIP-S=0.7167, DINO-sty=0.4762, DINO-con=0.8052, 1-LPIPS=0.7010, MUSIQ=43.23。

## Milestones

1. **M1 消融减法**: 逐个关闭 hp 机制，测 5 指标，删除对 DINO-sty 无贡献的机制
2. **M2 架构加法**: 实现 5 个方向，每个训练到收敛，保留提升 DINO-sty 且不损核心指标的
3. **M3 参数调整**: 基于最优架构组合，网格调参
4. **M4 主表定稿**: 最终配置 + 完整 5 指标 + 文档

## Success Criteria

- DINO-sty > 0.50 (当前 0.4762, +0.024 目标)
- CLIP-S >= 0.715 (不跌破 WEAVE)
- DINO-con >= 0.78 (不显著损内容)
- 1-LPIPS >= 0.68
- 训练 VRAM <= 11.2GB

## DINO-style 可行性确认

| 维度 | 证据 | 结论 |
|------|------|------|
| 区分度 | semantic SWD 0.4584-0.4637 vs global 0.4762, Δ=0.018 | ✓ 能区分好坏 |
| 稳定性 | DINOv2-small, 750img, 30refs/style, 可复现 | ✓ 稳定 |
| 诊断价值 | 能区分 global vs semantic, 轻度 vs 重度后处理 | ✓ 有诊断力 |
| 比较轴 | WEAVE 无 DINO 数据 | ✓ 新轴 |
| 基础设施 | _compute_dino.py 已就绪 | ✓ 可执行 |

**结论: DINO-style 确认可行，切换为主指标。**

## 当前 hp 架构组成 (SpectralODEObjective620)

损失项:
- 3 per-subband FM losses: w_ll=0.3, w_lh=1.0, w_hl=1.0
- HH velocity head: w_hh=2.0
- endpoint SWD: single_step_swd_weight=12.0 (global, squared)
- endpoint content: w_endpoint_content=1.0
- endpoint style: w_endpoint_style=8.0
- edge L1: single_step_edge_weight=0.1
- terminal SWD: terminal_swd_weight=0.1

架构:
- DWT route train prob=0.8 (stochastic)
- spectral_ode_enabled=true
- swd_semantic_mode=off
- kinetic_penalty_mode=off

训练: bs=160, 15ep, patience=2

## 阶段1: 消融减法清单

每个消融训练到收敛(Patience=2, max=10)，评估5指标。
判定: 去掉后 DINO-sty 不降(Δ>=-0.002)或升高 → 机制无效，删除。

| ID | 机制 | 关闭方式 | 假设 |
|----|------|----------|------|
| A3 | edge L1 | single_step_edge_weight=0 | 边缘约束可能冗余(SWD已约束分布) |
| A4 | terminal SWD | terminal_swd_weight=0 | 终端SWD可能冗余(single_step=12已很强) |
| A7 | endpoint content | w_endpoint_content=0 | 内容端点可能冗余(LL已锁死保护结构) |
| A8 | LL FM loss | spectral_w_ll=0 | 低频FM可能冗余(LL已锁死) |

注: A1(spectral_ode)/A2(dwt_route)是架构级开关,由contract_family控制,关闭不安全,跳过。
A5(HH head)取决于enable_hh_head,需先确认是否启用。
A6(style endpoint)降低可能损CLIP-S,不属于"减法"范畴。

## 阶段2: 架构加法方向 (5个, 不引入外部预训练模型避免先验污染)

**硬约束: 训练中禁止引入 DINO/CLIP 等外部预训练模型作为 loss, 会先验污染。**
**评估期用 DINO/CLIP 是 OK 的 (评估不污染训练)。**

### D1: Gram matrix 风格损失 (Loss方向)
- 机制: 对 z_hat1 和 target_style 的 latent 计算自相关 Gram matrix (C×C), 加 L1 距离
- 理论: Gram matrix 捕获通道间相关性 (纹理/笔触), 是经典风格迁移损失; 当前 SWD 只匹配边缘分布, 丢失通道相关结构
- 风险: Gram 对低频敏感, 可能破坏内容; 需要只在 HF 子带应用
- 预期: DINO-sty +0.015 (纹理相关性提升风格相似度)
- 实现位置: spectral_losses620.py compute() 内, 在 z_hat1 计算后

### D2: 高阶矩分布匹配 (Loss方向)
- 机制: SWD 之外, 加三阶矩 (skewness) 和四阶矩 (kurtosis) 的对齐损失
- 理论: SWD 通过 projection 匹配分布, 但少量 projection (64) 采样不足; 显式高阶矩约束补充分布形状
- 风险: 高阶矩方差大, 可能不稳定; 需要 small weight
- 预期: DINO-sty +0.01
- 实现位置: spectral_losses620.py _sliced_wasserstein 旁

### D3: 多级 DWT 分频 (分频方向)
- 机制: 当前单级 Haar DWT (LL/LH/HL/HH), 升级为 2-3 级 (LL1→LL2/LH2/HL2/HH2), 每级独立 FM + SWD
- 理论: 单级分频粗, 中频和细节混在一个子带; 多级让 style 在不同尺度独立注入
- 风险: 显存增加 (多级激活); 需要降 bs
- 预期: DINO-sty +0.02 (多尺度风格匹配)
- 实现位置: spectral620.py dwt2_haar, spectral_bridge620.py forward

### D4: style cross-attention 注入 (Backbone方向)
- 机制: ODE 速度场网络中插入 cross-attention 层, style_latent 编码为 K/V, content 特征为 Q
- 理论: 当前 style 注入靠 style_memory (class-level), instance-level 风格信息不足; cross-attn 做空间相关的 style 注入
- 风险: 增加参数量; 训练时间增加
- 预期: DINO-sty +0.02, CLIP-S +0.005
- 实现位置: spectral_bridge620.py, 在 backbone blocks 间插入

### D5: frequency-adaptive style injection (分频+Backbone方向)
- 机制: 每个子带独立 style gate (可学习), LL gate 小 (保内容), HH gate 大 (强风格); gate 由 style_latent 条件化
- 理论: 当前所有子带共享 style_memory, 无法区分低频/高频的风格强度需求; 自适应 gate 让模型学会分频风格注入
- 风险: gate 可能退化为常数; 需要正则
- 预期: DINO-sty +0.015
- 实现位置: spectral_bridge620.py head_ll/lh/hl/hh 前加 gate

## 阶段3: 参数调整
基于 M1+M2 最优架构组合, 网格调:
- SWD weight (8-16), spectral weights, endpoint weights
- 新损失权重, 训练超参 (epochs, lr)

## Constraints
- VRAM <= 11.2GB (RTX 3060 12GB)
- bs 控制: K<=4 用 128, K>=8 用 96, 无 semantic 用 160
- 评估 VRAM <= 7GB (bs=2)
- 数据集: D5-512 (samam_512), 750 test images
- 所有实验训练到收敛 (Patience=2, max=10, >=5ep)
