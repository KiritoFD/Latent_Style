# 625 FC-SB推理优先实验探索 - Product Requirement Document

## Overview
- **Summary**: 基于FC-SB（纤维约束薛定谔桥）理论和之前620消融实验的教训，采用"推理优先"策略在远程3060 WSL上推进实验。核心洞察："Fiber-SDE σ=0.08 (不训练)"已达到clip_style=0.711, LPIPS=0.337，证明推理期FC-SB机制本身就是突破帕累托前沿的关键。本项目分阶段验证：0训练成本的推理参数扫描 → 干净基线训练 → FC-SB增强训练 → 全FC-SB组合。
- **Purpose**: 避免之前Phase 2实验同时修改过多训练期机制导致模型学坏的错误，通过"推理优先"快速验证FC-SB理论预测，以最小训练成本冲击clip_style>0.73且LPIPS<0.35的目标。
- **Target Users**: 研究人员（本项目开发者）

## Goals
- 验证FC-SB四个推理期机制（Fiber Velocity Projection, Base Locking, Fiber SDE Noise, Fiber-Only Endpoint）的独立和组合效果
- 在0训练成本下（仅推理）复现"不训练Fiber-SDE"的奇迹结果
- 训练一个干净的620推荐配置基线（3 epochs），作为FC-SB推理的基座
- 找到推理期σ和kernel的最优组合
- 验证curriculum sigma schedule在推理期的效果
- 探索CFG外推与FC-SB的协同效应
- 冲击目标：clip_style > 0.72 且 LPIPS < 0.40（铜牌），目标clip_style > 0.73 且 LPIPS < 0.35（金牌）

## Non-Goals (Out of Scope)
- 不修改模型架构（不替换RMSNorm、不改gate初始化——这些是训练期改动，留待后续）
- 不开启激进的训练期投影（training_target_projection_mode保持legacy）
- 不测试DINO条件（已证明DINO导致严重白化WFI=0.64）
- 不测试Text条件（理论预测gate修复后才有效）
- 不进行超过3 epochs的长训练（基于之前发现：WFI随训练单调恶化）
- 不尝试per-style模型（保持universal模型设定）

## Background & Context

### 理论基础
- **FC-SB理论**（docs/622/FC.md）：将潜空间物理解耦为Base（结构，死寂Dirac分布）和Fiber（纹理，热力学扩散）
- **四机制退化吸引子**（docs/622/history/10_unified_mathematical_model.md）：Gate Collapse (g→0.05), Endpoint Shrinkage (α→0.16), GN白化, Training-Output Mismatch
- **关键实验现象**：未训练的Fiber-SDE σ=0.08达到clip_style=0.711, LPIPS=0.337，比训练过的模型更好

### 代码状态
- model620.py `integrate_transport()` 已完整实现FC-SB推理机制（L535-626）：
  - Fiber Velocity Projection (v_fiber = v_pred - lp(v_pred))
  - Base Locking (h = x_base_lock + (h - lp(h)))
  - Fiber SDE Noise (noise_fiber = noise - lp(noise))
  - Fiber-Only Endpoint (endpoint = x_base_now + ep_fiber)
  - Curriculum sigma schedule (t∈[0,0.33]: 0.25σ; [0.33,0.66]: 0.6σ; [0.66,1]: 1.0σ)
  - 支持avg_pool和wavelet两种lowpass模式
- config_schema.py已添加所有必要配置字段
- 基线配置configs/620_spatial_bridge_ablation_recommended.json已存在（WFI=0.3757, CLIP-S=0.6995, LPIPS=0.3422）

### 之前Phase 2实验失败的教训
- F3-F7实验clip_style下降~14%，LPIPS恶化~29%
- 失败原因：同时开启了训练期投影（pure_vertical_flow_wavelet）、修改了模型架构（RMSNorm、gate=0.5），多个改动耦合导致模型学坏
- 正确路径：先验证推理期机制（不需要训练），再逐步引入训练期改动

### 硬件约束
- 远程：RTX 3060 12GB WSL
- batch_size=24安全（峰值显存~8.9GB）
- 每epoch训练时间约50分钟（batch=24, 5类风格）

## Functional Requirements
- **FR-1**: 现有checkpoint的推理期FC-SB参数扫描脚本
  - 自动加载指定checkpoint
  - 支持4个FC-SB开关的所有组合（2^4=16组，但分阶段增量添加）
  - 支持σ值扫描：0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12
  - 支持kernel扫描：3, 5, 7
  - 支持lowpass模式：avg_pool, wavelet
  - 支持curriculum/constant/linear_ramp三种sigma schedule
  - 自动运行full_eval并提取clip_style, LPIPS, WFI
- **FR-2**: 干净基线训练配置生成
  - 基于620_spatial_bridge_ablation_recommended.json
  - 修正远程路径（/mnt/i/...）
  - batch_size=24, num_epochs=3, virtual_length_multiplier=1.0
  - DataLoader: num_workers=0, pin_memory=False, persistent_workers=False
  - 评估延迟到训练结束（full_eval_defer_until_training_end=True）
- **FR-3**: FC-SB增量实验配置生成
  - G0基线（无FC-SB推理开关）
  - G1: +Fiber Velocity Projection
  - G2: +Base Locking
  - G3: +Fiber SDE Noise (σ=0.04)
  - G4: +Fiber-Only Endpoint
  - G5: Full FC-SB (G1+G2+G3+G4, σ=0.06)
  - G6: Full FC-SB σ=0.08（FC.md魔法阈值）
  - G7: Full FC-SB + curriculum sigma
- **FR-4**: 远程实验批量运行脚本
  - 顺序运行所有实验，错误处理（单个失败不中断）
  - 完整日志捕获（保存到exp/625_fc_sb/&lt;name&gt;/train.log）
  - 训练完成后自动运行统一评估
  - 实验结果汇总到CSV
- **FR-5**: 结果分析与对比脚本
  - 自动提取所有实验的metrics
  - 生成帕累托前沿图
  - 与历史最佳结果对比
  - 识别最优配置

## Non-Functional Requirements
- **NFR-1**: 单实验训练显存≤10.8GB（12GB卡留安全余量）
- **NFR-2**: 0训练成本推理扫描阶段&lt;2小时完成
- **NFR-3**: 所有训练实验3 epochs内完成，总训练时间≤12小时
- **NFR-4**: 实验配置可复现（固定seed=42）
- **NFR-5**: 所有路径自动适配远程WSL（F盘→/mnt/i/）

## Constraints
- **Technical**: Python/PyTorch现有代码库，不引入新依赖
- **Business**: 单卡RTX 3060 12GB，需要在~24小时窗口内完成
- **Dependencies**: 现有VAE latent缓存、DINO pairing缓存必须在I盘可用
- **Hard Constraint**: WFI必须<0.40才能通过白化验收

## Assumptions
- 远程服务器I盘已挂载wikiart_distinct5_samam_512_latents_ema数据集
- DINO pairing缓存已构建完成
- eval_cache（HF CLIP模型）已缓存
- 现有最佳checkpoint（如E4-long或620推荐配置checkpoint）可在远程找到或快速训练
- num_steps=12（NFE）对FC-SB推理足够

## Acceptance Criteria

### AC-1: 推理期σ扫描完成
- **Given**: 可用的620推荐配置checkpoint
- **When**: 运行推理参数扫描脚本
- **Then**: 得到至少7个σ值（0.0-0.12）的clip_style/LPIPS/WFI数据点
- **Verification**: `programmatic`
- **Notes**: σ=0.0应接近baseline，σ=0.08应接近或超过0.71/0.34

### AC-2: 干净基线G0训练完成
- **Given**: 正确的远程配置
- **When**: 训练G0基线3 epochs
- **Then**: clip_style≈0.70±0.01, LPIPS≈0.34±0.02, WFI<0.40
- **Verification**: `programmatic`
- **Notes**: 这是所有后续实验的对照基准

### AC-3: FC-SB四个机制独立贡献量化
- **Given**: G0基线checkpoint
- **When**: 分别开启G1-G4的FC-SB推理开关
- **Then**: 每个机制的Δclip_style和ΔLPIPS可量化，且至少一个机制带来正向增益（clip_style+LPIPS综合改善）
- **Verification**: `programmatic`
- **Notes**: 重点关注G3 (Fiber SDE Noise)，这是FC.md的核心

### AC-4: Full FC-SB组合效果
- **Given**: G5-G7配置checkpoint
- **When**: 评估Full FC-SB组合
- **Then**: 达到铜牌标准clip_style>0.72且LPIPS<0.40
- **Verification**: `programmatic`
- **Notes**: 如果σ=0.08+curriculum能达到0.73/0.35就是金牌

### AC-5: 实验结果可复现
- **Given**: 所有配置文件和脚本
- **When**: 重新运行任意实验
- **Then**: metrics差异<0.005（随机性导致的波动）
- **Verification**: `programmatic`

### AC-6: 帕累托前沿更新
- **Given**: 所有新实验数据
- **When**: 与历史17000+实验点对比
- **Then**: 至少1个新数据点位于现有帕累托前沿上方（dominates现有最佳）
- **Verification**: `human-judgment` + `programmatic`

## Open Questions
- [ ] 远程服务器当前是否有可用的620推荐配置checkpoint？还是需要从0训练？
- [ ] 之前训练的fc_sb_sigma04 (E2, clip_style=0.708, LPIPS=0.540)的checkpoint是否还在？LPIPS=0.540太高，可能模型已坏
- [ ] Fiber-Only Endpoint在训练时是否需要特殊处理？还是只在推理时投影即可？
- [ ] wavelet lowpass是否真的比avg_pool5x5好？需要对比验证
- [ ] CFG外推（cfg_target_scale>1.0）与FC-SB SDE噪声是否冲突或协同？
