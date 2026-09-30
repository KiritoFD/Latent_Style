# AAAI 2027 论文 Strong Accept 重构规范

## Why
当前论文 v3 的叙事方向错误：正文中暴露了 645 次消融、22 个被移除模块等"过程问题"，这些是研发过程，不应在正文出现任何一点，最多放补充材料。正文应从目前模型的难处出发，提出方案，展示结果。此外，VAE latent 空间的使用不只是降低计算开销，它本身就是更好的语义表示，有利于风格表征——这一动机需要明确阐述。用户建议尝试在像素空间跑 SFM 做对比，以实证 VAE latent 空间的优越性。

## 叙事核心（不可妥协）
标题：**Affordable Real Style Transfer: Training Spectral Flow Matching on an RTX 3060 in Minutes**

叙事逻辑（从难处出发，提出方案，展示结果）：
1. **难处（The Difficulty）**：当前无监督风格迁移方法在消费级硬件上面临困难。Diffusion 模型（CSGO 等）需要 A100 集群和 21 万配对数据。CUT、SaMam 等消费级方法训练慢，且 SaMam 的 CLIP-S=0.5816 低于 identity baseline 0.6933——它实际在做内容重建而非风格迁移。
2. **方案（The Method）**：Spectral Flow Matching (SFM)。在 SDXL VAE latent 空间操作——VAE latent 不只降低计算开销，其压缩的语义表示本身就更有利于风格表征（相比像素空间的低层纹理）。在 latent 上应用 Haar DWT：LL（内容锚定）+ 高频子带（LH/HL/HH，笔触载体）。Base locking 锁定 LL 速度场，保 LPIPS。Fiber flow 在高频子带上做流匹配，100% 算力学习笔触。EOTA 在轨迹末端注入风格统计矩。
3. **结果（The Result）**：903K 参数，3 分钟训练（RTX 3060 12GB），CLIP-S=0.7213，LPIPS=0.2868。比 SaMam 快 141×，CLIP-S 高 0.14。真实笔触迁移，非颜色滤镜。

## What Changes

### BREAKING: 正文删除所有"过程问题"
- 从正文（§1, §4）完全删除 645 次消融、22 个被移除模块的任何提及
- 这些内容只放 supplement §B 作为补充材料，且语调客观（"supplementary ablation audit"），非"邀功"
- 正文 §4.4 Ablation 只保留方法组件消融（DWT 级数、base locking、EOTA、stochastic routing），不提"减法消融"

### BREAKING: 叙事重构（从难处出发）
- 重写 §1 Introduction：从消费级硬件上风格迁移的难处出发 → 提出 SFM 方案 → 展示硬结果
- 重写 Abstract：第一句直接亮出最终结果
- 删除任何"邀功"语调（"史无前例"、"长达半年"、"645 次"等）

### BREAKING: VAE latent 动机阐述
- 在 §3.1 Problem Setup 明确阐述：选择 VAE latent 空间不仅为降低计算开销，更因为 VAE 的压缩表示提供更好的语义抽象，有利于风格表征（相比像素空间的低层纹理）
- 新增 Task：像素空间对比实验（若可行）

### 新增：像素空间对比实验
- 在像素空间跑 SFM（相同架构，输入改为像素而非 latent），对比 CLIP-S/LPIPS
- 若像素空间性能显著低于 latent 空间，则实证 VAE latent 的语义优势
- 结果放 §4.6 或 Table 1 的一行

### Supplement 术语全面切换
- 重写 `supplement_aaai2027.tex`：从 LBM 切换到 SFM 术语
- §A 方法详述
- §B 减法消融审计（645 配置、22 模块）——客观补充材料，非邀功
- §C 3-seed 稳定性
- §D 失败案例

### Baseline 数值口径与统计显著性
- 核实 Table 1 所有 baseline 数值与 `docs/baseline/README.md` v5 一致
- 为 SFM/SaMam/Seedream 添加 bootstrap 95% CI
- 为 SFM 补 3-seed std
- Table 1 脚注声明口径

### 语言全文审查
- 每句 ≤25 词，主动语态，无嵌套从句
- 无内部代号（LBM, FC-SB 等）
- 术语首次出现有清晰定义

### 编译验证与提交
- 主文：7 正文 + 2 引文/checklist，0 errors
- supplement：独立编译 0 errors
- git commit

## Impact
- Affected code:
  - `aaai2027_v2/paper.tex`（§1/Abstract 重写、删除过程问题、VAE 动机阐述、Table 1 加 CI+std、像素对比）
  - `aaai2027_v2/supplement_aaai2027.tex`（完全重写）
  - `aaai2027_v2/refs.bib`（若需新引文）
  - `aaai2027_v2/distinct5_idt_bootstrap_extended.csv`（只读数据源）
  - `docs/baseline/README.md`（只读数据源）
  - 可能新增像素空间实验配置和日志

## ADDED Requirements

### Requirement: 正文无过程问题
系统 SHALL 确保正文（§1-§6）中不出现任何"过程问题"：不提 645 次消融、不提 22 个被移除模块、不提"减法消融"、不提"长达半年"。这些内容只在 supplement §B 作为客观补充材料出现。

#### Scenario: 审稿人阅读正文
- **WHEN** 审稿人阅读 §1-§6
- **THEN** 看到的是"难处→方案→结果"，无任何研发过程的自我表白

### Requirement: 从难处出发叙事
系统 SHALL 在 §1 从消费级硬件上风格迁移的难处出发：diffusion 太重（A100）+ CUT/SaMam 慢 + SaMam 不如 identity。然后提出 SFM 方案，展示硬结果。

#### Scenario: 审稿人理解动机
- **WHEN** 审稿人阅读 §1 第一段
- **THEN** 清楚理解消费级硬件上风格迁移的具体难处，而非自我表白

### Requirement: VAE latent 语义优势阐述
系统 SHALL 在 §3.1 明确阐述：选择 VAE latent 空间不仅为降低计算开销，更因为 VAE 压缩表示提供更好的语义抽象，有利于风格表征。

#### Scenario: 审稿人质疑为何用 VAE
- **WHEN** 审稿人质疑 VAE latent 是否只是为降低开销
- **THEN** §3.1 明确回答 VAE latent 的语义优势，且 §4 有像素空间对比实验佐证

### Requirement: 像素空间对比实验
系统 SHALL 尝试在像素空间跑 SFM（相同架构，输入改为像素），对比 latent 空间的 CLIP-S/LPIPS，以实证 VAE latent 的优越性。若可行，结果放 §4.6 或 Table 1。

#### Scenario: 审稿人评估 VAE 必要性
- **WHEN** 审稿人判断 VAE latent 是否真的比像素空间好
- **THEN** §4 提供像素 vs latent 的直接对比数据

### Requirement: 反差感 Abstract
系统 SHALL 在 Abstract 第一句直接亮出最终结果（3 分钟、RTX 3060、CLIP-S 0.7213），配合"真实笔触迁移而非颜色滤镜"。

### Requirement: 语言简洁性
系统 SHALL 确保全文每句 ≤25 词，主动语态，无内部代号，术语首次出现有定义。

### Requirement: Supplement 术语一致性
系统 SHALL 保持 supplement 与主文术语一致，使用 SFM 叙事，不用 LBM 旧术语。

### Requirement: Baseline 数值口径声明
系统 SHALL 在 Table 1 脚注声明口径，数值可追溯到 docs/baseline/README.md v5。

### Requirement: 统计显著性
系统 SHALL 在 Table 1 为 SFM/SaMam/Seedream 添加 bootstrap 95% CI。

### Requirement: 多 seed 稳定性
系统 SHALL 为 SFM 主结果提供 3-seed mean±std，std < 0.005。

## MODIFIED Requirements

### Requirement: AAAI Reproducibility Checklist
更新第 8 项：声明 "3-seed mean±std reported in Table 1"。

## REMOVED Requirements
（无删除项）
