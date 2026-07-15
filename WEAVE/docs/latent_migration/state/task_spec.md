# Task: SAMST & SaMam Latent Migration + Comprehensive Metrics Table

**Created**: 2026-07-04 13:20 (Asia/Shanghai)
**Owner**: TRAE agent
**Framework**: Deli_AutoResearch
**Mode**: Zero-interaction, autonomous

## 1. Goal

1. 尝试将 SAMST 和 SaMam 两个 baseline 迁移到 latent 空间（4×16×16，256 输入 VAE 编码）
2. 生成巨长总表：512 和 256 同行两列，含 clip-s / clip-t / lpips / musiq / art-fid

## 2. Scope

### 2.1 Latent 迁移
- **SAMST-latent**: 修改 TransformerNet 输入/输出通道 3→4，调整 kernel_size 和 stride 以适配 4×16×16
- **SaMam-latent**: 修改 patch embedding 和 unpatchify 层以适配 4 通道 latent
- 训练数据: `I:/wikiart_distinct5_samam_512_latent256/train`（5×1000 张 .pt VAE latent packed）
- 测试集: `I:/wikiart_distinct5_samam_512_classview/test`

### 2.2 总表指标
- **CLIP-S**: 风格相似度（已有部分数据）
- **CLIP-T**: 内容相似度（text-image CLIP，部分已有）
- **LPIPS**: 内容保留（已有）
- **MUSIQ**: 多尺度图像质量（需实现/查找）
- **ART-FID**: 艺术 FID（需实现/查找）

### 2.3 方法列表（总表行）
- AdaIN (train-free)
- WCT (train-free)
- SAMST (pixel)
- SaMam (pixel)
- SAMST-latent (新)
- SaMam-latent (新)
- Our pixel256 / pixel512
- Our latent256 / latent512 (spectral_ode 主线)
- Identity (baseline)
- 其他 512 baseline（SDEdit, StyleID, CUT, SeeDream, SD-Turbo）

## 3. Success Criteria

1. SAMST-latent 和 SaMam-latent 至少有一个完成训练+评估
2. 若迁移确实不可行，记录详细原因（架构冲突、训练不收敛等）
3. 总表包含所有可用方法的 5 个指标（512 和 256 两列）
4. 缺失指标标注 N/A 并说明原因

## 4. Hard Constraints

- 显存 ≤ 7G（评估）/ ≤ 11G（训练）
- 远程: `ssh -p 2222 administrator@100.115.18.62`，RTX 3060 12GB
- WSL venv: `/home/xy/venvs/samam312/bin/python`
- 命令加 30s timeout
- 训练 Patience=2, max=10

## 5. Milestones

| M | 内容 | 状态 |
|---|---|---|
| M1_explore | 探索 musiq/art-fid 脚本 + baseline 架构 + latent 数据格式 | pending |
| M2_samst_latent | SAMST 架构修改 + 训练 + 评估 | pending |
| M3_samam_latent | SaMam 架构修改 + 训练 + 评估 | pending |
| M4_metrics_impl | musiq / art-fid 评估脚本实现 | pending |
| M5_eval_all | 评估所有方法的缺失指标 | pending |
| M6_final_table | 生成巨长总表 | pending |
