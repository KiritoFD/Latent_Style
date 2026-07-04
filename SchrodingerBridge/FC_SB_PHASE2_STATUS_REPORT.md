# FC-SB Phase 2 远程实验状态报告
# 检查时间: 2026-06-25 03:54 (Asia/Shanghai)

## 📊 总体状态概览

| 检查项 | 状态 | 详情 |
|--------|------|------|
| **Tmux Session** | ✅ 运行中 | phase2 (创建于 2026-06-25 03:49:00) |
| **实验进程** | ✅ 运行中 | Python/Torch 进程活跃 |
| **当前实验** | p3_remote_10h | FC-SB Kernel=7 实验 |
| **训练进度** | ⏳ 进行中 | Epoch 1/3 完成 |
| **评估状态** | ⏳ 部分完成 | 仅生成图片，无指标文件 |

---

## 🔬 当前实验配置

| 参数 | 值 |
|------|-----|
| **实验名称** | p3_remote_10h |
| **配置名称** | fc_sb_kernel7 |
| **Ablation 标签** | F3: FC-SB kernel=7 (larger fiber projection) |
| **总训练轮数** | 3 epochs |
| **当前进度** | Epoch 1 完成 |
| **Batch Size** | 12 |
| **学习率** | 0.0002 |
| **优化器** | AdamW (fused) |
| **混合精度** | AMP (bf16) |
| **两阶段训练** | ✅ 启用 |
| **数据集风格** | Early_Renaissance, Impressionism, Minimalism, Rococo, Ukiyo_e (5类) |
| **数据路径** | /mnt/i/wikiart_distinct5_samam_512_latents_ema/train |

---

## 📁 文件系统状态

### Checkpoints 目录结构
```
fc_sb_kernel7/checkpoints/
├── epoch_0001.pt          (23 MB) ✅ 已保存
├── config.json            (22 KB)  ✅ 配置文件
├── full_eval/
│   └── epoch_0001/
│       └── images/        ⏳ 仅图片，无指标
├── logs/                  📝 训练日志目录
└── src/                   💻 源代码副本
```

### 磁盘使用情况
| 目录 | 大小 |
|------|------|
| fc_sb_kernel7/ | 24 MB |

---

## 📈 训练与评估详情

### 已完成项目
| 项目 | 状态 | 说明 |
|------|------|------|
| Epoch 1 Checkpoint | ✅ 完成 | epoch_0001.pt (23MB) |
| Epoch 1 图片生成 | ✅ 完成 | full_eval/epoch_0001/images/ |
| 配置文件保存 | ✅ 完成 | config.json |

### 待完成/进行中
| 项目 | 状态 | 预期输出 |
|------|------|----------|
| Epoch 1 指标计算 | ⏳ 可能进行中 | clip_style, LPIPS 等 |
| round2_convergence.json | ❌ 不存在 | 聚合指标文件 |
| Epoch 2 训练 | ⏳ 待开始 | 下一个 checkpoint |
| Epoch 3 训练 | ❌ 未开始 | 最终 checkpoint |

---

## ⚠️ 注意事项

1. **实验刚启动**: 该实验仅在 ~5 分钟前启动 (03:49:00)，目前处于早期阶段

2. **评估未完成**: 
   - full_eval/epoch_0001/ 目录只包含 images 子目录
   - 未找到 summary/metrics/convergence JSON 文件
   - 可能评估脚本还在运行或配置为延迟评估

3. **无失败记录**: 
   - 未发现 error log 或 failed 标记文件
   - 进程正常运行中

4. **资源占用**:
   - 实验目录仅 24MB（早期阶段正常）
   - 单个 checkpoint 23MB

---

## 🎯 预期时间线（基于配置推测）

基于配置参数估算：
- **总 epochs**: 3
- **每 epoch 评估**: full_eval_each_epoch=true
- **预计完成时间**: 取决于数据集大小和 GPU 性能

---

## 📋 建议操作

1. **等待完成**: 实验刚启动，建议等待更长时间后再次检查
2. **监控日志**: 可查看 tmux session 实时输出了解训练进展
3. **检查评估**: 如需立即获取指标，可检查是否需要手动触发评估

---

**报告生成时间**: 2026-06-25 03:54:33 (UTC+8)
**远程服务器**: administrator@100.115.18.62:2222
**项目路径**: /home/xy/Latent_Style/SchrodingerBridge
