# 实验目录详细记录 (04-03 → 04-09 新增)

> 生成时间: 2026-04-09
> 记录范围: 所有新增实验目录及其评估数据

> 说明: 这篇文档保留“目录层面的总清单”角色。  
> 如果要看更可靠的配置-结果对照，请优先读：
> - `exp_schedule.md`
> - `exp_46.md`
> - `exp_Aline120.md`
> - `exp_chess.md`
> - `exp_freq.md`
> - `exp_repulse.md`

---

## 0. 这篇文档的定位

这篇文档不是主时间线，也不是最细的实验台账。  
它的作用更接近“目录地图”：

1. 哪些实验目录在这几天出现了
2. 哪些目录已经有 `full_eval`
3. 哪些目录只是占位或半成品

所以这篇文档适合用来找材料位置，不适合单独承担结论文档的角色。

## 1. 46 系列实验 (7个)

| 目录名 | 有full_eval | 评估文件位置 |
|--------|-------------|--------------|
| 46 | ❌ | - |
| 46_08_natural_spectrum_reset | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\46_08_natural_spectrum_reset\full_eval\epoch_0080_tokenized_distill_epochs200\summary.json |
| 46_09_real_holy_grail | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\46_09_real_holy_grail\full_eval\epoch_0040_tokenized_distill_epochs200\summary.json |
| 46_in-idt | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\46_in-idt\full_eval\epoch_0040_tokenized_distill_epochs200\summary.json |
| 46_Layer-Norm | ❌ | - |
| 46_repulse | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\46_repulse\full_eval\epoch_0040_tokenized_distill_epochs200\summary.json |
| 46_splash | ✅ | - |

### 实验配置说明
- **46**: 基础46系列实验
- **46_08_natural_spectrum_reset**: 自然频谱重置
- **46_09_real_holy_grail**: 圣杯配置
- **46_in-idt**: 使用新的IN-only IDT Loss (见mode_in_idt_loss.md)
- **46_Layer-Norm**: Layer Norm配置
- **46_repulse**: Soft Repulsive Loss
- **46_splash**: Splash配置

---

## 2. freq 系列实验 (9个)

| 目录名 | 有full_eval | 评估文件位置 |
|--------|-------------|--------------|
| freq | ❌ | - |
| freq_01_conservative_baseline | ❌ | - |
| freq_02_brush_frenzy | ❌ | - |
| freq_03_large_view_awareness | ❌ | - |
| freq_04_no_idt_abyss | ❌ | - |
| freq_05_idt_iron_fist | ❌ | - |
| freq_06_yuv_dictatorship | ❌ | - |
| freq_07_remove_blast_wall | ❌ | - |
| freq_08_extreme_asymmetry | ❌ | - |

### 实验配置说明
- **freq**: 频域实验基础目录
- **freq_01_conservative_baseline**: 保守基线
- **freq_02_brush_frenzy**: 笔触狂野
- **freq_03_large_view_awareness**: 大视角感知
- **freq_04_no_idt_abyss**: 无IDT深坑
- **freq_05_idt_iron_fist**: 铁拳IDT
- **freq_06_yuv_dictatorship**: YUV独裁
- **freq_07_remove_blast_wall**: 移除blast墙
- **freq_08_extreme_asymmetry**: 极端不对称

---

## 3. 45 系列实验 (2个)

| 目录名 | 有full_eval | 评估文件位置 |
|--------|-------------|--------------|

### 实验配置说明
- **45_01_golden_funnel**: 金色漏斗架构首次实验
- 包含ma_probe分析（中间激活探测）

---

## 4. 其他新增实验

| 目录名 | 有full_eval | 评估文件位置 |
|--------|-------------|--------------|
| Aline120 | ❌ | - |
| in-idt | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\in-idt\full_eval\epoch_0040_tokenized_distill_epochs200\summary.json |
| Layer-Norm-repulse | ✅ | G:\GitHub\Latent_Style\Cycle-NCE\Layer-Norm-repulse\full_eval\epoch_0100_tokenized_distill_epochs200\summary.json |
| repuls | ❌ | - |
| splash | ✅ | - |
| video | ❌ | - |
| fewshot_ukiyoe_runs | ❌ | - |

### 实验配置说明
- **in-idt**: 独立验证IN-only IDT Loss的版本
- **chess_01_baseline**: chess系列实验，Style=0.72突破
- **base_01_idt_08**: 消融实验，Style=0.6867, Content=0.7371
- **Aline120**: 待确认
- **video**: 视频相关实验

---

## 📊 已获取评估数据的实验

### chess_01_baseline (Epoch 30)
| 风格 | clip_style | clip_content |
|------|------------|--------------|
| photo | 0.7962 | 0.8294 |
| 平均 | ~0.72 | ~0.65 |

### base_01_idt_08 (Epoch 80)
| 指标 | 数值 |
|------|------|
| clip_style | 0.6867 |
| clip_content | 0.7371 |

### in-idt (Epoch 40)
| 指标 | 数值 |
|------|------|
| clip_style | 0.6787 |
| clip_content | 0.8464 |

---

*文档生成时间: 2026-04-09*
