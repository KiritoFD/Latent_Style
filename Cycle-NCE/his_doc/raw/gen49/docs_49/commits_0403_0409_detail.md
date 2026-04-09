# Commit 详细记录 (04-03 → 04-09)

> 生成时间: 2026-04-09 04:15 CST
> 来源: G:\GitHub\Latent_Style\.git
> 记录范围: Cycle-NCE/src/ 目录下的所有变更

---

## Commit 1: 984d01415 (2026-04-08)
**消息**: Style_Clip达到0.72 同时1-lpips=0.5 全面跑通占优

### 变更文件 (7 files, +1846 lines)
- M Cycle-NCE/src/batch_distill_full_eval.py (+2/-1)
- A Cycle-NCE/src/chess.csv (+41)
- A Cycle-NCE/src/chess/chess_01_baseline/config.json (+153)
- A Cycle-NCE/src/chess/chess_01_baseline/dataset.py (+266)
- A Cycle-NCE/src/chess/chess_01_baseline/full_eval/epoch_0030/metrics.csv (+751)
- A Cycle-NCE/src/chess/chess_01_baseline/full_eval/epoch_0030/summary.json (+634)
- A Cycle-NCE/src/chess/chess_01_baseline/full_eval/epoch_0030/summary_grid.png (2.5MB)

### 实验数据 (chess_01_baseline, Epoch 30)
| 风格 | clip_style | clip_content |
|------|------------|--------------|
| photo | 0.7962 | 0.8294 |
| 平均 | ~0.72 | ~0.65 |

### 关键变更说明
- 创建了新的 chess 系列实验
- Style_Clip 首次达到 0.72
- 1-lpips = 0.5 (内容保留优秀)

---

## Commit 2: 62f2f3833 (2026-04-07)
**消息**: base01都出现了棋盘格和黑框，还是数值问题，改chess系列10个实验，看看是不是位置编码的问题

### 变更文件 (64 files, +19974 lines)
- M Cycle-NCE/src/ablate.py (+211/-???)
- A 10+ config 文件 (config_01 到 config_16)
- A base_01_idt_08/ (完整实验目录)
- A base_02_idt_12/ (完整实验目录)
- A Cycle-NCE/src/base.bat, chess.bat
- M Cycle-NCE/src/model.py (+18/-?)

### 实验数据 (base_01_idt_08, Epoch 80)
| 指标 | 数值 |
|------|------|
| clip_style | 0.6867 |
| clip_content | 0.7371 |

### 关键变更说明
- 创建了 16 个消融实验配置 (config_01 到 config_16)
- 测试不同位置编码 (pos_emb)、idt 权重、patch 大小等
- **问题发现**: 棋盘格和黑框问题 -> 怀疑是位置编码问题
- 模型代码有 18 行修改

---

## Commit 3: d7ee72f17 (2026-04-07)
**消息**: 较好的变化，结构需要加强

### 变更文件 (3 files, +161 lines)
- A Cycle-NCE/src/Layer-Norm-repulse/config.json (+153)
- M Cycle-NCE/src/Layer-Norm.json (+2/-1)
- A Cycle-NCE/src/rep47.csv (+7)

### 关键变更说明
- 创建 Layer-Norm-repulse 实验
- 修改 Layer-Norm 配置
- 新增 rep47.csv 记录

---

## Commit 4: 97de16534 (2026-04-07)
**消息**: 开始学到东西，低idt导致视觉效果不好，但是确实有区别了

### 变更文件 (20+ files)
- A Cycle-NCE/src/Layer-Norm-idt_schedule/ (完整实验目录)
- 包含 Epoch 40/80/100 的完整评估数据

### 关键变更说明
- 引入 idt_schedule (identity loss 调度)
- 低 idt 导致视觉效果不好，但有区别
- 说明模型开始学到东西

---

## Commit 5: 61ae584de (2026-04-07)
**消息**: 回滚后可以正常训练

### 变更文件 (15+ files)
- A Cycle-NCE/src/46_in-idt/ (完整实验目录)
- M Cycle-NCE/src/batch_distill_full_eval.py

### 关键变更说明
- 创建 46_in-idt 实验
- 回滚后训练恢复正常

---

## Commit 6: 3ad9977e8 (2026-04-06)
**消息**: idt loss改用IN只保留内容轮廓

### 变更文件 (20+ files)
- M Cycle-NCE/src/ablate.py
- A config_00_l1_mean_filter.json 到 config_07_edge_rebel.json (8个新配置)
- M Cycle-NCE/src/config_in-idt.json
- M Cycle-NCE/src/config_repulse.json
- M Cycle-NCE/src/losses.py
- M Cycle-NCE/src/model.py
- M Cycle-NCE/src/run.py
- M Cycle-NCE/src/trainer.py

### 关键变更说明
- **核心变更**: idt loss 从 GN 改为 IN (Instance Norm)
- 只保留内容轮廓，不保留纹理
- 这是重要的架构调整

---

## Commit 7: 486f9cc09 (2026-04-06)
**消息**: 去掉SWD的滤波

### 变更文件 (30+ files)
- M Cycle-NCE/src/46.bat
- M Cycle-NCE/src/ablate.py
- M Cycle-NCE/src/batch_distill_full_eval.py
- A bench_onnx_batch4.py
- M 多个 config 文件 (config_00 到 config_09)
- M Cycle-NCE/src/dataset.py
- M Cycle-NCE/src/distill_cartridge.py
- A export_onnx_batch4.py
- A infer_manual_parallel.py
- M Cycle-NCE/src/losses.py
- M Cycle-NCE/src/model.py
- M Cycle-NCE/src/prob.py
- M Cycle-NCE/src/run.py

### 关键变更说明
- **去掉 SWD 的滤波** - 这是重要的损失函数调整
- 新增推理脚本 (infer_manual_parallel.py, export_onnx_batch4.py)
- 大量配置调整

---

## Commit 8: 28e6d0719 (2026-04-06)
**消息**: 结构调整，bottle neck上面做局部涂色，和output做skip connection；skip上面用IN，只保留线稿；decoder上面用小patch+AdaGN做纹理，蒸馏中带上了color_map

### 变更文件 (50+ files)
- A Cycle-NCE/src/45/45/45_01_golden_funnel/ (完整实验目录)
- 包含 Epoch 30 完整评估数据
- 包含 ma_probe (middle activation probe) 分析

### 关键变更说明 (重大架构调整)
1. **Bottle neck 局部涂色**: 在瓶颈层做局部纹理注入
2. **Skip Connection 改动**: 
   - 和 output 做 skip connection
   - skip 上用 IN，只保留线稿
3. **Decoder 改动**:
   - 用小 patch + AdaGN 做纹理
   - 蒸馏中带上 color_map

### 实验数据 (45_01_golden_funnel, Epoch 30)
- 这是 golden_funnel 架构的首次实验
- 包含 ma_probe 分析（中间激活探测）

---

## Commit 9: 92d7012c3 (2026-04-05)
**消息**: 用ma观测，调整结构

### 变更文件 (40+ files)
- D 多个 42_A 系列实验目录 (删除)
- A Cycle-NCE/src/45.bat
- M Cycle-NCE/src/ablate.py
- M Cycle-NCE/src/config.json
- A config_01_golden_funnel.json
- R (重命名) 42_A07_NoSkip_Conv2_LR3e4/config.json -> Cycle-NCE/src/config_01_nuke.json
- A config_02_high_temp_macro.json
- A config_02_naked_fusion.json
- A Cycle-NCE/src/45/ (新实验)

### 关键变更说明
- **删除 42_A 系列实验** - 说明该方向被放弃
- **新增 45 系列实验** - 新的研究方向
- 用 MA (middle activation) 观测来调整结构

---

## Commit 10: 6b34f2209 (2026-04-04)
**消息**: paper&ppt

### 变更文件 (2 files)
- M Cycle-NCE/src/ablate.py
- M Cycle-NCE/src/model.py

### 关键变更说明
- 论文和 PPT 相关修改
- 模型代码调整

---

## 总结: 10 天内的关键变化

### 架构层面
1. **IDT Loss 改进**: 从 GN 改为 IN (只保留内容轮廓)
2. **去除 SWD 滤波**: 简化损失函数
3. **结构调整**: 
   - Bottle neck 局部涂色
   - Skip connection 用 IN
   - Decoder 用小 patch + AdaGN
4. **新增 45/46 系列实验**: 替代原有的 42 系列

### 实验层面
- 创建了 30+ 个新实验目录
- 测试了 16+ 个消融配置
- 引入 ma_probe (中间激活探测) 进行分析

### 评估指标
- chess_01: Style 0.72, Content ~0.65
- base_01_idt_08: Style 0.6867, Content 0.7371

