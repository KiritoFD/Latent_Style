# AAAI 2027 论文深度清洗重写任务列表

## 阶段一：术语规范化与文本清洗

### Task 1: 创建术语映射表
- [x] 1.1 整理所有内部代号（I7, U4, V6, V3, LBM-K, LBM-Knee, LBM-PS, LBM-PS-v2）
- [x] 1.2 为每个代号定义规范学术表达
- [x] 1.3 整理工具特定词汇（live-dashboard, HTML payload, pairing cache 等）
- [x] 1.4 定义替代的正式表达

### Task 2: 重写 Abstract
- [x] 2.1 移除变体细节（LBM-K, LBM-Knee, LBM-PS-v2 等）
- [x] 2.2 聚焦核心贡献：IDT 校准 + LBM 方法 + 效率优势
- [x] 2.3 使用规范术语
- [x] 2.4 确保逻辑链清晰：问题 → 方法 → 结果

### Task 3: 重写 Introduction
- [x] 3.1 按核心叙事组织：风格迁移目标 → 之前方法 → IDT 问题 → LBM 方法 → 理论支撑 → 未来启发
- [x] 3.2 移除所有内部代号
- [x] 3.3 避免自我表扬和防御性表达
- [x] 3.4 确保每段紧扣中心故事

### Task 4: 重写 Method
- [x] 4.1 正式定义所有技术概念（endpoint transport, kinetic regularization, terminal distribution matching）
- [x] 4.2 移除内部实现细节（pairing cache, successor family, Stokes coefficient）
- [x] 4.3 清晰阐述三个定理的作用
- [x] 4.4 使用规范数学符号和术语

### Task 5: 重写 Experiments
- [x] 5.1 规范化评估术语（IDT calibration, style gain, content preservation）
- [x] 5.2 移除内部代号，使用描述性名称
- [x] 5.3 清晰阐述实验设计逻辑
- [x] 5.4 客观陈述结果，避免过度解读

### Task 6: 重写 Discussion 和 Conclusion
- [x] 6.1 提炼核心洞察
- [x] 6.2 直接陈述局限性，避免防御性表达
- [x] 6.3 总结贡献，避免自我表扬
- [x] 6.4 展望未来方向

## 阶段二：图表重绘（根据视觉质量和核心叙事）

### Task 7: 重绘 Figure 1 (Page 1 summary) - 核心叙事图
- [x] 7.1 设计面板 (a)：IDT 校准散点图
  - x轴：Content Preservation (1-LPIPS) ↑
  - y轴：Style Score (CLIP-S) ↑
  - 标注 IDT 基线（水平虚线）
  - 突出 LBM 操作点（在 IDT 之上）
  - **关键**：使用顶会标准的配色和标注，避免内部代号
- [x] 7.2 设计面板 (b)：效率对比柱状图
  - x轴：Training Time (minutes, log scale)
  - y轴：Method Name
  - 突出 LBM 的分钟级训练优势
  - **关键**：所有方法名称使用规范学术表达
- [x] 7.3 编写生成脚本（gen_fig1_summary.py）
- [x] 7.4 导出高质量 PDF/PNG（300 DPI）
- [x] 7.5 修复 Figure 1 caption（Panel b 描述与实际图不符）- 已更新为描述训练时间对比- 已更新为描述训练时间对比

### Task 8: 重绘 Figure 2 (Framework)
- [ ] 8.1 简化为三阶段流程
  - Stage 1: Style-ID Encoding
  - Stage 2: Latent Transport
  - Stage 3: Training Objectives
- [ ] 8.2 移除所有内部实现细节
- [ ] 8.3 使用清晰的图示和标注
- [ ] 8.4 编写生成脚本或手动设计

### Task 9: 重绘 Figure 3 (Qualitative)
- [ ] 9.1 选择最具代表性的案例
  - 包含 IDT 失败案例
  - 包含 LBM 成功案例
  - 包含与 baseline 的对比
- [ ] 9.2 组织为清晰的网格布局
- [ ] 9.3 添加描述性标注

### Task 10: 规范化所有表格
- [x] 10.1 Table 1: 主要结果（已完成列名规范化）
- [x] 10.2 Table 2: 辅助诊断指标（已完成列名规范化）
- [x] 10.3 Table 3: 训练成本对比（已完成列名规范化）
- [x] 10.4 Table 4: 推理时改进（已移除内部代号，使用参数描述）

## 阶段三：关键问题修复（2026-06-27）

### Task 11: Bootstrap 过程描述补充
- [x] 11.1 在 Line 260 附近添加 Bootstrap resampling 的具体描述
  - 说明 resampling 方法（如 percentile bootstrap）
  - 说明 resampling 次数（如 1000 次）
  - 说明置信区间计算方式
  - 确保可重复性

### Task 12: CycleGAN-256 处理
- [x] 12.1 决策：移除 CycleGAN-256 提及 OR 补充结果
  - 选项 A：移除 Line 188 和 201 的 CycleGAN-256 提及
  - 选项 B：补充 CycleGAN-256 实验结果表格
- [x] 12.2 确保实验设置部分与实际报告结果一致

### Task 13: Line 315 自引用修复
- [x] 13.1 修复 Section 自引用问题
  - 当前：Section~\ref{sec:fcsb} 指向自身
  - 修复：改为 "the preceding analysis" 或具体描述

### Task 14: 全文术语一致性检查
- [x] 14.1 搜索所有内部代号，确保已替换
- [x] 14.2 搜索所有缩写，确保已定义
- [x] 14.3 检查术语使用一致性（已完成 "effective" → "real" 核心术语修正）

### Task 15: 叙事连贯性检查
- [x] 15.1 检查每个章节是否紧扣核心故事
- [x] 15.2 检查段落之间的逻辑过渡
- [x] 15.3 确保没有偏离主题的内容
- [x] 15.4 精简 Abstract 和 Introduction 的重复数据
- [x] 15.5 精简 Experiments 的重复叙述

### Task 16: 学术写作规范检查
- [x] 16.1 检查是否有自我表扬表达
- [x] 16.2 检查是否有防御性表达
- [x] 16.3 检查是否有非正式用语
- [x] 16.4 检查语法和拼写

### Task 17: 编译和最终验证
- [x] 17.1 编译 LaTeX，确保无错误
- [x] 17.2 检查图表引用正确
- [x] 17.3 检查表格引用正确
- [x] 17.4 生成最终 PDF 并检查视觉效果
- [x] 17.5 修复 sec:experiments 未定义引用问题
