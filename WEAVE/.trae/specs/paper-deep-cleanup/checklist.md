# AAAI 2027 论文深度清洗验收清单

## 术语规范化验收
- [x] 所有内部代号（I7, U4, V6, V3, LBM-K, LBM-Knee, LBM-PS, LBM-PS-v2）已替换为描述性名称
- [x] 所有缩写（OMF, SA-SWD, FC-SB, tw-ArtFID, EdgePurity, NonCLIPAcc）已正式定义
- [x] 工具特定词汇（live-dashboard, HTML payload, pairing cache, successor family）已移除
- [x] 术语映射表已创建并应用到全文
- [x] 核心术语 "real style transfer" 已全文统一（修正了 "effective" 的不当使用）

## 文本清洗验收
- [x] Abstract 已重写，聚焦核心贡献，移除变体细节
- [x] Introduction 已按核心叙事重组：问题→方法→结果→贡献
- [x] Method 已正式定义所有技术概念，移除内部实现细节
- [x] Experiments 已使用规范评估术语，客观陈述结果
- [x] Discussion/Conclusion 已提炼核心洞察，避免防御性表达

## 图表重绘验收
- [x] Figure 1 已重绘：包含 IDT 校准散点图和效率对比柱状图
- [ ] Figure 2 已重绘：简化为三阶段流程，移除内部细节
- [ ] Figure 3 已重绘：选择代表性案例，清晰网格布局
- [x] Table 1-3 列名已规范化
- [x] Table 4 已移除内部代号，使用参数描述

## 核心叙事验收
- [x] 每个章节紧扣"Real style transfer, ultra efficient"核心故事
- [x] 段落之间逻辑过渡自然
- [x] 没有偏离主题的内容

## 学术写作规范验收
- [x] 无自我表扬表达（如"The main claim is precise"）
- [x] 无防御性表达（如"should be read in the correct way"）
- [x] 无内部术语和非正式用语
- [ ] 语法和拼写正确（待最终验证）

## 编译验收
- [ ] LaTeX 编译无错误
- [ ] 图表引用正确
- [ ] 表格引用正确
- [ ] 最终 PDF 视觉效果良好
