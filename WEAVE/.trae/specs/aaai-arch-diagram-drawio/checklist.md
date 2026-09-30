# Checklist — AAAI 主架构图 draw.io 绘制

## 代码分析验证
- [ ] 所有模块名称与代码中的类/函数名一致
- [ ] 所有维度标注与代码中的张量形状一致
- [ ] 数据流方向与 forward() 方法执行顺序一致
- [ ] 训练路径与 spectral_losses620.py 的 compute() 方法一致

## 架构图完整性
- [ ] 包含所有核心模块：DWT, Style Conditioner, Backbone, Velocity Heads, iDWT, Endpoint AdaIN
- [ ] 包含训练路径：Interpolation, Spectral FM Loss
- [ ] 包含推理路径：ODE Integration Loop
- [ ] 展开一个 Backbone Block 的内部细节（AdaLN, Self-Attn, Cross-Attn, FFN）

## 视觉质量
- [ ] 节点大小统一，间距合理
- [ ] 箭头使用正交连线（orthogonalEdgeStyle）
- [ ] 配色符合学术论文标准（柔和色块，不刺眼）
- [ ] 数学符号使用正确（z₀, v_LL, α, β, μ, σ 等）
- [ ] 容器分组清晰（Backbone, Training Objective）

## 子 Agent 评审
- [ ] 第一轮评审完成，收集修改意见
- [ ] 根据评审意见修改 XML
- [ ] 第二轮评审通过（架构准确性 + 视觉美观性）
- [ ] 必要时进行第三轮评审

## 最终输出
- [ ] draw.io XML 文件生成成功
- [ ] 文档 docs/630/arch_diagram_design.md 创建
- [ ] git commit 提交所有变更
