# Tasks — AAAI 主架构图 draw.io 绘制（理论驱动版）

## Task 0: 理论研究 — 理解为什么这样设计、为什么 work
- [ ] 0.1 读取 `docs/archive/theory/SpectralODE_Bridge.md` — 官方理论文档
- [ ] 0.2 读取 `docs/72/02_theory.md` — Phase 72 的理论总结
- [ ] 0.3 阅读关键 fog theory 文档：
  - `docs/archive/620/fog/theory/overall_dynamics.md`
  - `docs/archive/620/fog/theory/cross_attn_analysis.md`
  - `docs/archive/620/fog/theory/train_infer_mismatch.md`
- [ ] 0.4 提炼核心洞察：频域解耦、LL 保护、高频风格传输、Endpoint AdaIN、ReLU² Attention
- [ ] 0.5 写出“故事线”：用一段话讲清楚模型 why it works

## Task 1: AAAI 论文风格调研
- [ ] 1.1 读取 `F:\papers\latent_style_cited_20260605\AAAI\2024_huang2024aesfa_AesFA_*.pdf` 的主架构图
- [ ] 1.2 读取 `F:\papers\latent_style_cited_20260605\AAAI\2024_xia2024s2wat_S2WAT_*.pdf` 的主架构图
- [ ] 1.3 读取 `F:\papers\latent_style_cited_20260605\AAAI\2024_zhang2024artbank_ArtBank_*.pdf` 的主架构图
- [ ] 1.4 读取 `F:\papers\latent_style_cited_20260605\AAAI\lancet.pdf` 的主架构图（如相关）
- [ ] 1.5 总结调研结果：
  - 每篇论文主架构图的类型（流程图 / 模块图 / 公式+模块混合）
  - 如何表达 multi-scale / frequency / attention
  - 训练 vs 推理路径如何可视化
  - 配色、字体、箭头风格、图注位置
- [ ] 1.6 输出 `docs/630/aaai_arch_diagram_style_survey.md`

## Task 2: 确定主架构图的故事与草图
- [ ] 2.1 基于 Task 0 的理论洞察，选择 1-2 个核心故事方向
- [ ] 2.2 手绘/文本草图：模块位置、箭头、颜色、标注
- [ ] 2.3 明确每个模块旁边的“一句话理论说明”
- [ ] 2.4 确定配色方案（参考 Task 1 的 AAAI 风格）

## Task 3: 生成 draw.io XML — 第一版
- [ ] 3.1 使用 `open_drawio_xml` MCP 工具生成初始架构图
- [ ] 3.2 图必须体现理论故事，而非代码逐行映射
- [ ] 3.3 使用正交连线（edgeStyle=orthogonalEdgeStyle）和学术配色
- [ ] 3.4 包含图例和核心公式标注

## Task 4: 子 Agent 视觉评审 — 第一轮
- [ ] 4.1 启动 sub-agent 使用视觉能力评审
- [ ] 4.2 评审维度：
  - 理论准确性：是否准确传达了核心洞察
  - 视觉清晰度：节点、箭头、标注是否易读
  - AAAI 风格一致性：与调研的论文图是否风格协调
  - 故事性：是否能在 30 秒内看懂模型 why it works
- [ ] 4.3 收集评审意见，生成修改清单

## Task 5: 迭代优化 — 根据评审修改
- [ ] 5.1 根据评审意见修改 XML
- [ ] 5.2 重新调用 `open_drawio_xml` 生成改进版
- [ ] 5.3 再次启动 sub-agent 评审
- [ ] 5.4 重复 5.1-5.3 直到评审通过（目标：绝对准确 + 漂亮 + 风格一致）

## Task 6: 最终输出与文档
- [ ] 6.1 生成最终版 draw.io XML 文件
- [ ] 6.2 在 `docs/630/` 下创建文档记录：
  - 理论依据
  - AAAI 风格调研结果
  - 设计决策
- [ ] 6.3 git commit 提交所有变更

## Task Dependencies
- Task 1 parallel with Task 0
- Task 2 depends on Task 0 and Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3
- Task 5 depends on Task 4
- Task 6 depends on Task 5
