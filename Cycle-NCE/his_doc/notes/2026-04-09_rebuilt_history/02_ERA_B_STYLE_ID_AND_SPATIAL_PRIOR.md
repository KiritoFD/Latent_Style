# Era B：Style-ID 蒸馏与空间先验

时间范围：2026-02-08 到 2026-02-26  
代表提交：`d916277`, `83ffe10`, `1f818cc`, `adb274a`, `f7b328c`

## 1. 关键转折

`d916277` 的提交信息是这一阶段的总开关：

> 蒸馏把风格放进模型，推理不需要参考图

这意味着项目从“reference-conditioned translator”转向“style-id conditioned latent editor”。

这是整个历史里最重要的结构切换之一。

## 2. 结构上发生了什么

### 2.1 从 `style_ref encoder` 转向 `style_emb`

后续版本逐渐出现并稳定下来的部件：

- `self.style_emb = nn.Embedding(num_styles, style_dim)`
- `self.style_spatial_id_16`

也就是把风格拆成：

- 全局离散 style embedding
- 每个 style 的空间先验图

这两个部件一起替代了早期“必须输入 style_ref”的流程。

### 2.2 map16 / map32 注入逻辑清晰化

`83ffe10` 的提交信息直接写了设计结论：

- `map16` 负责中频大块风格
- `map32` 负责高频笔触

这不是简单调参，而是开始把“风格的频段职责”明确分工。

对应的工程含义是：

- 16x16 主干负责整体风格基调、布局级纹理
- 32x32 高分辨率路径负责细碎笔触和局部纹理

## 3. 这一阶段的核心模块

### 3.1 `style_emb`

作用：

- 用离散 `style_id` 替代 `style_ref` 编码器输出
- 让推理接口更轻
- 让每个风格拥有稳定的全局向量表示

设计意义：

- 降低推理依赖
- 降低“reference 图像内容污染风格表示”的风险

### 3.2 `style_spatial_id_16`

作用：

- 给每个风格一个 16x16 的可学习空间先验
- 在 body 分辨率上直接对 feature 做预注入

从后续代码看，这种设计被长期保留，说明它不是临时 hack，而是稳定有效的骨架。

### 3.3 多 patch SWD

`adb274a` 的提交信息是 `patch 1,3,5`，这说明 SWD 的 patch 尺度已经被当成主控变量。

后来 `EXPERIMENT_RECORD_FULL_DATA.csv` 里 A 系列实验也明确围绕：

- `p5`
- `p7`
- `p11`

去测 patch 尺度影响。

## 4. 这一阶段的 Loss 演化

### 4.1 SWD 开始占据主位

`1f818cc` 给出了非常关键的结论：

- Domain 明显优于 Instance
- 1x1 / 3x3 / 平滑变体都做了对比
- 最优点落在 Domain 1x1, 512 projections 附近

这在项目历史里的含义很大：

- 风格约束从 Gram / moment 的“二阶统计”逻辑，转向分布匹配逻辑
- 并且不是实例级，而是域级更有效

### 4.2 AdaGN 被重新调教

`f7b328c` 的提交信息是“改动 AdaGN，观察到笔触明显变化”。

这说明：

- AdaGN 不只是稳定训练的层
- 它直接决定笔触表现
- 风格质量的关键，不只在 loss，也在调制器本身

## 5. 与实验台账的映射

这一阶段能比较稳地映射到下面几类实验：

- A 系列参数消融组
  - patch / identity / TV 单因素扰动
- 部分 EXP 主线组
  - 如 `exp_1_control`, `exp_2_zero_id`, `exp_3_macro_strokes`
- 早期 SWD 路线实验

在 `EXPERIMENT_RECORD_FULL_DATA.csv` 中，A 系列是很典型的这一时代实验风格：

- 固定主结构
- 单独扫 `p`, `id`, `tv`
- 追踪 style / lpips / fid / art_fid

## 6. 这一阶段得到的几个硬结论

### 6.1 identity 是双刃剑

后续主线中 `exp_S1_zero_id` 的 style 很高，但 LPIPS / FID / art_fid 明显恶化。  
这说明把 identity 完全拿掉，确实能提高风格分数，但会让内容与分布稳定性崩坏。

### 6.2 patch 不是越小越好，也不是越大越好

旧报告里已经明确写过：

- `p5 -> p7 -> p11` 有连续趋势
- patch 尺度影响的是“分布匹配的观测尺度”

所以 patch size 在这个项目里不是纯超参，而是“模型看待风格纹理的粒度”。

### 6.3 map-based spatial prior 是后续所有版本的骨架

哪怕后面加了 texture dictionary、cross-attn、feature attention，这条路也没彻底被丢掉。

## 7. 阶段总结

Era B 的贡献，不是把最终模型做好，而是把项目从“要参考图才能风格化”的实验系统，推进成“有固定 style code / style prior 的可部署架构”。

这是后面所有结构复杂化之前，最重要的一次收敛。

