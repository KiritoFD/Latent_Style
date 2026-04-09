# Era A：Reference-Conditioned Bootstrap

时间范围：2026-01-27 到 2026-02-08  
代表路径：`Thermal/src` -> `NCE_SWD/src` -> `Cycle-NCE/src`  
代表提交：`59cffe2`, `1a8b94b`, `47ed14b`, `ae596d1`

## 1. 这个阶段在解决什么

这个阶段最核心的问题不是“风格是否够强”，而是三件基础问题：

1. latent 空间里能不能稳定做风格迁移。
2. style signal 能不能被模型真正利用，而不是只学 identity。
3. 训练和评估链条能不能跑通。

从 `Thermal` 的历史看，早期探索包括 LoRA、cross-attn、MSE、不同 patch 策略、显存与 batch-size 调整。到了 `Cycle-NCE/src` 初版，结构终于开始收敛成一个专用的小型 latent U-Net。

## 2. 代表模型结构

以 `ae596d1` 的 `Cycle-NCE/src/model.py` 为代表，可以看到这一版还是标准的 reference-conditioned 设计：

- `style_enc`
  - 直接编码 `style_ref` latent，得到全局 `style_code`。
- `enc_in -> hires_body -> down -> body -> dec`
  - 一个轻量 U-Net 主体，没有复杂 skip 过滤。
- `AdaGN`
  - 每个 `ResBlock` 通过 `GroupNorm + Linear(style_code)` 生成 scale / shift。
- `encode_style_feats`
  - 从 `style_ref` 抽多尺度 feature。
- `_extract_style_spatial_maps`
  - 从 style feature 构造高频空间 map。

这一版的关键特征是：

- 推理需要 `style_ref`
- style 表达主要来自 reference encoder
- 风格注入是“全局 style code + 空间高频图”的组合

## 3. 核心模块

### 3.1 `AdaGN`

实现特征：

- `GroupNorm(affine=False)` 做内容归一化
- 线性层把 `style_code` 投成 `2 * dim`
- 前半做 scale，后半做 shift
- 初始化为恒等映射，避免训练一开始破坏内容

这一点很重要，因为后面几乎所有版本都没有放弃“style-conditioned normalization”这个大方向，只是在其上不断改写。

### 3.2 `ResBlock`

结构比较直接：

- 两次 `AdaGN`
- 两层 `3x3 conv`
- 残差回连

这个块在历史上非常稳定，后面变化主要是“norm 的具体形式”和“block 内是不是插 attention”。

### 3.3 `style_ref` 编码器

这部分是早期版本和后期版本最大的分界线之一。

早期逻辑：

- style 不靠离散 embedding
- style 由输入参考图 latent 直接编码得到
- 模型更像“带示例的 conditional translator”

后期逻辑：

- style 被蒸馏进 `style_id` embedding / spatial prior
- 推理不再必须输入参考图

## 4. 早期 loss 思路

从 `ae596d1` 以及更早的 `Thermal` / `NCE_SWD` 历史能看到，早期重点还在“找到可用的风格信号”：

- `Gram loss`
- `Moment loss`
- `Code loss`
- `NCE loss`
- `Cycle loss`
- 若干高频/概率门控/空间原型约束

这一时期的总体特征是：loss 很多，模型本体还没完全定型。

## 5. 这个阶段的主要问题

### 5.1 风格有效，但结构和画面不稳定

`ae596d1` 的提交信息已经写得很直白：

- 分类成绩很好
- 画面有点崩
- 说明风格信号“可分”，但生成质量未达标

### 5.2 过度依赖 reference-conditioned 路径

这一代必须带 `style_ref`，这会带来两个问题：

- 推理接口重
- 风格表征容易和 reference 图像内容纠缠

### 5.3 loss 设计偏重“把信号做出来”

这使得早期系统很像实验平台，而不像最终可部署结构。后面很多删减都来自这里。

## 6. 与实验的对应关系

这个阶段在 `Y:\experiments` 中的对应，大多还散落在旧目录和快照里，不像后期那么规范。能明确对应上的主要是早期 overfit / style-distill / strong-style 轨迹。

我更倾向把它理解为：

- “验证 latent 风格迁移可行”
- “验证 style 信号能被编码”
- “验证哪些 loss 至少不会完全失效”

而不是成熟 ablation 体系的一部分。

## 7. 这个阶段留下来的遗产

真正活到后面的，不是具体实现，而是三个原则：

1. 风格必须通过可控调制进入主干，而不是只靠输出端修补。
2. 单纯内容保持会把模型拖回 identity。
3. 高频与局部统计一定要有专门机制约束，否则“有风格分数”不等于“有画面质量”。

