# 模型结构与风格注入改进说明

## 1. 当前模型结构（简述）
- 主体是 latent 空间的 U-Net 风格生成器，输入为 `content latent + style condition`。
- 风格条件由 style encoder 产生，当前包含全局 code 与空间特征注入路径。
- 解码端（decoder）有 AdaGN/空间调制增强版本，是目前最有效的注入入口。
- 训练中同时存在结构保持项（cycle/idt/低频约束）与风格项（分类器/统计项）。

## 2. 为什么会出现“像缓存没清理”的现象
- 近几轮实验里 `photo->Hayao classifier_acc` 持续接近 `0.06`，`clip_style` 也几乎不动，表面上像复用了旧结果。
- 实际上更常见原因是模型坍缩到“近恒等/弱色调变化”，多组配置都落到同一坏局部最优，指标会非常接近。
- 但评估侧确实有一个风险：`metrics.csv` 若追加写入，会混入历史行，导致 summary 被污染。

## 3. 已做的评估链路修复
- 文件：`src/utils/run_evaluation.py`
- 新增参数：`--append_metrics`
- 默认行为改为每次重写 `metrics.csv`（`csv_mode='w'`），仅在显式传入 `--append_metrics` 时才追加。
- 这样可以避免同一输出目录下的历史指标混入，排除“缓存污染”假象。

## 4. 现阶段结论
- `d2/d3` 说明“decoder 端 + 空间注入”方向是有效入口。
- `d4+` 之后多组接近同一失败指标，主要问题不是缓存，而是风格信号在优化中被结构项/低频捷径压制。
- `proto separation / push` 仅改善原型几何，并未显著提升跨域风格注入结果。

## 5. 改善风格注入的优先级（按收益排序）
- 第一优先：把风格监督从低维统计迁到高维多尺度特征（避免只学亮度/对比度）。
- 第二优先：保留低频结构约束，但把风格变化能量显式推到中高频（作用在主 delta 分支）。
- 第三优先：把分类器从“主导目标”降级为“门槛约束”，防止被 classifier shortcut 牵着走。
- 第四优先：增加条件敏感性门控（同一输入对不同 style_id 的输出差异与高频占比）作为早停/准入指标。

## 6. 下一轮实验最小矩阵
- E1 基线：当前 d2 稳定配置（修复后的评估链路）。
- E2 只改监督：多尺度高维风格特征损失。
- E3 在 E2 上加：delta 低频抑制与中高频偏置。
- 每个实验都必须输出：
- full_eval summary
- condition sensitivity（`delta_abs`, `delta_high_ratio`）
- collage 对比图

## 7. 实验准入门槛（overfit50）
- `photo->Hayao classifier_acc >= 0.60`
- `style_transfer_ability classifier_acc >= 0.70`
- `photo_to_art clip_style >= 0.53`
- 拼图可见稳定纹理变化，且不是纯全局色调偏移。
