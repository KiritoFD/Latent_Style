# 源码证据专档：关键提交差异摘要

用途：把“我为什么说发生了某次结构切换”落到具体 diff 证据上。

## 1. `ae596d1 -> d916277`：reference-conditioned 向 style-id 蒸馏切换

这组 diff 是最关键的一条证据链。

### 1.1 直接新增的核心字段

从 `Cycle-NCE/src/model.py` diff 可以直接看到新增：

- `self.body_channels = int(base_dim * 2)`
- `self.style_emb = nn.Embedding(self.num_styles, style_dim)`
- `self.style_spatial_id_32`
- `self.style_spatial_id_16`

这说明模型第一次正式拥有：

- 全局离散 style embedding
- 每个 style 的可学习空间先验

### 1.2 `_style_code()` 语义变化

旧逻辑：

- 没有 `style_ref` 就报错
- 完全是 reference-conditioned

新逻辑：

- 支持 `style_id`
- 支持 `style_ref`
- 支持 `style_mix_alpha`
- `code_ref` 和 `code_id` 可以混合

这说明过渡期不是简单“一刀切”，而是：

- 从 reference-only 过渡到 id/ref 可混合
- 便于蒸馏和迁移

### 1.3 空间 style map 也从 reference-only 变成 mixed

新增函数：

- `encode_style_spatial_ref`
- `encode_style_spatial_id`
- `_blend_style_maps`

这直接坐实了：

- 不是只有全局 style code 被蒸馏
- 空间风格先验也被显式建模

## 2. `c619fda -> c8577e0`：loss 主线与注意力扩展

这一组 diff 更像“系统重构”，不只是一个小改动。

## 2.1 loss 侧：color 逻辑收束

从 `losses.py` diff 可以直接看到：

- 删掉了早期更散乱的 TV / 高频附加路径
- color 逻辑被收束到 `calc_spatial_agnostic_color_loss`
- 并且只保留 `latent_decoupled_adain` 主线
- 新增 `luma_range_weight` / `luma_quantiles`

这说明到这个阶段，color loss 的认知已经从“多路尝试”收敛成：

- 低频颜色统计
- 亮度范围约束
- 有限的主干实现

## 2.2 SWD 侧：实现更模块化

从 diff 看出：

- `_swd_distance_from_projected()` 被抽出来
- `calc_swd_loss()` 代码被整理
- projection cache / color weight cache 更明确

这说明：

- SWD 不再是临时实验代码
- 已经进入可维护、可优化、可复用的主损失地位

## 2.3 trainer 侧：从“功能堆叠”转向“训练系统”

同一组 diff 中，`trainer.py` 出现了这些明显变化：

- `OneCycleLR` 延迟初始化
- `warmup_steps` / `warmup_ratio` / `warmup_start_factor`
- batch 级 scheduler step
- gradient direction probe
- 去掉 `delta_tv` 在日志中的中心地位
- 完整的 base lr 管理与 scheduler 恢复逻辑

这说明 trainer 的关注点已经变成：

- 学习率形状
- 长程训练稳定性
- 不同 loss 方向之间是否冲突

## 3. `c8577e0`：cross-attn 不是口号，是真的换了 modulator

这一点从 `code/model.py` 及后续 `src/model.py` 可以直接坐实：

- `CrossAttnAdaGN` 成为新模块
- style tokens、q/k/v、位置编码、FFN 都是新增实体

所以文档里如果写“2026-03-26 开始进入 cross-attn 风格调制时代”，这是有直接源码依据的。

## 4. `426ae0a -> cfdbaba -> c405b9d`：attention 从模块试探走向 backbone 搜索

这段历史从提交信息和 diff 组合上能说明三层变化：

### 4.1 `426ae0a`

- attention 加入后效果明显
- 说明 attention 不再只是探索性插入

### 4.2 `cfdbaba`

- 全部换到 c-g-w backbone
- attention 已经进入骨架级别

### 4.3 `c405b9d`

- channel-last 问题修复
- window attention 增加 shift

这说明：

- window attention 已进入实际主线训练
- 并且开始处理它的工程副作用和细节正确性

## 5. 这份 diff 证据对后续文档的作用

后面如果继续扩写正式历史报告，这份文档可以充当“防止只凭印象写历史”的底稿，因为它回答的是：

- 哪个说法有源码依据
- 哪个说法只是从实验和命名反推

