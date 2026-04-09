# Era D：Cross-Attn、Feature Attention、CGW 与 Micro-Batch

时间范围：2026-03-26 到 2026-04-02  
代表提交：`c8577e0`, `426ae0a`, `cfdbaba`, `c405b9d`, `1e25659`, `4e166f0`

## 1. 这一阶段的主线

这段历史的主题是：

- 在已有 AdaGN / TextureDict / skip-routing 基础上，继续上更强的特征交互机制。

主要尝试包括：

1. cross-attn style modulator
2. global / window feature attention
3. c-g-w backbone
4. micro-batch 训练范式

## 2. `CrossAttnAdaGN`

`c8577e0` 引入的关键模块之一是：

- `CrossAttnAdaGN`

这个模块保留了 AdaGN API，但内部已经变成：

- `style_tokens_basis`
- `style_proj`
- `pos_proj`
- `q/k/v`
- `scaled_dot_product_attention`
- FFN

### 2.1 它解决什么问题

TextureDict 版本本质上还是一种局部读写调制。  
Cross-attn 版本想解决的是：

- 让风格 token 和空间查询显式对齐
- 让不同空间位置从 style token 中读取不同信息

这比“同一个 style code 到处 broadcast”更强。

### 2.2 亮度约束为何与它一起出现

`c8577e0` 的提交信息把两件事写在一起：

- 加亮度约束
- 换 cross_attn

这很说明问题：

- 注意力增强了风格改写能力
- 同时也更容易把亮度 / 色调整体拉偏
- 所以必须引入更明确的 color / brightness 约束

## 3. Feature Attention Block

到了 `426ae0a` 附近，`model.py` 进一步出现：

- `SpatialSelfAttention`
- `AttentionBlock`
- `global_attn`
- `window_attn`

这意味着模型已经不再只是“卷积骨架 + style modulation”，而是在主干 / decoder block 里尝试显式建模空间长程关系。

## 4. CGW Backbone

`cfdbaba`：

> 全部换用 c-g-w 的 backbone

`c405b9d`：

> 修改 channel last 问题，对 windows attention 加上 shift

结合代码可以把 `c-g-w` 粗略理解为：

- conv / global attention / window attention 的混合骨架

这一阶段的结构重点是：

- 不同层不再统一用卷积
- 而是把 block type 当作架构搜索空间

这也解释了为什么当时会出现大量：

- `arch_ablate_*`
- `ca_pram_*`
- `cgw/*`

这样的目录。

## 5. Micro-batch 革命

`4e166f0` 的提交信息是：

> micro batch 效果大好

旧草稿里还有一个很强的旁证：

- `trainer.py` 大幅瘦身
- 许多复杂机制被移除
- 保留了 SWD + Color + Identity 的核心三件套

这很像一次“系统反思”：

- 不是功能越多越好
- 不是大而全 trainer 越复杂越好
- 保留最稳定的主损失与训练循环，反而性能更稳

## 6. 这一阶段的代码特征

### 6.1 `model.py` 变成架构搜索平台

当前版本可见大量开关：

- `style_modulator_type`
- `hires_block_type`
- `body_block_type`
- `decoder_block_type`
- `feature_attn_num_heads`
- `window_attn_window_size`
- `skip_fusion_mode`
- `skip_routing_mode`
- `attn_gate_mode`

这说明模型已经从“单一设计”变成“可组合架构空间”。

### 6.2 `trainer.py` 更强调训练吞吐与稳定性

这一代 trainer 更像是围绕实际训练效率做的工程化版本：

- attention 指标记录
- gradient accumulation
- onecycle / warmup
- channels-last
- 更干净的 epoch metrics

## 7. 对应实验组

这一阶段最强对应的是：

- `arch_ablate_*`
- `cgw/*`
- `ca_pram_*`
- `style_oa/*`
- `optuna_hpo/*`
- `micro*`
- `abl_*`

也就是：

- 架构搜索
- 结构消融
- 参数联合优化
- 快速回归验证

全部汇到一起。

## 8. 阶段评价

这是项目里最“豪华”、也最复杂的一段历史。

它的贡献有两层：

### 8.1 正面贡献

- attention 机制被正式纳入骨架
- window/global 混合骨架被验证
- 微批训练带来稳定收益
- 架构搜索空间被系统化

### 8.2 负面教训

- 复杂架构很容易诱导 identity shortcut
- 复杂模块叠加后，归因变困难
- 单看 style 指标很容易误判为进步

这也是为什么后续文档里要把实验和结构强绑定，否则很难复盘哪一步真的有效。

