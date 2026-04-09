# 模块 - 提交矩阵

用途：回答“哪个模块在哪个提交点真正进入主线”。

## 1. 阅读方式

- “首次明确出现”不一定代表第一次想过这个概念，而是指能从主线 `src/` 或关键分支中直接看到它成为结构实体。
- 某些模块后续会重写，这里记录的是主线进入点和主导阶段。

## 2. 矩阵

| 模块/主题 | 关键提交 | 主导阶段 | 说明 |
| --- | --- | --- | --- |
| `AdaGN` 基础调制 | `ae596d1` | Era A | 参考图条件风格调制基座 |
| `style_emb` | `d916277` | Era B | 风格蒸馏进模型的全局 id 表示 |
| `style_spatial_id_16/32` | `d916277` | Era B | 每个风格的空间先验图 |
| `map16 / map32` 分工明确 | `83ffe10` | Era B | map16 管中频大块，map32 管高频笔触 |
| 多 patch SWD 主线化 | `1f818cc`, `adb274a` | Era B | domain 1x1 / patch 多尺度结论固定 |
| AdaGN 笔触强化调整 | `f7b328c` | Era B | 调制器本身开始被视为风格主控器 |
| `TextureDictAdaGN` | `c619fda` 附近 | Era C | 低秩纹理读写式 style modulation |
| `NormFreeModulation` | `c619fda` 附近 | Era C | decoder 侧 no-norm 风格调制 |
| `StyleAdaptiveSkip` / `StyleRoutingSkip` | `c619fda` 后稳定 | Era C | skip 过滤与改写正式进入主线 |
| NCE 再确认有效 | `4992e06` | Era C | 内容结构保护的主辅助损失 |
| tokenizer / `prob.py` 解耦 | `4699637` | Era C | style embedding 与 tokenizer 路线拆开 |
| 伪 RGB color 路线获胜 | `ed596c0` | Era C | color loss 从多方案走向收敛 |
| `CrossAttnAdaGN` | `c8577e0` | Era D | token 级风格注意力调制 |
| brightness / luma 约束 | `c8577e0` | Era D | 与 cross-attn 同期加强 |
| `SpatialSelfAttention` / `AttentionBlock` | `426ae0a` | Era D | feature attention 成为骨架部件 |
| `SemanticCrossAttn` | `426ae0a` 后主线可见 | Era D | 语义层面的空间改写块 |
| c-g-w backbone | `cfdbaba` | Era D | conv/global/window 混合骨架主线化 |
| shifted window 修正 | `c405b9d` | Era D | window attention 工程修补完成 |
| micro-batch 训练范式 | `4e166f0` | Era D | 训练组织方式成为性能关键 |

## 3. 这个矩阵最重要的三个判断

### 3.1 主线最稳定的不是 backbone，而是调制器

从 `AdaGN -> TextureDictAdaGN -> CrossAttnAdaGN` 可以看出，真正持续演化的是 style modulator。

### 3.2 skip 与 decoder 是第二主线

它们不是附属配件，而是决定模型会不会偷 identity shortcut、会不会泄漏 source 高频的关键。

### 3.3 attention 真正骨架化是在 2026-03-29 之后

在此之前 attention 更像增强模块；到 `426ae0a / cfdbaba / c405b9d` 之后，attention 才进入 backbone 搜索空间。

