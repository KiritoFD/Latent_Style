# module_index_49.md

> 生成时间: 2026-04-09  
> 用途: 给 2026-02-08 到 2026-04-09 这段 `model.py` 演化建立模块总索引，确保“每看到一个模型模块，都单独写文档”这条规则可以被执行和复查。

---

## 1. 写法规则

从这一版开始，模块文档统一按下面四个问题来写，不以贴代码为主：

1. **这个模块在干什么**
2. **为什么要这样写**
3. **它解决了当时什么问题**
4. **它和实验现象、后续替代模块是什么关系**

代码片段只保留少量锚点，用来证明结论来自真实实现，而不是凭印象描述。

---

## 2. 历史模块总表

### 2.1 2026-02-08 初始基线期

| 模块 | 角色 | 状态 |
|------|------|------|
| `AdaGN` | 最早的风格调制器，负责 GroupNorm 后的 style affine | 已写 `mode_AdaGN.md` |
| `ResBlock` | 基础残差块 | 已写 `mode_ResBlock.md` |
| `LatentAdaCUT` | 主模型框架 | 已写 `mode_LatentAdaCUT.md` |

### 2.2 2026-02-11 到 2026-02-15 StyleMaps 阶段

| 模块 | 角色 | 状态 |
|------|------|------|
| `StyleMaps` | 风格图/中间风格映射载体 | 已写 `mode_StyleMaps.md` |

### 2.3 2026-03-26 纹理字典与 cross-attn 并存阶段

| 模块 | 角色 | 状态 |
|------|------|------|
| `_BaseStyleModulator` | 调制器抽象基类，统一 gate 与极端消融开关 | 已写 `mode__BaseStyleModulator.md` |
| `TextureDictAdaGN` | 主力纹理字典调制器 | 已写 `mode_TextureDictAdaGN.md` |
| `GlobalDemodulatedAdaMixGN` | 历史兼容别名类 | 已写 `mode_GlobalDemodulatedAdaMixGN.md` |
| `CrossAttnAdaGN` | 用 style tokens 做 cross-attn 的调制器 | 已写 `mode_CrossAttnAdaGN.md` |
| `NormFreeModulation` | decoder 侧无归一化调制 | 已写 `mode_NormFreeModulation.md` |
| `StyleAdaptiveSkip` | 早期风格自适应 skip 过滤器 | 已写 `mode_StyleAdaptiveSkip.md` |

### 2.4 2026-03-29 到 2026-04-09 attention / routing 扩展期

| 模块 | 角色 | 状态 |
|------|------|------|
| `SpatialSelfAttention` | 空间自注意力 | 已写 `mode_SpatialSelfAttention.md` |
| `AttentionBlock` | attention 组合块 | 已写 `mode_AttentionBlock.md` |
| `SemanticCrossAttn` | 语义 cross-attention | 已写 `mode_SemanticCrossAttn.md` |
| `DecoderTextureBlock` | decoder 端纹理块 | 已写 `mode_DecoderTextureBlock.md` |
| `StyleRoutingSkip` | 新一代 skip 路由模块 | 已写 `mode_StyleRoutingSkip.md` |

### 2.5 不是类，但必须保留的结构主题

| 主题 | 原因 | 状态 |
|------|------|------|
| `in_idt_loss` | 04-06 之后是核心实验变量，虽然不是类，但已经是结构级概念 | 已写 `mode_in_idt_loss.md` |
| `structure_tuning` | 04-05 到 04-06 的结构调整是 45/46/chess 的直接前情 | 已写 `mode_structure_tuning.md` |
| `attn_gate_mode` | 04-08 之后成为 `Gate` 系列最核心的 attention 注入控制旋钮，虽然不是独立类，但已经是结构级概念 | 已写 `mode_attn_gate_mode.md` |

---

## 3. 当前覆盖判断

截至目前，**能在历史提交里直接看到的主要模型类，已经全部有对应单文档**。  
后面需要继续做的，不是“有没有文件”，而是两件更难的工作：

1. 把已有偏代码型文档改成解释优先。
2. 把每个模块和具体实验结果、版本阶段再绑得更紧。

---

## 4. 后续补强顺序

### 第一优先级

- `CrossAttnAdaGN`
- `StyleRoutingSkip`
- `DecoderTextureBlock`
- `SemanticCrossAttn`

原因很直接：这几个模块最接近 4 月上旬的核心实验线，和 `46`、`chess`、`Gate` 的关系最强。

### 第二优先级

- `LatentAdaCUT`
- `ResBlock`
- `NormFreeModulation`
- `StyleMaps`

这些模块虽然文档已有，但还要进一步补“版本差异”和“实验对应”。

### 第三优先级

- 历史兼容链文档
  - `GlobalDemodulatedAdaMixGN`
  - 旧命名 alias

这部分对理解 checkpoint 兼容和命名演化很重要，但优先级略低于当前主战场模块。

---

## 5. 这份索引的作用

后面只要继续翻到新的模型模块，就在这里补一行，再去落单独文档。  
这样做的目的不是好看，而是防止考古过程中出现最常见的问题：

- 在总报告里提了一嘴
- 过两小时忘了补单文档
- 最后变成“知道它存在，但说不清它干了什么”

这份索引就是用来防止这种断档的。
