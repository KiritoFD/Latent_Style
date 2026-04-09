# Model_49: 当前架构状态 (2026-04-09)

> **基准时间**: 2026-04-03 考古整理结束
> **当前时间**: 2026-04-09 02:06 CST
> **文档目的**: 记录当前模型架构状态

---

## 先看结论：49 版模型现在在干什么

当前这版模型已经不是单一的“卷积 + 风格注入”结构，而是一台职责分得比较细的 latent 风格迁移器：

1. 由 `style modulator` 决定风格怎样进入特征流
2. 由 `attention` 决定空间和语义关系怎样重排
3. 由 `skip routing` 决定浅层结构信息怎样回流
4. 由 `decoder texture` 决定最后一公里纹理怎样落地

所以 49 版的关键不在“某个神奇新层”，而在于它已经把颜色、结构、纹理、语义路由这些职责分散到不同部件上了。

## 为什么这一版会长成这样

从 04-03 到 04-09 的实验可以看出，团队已经不再满足于简单地给 U-Net 加一个风格向量。  
真正推动 49 版成形的，是一连串更具体的问题：

- skip 到底在保结构还是泄漏源纹理
- decoder 到底在恢复还是抢权
- color highway 是约束还是负担
- attention 是帮助空间关系还是制造不稳定
- patch 频谱和 identity 权重到底在把模型往哪边推

49 版就是这些问题暂时汇总后的结构答案。

---

## 📊 核心文件概览

| 文件 | 大小 | 行数 | 主要内容 |
|------|------|------|----------|
| `model.py` | 63.4 KB | 1472 | 模型架构定义 |
| `trainer.py` | 25.1 KB | 599 | 训练循环 |
| `losses.py` | 28.9 KB | 683 | 损失函数 |
| `ablate.py` | 7.1 KB | - | 消融实验配置 |

---

## 🏗️ model.py 架构详解

### 类定义 (10 个)

| 类名 | 行数 | 功能 |
|------|------|------|
| **CrossAttnAdaGN** | 81-230 (150行) | Cross Attention + AdaGN 调制模块 |
| **ResBlock** | 231-272 (42行) | 残差块 |
| **SpatialSelfAttention** | 273-361 (89行) | 空间自注意力 |
| **AttentionBlock** | 362-403 (42行) | 注意力块封装 |
| **SemanticCrossAttn** | 404-516 (113行) | 语义交叉注意力 |
| **NormFreeModulation** | 517-541 (25行) | 无归一化调制 |
| **DecoderTextureBlock** | 542-574 (33行) | 解码器纹理块 |
| **StyleRoutingSkip** | 575-652 (78行) | 风格路由跳跃连接 |
| **StyleMaps** | 653-656 (4行) | 风格地图容器 |
| **LatentAdaCUT** | 657-1473 (817行) | **主模型** |

### LatentAdaCUT 主模型参数

```python
def __init__(
    self,
    latent_channels: int = 4,          # VAE潜空间通道数
    num_styles: int = 3,               # 风格数量
    style_dim: int = 256,              # 风格向量维度
    base_dim: int = 64,                # 基础通道数
    num_hires_blocks: int = 2,         # 高分辨率块数量
    num_res_blocks: int = 4,           # 残差块数量
    num_decoder_blocks: int = 1,       # 解码器块数量
    
    # 风格相关
    style_attn_num_tokens: int = 16,   # 风格注意力token数
    style_attn_num_heads: int = 4,     # 风格注意力头数
    style_spatial_pre_gain_16: float = 0.35,
    
    # 注意力相关
    semantic_attn_temperature: float = 0.08,
    feature_attn_num_heads: int = 4,
    window_attn_window_size: int = 8,
    
    # Skip连接
    skip_fusion_mode: str = "concat_conv",
    skip_routing_mode: str = "normalized",
    skip_residual_weight: float = 0.1,
    
    # 其他
    use_checkpointing: bool = False,
    output_moment_match: bool = False,
)
```

### Forward 流程

```python
def forward(x, style_id, ...):
    1. resolve_style_strength()     # 风格强度
    2. _prepare_style_context()     # 准备风格上下文
    3. _predict_delta_from_context() # 预测 delta
    4. _perturb_anchor_if_needed()  # 输入扰动
    5. output = anchor + delta * step_size * step_scale
    6. _apply_output_moment_match() # 输出归一化
```

---

## 🏃 trainer.py 架构详解

### AdaCUTTrainer 类

**核心方法**:
- `__init__`: 初始化训练器
- `train_epoch()`: 训练一个 epoch
- `step_scheduler()`: 学习率调度
- `log_epoch()`: 记录日志

**特点**:
- 单文件训练器（599 行）
- 支持梯度检查点
- 支持 OneCycle LR 调度

---

## 📉 losses.py 架构详解

### AdaCUTObjective 类

**核心损失函数**:
1. `soft_repulsive_loss` - 软斥力损失
2. `calc_spatial_agnostic_color_loss` - 空间无关颜色损失
3. `calc_swd_loss` - Sliced Wasserstein Distance 损失

**特点**:
- 无 Gram Matrix 损失（已在 04-03 前移除）
- SWD 作为核心纹理损失
- Color Loss 用于色彩锚定

---

## 🔄 与 04-03 架构对比

**待补充**: 需要从 Git 记录还原 04-03 时的架构状态

---

## 📝 下一步

1. 创建 `model_43.md` - 记录 04-03 时的架构
2. 对比两个版本的差异
3. 逐个 commit 分析变化
