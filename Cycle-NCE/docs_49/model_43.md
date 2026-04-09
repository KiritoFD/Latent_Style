# Model Architecture @ 2026-04-03 (Commit: 58831eb69 "micro batch效果大好")

## 📊 基本信息
- **Commit**: 58831eb69
- **日期**: 2026-04-02
- **model.py 行数**: 1341 行
- **类数量**: 11 个

## 🏗️ 完整类列表

| 序号 | 类名 | 功能描述 |
|------|------|----------|
| 1 | `_BaseStyleModulator` | 基础风格调制器抽象类 |
| 2 | `TextureDictAdaGN` | 纹理字典自适应组归一化（核心） |
| 3 | `GlobalDemodulatedAdaMixGN` | 向后兼容别名 |
| 4 | `CrossAttnAdaGN` | 交叉注意力风格调制（新引入） |
| 5 | `ResBlock` | 残差块 |
| 6 | `SpatialSelfAttention` | 空间自注意力 |
| 7 | `AttentionBlock` | 注意力块 |
| 8 | `NormFreeModulation` | 无归一化调制（解码器用） |
| 9 | `StyleRoutingSkip` | 风格路由跳跃连接 |
| 10 | `StyleMaps` | 风格映射管理器 |
| 11 | `LatentAdaCUT` | 主模型类 |

---

## 1. TextureDictAdaGN (核心类)

### 参数
- `dim`: 通道数
- `style_dim`: 风格嵌入维度 (默认512)
- `num_groups`: GroupNorm 分组数 (默认4)
- `rank`: 纹理字典条目数 (默认16)

### 工作原理
1. **GroupNorm**: 对输入特征进行分组归一化
2. **global_proj**: 将 style_dim 映射到 dim*2 (scale + bias)
3. **rank**: 纹理字典的维度，控制风格表达能力

### 关键代码片段
```python
class TextureDictAdaGN(nn.Module):
    def __init__(self, dim: int, style_dim: int, num_groups: int = 4, rank: int = 16) -> None:
        super().__init__()
        groups = max(1, min(int(num_groups), int(dim)))
        while dim % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, dim, affine=False)
        self.rank = max(1, int(rank))
        self.global_proj = nn.Linear(style_dim, dim * 2)
```

---

## 2. CrossAttnAdaGN (新引入)

### 参数
- `num_tokens`: 64 (可学习风格 tokens 数量)
- `num_heads`: 4 (注意力头数)
- `sharpen_scale`: 2.0

### 关键改进
- 使用可学习的 style tokens 进行交叉注意力计算
- 可替换 TextureDictAdaGN 通过配置切换

---

## 3. ResBlock

### 特点
- 两层卷积 + 两层风格调制
- 支持 `texture_dict` 或 `cross_attn` 调制类型

---

## 4. SpatialSelfAttention

### 模式
- `global_attn`: 全局注意力
- `window_attn`: 窗口注意力 (window_size=8)

---

## 5. NormFreeModulation

### 特点
- 无归一化: 直接使用 gamma/beta 调制
- 恒等初始化: 权重初始化为0，训练开始时为恒等映射

---

## 6. LatentAdaCUT (主模型)

### 结构组成
- Encoder: 多个 ResBlock
- Middle: AttentionBlock
- Decoder: 多个 ResBlock
- Skip: StyleRoutingSkip

---

## 📊 与 04-09 版本的差异

| 特性 | 04-03 (58831eb) | 04-09 (当前) |
|------|-----------------|--------------|
| TextureDict rank | 16 | 待查 |
| CrossAttn | 已存在 | 已存在 |
| AttentionBlock | 已存在 | 已存在 |
| NormFreeModulation | 已存在 | 已存在 |

---

*文档生成时间: 2026-04-09*
*数据来源: Git Commit 58831eb69*
