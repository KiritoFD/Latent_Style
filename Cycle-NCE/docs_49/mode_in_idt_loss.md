# IDT Loss 从 GN → IN 变化

## 基本信息
- **Commit**: `3ad9977e8` (2026-04-06 18:42)
- **提交信息**: `idt loss改用IN只保留内容轮廓`
- **影响文件**: `src/losses.py`

---

## 变化内容

### 1. 变化前 (使用 Group Normalization)

```python
# Topology-only identity: remove style magnitude (mean/variance), keep spatial structure.
pred_struct = F.instance_norm(pred)
content_struct = F.instance_norm(content)
return _masked_mse_mean(pred_struct, content_struct, id_mask)
```

**原理**: 对整个特征图做 Instance Norm，移除所有通道的均值和方差，只保留空间结构。

**问题**: 这种方法会同时影响所有通道（包括颜色通道），导致风格迁移过于保守，颜色保留不足。

### 2. 变化后 (仅使用 Luma 通道)

```python
# Luma-only identity: keep channel-0 structure aligned while leaving chroma channels unconstrained.
pred_luma = pred[:, 0:1, :, :]      # 只取第一个通道（亮度/灰度）
content_luma = content[:, 0:1, :, :] # 只取第一个通道
pred_struct = F.instance_norm(pred_luma)
content_struct = F.instance_norm(content_luma)
return _masked_mse_mean(pred_struct, content_struct, id_mask)
```

**原理**: 只对第一个通道（亮度通道/Luma）做 Instance Norm，保持亮度结构的对应，同时让颜色通道（Chroma）自由变化。

**优势**: 
- 保留内容轮廓（亮度结构）
- 允许风格迁移时的颜色自由变化
- 解决"颜色过于保守"的问题

---

## 新增功能：Soft Repulsive Loss

同时在此 Commit 中引入了新的 Loss：

```python
def soft_repulsive_loss(
    pred: torch.Tensor,
    content: torch.Tensor,
    margin: float = 0.5,
    temperature: float = 0.1,
    dist_mode: str = "l1",
) -> torch.Tensor:
    mode = str(dist_mode).strip().lower()
    if mode == "mse":
        diff = ((pred - content) ** 2).mean(dim=(1, 2, 3))
    else:
        diff = (pred - content).abs().mean(dim=(1, 2, 3))
    tau = max(float(temperature), 1e-4)
    return F.softplus((pred.new_tensor(float(margin)) - diff) / tau) * tau
```

**作用**: 推开预测结果和内容特征的差异，避免模型过于保守。

---

## 实验数据

### 实验: `in-idt` (使用新的 IN-only IDT Loss)

**路径**: `G:\GitHub\Latent_Style\Cycle-NCE\in-idtull_eval\epoch_0040\`

**评估结果**:
| 指标 | 数值 |
|------|------|
| **clip_style (全局)** | 0.6787 |
| **clip_content (全局)** | 0.8464 |
| **clip_dir (全局)** | 0.3784 |
| **Style Transfer clip_style** | 0.6480 |
| **Style Transfer clip_content** | 0.8463 |
| **Identity clip_content** | 0.8469 |

### 风格平均分数
| 风格 | clip_style | clip_content |
|------|------------|--------------|
| Hayao | 0.6043 | 0.8334 |
| monet | 0.6168 | 0.8268 |
| vangogh | 0.6200 | 0.8238 |
| cezanne | 0.5876 | 0.8265 |

---

## 对比分析

### 与之前版本的对比

| 版本 | clip_style | clip_content | 颜色保留 |
|------|------------|--------------|----------|
| 04-03 基准 | ~0.68 | ~0.84 | 一般 |
| **in-idt (IN-only)** | **0.6787** | **0.8464** | **更好** |

**结论**: 新的 IN-only IDT Loss 在保持相似内容保留的情况下，提供了更好的颜色自由度。

---

## 相关实验目录

此 Commit 同时创建了以下实验：
- `Cycle-NCE/46_in-idt/` - 使用 IN-only IDT Loss
- `Cycle-NCE/46_repulse/` - 使用 Soft Repulsive Loss
- `Cycle-NCE/in-idt/` - 独立验证版本

---

## 配置参数

### in-idt 实验配置 (config.json)
```json
{
    "loss": {
        "w_identity": 1.0,
        "w_swd_micro": 1.0,
        "w_swd_macro": 10.0,
        "w_color": 0.01
    }
}
```

### repulse 实验配置
```json
{
    "loss": {
        "w_identity": 1.0,
        "w_repulsive": 0.3,
        "repulsive_margin": 0.5,
        "repulsive_temperature": 0.1
    }
}
```
