# 2026-05-22 Diffeomorphic Grid Regression Fix

## 结论

`t00/t01` 历史 `clip_style ~= 0.726` 复现失败的根因，不在 evaluator、JPEG、seed、CLIP cache，也不在 checkpoint 本身。

真正的回归点是：

- 文件：`SchrodingerBridge/src/utils/diffeomorphic.py`
- 函数：`_base_grid_like()`
- 问题：warp 采样基网格从历史版本的 `torch.linspace(-1, 1, H/W)`，漂移成了像素中心定义

这会直接改变 `grid_sample(..., align_corners=False)` 的坐标语义，导致所有基于 tangent diffeomorphic stroke 的旧 checkpoint 在当前代码上推理时风格分数系统性掉点。

## 排查过程

先后排除了以下假设：

1. `--reuse_generated` 读回 JPEG 导致掉点
   - 这条链确实会掉点，但只是“磁盘重读协议”和“原始 in-memory full eval 协议”不同。
   - 不能解释“当前 checkpoint 直推也只有 0.703”。

2. VAE encode 采样随机性
   - 对 `t00 epoch8` 做多 seed 复评，`clip_style` 波动只有 `~5e-5`。

3. source subset 漂移
   - 对比历史 `t00` 和当前重跑的 source stem，完全一致。

4. evaluator 全局失效
   - `SaMST` 用当前 evaluator 重评后，和原有 summary 基本一致。

## 关键隔离实验

### 1. 当前正式代码直推旧 `t00 epoch8`

- `clip_style = 0.703887`
- `clip_content = 0.769999`
- `content_lpips = 0.495514`

### 2. 只恢复历史 grid 语义

在临时 probe 中，仅把 `_base_grid_like()` 改回：

```python
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
    torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
    indexing="ij",
)
```

其余保持当前代码不变，得到：

- `clip_style = 0.725958`
- `clip_content = 0.760257`
- `content_lpips = 0.516626`

这与历史 summary 几乎逐位重合：

- 历史 `t00`：`0.725912 / 0.760208 / 0.516601`

### 3. 正式源码修复后复评

修复 `SchrodingerBridge/src/utils/diffeomorphic.py` 后：

#### `t00_ws0p03_g6_nl0`

- 历史：`clip_style=0.7259116765658061`
- 修复后：`clip_style=0.7258791350523631`

- 历史：`clip_content=0.7602079883733333`
- 修复后：`clip_content=0.7599943759066667`

- 历史：`content_lpips=0.5166011267333332`
- 修复后：`content_lpips=0.5166116979066667`

#### `t01_ws0p03_g6_nl0p05`

- 历史 ledger：`clip_style=0.7263630538781485`
- 修复后：`clip_style=0.7264026194016138`

- 历史 ledger：`clip_content=0.7569881064`
- 修复后：`clip_content=0.7567931872`

- 历史 ledger：`content_lpips=0.5169742019866667`
- 修复后：`content_lpips=0.5169946044533333`

## 最终修复

正式恢复历史语义：

- 文件：`SchrodingerBridge/src/utils/diffeomorphic.py`
- 修改：`_base_grid_like()` 改回历史 `linspace(-1, 1)` 版本

当前实现：

```python
def _base_grid_like(x: torch.Tensor) -> torch.Tensor:
    b, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
```

## 影响与后续

1. 旧的 tangent sweep checkpoint 现在可以在当前正式代码上复现历史 `0.726` 指标。
2. 后续所有 diffeomorphic/tangent 相关实验，都应该基于这个修复后的坐标语义继续。
3. 如果将来确实需要“像素中心 grid”这套定义，必须作为显式新开关引入，不能静默替换历史默认值。
