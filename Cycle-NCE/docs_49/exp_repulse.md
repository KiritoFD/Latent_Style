# exp_repulse.md

> 生成时间: 2026-04-09  
> 数据来源: `46_repulse/config.json`、`Layer-Norm-repulse/config.json` 及对应 `summary.json`  
> 关联提交: `3ad9977e8`、`97de16534`

---

## 1. 这条线在研究什么

`repulse` 线的核心不是改结构主干，而是引入一种新的损失形态：

- repulsive loss

它想测试的不是“模型能不能学风格”，而是：

**如果显式把生成结果从某些错误纹理/错误局部结构上推开，会不会更容易得到目标风格。**

这条线很像给主线模型加了一种新的训练拉力。

---

## 2. 当前已确认的代表点

| 目录 | 关键配置 | style | content | clip_dir | p2a_style | p2a_content |
|------|----------|-------|---------|----------|-----------|-------------|
| `46_repulse` | `w_repulsive=25`, `margin=0.8`, `temp=0.2`, `w_id=5`, `w_color=25`, patch=`3,5,7,11,15` | 0.6474 | 0.8251 | 0.4024 | 0.6117 | 0.8098 |
| `Layer-Norm-repulse` | `repulsive_mode='l1'`, `w_repulsive=15`, `margin=1.5`, `temp=0.1`, `w_id=5`, `w_color=50`, patch=`3,5,15,19`, `cdf` | 0.6881 | 0.6529 | 0.5565 | 0.6696 | 0.6524 |

---

## 3. 最关键的现象

### 3.1 repulsive loss 不是单向提高 style 的按钮

这两个结果差异非常大：

- `46_repulse` 明显偏 content-first
- `Layer-Norm-repulse` 明显偏 style-first

这说明 repulsive loss 本身并不决定方向，真正决定方向的是它接在什么基线上、再配什么 patch 与 color/identity 权重。

### 3.2 `Layer-Norm-repulse` 很像“用 repulse 强行推风格”

`Layer-Norm-repulse` 的结果：

- style 高到 `0.6881`
- content 掉到 `0.6529`
- clip_dir 也很高

这说明在这条支线上，repulsive loss 确实能成为一种风格推进器，但代价是整体均衡性改变。

### 3.3 `46_repulse` 更像主线结构上的保守接枝

`46_repulse`：

- style 没有大涨
- content 反而很高

这说明把 repulsive loss 接在 46 主线附近时，它更像一种保守修补，而不是激进推动器。

---

## 4. 这条线在大历史里的位置

`repulse` 线的重要性在于，它给了项目一个和结构改动不同的尝试方向：

- 不一定改模块；
- 改损失形状也可能重排 style/content 平衡。

所以这条线应该被视为“训练目标演化史”的一部分，而不只是结构史的附录。

---

## 5. 当前结论

当前最值得保留的结论是：

1. repulsive loss 的效果高度依赖基线结构。
2. 它可以成为 style 推进器，也可以成为保守修补器。
3. 因此后续如果继续写训练目标演化，`repulse` 必须单独占一节，不能只在实验目录里顺手一提。

