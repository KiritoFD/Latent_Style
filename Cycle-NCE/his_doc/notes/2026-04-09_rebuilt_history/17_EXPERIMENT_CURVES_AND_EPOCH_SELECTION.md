# 实验曲线与 Epoch 选择专档

用途：把“最佳配置”拆成“最佳 epoch”问题，不再默认最后一个 checkpoint 最好。

## 1. 为什么这份文档必要

很多实验目录不是单点评估，而是有多轮 `summary_history`。  
这意味着：

- 同一个实验，style 最佳 epoch
- 内容最佳 epoch
- FID 最佳 epoch

可能完全不是同一个点。

## 2. `style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10`

数据来源：

- `Y:\experiments\style_oa\style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10\full_eval\summary_history.json`

### 2.1 关键点

- 最新轮次：`epoch 120`
  - style `0.7264517831`
  - lpips `0.5217467501`
- best transfer style：`epoch 100`
  - style `0.7297230264`
  - lpips `0.5095066082`
- best transfer LPIPS：`epoch 30`
  - style `0.6981279703`
  - lpips `0.4249562226`

### 2.2 结论

这条曲线非常典型：

- 30 epoch 是内容/保真甜点
- 100 epoch 是 style 峰值
- 继续训到 120 epoch，style 反而略掉，LPIPS 继续变差

所以这条线直接证明：

- “最后一轮最好”并不成立
- 这个项目的很多结果应该按目标选 checkpoint，而不是固定最后 epoch

## 3. `abl_no_adagn`

数据来源：

- `Y:\experiments\abl\abl_no_adagn\full_eval\summary_history.json`

### 3.1 关键点

- `epoch 40`
  - style `0.6855056301`
  - lpips `0.5233671738`
  - fid `304.9437813`
  - art_fid `475.6556875`
- `epoch 80`
  - style `0.6973906821`
  - lpips `0.5414757411`
  - fid `312.5094776`
  - art_fid `492.0466261`

### 3.2 结论

这是典型的“后期 style 继续涨，但整体变坏”的曲线。  
它支持两个判断：

1. 去掉 AdaGN 后，模型更容易向高风格分数方向漂移。
2. 这种漂移并不代表更好的迁移，只代表约束失效。

## 4. `abl_naive_skip`

数据来源：

- `Y:\experiments\abl\abl_naive_skip\full_eval\summary_history.json`

### 4.1 关键点

- `epoch 40`
  - style `0.6984118814`
  - lpips `0.5939234347`
  - fid `326.3586001`
  - art_fid `526.6445349`
- `epoch 80`
  - style `0.7061084746`
  - lpips `0.6041445841`
  - fid `330.4055816`
  - art_fid `538.0465795`

### 4.2 结论

和 `abl_no_adagn` 一样，这条线也是：

- style 上升
- 其余质量指标恶化

这进一步强化了 skip 过滤的重要性。

## 5. `abl_no_residual`

数据来源：

- `Y:\experiments\abl\abl_no_residual\full_eval\summary_history.json`

### 5.1 关键点

- `epoch 40`
  - style `0.6285932735`
  - lpips `0.2966944365`
  - fid `286.4353868`
  - art_fid `372.4162687`
- `epoch 80`
  - style `0.6285974741`
  - lpips `0.2965136992`
  - fid `286.4885003`
  - art_fid `372.4398673`

### 5.2 结论

这条线几乎不动，说明：

- 去掉 residual 后，模型很快卡在一个“保守但风格弱”的区域
- 它不是后期过拟合问题，而是表达能力上限本来就低

## 6. `micro05_id_anchor`

数据来源：

- `Y:\experiments\micro05_id_anchor\full_eval\summary_history.json`

### 6.1 关键点

- `epoch 20`
  - style `0.6741548417`
  - lpips `0.4830329603`
  - fid `293.6816932`
  - art_fid `443.8759693`
- `epoch 80`
  - style `0.6918713432`
  - lpips `0.5186884047`
  - fid `310.4382559`
  - art_fid `482.9165670`

### 6.2 结论

这是典型的“早期就有甜点”的 micro-run：

- 20 epoch 是更均衡的点
- 80 epoch 风格更高，但整体更差

这说明 micro 系列很适合拿来做“方向筛选”，不适合直接拿最后一轮当最终答案。

## 7. 这份曲线材料的整体结论

如果把这几条曲线放在一起，能得到一个很稳的元结论：

1. 这个项目里 style 常常随 epoch 上升得更久。
2. LPIPS / FID / art_fid 往往更早达到甜点。
3. 因此 checkpoint selection 是方法的一部分，而不是训练结束后的附带操作。

## 8. 后续扩展建议

下一轮如果继续做，可以把更多 `summary_history.json` 归纳成三个清单：

- “style 后期持续上升型”
- “中期甜点型”
- “几乎不动型”

这会让后面做正式汇报时非常有说服力。

