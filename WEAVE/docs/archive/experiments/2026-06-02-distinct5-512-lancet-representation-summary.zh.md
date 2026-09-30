# Distinct5-512 LANCET 表征探索中文总结

本文档总结 2026-06-02 这一轮 Distinct5-512 LANCET 表征与速度实验。英文原始长日志见：

- `docs/experiments/2026-06-02-distinct5-512-lancet-representation-speed.md`

本文档不替代原始日志；它用于说明我们到底尝试了哪些设计、每条线的实验结果如何、哪些结论可以继续依赖。

## 1. 数据集与评估口径

本轮实验使用的数据集固定为：

- 原始数据：`F:\wikiart_distinct5_samam_512`
- 图像 classview：`F:\wikiart_distinct5_samam_512_classview`
- EMA VAE latent：`F:\wikiart_distinct5_samam_512_latents_ema`

五个风格类别固定为：

- `Early_Renaissance`
- `Impressionism`
- `Minimalism`
- `Rococo`
- `Ukiyo_e`

数据规模：

| Split | 每类数量 | 总数 | 用途 |
|---|---:|---:|---|
| train | 1000 | 5000 | LANCET 训练与 latent queue 构建 |
| test | 30 | 150 | all 5x5 full eval |

正式评估统一使用 `750` 图 all-pairs 口径：

```text
5 个 source style x 每类 30 张 source image x 5 个 target style = 750
```

主指标只看：

- `clip_style`：越高越好，用于衡量目标风格相似度。
- `content_lpips`：越低越好，用于衡量内容保持。

训练期限制：

- 不引入 DINO。
- 不引入 CLIP loss。
- 不引入分类器监督。
- 不引入外部语义特征监督。
- CLIP 和 LPIPS 只作为评估指标，不进入训练 loss。

训练信号只来自：

- VAE latent。
- LANCET/SWD/OT。
- 动力学正则。
- 内部 tokenizer/prototype/queue 结构。

## 2. 工程与数据准备

本轮先补齐了 Distinct5-512 的工程基础：

| 工具 | 作用 |
|---|---|
| `tools/prepare_distinct5_classview.py` | 将 flat 文件按 `StyleName__...jpg` 前缀整理成 class folder |
| `tools/build_latent_packed_cache.py` | 将每类 latent 打包为 packed cache，减少训练启动与 DataLoader 开销 |
| `tools/build_latent_prototype_pairing_cache.py` | 用 VAE latent 内部统计构建 prototype-aware target queue |
| `tools/diagnose_wikiart_latents.py` | 诊断 latent 统计、梯度、频域能量等内部风格差异 |
| `tools/probe_style_representation.py` | 探查 tokenizer code geometry 和生成 residual geometry |
| `src/utils/run_evaluation.py` | 统一 full eval 入口，支持 timing profile 与只算 LPIPS/CLIP style |

训练默认使用 packed latent cache：

- `F:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\manifest.json`
- `F:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\packed\*.pt`
- `F:\wikiart_distinct5_samam_512_latents_ema\train\.latent_cache\prototype_pairing_top8.pt`

远程 3060 标定后的正式训练 batch 是 `44`。这个 batch 的显存使用约为：

- PyTorch reserved：约 `9.1-9.2 GB`
- `nvidia-smi` total used：约 `9.6-9.7 GB`

这符合之前设定的正式训练显存区间，低于 9GB 的 run 只作为 smoke/calibration。

## 3. 总体实验路线

本轮核心问题是：单一 style embedding 或简单 tokenizer 是否能把 Distinct5 的风格差异表达出来，并让 LANCET 真正执行出有区分度的风格变化。

我们从 baseline 开始，逐步尝试了三类设计：

1. tokenizer 表征改造。
2. content-guided 空间路由。
3. prototype-aware latent OT target queue。

后续又在 queue 和 tokenizer 上做了更细的消融：

1. easy-to-hard target curriculum。
2. sparse hard target exploration。
3. dual target latent mix。
4. auxiliary hard-target SWD。
5. content-adaptive VQ atom routing。
6. style-gated content router。

所有正式变体均基于 EMA VAE、packed latent cache、`virtual_length_multiplier=1.0`、每 epoch checkpoint 和 full eval。

## 4. 逐变体设计与结果

### Baseline：Direct Atom Residual

设计：

- 保留当前 direct style code。
- 加一个 shared atom residual。
- 每个 style 仍主要由 style id 直接索引得到。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 8 | 0.687649 | 0.452743 |
| best LPIPS | 1 | 0.686958 | 0.446756 |

结论：

- 这是一个弱 baseline。
- Distinct5 上 style 不够，LPIPS 也不够低。
- 它证明简单 direct atom residual 不足以解决表征问题。

### Variant A：Per-Class Prototype Tokenizer

设计：

- 每个 style 拥有类内 K 个 prototype。
- style code 由类内 prototype mixture 得到。
- 目标是表达类内方差，避免每类只剩一个均值向量。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 8 | 0.684946 | 0.462296 |
| best LPIPS | 1 | 0.681630 | 0.446381 |

结论：

- style 低于 baseline。
- LPIPS 只比 baseline 好 `0.000375`，没有实际意义。
- 类内 prototype 增加了容量，但没有形成更好的共享风格几何。
- 这条线拒绝。

### Variant B：Global VQ Codebook

设计：

- 所有 style 共享一个全局 VQ-style atom codebook。
- 每类只学习对全局 atoms 的分布。
- 目标是让风格成为共享原子的组合，而不是每类孤立向量。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 8 | 0.687321 | 0.444600 |
| best LPIPS | 8 | 0.687321 | 0.444600 |

结论：

- style 仍略低于 baseline。
- LPIPS 比 baseline 小幅下降约 `0.00216`。
- 只能弱保留：全局 VQ 几何比 class-local prototype 稍微更稳，但还没有解决 style 表达。

### Variant C：Content-Guided Spatial Routing

设计：

- 保留 class prototype tokenizer。
- 让 content latent 特征参与 16x16 spatial style prior 的路由。
- 目标是让不同内容区域选择不同空间风格先验，而不是全图固定一张 style map。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 2 | 0.690659 | 0.422593 |
| best LPIPS | 2 | 0.690659 | 0.422593 |

结论：

- 这是第一条同时提升 style 和 LPIPS 的路线。
- 相比 baseline，`clip_style` 从 `0.687649` 升到 `0.690659`。
- `content_lpips` 从 `0.446756` 降到 `0.422593`。
- 真正有用的是 content-conditioned spatial routing，不是单纯增加 class prototype 容量。

### Variant D：Global VQ + Content-Guided Spatial Routing

设计：

- 将 Variant B 的 global VQ style atoms 和 Variant C 的 content-guided spatial routing 合并。
- 目标是同时获得共享风格字典和内容自适应空间执行。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best point | 1 | 0.689761 | 0.415599 |

结论：

- LPIPS 进一步低于 C。
- style 没有超过 C。
- 说明 VQ + spatial routing 对内容保持有帮助，但 style 表达仍没有突破。
- 可保留，但不是当前主攻点。

### Variant E：Prototype-Aware Latent OT Queue

设计：

- 不再随机或均匀选择 target latent。
- 用 VAE latent 内部统计、梯度、频域能量构建 prototype-aware target queue。
- 训练仍然只用 LANCET/SWD 目标，不引入外部监督。
- 目标是降低 target distribution 噪声，让 SWD 拉到更合理的目标样本。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 1 | 0.697347 | 0.340965 |
| best LPIPS | 3 | 0.696186 | 0.333086 |

结论：

- 这是一次关键跃迁。
- style 从 C/D 的 `0.69` 区间提升到 `0.697`。
- LPIPS 从 D 的 `0.415599` 大幅降到 `0.333086`。
- 有效设计不是外部语义模型，而是“内部 latent 表征感知的 target queue”。
- E 是后续 F/H/K 系列的基础。

### Variant F：Annealed Prototype-Aware Latent OT Queue

设计：

- 继承 E。
- target queue 采用 easy-to-hard curriculum。
- active top-k 随 epoch 从干净 target 扩展到更难 target。
- 目标是保留 E 的 clean target 优势，同时逐步增加风格难度。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best point | 1 | 0.696915 | 0.318645 |

结论：

- F 是当前 Distinct5 LANCET 的 LPIPS 最优点。
- `content_lpips=0.318645`，明显优于 E。
- style 与 E 基本同级，略低但可接受。
- 说明 easy-to-hard curriculum 对内容保持非常有效。
- 当前保留 F 作为 LPIPS 压力基线。

### Variant G：Rank-Stratified Queue

设计：

- 继承 F 的 queue curriculum。
- 将随机 rank-biased sampling 改为 deterministic rank-stratified sampling。
- 目标是减少 batch 内 target hardness 抖动。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 2 | 0.697271 | 0.340381 |
| best LPIPS | 3 | 0.696674 | 0.332391 |

结论：

- 没有超过 F/E 的 Pareto。
- 去掉随机性后，style 稳定但 LPIPS 没有 F 好。
- 这说明 queue 需要 controlled stochastic exposure，不是完全确定性的 hardness coverage。
- 拒绝。

### Variant H：Fixed Clean Top-2 + Sparse Hard Exploration

设计：

- 固定使用 clean top-2 target。
- 以 15% 概率从 top-8 注入 hard target exploration。
- 目标是在 F 的 clean target LPIPS 优势和更强 style pressure 之间折中。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 2 | 0.699383 | 0.348407 |
| best LPIPS | 1 | 0.697363 | 0.321333 |

结论：

- H 是当前较均衡的点。
- `clip_style=0.699383`，接近 K 之前的 style 最优。
- LPIPS 最好 `0.321333`，只比 F 的 `0.318645` 略差。
- 设计判断：稀疏 hard target exposure 比 deterministic stratification 更有用。
- 当前保留 H 作为均衡基线。

### Variant I：Dual Target Latent Mix

设计：

- clean top-2 target 作为主目标。
- 从 top-8 取 hard target。
- 在 latent 空间中将 clean target 和 hard target 做 convex mix。
- 目标是平滑引入 hard target pressure。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best style | 2 | 0.696633 | 0.384613 |
| best LPIPS | 1 | 0.696485 | 0.347966 |

结论：

- style 和 LPIPS 都明显差于 F/H。
- latent premix 会制造一个不自然的中间 target manifold 点。
- 这不是好的 hard-target 表达方式。
- 拒绝。

### Variant J：Auxiliary Hard-Target SWD

设计：

- 不混合 latent。
- clean top-2 仍作为主 target。
- hard top-8 target 作为单独 auxiliary terminal SWD。
- 目标是用第二个 loss 表达 hard pressure，而不是改 target latent。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best point | 1 | 0.697653 | 0.332274 |

结论：

- 比 I 干净，但仍未超过 F/H。
- auxiliary SWD 更像额外终端拉力，不像更好的表征。
- hard target 最有效的方式仍是 sampler 里的稀疏暴露，而不是同时拉两个目标。
- 拒绝。

### Variant K：Content-Adaptive Global VQ Atom Routing

设计：

- 继承 H 的 fixed top-2 + 15% hard exploration。
- 在 global VQ atom logits 上加入 content-adaptive residual。
- router 只读取内部 VAE latent/content feature 统计。
- 最后一层 zero-init，初始行为等价 H。
- 目标是让同一个目标 style 根据 source/content 状态选择不同 style atom 组合。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best point | 1 | 0.700995 | 0.362294 |

诊断：

- `tokenizer_content_atom_delta_abs` 训练后到约 `0.034-0.037`。
- active atoms 约 `7.8`。
- router 确实学到了非零内容自适应残差。

结论：

- K 是当前 Distinct5 LANCET 的 style 最优点。
- 第一次突破 `clip_style=0.700`。
- 但 LPIPS 明显变差，`0.362294` 远差于 F/H。
- 这说明 content-adaptive atom routing 能增强风格区分，但会增加 endpoint movement。
- 保留为 style-only 方向，不作为综合最优。

### Variant L：K Router + F Annealed Queue

设计：

- 尝试把 K 的 content-adaptive router 和 F 的 easy-to-hard curriculum 合并。
- 目标是同时拿到 K 的 style boost 和 F 的 LPIPS 优势。

结果：

| 最优项 | epoch | clip_style | content_lpips |
|---|---:|---:|---:|
| best point | 1 | 0.697777 | 0.339710 |

诊断：

- router 确实学习：`tokenizer_content_atom_delta_abs` 均值约 `0.031`，最大约 `0.0395`。
- 但 style 没有保住 K 的提升。
- LPIPS 也没有回到 F/H。

结论：

- 简单合并 K 和 F 不成立。
- K 的 style gain 依赖 H 的 sparse hard-target exposure，而不是 F 的 easy-to-hard curriculum。
- 拒绝。

### Variant M：Style-Gated Content Router

设计：

- 继承 K。
- 给 content-adaptive atom-logit residual 加 target-style scalar gate。
- 设计动机：K 的 LPIPS 退化主要集中在 `Minimalism` target，希望让不同目标风格自己调节 router 强度。
- gate 初始化为 `1.0`，初始行为等价 K。

可用结果：

| epoch | clip_style | content_lpips |
|---:|---:|---:|
| 1 | 0.698726 | 0.346543 |
| 2 | 0.696810 | 0.345800 |

本地 smoke：

- Windows Python 3.12。
- `torch 2.11.0+cu128`。
- 本地 RTX 4070 Laptop GPU。
- batch 8。
- 12 step finite。
- checkpoint 正常保存。
- peak reserved memory 约 `1.83 GB`。

结论：

- M 没有进入 Pareto。
- style 低于 K/H。
- LPIPS 低于 K，但仍差于 F/H/J。
- style gate 不是已经解决的 LPIPS repair。
- 如果未来重做，需要更强的 gate prior 或正则，而不是默认沿用。

## 5. 当前最佳结果

| 模型 | 最优 epoch | clip_style | content_lpips | 决策 |
|---|---:|---:|---:|---|
| Baseline direct atom residual | 8 / 1 | 0.687649 | 0.446756 | 弱 baseline |
| Variant A class prototypes | 8 / 1 | 0.684946 | 0.446381 | 拒绝 |
| Variant B global VQ | 8 | 0.687321 | 0.444600 | 弱保留，仅 LPIPS 小幅好 |
| Variant C content-guided spatial | 2 | 0.690659 | 0.422593 | 保留 |
| Variant D VQ + content-guided | 1 | 0.689761 | 0.415599 | 保留 |
| Variant E latent prototype OT queue | 1 / 3 | 0.697347 | 0.333086 | 强保留 |
| Variant F annealed prototype OT queue | 1 | 0.696915 | 0.318645 | 当前 LPIPS 最优 |
| Variant G stratified prototype OT queue | 2 / 3 | 0.697271 | 0.332391 | 拒绝 |
| Variant H hard-explore prototype OT queue | 2 / 1 | 0.699383 | 0.321333 | 当前均衡点 |
| Variant I dual-target latent mix queue | 2 / 1 | 0.696633 | 0.347966 | 拒绝 |
| Variant J auxiliary hard-target SWD queue | 1 | 0.697653 | 0.332274 | 拒绝 |
| Variant K content-adaptive VQ atom routing | 1 | 0.700995 | 0.362294 | 当前 style 最优，style-only 保留 |
| Variant L content-adaptive annealed queue | 1 | 0.697777 | 0.339710 | 拒绝 |
| Variant M style-gated content router | 1 / 2 | 0.698726 | 0.345800 | 拒绝，部分结果 |

当前 Pareto 结论：

- 如果优先 `clip_style`：选 K epoch 1，`0.700995 / 0.362294`。
- 如果优先 `content_lpips`：选 F epoch 1，`0.696915 / 0.318645`。
- 如果要相对均衡：选 H，style 可到 `0.699383`，LPIPS 最好 `0.321333`。

## 6. 速度与瓶颈结论

正式远程 b44 训练速度：

- baseline 8 epoch 训练约 `8.4 min`。
- epoch 1 因 warmup/DataLoader 较慢，约 `70 s`。
- 后续 epoch 约 `60-62 s/epoch`。
- F/H/K/L/M 三 epoch短程实验基本都在这个速度区间。

full eval 速度：

- 每个 checkpoint 的 trainer-level full eval wall 约 `145-153 s`。
- stable profile 中：
  - LANCET latent generation：约 `5.1-5.6 s / 750`。
  - VAE decode：约 `52.6-52.8 s / 750`。
  - eval metric loop：约 `24.1-24.6 s / 750`。

关键瓶颈：

- LANCET 本身不是 eval 主要瓶颈。
- VAE decode 是最大稳定瓶颈。
- PNG 保存、CPU copy、metric loop 也有明显成本。
- tokenizer/router 变体对训练/推理速度影响很小，至少不是当前主要 infra 瓶颈。

## 7. 表征层面的阶段性结论

本轮最重要的结论不是“哪个 tokenizer 参数更大”，而是以下几点：

### 7.1 单纯增加 tokenizer 容量没有用

Variant A 的 class-local prototypes 失败，说明只给每类更多 prototype 不会自动得到更强风格表达。

Variant B 的 global VQ 只带来很小 LPIPS 改善，也没有 style 突破。

结论：风格表征必须被 LANCET 执行路径读到，不能只停留在 style code 空间。

### 7.2 Content-guided spatial routing 是有效方向

Variant C/D 明显降低 LPIPS，并带来一定 style 提升。

这说明“风格怎么作用在不同内容区域上”比“每类有几个向量”更重要。

### 7.3 Prototype-aware latent queue 是最大提升来源

Variant E/F/H 的跃迁主要来自 target queue。

这说明 Distinct5 的训练难点不只是模型容量，而是 target distribution 噪声。用内部 VAE latent 统计构建更干净的 target queue，能显著改善 SWD 训练信号。

### 7.4 Hard target pressure 需要稀疏、随机、离散

F 的 easy-to-hard curriculum 给了最好 LPIPS。

H 的 sparse hard exploration 给了更强 style。

I 的 latent mix 和 J 的 auxiliary SWD 都没有超过 H/F。

结论：hard target 最好以 sampler 中的稀疏暴露出现，而不是把多个 target 混成一个 latent，也不是同时施加两个终端 SWD。

### 7.5 Content-adaptive atom routing 能提 style，但会伤 LPIPS

K 证明 content-adaptive atom routing 能把 style 推到 `0.700995`。

但它明显提高 LPIPS，说明 router 增强了 endpoint movement。后续不能简单加大 router gain。

M 的 style gate 没有修复这个问题，说明“给 gate 自己学”还不够。

## 8. 下一步建议

短期应保留三条基线：

- F：LPIPS 最优。
- H：均衡点。
- K：style-only 最优。

下一轮实验应该围绕 E/H/K 系列做小网格，而不是重新发明大 tokenizer：

1. hard exploration probability：例如 `0.05 / 0.10 / 0.15 / 0.20`。
2. active top-k：例如 fixed `1 / 2 / 3 / 4`。
3. content-adaptive router gain：降低 K 的 router 强度，看能否保住部分 style boost 同时降低 LPIPS。
4. route temperature：控制 atom mixture 稀疏度。
5. prototype count：只在 E/H/K queue 已稳定后调整。
6. generated-delta rank 诊断：验证风格是否真的变成不同执行方向，而不是共享改色方向。

不建议继续的方向：

- class-local prototype tokenizer 单独加容量。
- deterministic rank stratification。
- clean/hard target latent premix。
- simultaneous auxiliary hard-target SWD 作为默认机制。
- 直接把 K router 和 F curriculum 粗暴相加。
- 依赖 DINO/CLIP/classifier 训练监督。

## 9. 最终阶段判断

Distinct5-512 这一轮已经明确了一个较可靠的研究方向：

```text
内部 VAE latent prototype queue
+ sparse hard target exposure
+ content-aware spatial/style routing
```

当前还没有得到一个同时支配 F/H/K 的模型：

- F 守住内容。
- H 是折中。
- K 提升 style 但牺牲内容。

因此下一步不是继续堆 tokenizer 参数，而是约束“风格表征如何被执行”。如果新的表征不能提升 generated-delta 的可分性，或者只是在 endpoint 上增加无组织位移，就很可能复现 K 的问题：style 指标上升，但 LPIPS 变差。
