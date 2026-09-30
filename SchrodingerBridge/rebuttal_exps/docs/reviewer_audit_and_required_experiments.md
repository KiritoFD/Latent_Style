# WEAVE 审稿视角实验审计与补充实验计划

**日期**: 2026-07-16  
**审计对象**:

- `SchrodingerBridge/rebuttal_exps/docs/stability_experiments_summary.md`
- `WEAVE/aaai2027_v4/paper.tex`
- `WEAVE/aaai2027_v4/supplement.tex`
- `WEAVE/config.json`, `WEAVE/inference.json`, `WEAVE/flow.py`
- 已有稳定性、早停、TGT 敏感性和 ArtFID 原始记录

本文档只定义证据、实验和写作修正，不直接修改论文。目标是用尽量少的新增训练，形成审稿人可以复核的证据链。

## 1. 执行结论

现有补充实验不能原样写入论文。主要问题不是结果差，而是部分结论超出了实验协议能够支持的范围。

必须优先完成的工作如下，按审稿风险排序：

1. **重做当前 1.04M 主模型的核心机制消融。** 当前正文消融表的 baseline 数值不是主表 checkpoint，却在 caption 中被描述成当前 D5 模型。这是最严重的可比性问题。
2. **验证内部早停是否真的选择接近外部指标峰值的 checkpoint。** 现有结果说明规则在 3 个 seed 和 3 个 probe batch 上选择 epoch 3-4，但没有给出每个 seed 的完整外部指标曲线和选择 regret。
3. **重做参考池与 IDT-TGT 稳定性分析。** 当前 Exp2 实际测量 WEAVE 的 DINO-S 对参考池重采样的敏感性，并没有测量 TGT 内容边界；其中所谓 `IDT floor` 也不是真正的 identity 输出。
4. **统一 ArtFID 的输入 manifest，并拆开报告 FID 与 LPIPS 项。** 当前 WEAVE 与 StyleAligned/Z-STAR 使用的 source manifest 不同，不能作为最终公平比较。现有数据也不能只解释成“ArtFID 奖励 IDT”，因为 WEAVE 的原始 FID 项同样比 IDT 差。
5. **若正文保留“matched HH-head ablation”这一表述，必须在当前架构上补一个真正匹配的 HH-head run。** 旧 base-WEAVE 上的负结果不足以支持当前 1.04M 架构的匹配消融声明。

推荐但并非必须的工作：在当前架构上重做 Haar level/basis 小消融；补齐 AdaIN scale 在 2.0 两侧的局部扫描；加入一个中性措辞的高频压力测试定性图。

不建议优先做的工作：完整人类研究、为所有最新方法重新调参、把所有历史架构探索都塞进补充材料。它们的成本高，且不能修复当前最核心的证据一致性问题。

## 2. 现有 Exp1-Exp6 的证据审计

### 2.1 Exp1a: `lambda_LL` 扫描

**可以保留的证据**:

- 新训练的中间点使用同一当前配置，并读取 `epoch_0004.pt`。因此，在确认 resolved config 只差 `spectral_w_ll`、输出目录和早停开关后，它们可以作为“固定 epoch 4 训练预算下的敏感性扫描”。
- `lambda_LL` 在中等区间内没有出现灾难性失稳，CLIP-S 与 DINO-C 呈现可解释的相反变化。

**不能直接保留的表述**:

- 文档把数值跨度写成“方差”。例如 `0.4850-0.4857` 是 range，不是 variance；而且表内还存在 `0.4847` 和 `0.4848`，完整 range 应按原始结果重新计算。
- `lambda_LL=0` 和 `2.0` 来自旧 `ablation_v2` 的 epoch 5 checkpoint，不能与当前架构的 epoch 4 中间点混成一条严格匹配的曲线。
- 固定 epoch 4 扫描只能说明固定预算敏感性。若声称“完整 internal-stop pipeline 对所有参数都稳定”，则每个点必须应用同一个冻结早停规则。
- `0.3` 可以称为当前选择的 operating point，不能仅凭这组混合来源数据称为经过证明的唯一 sweet spot。

**最小处理**:

1. 对每个中间点保存 resolved config 和 SHA-256。
2. 自动 diff 配置，确认除被扫参数、保存目录、`internal_early_stop_enabled` 外没有差异。
3. 仅报告当前 epoch 4 的 `0.1-0.5` 点；旧 `0`/`2.0` 单独标为 historical extremes，或在当前架构上重跑。
4. 同时报告 mean、standard deviation 和 range，不再混用“方差”和“跨度”。

### 2.2 Exp1b: LL blend `alpha` 扫描

**可以保留的证据**:

- 当前架构在固定 epoch 4 下，`alpha=0.1-0.5` 没有训练崩溃。
- DINO-S 在 `0.3` 处达到本扫描最高值，CLIP-S 随 alpha 增加。

**必须修正的解释**:

- 文档称“LPIPS 单调递减表示内容保留变差”，这与论文指标定义相反。LPIPS 越低通常表示越接近 source，按该指标应解释为内容保留更强。
- 同时，DINO-C 随 alpha 增加而下降。因此这组扫描显示的是 **两个内容代理发生分歧**，不能写成“exactly as theory predicts”。
- `0.4843-0.4918` 也是 range，不是 variance。
- “Pareto-optimal”需要明确指标方向和支配关系。不要把一个经验选择包装成无条件最优。

**最小处理**:

- 保留表格，改成逐指标描述。
- 明确写出 LPIPS 与 DINO-C 在该方向上不一致，并把这种分歧作为使用多指标的理由。
- 若配置 provenance 不一致，则重跑所有 5 个点；若一致，不需要重新训练。

### 2.3 Exp1c: stepwise AdaIN scale 扫描

**可以直接使用的部分**:

- 四个点来自同一个 checkpoint，仅改变推理 scale，协议是匹配的。
- 结果可以说明 `1.0-1.5` 区间变化较小，而 `2.0` 发生明显非线性变化。

**当前证据不足的部分**:

- 最高只测到 `2.0`，所以不能称 `2.0` 为 sweet spot，也不能称其为“fully activates without over-stylization”。没有观察 2.0 之后的下降侧。
- `1.0-1.5` 的 DINO-S range 是 `0.0013`，不是“小于 0.001”。
- `2.0` 提升 DINO-S/DINO-C/LPIPS，但 CLIP-S 下降，不能称所有风格指标同时 Pareto 改进。

**最小补点**:

- 当前 checkpoint 上补 `1.75, 2.25, 2.5`。如 2.25 已明显恶化，可停止，不需要继续大范围调参。
- 正文不放整张扫描图；补充材料报告四个指标的原始值。

### 2.4 Exp2: 所谓 IDT-TGT 边界方差

当前脚本 `exp2_idt_variance.py` 实际执行两件事：

1. 重采样 target-style reference pool，重新计算 **WEAVE 输出的 DINO-S**。
2. 从 `metrics.csv` 中挑出 `src_style == tgt_style` 的生成结果，并对其 CLIP-style 分数做 bootstrap，称为 `IDT floor`。

这两件事都不是 TGT 内容边界，也不是严格的 IDT 输出：

- 真正的 IDT 是 `y_IDT=x`，不是同风格生成结果。
- 真正的 TGT 稳定性应改变 `r_s`，然后观察 source-to-TGT 的 LPIPS/DINO-C。
- DINO-S 使用 max cosine；有放回抽样造成重复 reference，会改变有效池大小和 max 分布。bootstrap 均值变化并不表示模型或边界变化。
- 30 次重复不足以稳定估计 2.5%/97.5% 分位点。

因此该实验必须改名为 **DINO-S reference-pool sensitivity**，或者按第 5 节重做。不能用当前结果声称 “TGT boundary sigma < 0.01”。

现有可用证据是 `WEAVE/docs/reproduction/tgt_reference_sensitivity.csv`：前 5 个确定性 TGT reference 给出 LPIPS `0.7522-0.7833`、DINO-C `0.1891-0.2455`。这直接支持“主要内容坍塌判断不依赖第一张 TGT 图”，但样本仍较少。

### 2.5 Exp3: 内部梯度门控鲁棒性

**真实实验规模**:

- seed 7/42/123，probe batch 固定为 4。
- probe batch 2/4/8，seed 固定为 42。
- 共 5 个 setting，是 one-factor-at-a-time 交叉，不是 3x3 的 9 个完整组合。

旧日志还混用了 absolute crossing rule 与后来修正的 relative drop rule。不能直接读取旧 `stop_requested` 字段后混成一张表。当前可信记录是：

| Setting | Selected epoch | DINO-S | CLIP-S | LPIPS | DINO-C |
|---|---:|---:|---:|---:|---:|
| seed42, batch4 | 4 | 0.4915 | 0.7126 | 0.2596 | 0.8103 |
| seed7, batch4 | 4 | 0.4910 | 0.7140 | 0.2668 | 0.8076 |
| seed123, batch4 | 3 | 0.4862 | 0.7144 | 0.2552 | 0.8040 |
| seed42, batch2 | 4 | - | - | - | - |
| seed42, batch8, relative rule | 4 | - | - | - | - |

这可以支持“在测试的 seed 上选择 epoch 3-4，在 batch 2/4/8 上选择 epoch 4”。它还不能证明选中的 epoch 是每个 seed 的外部风格峰值，也不能证明跨数据集泛化。需要按第 4 节补 oracle 对照。

### 2.6 Exp4: level 与 wavelet basis

该实验来自旧架构，只报告 CLIP-S/LPIPS：

| Setting | CLIP-S | LPIPS |
|---|---:|---:|
| Haar level 1 | 0.7261 | 0.3288 |
| Haar level 2 | 0.7301 | 0.3402 |
| db2 level 1 | 0.7258 | 0.3288 |
| db2 level 2 | 0.7298 | 0.3398 |

它只能作为 historical design diagnostic。现有数据不支持以下说法：

- db2 提升 CLIP-S。实际 level 1 上是 `-0.0003`。
- level 2 在当前模型上是 Pareto sweet spot。
- 训练时间增加 X%。没有测量值。
- VAE decoder 导致 ringing。没有定量或预先定义的定性评估。

若不想重跑，应将正文改为“we fix one-level Haar and leave basis/depth to future work”。若要保留比较结论，按第 8 节在当前模型上重跑。

### 2.7 Exp5: 高频压力测试

6 个 `[content | style | output]` 三联图只能用于展示行为和局限，不能证明：

- 省略 HH head 在整体上鲁棒。
- WEAVE 适用于“95% styles”。
- 被选中的 3 张 style image 具有总体代表性。

安全用法是放一张紧凑补充图，并在 caption 中写明样例是为了展示 phase-coherent texture 的失败模式，而不是随机抽样的总体性能证明。

### 2.8 Exp6: 最新 baseline 核实

“SCAdapter、STRDP、SkipInject 都不存在”这一结论是错误的，不能写入 rebuttal 或论文：

- [SCAdapter](https://arxiv.org/abs/2512.12963) 是真实论文。
- [STRDP](https://arxiv.org/abs/2410.01366) 是真实论文。
- [SkipInject](https://openreview.net/forum?id=FKQvt1yaEf) 是真实论文。

`FreqFlow`、`FA-VAE`、`LWD` 等名称存在同名或误名风险。它们必须按标题、任务、作者和官方实现逐一核实。可能相关的是 WF-VAE，而不是“FA-VAE”。不要把搜索不到的缩写直接判为 hallucination。

这部分首先是文献审计，不是 GPU 实验。只有满足“同任务、公开代码、可接收相同 content/style 输入”的方法才进入直接复现候选。

## 3. P0: 冻结可复现实验协议

在新增训练前先建立一个不可变实验清单。否则不同 agent、旧输出目录和缓存很容易再次混合。

每个正式实验必须记录：

```text
experiment_id
git_commit
resolved_config_path + sha256
inference_config_path + sha256
checkpoint_path + sha256
dataset_manifest_path + sha256
evaluator_commit/file hashes
seed
training epochs and selected epoch
generated file count
metric raw JSON/CSV paths
GPU, dtype, batch size, cache state, wall-clock definition
```

正式 D5 board 固定为同一 750 个 source-target 请求。所有方法必须使用同一个 pair manifest，不能只保证“每类都是 30 张”而实际图片不同。

### 当前需要先消除的 artifact drift

1. `WEAVE/config.json` 的 `_main_table_metrics` 仍指向旧 `brk_a` checkpoint，与当前 `0.4915/0.7126/0.2596/0.8103` 主结果不一致。
2. config 写有 `t_sampling_mode=logit_normal`，但当前 `WEAVE/flow.py::_sample_t` 实际直接调用 uniform sampling。正文的 uniform 公式与代码一致，错误在于配置存在无效或陈旧字段。
3. 当前训练确实使用 `bridge_sigma=0.02` 的轻微 Brownian-shaped 扰动输入；正文把路径写成完全无扰动直线会省略实现细节。velocity label 仍是与 t 无关的 endpoint displacement，因此快速收敛解释可以保留，但应称为 lightly perturbed rectified path。
4. 架构图必须继续使用用户指定的 `aaai_arch_diagram_v16_staggered_bundle.drawio.png`，但同一设计源内的文字需要与实现一致。当前图中 `903K`、`Endpoint-only Injection` 和 `Diagonal WCT` 分别与 1.04M、every-step AdaIN 和正文 AdaIN 定义冲突。

## 4. P0 实验 A: 当前主模型的内部早停有效性

### 审稿问题

冻结的内部梯度规则是否在不同 seed 上选择接近 DINO-S 峰值的 checkpoint，而不是只在设计规则所用的 seed 42 上碰巧吻合？

### 最小协议

1. 冻结当前 relative rule，不再根据结果改阈值。
2. 使用 seed 42、7、123 的完整 15-epoch checkpoint 与 internal probe 日志。若 checkpoint 已存在，不重新训练。
3. 对每个 seed 的 epoch 1-15 离线评估 DINO-S、CLIP-S、LPIPS、DINO-C。
4. 用内部规则独立选出 `e_internal`。
5. 计算：

```text
e_oracle = argmax_epoch DINO-S
DINO regret = DINO-S(e_oracle) - DINO-S(e_internal)
epoch offset = e_internal - e_oracle
content delta at selected epoch
```

6. 在 seed 42 的同一 checkpoint sequence 上，离线重算 probe batch 2/4/8。由于 probe 使用 `fork_rng` 且不更新模型，batch 变化不应要求重新生成图像；应额外比较 checkpoint hash，确认训练轨迹未被 probe 改写。

### 报告方式

- 表格按 seed 报 `e_internal`, `e_oracle`, regret 和 selected metrics。
- 报 3 个 seed 的 mean +/- standard deviation。
- 明确写成“offline validation of an online metric-free rule”。外部 DINO 只验证规则，不参与在线训练或 checkpoint 选择。
- 不写“architecture-independent”或“all 9 configurations”。

### 可选增强

在 P2A 或 R5 上用冻结阈值做一次完整曲线验证。只有在正文声称规则跨数据集泛化时才需要。

## 5. P0 实验 B: 正确的 IDT、TGT 与参考池稳定性

该问题要拆成两个统计对象，不能用一个 bootstrap 混在一起。

### B1. IDT 与 WEAVE 的 DINO-S margin 对 reference pool 的敏感性

对每个 target style 的 reference pool 做固定大小、无放回子集抽样。建议 pool size `m=8` 和 `m=16`，每个设置 1,000 次：

1. 同一个 reference subset 同时用于 WEAVE 和真正的 IDT 输出 `x`。
2. 每次记录 `DINO-S_WEAVE`, `DINO-S_IDT` 和配对 margin
   `Delta_style = DINO-S_WEAVE - DINO-S_IDT`。
3. 报 margin 的 median、95% interval、最小值，以及 `Delta_style > 0` 的抽样比例。
4. 保持 pool size 固定，避免 max-cosine 因重复 reference 改变有效样本数。
5. source image 若属于目标风格且出现在 reference pool，必须按当前 evaluator 规则排除 self-match。

这回答的是“超越 IDT 的结论是否依赖某一参考池”，而不是把 WEAVE 分数本身误称为 TGT boundary。

### B2. TGT 内容 anchor 对 reference image 的敏感性

1. 对每个 style，枚举所有可用候选 `r_s`，或至少使用预先固定的前 10 张，而不是只挑视觉上合适的样例。
2. 对每个候选计算 750 请求上的 LPIPS(source, TGT) 和 DINO-C(source, TGT)。
3. 报每个 style 和全 board 的 median、range、95% interval。
4. 对需要判断是否跨越 TGT 的方法，报告它在多少候选 TGT anchor 下仍越界。
5. 不报告 TGT 的 DINO-S 作为模型质量。若 `r_s` 同时属于 DINO-S reference pool，其 max cosine 为 1 是定义结果。

已有前 5 张 reference 的结果可作为预检查，但正式补充材料最好扩大到所有候选或固定前 10 张。

## 6. P0 实验 C: 同 manifest 的 ArtFID 审计

### 当前证据的正确解读

当前记录是：

| Method | ArtFID | raw FID | source LPIPS |
|---|---:|---:|---:|
| IDT | 216.51 | 215.51 | 0.000 |
| WEAVE | 295.27 | 230.52 | 0.283 |
| SaMam | 297.32 | 231.87 | 0.283 |
| Seedream 4.5 | 310.97 | 217.37 | 0.422 |
| Z-STAR | 332.91 | 251.99 | 0.321 |
| StyleAligned | 368.63 | 202.67 | 0.822 |

这些数据说明 ArtFID 的 LPIPS 因子确实强烈惩罚内容变化。例如 StyleAligned 的 raw FID 最好，但 composite 最差。然而 WEAVE 的 raw FID `230.52` 也差于 IDT 的 `215.51`。因此不能把 WEAVE 输给 IDT 完全归因于 content multiplier，更不能直接宣布指标无效。正确结论是 DINO/CLIP 与 Inception-FID 在该 art-to-art board 上给出不同的风格排序。

### 必须重做的协议

1. 为 WEAVE、IDT、SaMam、StyleAligned、Z-STAR 使用同一个 D5 pair manifest。
2. 使用同一 target reference set、相同 resize、相同 Inception implementation。
3. 每个方法验证 750/750 文件与 source 请求一一匹配。
4. 分别报告 per-style raw FID、source LPIPS 和 ArtFID，再报告 5 个 style 的平均值。
5. 对小样本 FID 同时报告 KID 或重复子采样区间。KID 对小样本更适合，不要把单个 FID 数字写成高精度稳定排名。
6. 图中不只给 composite bar；补充材料提供 component table。

### 写作边界

- 可以说 ArtFID 与 DINO-S/CLIP-S disagreement，且 composite 受 source-distance 项强烈影响。
- 可以说 IDT 显式暴露了 composite metric 可能偏好 no-op 的情况。
- 不应说 ArtFID 本身错误、SeeDream 的高 ArtFID 证明指标无效，或 WEAVE 在 raw FID 上也优于 IDT。
- API 输出缺失细节只需在补充材料的 protocol note 中说明，不在正文图上标 `720/750`。

## 7. P0 实验 D: 当前 1.04M 模型的匹配机制消融

### 为什么这是最高优先级

正文当前消融 baseline 是：

```text
CLIP-S 0.727, LPIPS 0.343, DINO-S 0.483, DINO-C 0.755
```

主表当前模型是：

```text
CLIP-S 0.7126, LPIPS 0.2596, DINO-S 0.4915, DINO-C 0.8103
```

两者不是同一个 checkpoint/architecture，但 caption 把前者描述为带 oriented target-HF route、epoch 4 的当前 D5 模型。审稿人一旦对照主表，会直接怀疑所有机制结论。

### 最小消融矩阵

**无需重训的 current-checkpoint inference ablations**:

| ID | Variant | 目的 |
|---|---|---|
| D0 | Full current WEAVE | 当前基准 |
| D1 | AdaIN scale = 0 | 验证 stepwise statistics 的贡献 |
| D2 | Disable oriented target-HF residual route | 验证新增图像级 HF 信息路径是否活跃 |

**需要当前架构从头训练的 matched ablations**:

| ID | Variant | 目的 |
|---|---|---|
| D3 | `lambda_LL=1.0` | 直接检验 unweighted LL dominance，而不是只测 `lambda_LL=0` |
| D4 | Direct target endpoint, no source-aligned LL | 检验 source-aligned endpoint 对内容保持的作用 |
| D5 | Enable learned HH velocity head, same HH target | 支持或推翻“不学习 HH head”的当前架构决定 |

所有训练 variant 使用同一 seed 42、同一初始化规则、同一 15-epoch schedule，并关闭 active stop 但保留 probe。之后用预先冻结的内部规则选择 checkpoint。这样每个 run 都保留完整曲线，同时选择不依赖外部图像指标。

### 必须输出的证据

- DINO-S、CLIP-S、LPIPS、DINO-C，完整 750 board。
- `shared LL/HF gradient ratio` 与各 band gradient mass。
- selected epoch 与训练时间。
- full vs D3 的梯度曲线，用来直接支撑“频率加权改变优化竞争”。
- full vs D5 的 HH loss、HH head gradient、最终 style/content metrics。

### 论文中只保留的核心行

主论文消融表不需要塞入几十个历史尝试。只保留 D0-D5 中最能对应三个方法贡献的行：source-aligned endpoint、band weighting、stepwise AdaIN、target-HF route、HH decision。历史 `lr`、单步 solver、旧 cross-attention 等诊断放补充材料或删除。

## 8. P1 实验 E: 当前架构的 wavelet level/basis

这是条件性实验：

- 如果正文只承认“we fix one-level Haar”，并在 Limitations 中说明多层/其他 tokenizer 未系统研究，则不必新增训练。
- 如果正文要声称 one-level Haar 是 measured Pareto choice，必须在当前 1.04M 架构上重做。

推荐最小 2x2：

| Basis | Level |
|---|---:|
| Haar | 1 |
| Haar | 2 |
| db2 | 1 |
| db2 | 2 |

固定 seed、训练预算和 checkpoint 选择规则，报告四个主指标、参数量、训练 wall time、推理 wall time。不要在没有定义评价协议时声称 ringing artifact。

## 9. P1 实验 F: HH/phase-coherent style 压力测试

### 定量子集定义

不要按结果手工选择 3 张图。先在所有 target reference 上计算：

```text
HH energy ratio = ||HH||^2 / (||LH||^2 + ||HL||^2 + ||HH||^2)
```

以训练前固定的 top quartile 作为 high-HH subset，其余作为 control subset。对 D0 和 D5 分别报告两个 subset 的四个主指标。

### 定性图

现有 6 个三联图可以压缩为一张 2x3 panel。caption 使用中性描述：

- 示例用于观察 phase-coherent curves/brush patterns。
- 点对点 HH phase 在 unpaired target 中没有 source registration。
- 输出可能匹配统计但不能复制精确相位。
- 不声称样例代表 95% 的风格。

### 可选 phase-insensitive variant

若 D5 的 learned HH MSE head 明显为负，可选做一个 pooled local-energy 或 channel-moment HH loss。该 variant 只用于回应“是否所有 HH 学习都无效”，不应在没有收益时扩张成新 method。

## 10. P1 实验 G: 超参数跨 board 迁移

Reviewer 真正关心的是参数是否每个 dataset 都重新调过，而不是 D5 上曲线是否平滑。

最省实验的回答方式：

1. 明确列出 D5、P2A、R5 是否共享 `lambda_LL=0.3`, `alpha=0.3`, AdaIN scale 2.0 和 8 steps。
2. 若确实共享且没有 board-specific search，直接作为 protocol fact 报告，不需要每个 board 完整 sweep。
3. 若希望更强证据，只在 P2A/R5 上测试 AdaIN scale `1.5, 2.0, 2.5`，无需重新训练。
4. 不要写“hyperparameters transfer without tuning”除非训练与推理配置记录能证明。

## 11. Baseline 公平性与是否需要新复现

### 公平性原则

- 所有可运行 baseline 使用作者推荐配置和公开 checkpoint，不针对 IDT-TGT 指标重新调参。
- IDT-TGT 是诊断，不是优化目标。要求其他方法“针对 sandwich 重新调优”反而会造成 benchmark tuning。
- 生成分辨率、content/style 请求、评价脚本和 reference pool 必须相同。
- 参数量、训练时间和推理时间明确区分 `source-reported`、`reproduced on RTX 3060` 和 `API/withheld`。
- 训练免费方法仍应报告 backbone 参数、峰值显存和推理时间，不能因为 trainable params 为 0 就把模型成本写成 0。

### 新 baseline 的决策门槛

只有同时满足以下条件才做直接实验：

1. 论文真实且任务是 arbitrary/reference-based style transfer。
2. 有作者官方代码和权重。
3. 可以接收当前 D5 content/style 请求，不需要文本 prompt 人工工程。
4. license 和环境可运行。
5. 可以生成同一 750 manifest。

STRDP、SkipInject、SCAdapter 首先加入 Related Work 并说明注入机制差异。完成官方仓库核实后，再选最多 1-2 个最接近且可复现的方法进入补充比较。FreqFlow/WF-VAE 等生成或 tokenizer 工作主要用于定位，不必强行作为风格迁移 baseline 跑主表。

## 12. 不建议追加的实验

### 大规模人类研究

人类研究不是当前最优先缺口。若不做，应避免声称 DINO/CLIP 等价于主观风格质量，并用随机、完整的定性 board 和多指标 disagreement 代替。若最终决定做，只接受预先固定样例、盲法随机顺序、明确问题和完整置信区间的 pairwise preference；不做小规模、可挑结果的“用户投票”。

### 全量 3x3 seed x probe batch 训练

probe batch 不改变模型优化，完整 9 次训练的新增信息有限。当前 3-seed + 3-batch 的交叉设计足够，只要准确描述为 5 个设置，并完成 oracle checkpoint 对照。

### 把历史 30 个 probe 全部写入论文

大量旧 run 使用不同 backbone、checkpoint family 或 fine-tune 起点。它们适合作为内部工程档案，不适合作为当前主模型的 matched evidence。

## 13. 当前论文的非实验性高风险问题

这些问题不需要 GPU，但应在加实验前修正：

1. **主消融表 baseline 与主表不一致。** 必须替换为当前 matched ablation 或明确标成 historical diagnostics。
2. **架构图文字与实现冲突。** 在用户指定的原图设计中修正 903K、endpoint-only 和 WCT 标签，不换另一张图。
3. **正文 basis 结论写反。** 旧数据中 db2 的 CLIP-S 略低，不是略高。
4. **AdaIN 叙述不一致。** 实现是每个 Euler step 后执行；Conclusion 中“without repeatedly perturbing the trajectory”与实现冲突。
5. **路径公式省略训练噪声。** uniform t 与代码一致，但应说明 `sigma=0.02` 的轻扰动输入，或在补充材料给出实际训练式。
6. **内部早停的证据被写得过强。** 当前只能说 tested seeds/batches 选择 epoch 3-4，不能说已证明跨架构、跨数据集稳定。
7. **ArtFID 解释过强。** 当前 raw FID 也不利于 WEAVE，必须承认 metric disagreement，而不是只归因于 LPIPS 项。
8. **“60+ GB”范围过宽。** 应限定为特定大 diffusion/FLUX/Qwen variants，并给引用或测量来源，不能让读者理解为所有 training-free 方法都需要 60+ GB。
9. **Related Work 的近邻工作需要核实后补齐。** 不要使用未核实缩写，也不要把真实论文称为 hallucination。
10. **补充材料像内部交接文档。** `Next Architecture Plan`、keep/avoid 工程决策、30 个目录 ledger、raw-to-public mapping 不应占据科学补充材料的主体。保留可复现实验协议、matched tables、统计分析和必要命令即可。
11. **补充 PDF 存在多处 overfull box。** 当前 log 最大超过 70 pt；需要做一次可视排版检查，尤其是长路径和宽表。

当前主文 PDF 没有 undefined citation/reference，但以上语义矛盾比交叉引用警告更严重。

## 14. 推荐写入位置

### 主论文

- 主消融表替换为当前 matched D0-D5 的精简结果。
- Generalization/Robustness 只用一句话概括 3-seed 和 probe-batch 结果，并引用 supplement。
- Figure 1 的 ArtFID caption 说明它是 composite，并指向 supplement 的 component table。
- Limitations 明确单层 Haar、固定 VAE、HH phase-coherent styles 和 metric disagreement。
- 不放完整超参数折线图，不扩张 abstract。

### 补充材料

建议按以下科学结构重排：

1. Frozen protocol and artifact hashes
2. Dataset and metric definitions
3. Current-model matched mechanism ablations
4. Internal stopping rule: full curves, seed robustness, oracle regret
5. Hyperparameter sensitivity with corrected statistics
6. IDT/reference-pool margin and TGT reference sensitivity
7. ArtFID component decomposition and matched-manifest audit
8. HH/high-frequency stress test and optional basis/level ablation
9. Full board results and qualitative examples
10. Reproduction commands

删除内部 roadmap 和无法映射到当前方法的历史实验目录列表。

## 15. 执行顺序与停止条件

### 第一批: 不需要重新训练

1. 冻结 manifest/config/checkpoint/evaluator hashes。
2. 验证 Exp1a/1b 每个 run 的 resolved config provenance。
3. 对现有 seed 7/42/123 checkpoint 做每 epoch 外部评估。
4. 重做 reference-pool paired margin 与 TGT candidate 枚举。
5. 统一 ArtFID manifest 并重算可用已有输出；若 source manifest 不同，则进入第二批重新生成。
6. 当前 checkpoint 做 no-AdaIN 和 no-oriented-route inference ablation。

### 第二批: 最少新增训练

1. 当前架构 `lambda_LL=1.0`。
2. 当前架构 direct target endpoint。
3. 当前架构 learned HH head。

每个 run 最多 15 epochs，保留完整 checkpoint，但 checkpoint 选择使用冻结内部规则。完成这三项后，先更新证据矩阵，不继续无目标架构搜索。

### 第三批: 条件性实验

- 只有保留 Haar optimality claim 时跑 2x2 basis/level。
- 只有保留 `scale=2.0 sweet spot` 时补 2.0 右侧点。
- 只有可获得官方实现且同任务时新增 STRDP/SkipInject/SCAdapter 中最多两个 baseline。

### 停止条件

满足以下条件即可停止新增实验并转入写作：

- 主模型和核心消融使用同一 protocol。
- 早停规则有独立 seed 的 oracle-regret 对照。
- WEAVE-IDT style margin 对 reference subset 选择稳定，或敏感性被诚实报告。
- TGT 内容判断对预定义 reference candidates 稳定，或敏感性被诚实报告。
- ArtFID 使用同 manifest，并同时报告 raw FID 与 LPIPS 项。
- HH claim 与当前架构证据一致。
- 所有表中数值都能追溯到 config、checkpoint、manifest 和原始 JSON/CSV。

## 16. 结果文件命名建议

```text
WEAVE/experiments/rebuttal_20260716/
  protocol_manifest.json
  configs_resolved/
  checkpoint_hashes.csv
  early_stop/
    per_epoch_metrics_seed{42,7,123}.csv
    oracle_regret.csv
  reference_sensitivity/
    dino_margin_pool8.csv
    dino_margin_pool16.csv
    tgt_candidate_sensitivity.csv
  artfid/
    canonical_pair_manifest.csv
    components_by_style.csv
    summary.csv
  matched_ablation/
    full_current.json
    wll1.json
    direct_endpoint.json
    hh_head.json
    metrics.csv
    probe_curves.csv
  optional_wavelet/
    metrics.csv
```

所有最终论文数字只从上述冻结目录导出，`SchrodingerBridge/rebuttal_exps` 中的临时脚本和旧 `ablation_v2` 结果不再直接进入论文表格。
