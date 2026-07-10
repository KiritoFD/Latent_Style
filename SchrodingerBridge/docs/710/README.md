# 710: WEAVE 四指标实验路线

日期：2026-07-10

## 1. 目标与评价协议

从本阶段开始，模型选择只使用四项闭合指标：

- `CLIP-S`：目标风格相似度，越高越好。
- `LPIPS`：生成图与内容图的感知距离，越低越好。
- `DINO-S`：目标风格结构/外观相似度，越高越好。
- `DINO-C`：内容结构保持，越高越好。

`MUSIQ` 不再参与模型选择、消融结论、早停或论文主张。所有推理后处理默认关闭，除非实验本身明确研究该后处理；禁止用后处理后的指标替代 checkpoint 原生结果。

## 2. 唯一基线

主基线固定为 `configs/630_local_t11_stochastic_dwt_p08.json`：

- 单层 Haar DWT；
- 高频 cross-attention routing；
- 训练 routing 概率 `p=0.8`；
- LL 不进入 style query；
- 终点 per-subband WCT；
- 4-block、width 64、903K 参数；
- 5 epochs、Heun、cosine schedule。

已闭合的 D5 指标为 `CLIP-S=0.7213`、`LPIPS=0.2868`。DINO-S/DINO-C 必须在统一评估脚本下重新回填，不能从论文中其他 operating point 拼接。

## 3. 判定规则

不再使用四指标加权总分。采用约束加 Pareto 判定：

1. 首先要求 `CLIP-S` 高于对应数据集的 IDT floor。
2. 新点若同时改善四项指标，判为 strict win。
3. 若 `CLIP-S` 或 `DINO-S` 改善，则要求 `LPIPS` 与 `DINO-C` 的退化均不超过基线 seed 标准差。
4. 若 `LPIPS` 或 `DINO-C` 改善，则要求 `CLIP-S` 与 `DINO-S` 的退化均不超过基线 seed 标准差。
5. 只在至少 3 个 seeds 的均值与标准差上做模块结论。

建议主报告同时给出两个 operating point：

- `WEAVE-C`：内容优先，最小 LPIPS / 最大 DINO-C；
- `WEAVE-S`：风格优先，最大 CLIP-S / DINO-S，同时保持正 IDT transfer。

## 4. Phase A：精简后等价性验证

目标：证明代码精简没有改变 T11 主路径。

| ID | 改动 | Seeds | 必须检查 |
|---|---|---:|---|
| A0 | 精简前 T11 已有 checkpoint 重评 | 1 | 四指标、750 张、无后处理 |
| A1 | 精简后 T11 原配置重新训练 | 42, 43, 44 | 四指标均值/标准差、参数量、训练时间 |
| A2 | 同 checkpoint 重复推理 | 1 | 输出 hash 或四指标数值一致性 |

通过标准：A1 相对历史 T11 的 CLIP-S/LPIPS 差异处于历史波动范围；无 NaN；参数量仍为 903,248；配置加载不依赖已删除的 remote/MUSIQ 文件。

## 5. Phase B：核心减法消融

只做能够验证论文机制的最小矩阵，不恢复大规模历史开关。

| ID | 单一变量 | 设置 |
|---|---|---|
| B0 | Full T11 | `p=0.8`, DWT route, endpoint WCT |
| B1 | No DWT route | `cross_attn_dwt_route=false` |
| B2 | Deterministic route | `p=1.0` |
| B3 | Insufficient routed exposure | `p=0.5` |
| B4 | No endpoint WCT | `endpoint_adain_mode=none` |
| B5 | Strong LL supervision | `spectral_w_ll=1.0` |
| B6 | No LL supervision | `spectral_w_ll=0.0` |
| B7 | Two residual blocks | `num_res_blocks=2` |
| B8 | Width 32 | `base_dim=32` |

每点先用 seed 42 筛选。只有进入四指标 Pareto 前沿或能直接验证理论主张的点再补 seeds 43/44。

## 6. Phase C：DINO 定向改进

优先级按“增加最少结构自由度”排列。

### C1. Hard-region SWD

当前只保留 `swd_semantic_mode=region`，扫描：

- regions：`4, 8`；
- blend：`0.25, 0.5, 0.75`；
- SWD weight：保持 T11 默认，最多补一个较低权重点。

目标：提高 DINO-S，同时不显著损伤 DINO-C/LPIPS。若 DINO-S 无稳定收益，删除 hard-region SWD 全部代码。

### C2. HH velocity head

只做一个严格 matched A/B：`enable_hh_head=false/true`。重点看 DINO-S 是否提高，以及 DINO-C、LPIPS 是否因对角高频位移恶化。若三 seeds 下没有稳定 Pareto 收益，删除该 flag 和 head。

### C3. Style-FiLM heads

只做一个严格 matched A/B：`style_film_heads=false/true`。它直接向 velocity head 注入 style，理论风险高于 C1/C2。只有 DINO-S 显著提升且内容指标退化不超过一个基线标准差时才保留。

### C4. 低维全局色调终点通道

若前三项不能提升 CLIP-S/DINO-S，新增的唯一架构方向是终点低维 color transform：

- 不恢复 LL cross-attention；
- 只预测每通道 mean/std 或低秩 `4x4` color matrix；
- 只在 `t=1` 应用；
- 参数与 DWT velocity backbone 分离。

该方向用于补足 HF routing 无法表达的低频色调，不允许引入空间形变自由度。

## 7. Phase D：训练长度与 Pareto 点

在结构确定后，仅扫描 `5/10/15 epochs`：

- 5 epochs：效率与内容优先候选；
- 10 epochs：预期平衡点；
- 15 epochs：风格优先候选。

训练长度不是模块消融，不与结构变化同时扫描。最终从同一结构中选择 `WEAVE-C` 和 `WEAVE-S`。

## 8. 实验记录要求

每个实验目录必须保存：

- 完整扁平化 config；
- git commit hash；
- seed；
- checkpoint 参数量；
- 每 epoch 四指标；
- 750 张 full-eval 汇总；
- IDT baseline 与 delta；
- 单卡训练和推理时间；
- 是否启用任何推理后处理。

汇总表固定列：

`run, seed, clip_s, lpips, dino_s, dino_c, clip_s_delta_idt, params, train_min, infer_sec, postprocess`

## 9. 停止标准

满足任一条件即停止该方向：

- 3 seeds 下没有进入四指标 Pareto 前沿；
- 风格收益完全可由 LPIPS 增长解释，DINO-S 无独立改善；
- DINO-S 改善但 DINO-C 下降超过两个基线标准差；
- 新增代码超过约 150 行仍没有 strict win；
- 需要重新引入已删除的 patch/Sinkhorn/hierarchical/adaptive/spectral-region SWD 分支。

## 10. 立即执行顺序

1. 完成 A0/A1，回填 T11 的统一 DINO-S/DINO-C。
2. 运行 B0-B6 单 seed，形成核心机制表。
3. 对 B 组关键点补 3 seeds。
4. 运行 C1 hard-region SWD 小矩阵。
5. 仅在 C1 失败时依次运行 C2、C3。
6. 固定结构后做 Phase D，确定 WEAVE-C / WEAVE-S。

