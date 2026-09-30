# 710: WEAVE 四指标实验路线

日期：2026-07-10

配套文档：

- Style 提升实验计划：`docs/710/STYLE_IMPROVEMENT_PLAN.md`；
- 训练与推理 Infra 优化方案：`docs/710/INFRA_OPTIMIZATION_PLAN.md`；
- 四指标评估事故复盘：`docs/710/EVALUATION_AUDIT_20260710.md`；
- Phase B 消融实验结果：`docs/710/PHASE_B_RESULTS.md`。

## 1. 目标与评价协议

从本阶段开始，模型选择只使用四项闭合指标：

- `CLIP-S`：目标风格相似度，越高越好。
- `LPIPS`：生成图与内容图的感知距离，越低越好。
- `DINO-S`：目标风格结构/外观相似度，越高越好。
- `DINO-C`：内容结构保持，越高越好。

`MUSIQ` 不再参与模型选择、消融结论、早停或论文主张。所有推理后处理默认关闭，除非实验本身明确研究该后处理；禁止用后处理后的指标替代 checkpoint 原生结果。

## 2. 唯一基线

历史 T11 仅保留为回归对照。新的主基线固定为
`configs/710_b0_weave_d5.json`，它从论文候选
`configs/clean_base_v2_local.json` 原样继承模型、loss 和数据采样协议，只显式锁定
seed、训练轮数、8-step Heun 推理和输出目录：

- 单层 Haar DWT；
- 高频 cross-attention routing；
- 训练 routing 概率 `p=0.8`；
- LL 不进入 style query；
- 终点 per-subband WCT；
- 4-block、width 64、903K 参数；
- 10 epochs、8-step Heun、cosine schedule。

目前最接近该候选的历史 750 图输出包给出
`CLIP-S=0.7292, LPIPS=0.3239, DINO-S=0.4874, DINO-C=0.7688`，但其 checkpoint
已经丢失，因此这些数值只能作为重训验收参考，不能直接充当可复现主结果。B0 必须重新训练并从同一个
checkpoint、同一批 750 图、同一版评估脚本产生四项指标。

论文草稿中的 `0.715 / 0.382 / 0.778 / 0.492` 不是一个闭环 operating point：前两项来自
`exp/swd_cm_sem_r8/full_eval/epoch_0005`，而后两项来自另一输出包
`results/D5-512/weave`。该拼接行作废，禁止作为 baseline 或论文结果继续引用。

统一 DINO 协议为：`DINO-C = cos(CLS(gen), CLS(source))`；
`DINO-S = max_ref cos(CLS(gen), CLS(target-style reference))`；另外记录倒数第二层 patch-token
self-similarity MSE 为 `DINO-structure`，但它不替代 `DINO-C`。实现唯一入口为
`src/utils/compute_dino_metrics.py`。

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
| B0 | Full WEAVE | `clean_base_v2_local`, seed 42, 10 epochs, 8-step Heun |
| B1 | No DWT route | `cross_attn_dwt_route=false` |
| B2 | Deterministic route | `p=1.0` |
| B3 | Insufficient routed exposure | `p=0.5` |
| B4 | No endpoint WCT | `endpoint_adain_mode=none` |
| B5 | Strong LL supervision | `spectral_w_ll=1.0` |
| B6 | No LL supervision | `spectral_w_ll=0.0` |
| B7 | Two residual blocks | `num_res_blocks=2` |
| B8 | Width 32 | `base_dim=32` |

每点先用 seed 42 筛选。只有进入四指标 Pareto 前沿或能直接验证理论主张的点再补 seeds 43/44。

## 6. Phase C：Style 定向改进

详细矩阵见 `docs/710/STYLE_IMPROVEMENT_PLAN.md`。本阶段按实际自由度从低到高执行：

1. 先验证 `style_strength` 是否真正进入 active endpoint 路径；
2. 先做 `spatial_fiber` 与 `per_subband` endpoint mode matched A/B；只有后者晋级时，才固定 LL 并轻度提高 LH/HL/HH；
3. 对正确方向做 matched retraining，并比较 10/15 epochs；
4. 再依次验证 endpoint style weight、SWD weight 和 LL supervision；
5. 最后才允许增加新模块。

Hard-region SWD 不再作为第一优先。对 `swd_cm_sem_r8` 的同源 canonical DINO 重算只有
`DINO-S=0.4661, DINO-C=0.6888`，没有支持其作为当前 style 改进主线。若后续重新验证，只允许 region 4/8
的最小 matched 矩阵。

HH velocity head 与 Style-FiLM heads 只保留一个 false/true A/B。若三 seeds 下不能进入四指标 Pareto 前沿，
删除对应 flag 和实现。

新增架构的第一选择是终点低维 color transform：

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
4. 运行 endpoint mode A/B；仅对晋级的 per-subband 路径运行分频 style 小矩阵。
5. 依次验证训练长度、endpoint style weight 和 SWD weight。
6. 固定结构后做 Phase D，确定 WEAVE-C / WEAVE-S。

## 11. 本地基础设施实测（RTX 4070 Laptop 8GB）

本节只记录机器相关的训练/推理基础设施结论，不作为模型质量结论。

### 11.1 训练

- Distinct5 packed latent、batch 24、bf16、fused AdamW 的 CPU 驻留基线为 `36.3s/epoch`。
- 将约 `0.31GB` packed latent 全量预加载到 GPU 后为 `37.6s/epoch`，无收益，因此正式训练不启用 `preload_to_gpu`。
- `cudnn_benchmark=true` 为 `35.3s/epoch`，相对同步修复后的配置明显回退，因此保持关闭。
- 根因优化是删除每个 optimizer step 无条件执行的整套指标 `.item()`，并避免 loss 调试字典逐标量回传 CPU；修复后为 `29.7s/epoch`，相对基线提速 `18.2%`，显存峰值不变。
- 干净 T11 seed 42 五 epoch 训练总 wall time 为 `153.0s`，epoch 2--5 稳定在 `28.1--29.3s`。
- 高频外部采样的训练稳态 GPU 利用率均值 `84.2%`、中位数 `91%`；功率均值 `58.8W`、中位数 `63.1W`、峰值 `66.7W`。该功率水平仍然异常偏低，现有采样不足以排除 kernel 碎片化、同步等待、dtype conversion 或 VAE/CPU 流水线空洞；后续必须用 Nsight/torch profiler 的 kernel timeline 定位，不能仅凭利用率宣称 infra 正常。
- profiler 仍显示 bf16 主干与 fp32 数值稳定 loss 之间存在大量 dtype conversion；这属于下一轮可验证优化，不能在没有质量/数值对照时直接删除。

### 11.2 推理

- 必须显式记录积分步数：新 checkpoint 默认值为 `1`，论文正文使用 `8-step Heun`，旧配置中还存在 `12-step`；三者不可混用。
- 750 个 latent 的桥接时间：1 step `7.16s`，8 steps `62.20s`，12 steps `76.76s`。后续速度表必须与质量指标使用同一步数。
- diffusers eager VAE、batch 16 解码 750 张为 `35.68s`，模型冷加载约 `13.32--13.89s`。
- `torch.compile` warm decoder 将解码降至 `23.90s`，但首次编译使单次进程 wall time 增至 `69.02s`；只适合常驻服务或多轮评测。
- 固定导出的 TorchScript VAE decoder 将跨进程加载降至 `0.52s`，输出 checksum 与 eager 完全一致；解码为 `32.16s`，单轮 wall time `37.99s`，适合一次性评测和批量离线生成。
- 固定 decoder 加 8-worker 异步 PNG 保存，1-step 生成并落盘 750 张为 `47.08s`：桥接 `6.95s`、VAE decode `34.12s`、PNG join `0.04s`。当前主要瓶颈已经收敛到 VAE decoder，而不是桥接 batch 或磁盘保存。
- 正式评测器原先虽然配置 `vae_decode_batch_size=16`，却在每个 4-latent style chunk 后立即解码，实际从未形成 batch 16。将 750 个生成 latent 在 GPU 上聚合后统一分批解码，使正式 VAE decode 从 `53.75s` 降至 `34.97s`（约 `35%`）；完整 8-step、750 PNG 落盘 wall time 为 `93.88s`，其中桥接 `57.42s`、VAE decode `34.31s`、PNG join `0.008s`。

### 11.3 本地 seed 42 四指标

`exp/710_infra_t11_distinct5_5ep/epoch_0005.pt` 使用统一 8-step、无后处理协议得到：

| CLIP-S | LPIPS | DINO-S | DINO-C |
|---:|---:|---:|---:|
| 0.7204 | 0.2857 | 0.4736 | 0.7759 |

该结果与历史 T11 的 `CLIP-S=0.7213, LPIPS=0.2868` 基本对齐。此前记录的
`DINO-S=0.3958, DINO-C=0.9752` 来自错误临时定义：前者使用平均风格 prototype，后者误将
`1 - structure MSE` 当作内容分数，现已作废。上表数值由统一 canonical 脚本重算；模块结论仍需
seeds 43/44。

### 11.4 推荐运行形态

- 单次离线 750 张：固定 TorchScript VAE decoder、decode batch 16、异步 PNG 保存。
- 连续多 checkpoint 评测：常驻进程加载一次 VAE，并使用 warm `torch.compile` decoder。
- 禁止继续盲目扩大 bridge/VAE batch；先保持 batch 2 / style chunk 2 / decode batch 16，在 8GB 显存上稳定运行。
