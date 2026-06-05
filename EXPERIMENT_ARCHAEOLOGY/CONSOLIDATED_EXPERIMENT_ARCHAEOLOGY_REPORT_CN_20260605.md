# 实验考古总归纳报告 - 2026-06-05

范围：

- 本地：`G:\GitHub\Latent_Style`
- 远程主仓：`I:\Github\Latent_Style`
- 远程 TokenizerClean：`I:\Github\Latent_Style_TokenizerClean`

这份报告是结论入口。逐目录明细继续看各 CSV ledger；这里回答四个问题：本地是什么状态、远程是什么状态、实验脉络是什么、还能怎么清。

覆盖矩阵见：

- `manual_coverage_matrix_20260605.csv`

## 1. 总结论

当前仓库不是一个单纯的“实验垃圾目录”，而是一个混合实验档案库。大体积 `.pt/.pth/.bin` 不能直接等价于训练 checkpoint，因为里面混有：

- per-image latent tensors；
- CLIP/DINO/VAE/ArtFID/LPIPS/Inception 依赖；
- eval reference feature cache；
- formal experiment epoch weights；
- fake eval placeholders；
- historical baseline curve checkpoints。

本地 checkpoint 清理已经做到“非主线训练 ckpt 基本清空”的状态；剩下的大头主要是数据/cache/依赖。远程主仓剩下的大权重集中在 `SchrodingerBridge/exp` 的当前 Distinct5/AAAI2027/SADD lineage 权重和 `Related_Works` 的 SaMAM 中央曲线 checkpoint。TokenizerClean 是当前 AAAI/tokenizer 工作树，不能按垃圾目录处理。

## 2. 本地状态

本地最重要的事实：真正还像“训练权重”的本地 `SchrodingerBridge/exp` 只剩 9 个 weight-like 文件，其中 8 个是 WikiArt512 formal anchor epoch 权重，1 个是 ArtFID metric dependency。此前怀疑的 Distinct5 vramprobe ckpt 当前已经不存在。

但是本地仍有很多 `.pt/.bin`，分类如下：

| 本地位置 | count | size MB | 归类 | 当前决策 |
|---|---:|---:|---|---|
| `root eval_cache` | 3069 cache files | 5747.939 current subtree MB | DINO/SDXL VAE/CLIP/ArtFID/offline pairing/ref feature cache/VAE compile/ONNX | 已打开到文件级；删除 55.994MB 无效 `.incomplete` 缓存，其余保留 |
| `SchrodingerBridge/scale` | 11350 | 3179.108 | scale dataset tensors | 保留，数据/cache |
| `clip-feats-vitb32` | 10361 | 1303.526 | per-image CLIP features | 保留，数据/cache |
| `Related_Works` | 31 | 814.913 | baseline deps + tiny placeholders | 保留依赖；placeholder 另定策略 |
| `SchrodingerBridge/exp` | 9 | 363.490 | WikiArt512 epoch weights + ArtFID dependency | 保留 |
| `SchrodingerBridge/datasets/horse2zebra` | 2661 | 336.076 | latent tensors | 保留，数据 |
| `Dataset` | 2920 | 188.731 | Distinct5/WikiArt latent tensors | 保留，数据 |
| `latent-256` | 10361 | 177.702 | legacy per-image latents | 保留，数据 |
| root `inception-2015-12-05.pt` | 1 | 91.179 | metric dependency | 保留 |
| `Cycle-NCE` | 3 | 1.054 | eval ref feature cache | 保留 |
| `seedream45_api` | 1 | 0.001 | fake eval placeholder | 可删但无空间收益 |

本地已清理并复查干净的 checkpoint 目标：

- `Related_Works/runs/cut_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5/checkpoints`
- `Related_Works/runs/cyclegan_5x5_smoke/checkpoints`
- `Related_Works/final_works/trial_0016`
- `Related_Works/final_works/trial_0019`
- `Related_Works/final_works/trial_0044`
- `SchrodingerBridge/exp/local_wsl_distinct5_512_ema_k_b16_step2min_ckptsync`

本轮继续清掉的只是空目录壳：

- `SchrodingerBridge/exp/seedream_distill_adapter`
- `SchrodingerBridge/exp/style_embedding_mainline_calibration`
- `SchrodingerBridge/exp/tmp_genonly_autonogrid_probe`
- `SchrodingerBridge/exp/vae_backend`
- `Related_Works/baseline_pipeline/results/samam_256_curve`
- `Cycle-NCE/eval_cache/hf`

这些都是删除前递归文件数 0、git tracked 0 的空壳，释放 0 字节，但能减少假线索。

本轮新增的 `eval_cache` 手工核验结论：

- `eval_cache/offline_pairing` 已打开日志和源码引用，确认是 `style_data/train` + `latent-256` 的 9636 行 DINOv2/offline pairing cache，不是训练 ckpt。
- `eval_cache/hf` 已打开 HF `refs/main`、VAE configs 和 ModelScope 文件，确认有效部分是 CLIP/VAE 模型依赖；只删除一个失败下载残留 `.incomplete` blob，释放 55.994MB。
- `eval_cache/manual_clip` 已打开 CLIP config 和 tokenizer/model 文件，确认是完整本地 CLIP 依赖。
- `eval_cache/ref_feats_*.pt` 已逐个只读加载，确认是 full-eval reference feature cache。
- `eval_cache/vae_compile` 和 `eval_cache/vae_onnx` 已打开文件类，归类为 VAE speed/export artifact，暂不删。

详见：

- `manual_local_eval_cache_policy_20260605.csv`
- `MANUAL_LOCAL_EVAL_CACHE_POLICY_20260605.md`
- `cleanup/manual_cache_cleanup_20260605.csv`

## 3. 远程主仓状态

远程主仓 `I:\Github\Latent_Style` 当前不是空盘，也不是未清理状态。核心分类：

| 远程位置 | count/size | 归类 | 当前决策 |
|---|---:|---|---|
| `SchrodingerBridge/exp` | 101 files / 5945.064 MB | current Distinct5/AAAI2027/SADD lineage weights | 保留，需 epoch thinning policy |
| `Related_Works/.../SaMAM.../step_checkpoints` | 19 files / about 5242 MB | SaMAM Distinct5 中央曲线 baseline evidence | 保留，需 cited-step thinning policy |
| `eval_cache` | 29 files / 6077.946 MB | eval deps/cache | 保留，cache policy |
| `latent-256-*` | 多个 root，各 1310-5195 MB | backend latent cache | 保留，数据/cache policy |
| `SchrodingerBridge/scale/datasets` | 11349 files / 2859.902 MB | dataset tensors | 保留，数据 |
| `Cycle-NCE` | 37 files / 937.553 MB | historical archive/cache/deps | archive policy |
| `experiments` | 3 files / 319.141 MB | Feb-Apr legacy archive | archive policy |
| `StarGAN` / `seedream45_api` placeholders | tiny | fake eval placeholder | 可删但无空间收益 |

远程 `SchrodingerBridge/exp` 已经单独打开：

- top-level inventory 124 行；
- 17 个顶层目录含权重；
- 101 个权重约 5945.064 MB；
- 打开过 `exp/README.md`、Distinct5 b44/A/J、AAAI2027 F/K、SADD exact/repro 的 config、训练日志和 full_eval summary。

远程 timing 代表例：

- Distinct5 b44 epoch 8：训练 `62.23699736595154s/epoch`，full-eval `wall_total=94.8003261089998s`。
- Distinct5 variant A epoch 8：训练 `62.34275555610657s/epoch`，full-eval `wall_total=95.04837335200136s`。
- Distinct5 variant J epoch 3：训练 `63.49859118461609s/epoch`，full-eval `wall_total=95.3742954019981s`。
- AAAI2027 F epoch 8：训练 `67.0106086730957s/epoch`，full-eval `wall_total=136.38432930499948s`。
- AAAI2027 K epoch 8：训练 `64.67221856117249s/epoch`，full-eval-artfid `wall_total=105.18277635000004s`。
- SADD exact/repro epoch 8：训练约 `42.16s/epoch` / `41.13s/epoch`，summary 未暴露同结构 full-eval wall time。

本轮没有远程删除，因为 101 个权重不是随机垃圾。要释放空间，下一步必须先决定“每个 family 留哪些 epoch”，不能按扩展名扫删。

## 4. TokenizerClean 状态

`I:\Github\Latent_Style_TokenizerClean` 是当前 AAAI/tokenizer 干净工作树，不是非主线垃圾目录。打开过的 docs 明确说明 `exp/` 仍被 master log 和实验文档直接引用。

已知权重分布：

- `SchrodingerBridge` total：334 files / 11822.873 MB。
- `SchrodingerBridge/exp` normal：326 files / 11375.355 MB。
- displayed `exp?saswd*` special paths：6 files / 403.558 MB。
- `artifacts`：1 file / 43.635 MB。
- `eval_cache`：1 file / 0.324 MB。

关键边界：

- 不能 mass delete `exp/`；
- `saswd` 路径显示异常但是真证据；
- flow-loss 第一组三臂已被 config audit 降级；
- repaired endpoint packet 是负结果闭环；
- H-family execution alignment 缺 H e1 payload，L e1 是 successor，不是同族 fallback。

## 5. 实验脉络

### Phase A：Jan-Feb，latent/style-transfer 原型

核心目标是证明 latent space style transfer 方向可行。主要证据在：

- `style_data`
- `latent-256`
- `clip-feats-vitb32`
- `Cycle-NCE`
- remote `experiments`

这阶段的结果很多，但命名混乱，timing 不完整，适合作为历史脉络，不适合作为当前论文主 claim。

### Phase B：Feb-Apr，Cycle-NCE / Latent AdaCUT 大规模架构考古

本地 `Cycle-NCE` 是这条线的主档案：

- 500 个 `summary.json`；
- 496 个 `metrics.csv`；
- 260 个 `training_*.csv`；
- 仅 3 个 weight-like 文件，均为 eval ref feature cache。

重要指标锚点：

- `exp.csv` best：`style_oa_5_lr5e4_wc2_swd60_id30_e120_interval10`，epoch 100，`transfer_clip_style=0.729723026394844`。
- `Aline120.csv` best：`Aline120_aline_03_ghost_wireframe`，epoch 20，`transfer_clip_style=0.7146436547239621`。
- `hf.csv` best：`p_base_hf_3p0_distill_epochs200_tokenized`，epoch 60，`transfer_clip_style=0.6734027210871381`。

### Phase C：Mar-Apr，baseline reproduction

主要在 `Related_Works` 和 `final_works`：

- CUT/CycleGAN/SaMST/SaMAM/StarGAN/SDEdit/SDTurbo/Seedream 等 baseline；
- 本地大权重多是 VGG/Inception/LPIPS 依赖；
- `final_works` 训练权重已清掉，只保留指标/placeholder。

### Phase D：May，SchrodingerBridge/LANCET phase-space

主要在：

- `SchrodingerBridge/exp/frontier`
- `SchrodingerBridge/exp/vae_backend`
- `SchrodingerBridge/exp/inference`
- `lambda_grid`
- `step_count_sweep`
- remote `SchrodingerBridge/review_additional_experiments`

关键纠偏：`lambda_grid` 和 `step_count_sweep` 的 root manifest 是 dry-run，`0.000/0.001s` 不能当训练/推理时间。

### Phase E：May 30-Jun 2，WikiArt512 和 Distinct5 formal evidence

本地可信锚点：

- `SchrodingerBridge/docs/experiments/2026-06-02-wikiart512-inference-speed.md`
- `SchrodingerBridge/docs/experiments/2026-06-05-timing-sidecar-inventory.md`
- `SchrodingerBridge/exp/local_wsl_wikiart512_hist_b32_e8`

远程可信锚点：

- `I:\Github\Latent_Style\SchrodingerBridge\exp\distinct5_512_ema_*`
- `I:\Github\Latent_Style\Related_Works\baseline_pipeline\results\samam_distinct5_512...`

这阶段才是当前 timing/efficiency claim 的核心证据。

### Phase F：Jun 3 后，AAAI2027 / TokenizerClean claim-closing

主要在：

- remote `Latent_Style_TokenizerClean`
- remote `SchrodingerBridge/exp/aaai2027_*`
- `docs/experiments/aaai2027_master_experiment_log.csv`

这阶段的目标是关闭 tokenizer、SA-SWD、flow-loss、endpoint metric 等 claim。

## 6. 清理策略

已经可以确认不该再做 broad delete。下一步清理必须分三类：

| 清理类别 | 可做什么 | 风险 | 前置条件 |
|---|---|---|---|
| Epoch thinning | 每个 formal family 只留 best/cited/last epoch | 误删可复核链 | 先写 family -> keep epochs policy |
| Baseline curve thinning | SaMAM 19 个 central step ckpt 只留 cited steps + last | 破坏曲线复现 | 先核对 comparison docs |
| Cache/archive policy | latent backend cache、eval cache、remote experiments/Cycle-NCE archive | 破坏复现或重评测 | 用户确认哪些 backend/dataset 不再复用 |

当前不建议删除：

- `eval_cache/offline_pairing`，虽然 3.6GB，但它是 DINO/offline pairing cache；
- `SchrodingerBridge/scale`，它是 dataset tensors；
- remote `SchrodingerBridge/exp` 的 101 个权重；
- remote SaMAM 19 个 central checkpoints；
- TokenizerClean `exp/`。

## 7. 8 小时级别执行计划

如果继续推进完整清理，按这个顺序做：

| block | 预计时间 | 动作 | 产物 |
|---|---:|---|---|
| 1 | 0.5h | 冻结当前所有 CSV/MD 入口，生成 coverage matrix | `manual_coverage_matrix_*.csv` |
| 2 | 1.0h | 本地 root-level 非主线文件逐个归档判断：hidden agent dirs、root scripts、root metric deps、page PNG、deb | root-level cleanup ledger |
| 3 | 1.0h | 本地 `eval_cache` 逐目录判断：offline_pairing/hf/manual_clip/artfid/ref_feats | cache retention policy |
| 4 | 1.0h | 本地 `SchrodingerBridge/scale` 和 `Dataset` 数据/cache policy | dataset/cache keep/delete table |
| 5 | 1.0h | 远程 `SchrodingerBridge/exp` epoch thinning policy | family -> keep epochs -> delete candidates |
| 6 | 1.0h | 远程 SaMAM 19 step checkpoints cited-step audit | SaMAM keep/delete candidate CSV |
| 7 | 1.0h | TokenizerClean master-log citation graph | referenced path map |
| 8 | 0.5h | 只执行已被 policy 证明安全的删除、校验、提交 | cleanup CSV + post-delete counts |

当前状态还不能宣称“全部完成”。已经完成的是：主要证据面被分层，非主线 checkpoint 清理边界变清楚，若要继续释放空间，下一步是 policy-driven thinning，而不是继续猜删。
