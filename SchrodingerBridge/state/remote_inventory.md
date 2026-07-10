# 远程 (I:) 目录分类普查 — 2026-07-09

按「数据集 → 方法/模型」分类。计数来自远程实际 image 目录（.png/.jpg/.jpeg）。
结论：**R5-WikiArt 的 adain/wct/weave/identity（各 12000）在远程不存在，也无 summary.json 清单**；它们是本地生成的。

## 远程根目录
- `I:\Github\Latent_Style\exp_baselines` — 各方法 baseline 实验（乱）
- `I:\exp_256_photo2art` — P256 协议 7 个干净基线（各 750）
- `I:\exp_our_models_eval` — 我们自己的模型评测
- `I:\exp_samst_latent_eval`, `exp_samam_latent`, `exp_samst_latent` — sam 系列
- `I:\Github\Latent_Style\seedream45_api` — seedream（D5-512 & P256 协议）
- `I:\latent_style_remote_curated\by_dataset` — 指标 CSV
- `I:\results` — 仅协议参考 (eval_protocol_750: 5风格×30) + 表格，**非模型输出**
- `I:\manifests` — 仅 ollama 注册表
- archaeology 4 根 — 空

## 分类映射

### D5-512 (distinct5_512, 5风格@512)
| 方法 | 远程源 | 计数 | 备注 |
|---|---|---|---|
| seedream | seedream45_api/distinct5_512_seedream45_..._repaired750/images | 750 | 与本地一致 |
| samst | samst_distinct5_512_*/eval_bundle/eval_step_000040_full | TBD | 训练运行，eval 在其内 |
| adain/wct/identity/weave/cut/stylealigned/styleid/styleshot/zstar | — | — | 远程未见，仅本地 |

### P256 (photo2art 256, 5风格 cezanne/Hayao/monet/photo/vangogh)
| 方法 | 远程源 | 计数 | 备注 |
|---|---|---|---|
| adain | exp_256_photo2art/adain_256/images | 750 | 干净 |
| identity | exp_256_photo2art/identity_256/images | 750 | 干净 |
| samam | exp_256_photo2art/samam_256/images | 750 | 干净 |
| samst | exp_256_photo2art/samst_256/images | 750 | 干净 |
| sdturbo | exp_256_photo2art/sdturbo_256/images | 750 | 干净 |
| styleid | exp_256_photo2art/styleid_256/images | 750 | 干净 |
| wct | exp_256_photo2art/wct_256/images | 750 | 干净 |
| cut | exp_baselines/_auxiliary_runs/cut_5x5/infer_5x5 | 2427 | 需按 P256 协议 5×150 筛选 |
| seedream | seedream45_api/protocol_a_800/images | 721 | 注意 <750（各风格 142-147） |
| stylealigned/styleshot/weave/zstar | — | — | 远程未见，仅本地 |

### R5-512 (wikiart 5风格@512) 与 R5-WikiArt (wikiart 5风格@512)
- 远程**无任何来源**。
- R5-WikiArt/adain/wct/weave/identity 各 12000 为**本地生成**，内容池独立于干净组（与 cut 等 750 零重叠），无远程清单，无法靠文件名/远程 manifest 取 750。

### 我们的模型 (SchrodingerBridge)
- exp_our_models_eval/latent512_e7/images = 750
- exp_our_models_eval/latent256_e10/images = 444
- exp_our_models_eval/pixel256_e3/images = (空)

## 关键结论
1. P256 7 个基线在 `exp_256_photo2art` 有干净 750，可与本地已存在的 750 互相印证（无需重取，除非本地损坏）。
2. P256 待取：cut (2427→筛750)、seedream (721，缺 29)。
3. R5-WikiArt 12000 组：远程不存在、无清单 → 必须本地解决（见决策）。
4. seedream 运行带 summary.json（例：seedream45_api/protocol_a_800/summary.json），印证部分方法有 manifest，但 adain/wct/weave/identity 12000 没有。
