# 远程实验结果目录分类清单（I:\ 全盘深入扫描）

> 扫描方式：在远程 `I:\` 全盘递归，找出**每一个含图片文件的目录**并统计图片数（jpg/png/webp/bmp/tif），并对图数≥50 的目录展开一级子目录拆分。
> 扫描时间：2026-07-09。共扫描目录 **16710** 个，其中含图片目录 **3453** 个。
> 原始数据：`_images_scan_out.txt`（每行 `DIR|IMG=<数>|<绝对路径>`，并以 `  SUB|IMG=<数>|<子目录名>` 标注内部拆分）。

## 一、全量按顶层子根聚合（含图目录数 / 总图数）

| 子根 | 含图目录数 | 总图数 | 性质 |
|---|---:|---:|---|
| `Github\Latent_Style\experiments_historical` | 788 | 335517 | 历史训练 full_eval 输出 |
| `Github\Latent_Style\exp_samam` | 178 | 115456 | SAMAM 训练曲线/评测 |
| `datasets` | 214 | 75634 | **源数据集**（输入，非生成结果） |
| `Github\Latent_Style\exp_ours` | 289 | 70464 | 我们的模型训练/评测 |
| `Github\Latent_Style_TokenizerClean\SchrodingerBridge` | 725 | 52078 | tokenizer 清洗实验 |
| `Github\Latent_Style\SchrodingerBridge` | 75 | 41131 | 主仓库实验 |
| `Github\Latent_Style\exp_baselines` | 82 | 18728 | **基线生成结果**（重点） |
| `Github\26AI-H\pj1` | 57 | 17663 | 无关项目 |
| `Github\Latent_Style\Related_Works` | 43 | 12547 | 相关工作 |
| `Github\Latent_Style\style_data` | 16 | 11011 | 源风格数据 |
| `exp_256_photo2art` | 7 | 5250 | **P256 干净基线 750** |
| `Github\Latent_Style_TokenizerClean\Related_Works` | 48 | 3783 | 相关工作 |
| `Github\Latent_Style\StarGAN` | 10 | 3111 | 对照方法 |
| `Github\Latent_Style\exp_baseline_256` | 3 | 2250 | **P256 干净基线 750**（adain/samst/wct 副本） |
| `Github\Latent_Style\seedream45_api` | 6 | 1442 | **P256 seedream** |
| `Github\Latent_Style_TokenizerClean\Cycle-NCE` | 852 | 1219 | 对照方法 |
| `exp_our_models_eval` | 3 | 1195 | **我们的模型评测** |
| `exp_samst_latent_eval` | 1 | 750 | **SAMST-latent 评测** |
| `results` | 5 | 150 | 评测协议参考（eval_protocol_750） |

> 说明：绝大多数 750 图目录是模型训练的 `full_eval`/`eval_bundle` 产物（如 `experiments_historical`、`exp_ours`、`exp_samam` 下大量 `epoch_XXXX\images`），它们并非论文协议用的“整理后 750 组”。下方第二节只列出**协议相关的整理结果集**。

## 二、协议相关整理结果集（按 数据集 → 方法 分类）

图例：✓ = 已齐 750（可直接用）；⚠ = 数量不符需处理；✗ = 远程不存在。

### D5-512（distinct5_512，5 风格 @512）
| 方法 | 状态 | 远程路径 | 图数 |
|---|---|---|---:|
| samst | ✓ | `Github\Latent_Style\exp_baselines\samst_distinct5_512_wsl_stepalign40_remote_20260605_r1\eval_bundle\eval_step_000040_full\step_000040_full\images` | 750 |
| samst(latent) | ✓ | `Github\Latent_Style\exp_baselines\samst_latent_distinct5_512_*\eval_bundle_fast\batch*_fast\images`（多 run，均 750） | 750 |
| samam | ✓ | `Github\Latent_Style\exp_samam\training\samam_distinct5_512_*\curve_eval_hf_750*\step_*\images`（多 run，均 750） | 750 |
| seedream | ✓ | `Github\Latent_Style\exp_baselines\seedream45_api\distinct5_512_seedream45_windhub_20260607_repaired750\images` | 750 |
| adain / wct / identity / weave / cut / styleid | ✗ | 远程扫描未命中（D5-512 分辨率下未见这些方法的生成目录） | — |

### P256（photo2art 256，cezanne/Hayao/monet/photo/vangogh）
| 方法 | 状态 | 远程路径 | 图数 |
|---|---|---|---:|
| adain | ✓ | `exp_256_photo2art\adain_256\images`（另 `Github\Latent_Style\exp_baseline_256\adain\step_000001\images`） | 750 |
| identity | ✓ | `exp_256_photo2art\identity_256\images` | 750 |
| samam | ✓ | `exp_256_photo2art\samam_256\images` | 750 |
| samst | ✓ | `exp_256_photo2art\samst_256\images`（另 `Github\Latent_Style\exp_baseline_256\samst\step_000001\images`） | 750 |
| sdturbo | ✓ | `exp_256_photo2art\sdturbo_256\images` | 750 |
| styleid | ✓ | `exp_256_photo2art\styleid_256\images` | 750 |
| wct | ✓ | `exp_256_photo2art\wct_256\images`（另 `Github\Latent_Style\exp_baseline_256\wct\step_000001\images`） | 750 |
| cut | ⚠ | `Github\Latent_Style\exp_baselines\_auxiliary_runs\cut_5x5\infer_5x5\images` | **2427**（需按 P256 协议 5风格×150 筛选 750） |
| seedream | ⚠ | `Github\Latent_Style\seedream45_api\protocol_a_800\images`（子目录 cezanne=147/Hayao=146/vangogh=144/monet=142/photo=142） | **721**（缺 29，需补齐） |
| stylealigned / styleshot / weave / zstar | ✗ | 远程扫描未命中 | — |

### R5-512 / R5-WikiArt（WikiArt，5 风格）
| 方法 | 状态 | 远程路径 | 图数 |
|---|---|---|---:|
| 全部（adain/wct/weave/identity/cut/samam/samst/styleid…） | ✗ | 远程**无任何 R5 生成产物**。扫描命中 `wikiart` 的仅 `I:\datasets\wikiart_distinct5_512_images\*`（源测试/训练集，30/1000 图），非生成结果 | — |

> 结论（与既有决策一致）：R5-WikiArt/R5-512 的 750 协议组为**本地生成、远程无源**；其中 adain/wct/weave/identity 的 12000（20风格）版本就不属于 750 协议，仅保留作 20 风格大图；R5-WikiArt 750 协议只纳入本地已干净的 cut/samam/samst/styleid。

### 我们的模型（OUR）
| 方法 | 状态 | 远程路径 | 图数 |
|---|---|---|---:|
| latent512_e7 | ✓ | `exp_our_models_eval\latent512_e7\images` | 750 |
| latent256_e10 | ⚠ | `exp_our_models_eval\latent256_e10\images` | **444**（不齐，需补齐至 750） |
| samst_latent_eval | ✓ | `exp_samst_latent_eval\step_000001\images` | 750 |
| pixel256 | ✗ | 远程未见（仅 `exp_ours\recent\*` 训练 full_eval，非整理协议集） | — |

## 三、待办（远程侧）
1. P256/cut：从 2427 图中按协议 5风格×150 筛选 750。
2. P256/seedream：补齐缺失的 29 张（protocol_a_800 各风格 142–147）。
3. 我们的模型 latent256_e10：补齐至 750。
4. R5-* 全部基线：远程无源，依赖本地生成结果。
5. D5-512 的 adain/wct/identity/weave/cut/styleid 及 P256 的 stylealigned/styleshot/weave/zstar：远程未命中，需确认是否在别处或需重跑。

## 四、源数据集（非生成，供参考）
- `I:\datasets\wikiart_distinct5_512_images` — D5-512 源（test 30/风格，train 1000/风格）
- `I:\datasets\wikiart_distinct5_samam_512*` — SAMAM 用 D5 源
- `I:\datasets\fewshot_data\5p*\test\*` — few-shot 测试源（30/风格）
- `I:\Github\Latent_Style\style_data\train\photo` — 风格参考源
