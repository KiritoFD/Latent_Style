# Baseline 补充进度与交接文档

> 目的：为论文主表补充近两年 3 个强 baseline（StyleAligned / Z-STAR / StyleShot）的推理 + 评估结果。
> 最后更新：2026-07-07

## 1. 远程机器

- SSH：`ssh -p 2222 administrator@100.115.18.62`（Windows 机器，远程 shell 为 **cmd**，**不是** PowerShell）
- GPU：RTX 3060（12GB）
- Python：`C:\Program Files\Python312\python.exe`（**务必用绝对路径**，schtasks 任务环境 PATH 可能不完整）
- 项目根：`I:\GitHub\Latent_Style\SchrodingerBridge\`
- 实验结果根：`I:\GitHub\Latent_Style\SchrodingerBridge\exp\`

### 已确认的数据集路径（远程）
| 数据集 | 远程路径 | 结构 | 每风格图数 | 总对数 |
|--------|----------|------|-----------|--------|
| D5 (Distinct5) | `I:/datasets/wikiart_distinct5_samam_512_classview/test` | `<style>/<img>` | 30 | 750 (5×5×30) |
| P2A (Photo2Art-256) | `I:/datasets/legacy256_overfit50/test` | `<style>/<img>` | 30 | 750 |
| R5 (Random5 / hold-out) | `I:/datasets/wikiarts20_512_test` | `<style>/<img>` | 30 | 750 |

- P2A 风格：`cezanne, Hayao, monet, photo, vangogh`（256×256）
- R5 hold-out 风格：`Cubism, Expressionism, Pop_Art, Pointillism, Romanticism`（512×512）
- 命名约定：生成图 `{{src_style}}__{{src_stem}}__to__{{tgt_style}}.png`

### 执行约定（重要）
- **必须用 `schtasks` 启动长任务**，不要直接 `ssh ... python ...` 或 `Start-Process`——SSH 断开后进程会被杀。
- 每个任务用一个 `.bat` 包装脚本（含绝对 python 路径 + 输出重定向到 `.log`），再用 `schtasks /Create` 建任务 + `schtasks /Run` 立即触发。
- 检查进度：`ssh ... 'type <logfile>'`，或 `schtasks /Query /TN <name>` 看状态。
- 脚本均幂等（已存在输出图则跳过），可安全重跑/续跑。

## 2. 三个方法状态

### 2.1 StyleAligned（SD1.5, training-free）
- 代码：`tools/style_aligned/`（sa_handler_sd15.py + inversion_sd15.py，已上传远程）
- 推理脚本：
  - D5：`tools/run_stylealigned_distinct5.py`（本地已跑完，结果在 `exp/baseline_stylealigned_distinct5`，CLIP-S 0.8739 / LPIPS 0.7825）
  - P2A+R5：`tools/_run_stylealigned_remote.py`（已上传远程）
- 远程输出：`exp/baseline_stylealigned/photo2art256/images`、`exp/baseline_stylealigned/random5/images`
- **状态：⬜ P2A + R5 待跑（StyleAligned 推理刚用 schtasks 重启）**

### 2.2 Z-STAR（CVPR 2024, SD1.5, zero-shot attention rearrangement）
- 官方仓库：`github.com/HolmesShuan/Zero-shot-Style-Transfer-via-Attention-Rearrangement`
- 需克隆到远程 `tools/zstar/`，写 wrapper 批量跑 D5/P2A/R5（官方 `demo.py` 接口是 folder→folder，需逐对包装）
- 需 SD1.5 权重 `runwayml/stable-diffusion-v1-5`（远程已缓存）
- **状态：⬜ 未开始**（待建 `tools/_run_zstar_remote.py` + schtasks 任务）

### 2.3 StyleShot（AAAI 2025, SD1.5, learned style-aware encoder + IP-Adapter）
- 官方仓库：`github.com/open-mmlab/StyleShot`
- 需克隆到远程 `tools/styleshot/`，下载权重 `Gaojunyao/StyleShot`(HF)、`Gaojunyao/StyleShot_lineart`(HF)、`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
- 推理：逐对 `styleshot_image_driven_demo.py --style ref --content src --preprocessor Contour --prompt "a painting" --output out`
- **状态：⬜ 未开始**（待建 `tools/_run_styleshot_remote.py` + schtasks 任务）

## 3. 评估与主表

- 评估脚本（复用 IP-Adapter 模式）：`tools/_eval_baselines_v3.py`（待建）
  - CLIP-S：`openai/clip-vit-base-patch32`，算 (gen, style_ref) 余弦
  - LPIPS：`lpips.LPIPS(net='vgg')`，统一 resize 后逐对算
  - MUSIQ：`pyiqa.create_metric('musiq')`，D5 可在本地 G 盘跑
- 主表：`aaai2027_v4/paper.tex`，当前 10 行 → 增至 13 行
  - 新增编号：5 StyleAligned / 6 Z-STAR / 8 StyleShot
  - 同步更新 caption、baseline 列表、Related Work、首页散点图 `plot_page1_summary.py`

## 4. 已上传远程的文件清单
- `tools/style_aligned/sa_handler_sd15.py`
- `tools/style_aligned/inversion_sd15.py`
- `tools/_run_stylealigned_remote.py`
- `tools/_env_check_remote.py`、`tools/_scan_remote_datasets.py`、`tools/_verify_counts_remote.py`（工具脚本）
- `tools/_sa_launch.bat`（StyleAligned 的 schtasks 启动包装）

## 5. 重启任务命令（schtasks）
```bat
:: 在本地执行（已封装进工具脚本，这里仅作交接记录）
scp tools/_sa_launch.bat administrator@100.115.18.62:I:/GitHub/Latent_Style/SchrodingerBridge/tools/
ssh administrator@100.115.18.62 "schtasks /Create /TN StyleAligned_Runs /TR I:\GitHub\Latent_Style\SchrodingerBridge\tools\_sa_launch.bat /SC ONCE /ST 00:00 /RL HIGHEST /F"
ssh administrator@100.115.18.62 "schtasks /Run /TN StyleAligned_Runs"
:: 查进度
ssh administrator@100.115.18.62 "type I:\GitHub\Latent_Style\SchrodingerBridge\exp\baseline_stylealigned\stylealigned.log"
```

## 6. 待办（TODO）
1. ⬜ StyleAligned P2A+R5 推理完成并评估
2. ⬜ Z-STAR 克隆+wrapper+3 数据集推理+评估
3. ⬜ StyleShot 克隆+权重下载+wrapper+3 数据集推理+评估
4. ⬜ 统一评估脚本跑 CLIP-S/LPIPS/MUSIQ
5. ⬜ 结果填入 paper.tex 主表（编号/caption/baseline 列表/散点图）
