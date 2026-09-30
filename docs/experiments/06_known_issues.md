# 06 已知问题与待办

## A. 数字冲突

| # | 问题 | 现状 / 处理 |
|---|---|---|
| A1 | AAAI Table 1：Z-STAR P2A 的 DINO-S/DINO-C 抄成了 R5 的值 | ICME 版已更正为 0.498/0.552 |
| A2 | AAAI Table 1：StyleID R5 DINO-S/DINO-C 0.571/0.374 与 JSON 0.551/0.385 不符 | ICME 版已更正 |
| A3 | AAAI 正文称 R5 是 "20-family" | 实为 5 个随机家族；ICME 版已改 |
| A4 | AAAI 消融表 caption 称重训变体"用同一早停规则选 epoch"，实际报告的是 oracle epoch | ICME 版改 caption，regret 进 supplement Table S3 |
| A5 | † / ‡ 手工标记与定义不符（StyleShot/SaMST D5 LPIPS） | 改为脚本自动计算 |
| A6 | WEAVE D5 在不同文档中有 0.4918 / 0.4917 / 0.4915 / 0.4859 / 0.4867 五套数 | 见 02；只用 0.4918（expD D0_full） |
| A7 | `batch1_summary.md` 仍写 w/o AdaIN 0.4917 "negligible" | 该 run 配置未生效；正确值 0.4821（`d1_adain_corrected.json`） |
| A8 | `batch1_summary.md` 称 learned HH head "改善内容保持" | 错误：LPIPS 0.2670>0.2595，DINO-C 0.8061<0.8102 |
| A9 | seed 7 早停：batch1 说"未触发"，supplement 说 epoch 4 | 前者是旧绝对规则，后者是相对规则；已在 03 说明 |
| A10 | StyleAligned D5 CLIP/LPIPS：BASELINE_PROGRESS 0.8739/0.7825 vs 论文 0.780/0.869 | **未解决**，需重跑评测确认 |
| A11 | CUT R5 的 DINO-C / structure 与 D5 完全相同 | **未解决**，疑似缓存复用 |
| A12 | Latent-WCT DINO 无原始文件；仓库内 `*__wct.json` 是另一个 WCT | **未解决** |
| A13 | Task1+（VAE × Haar 层数）"DINO-S 0.912" 实为 DINO-C 口径、仅 5 对 | 不可引用，已在 03 标注 |
| A14 | ArtFID 中 WEAVE LPIPS 0.283 vs 主表 0.2595 | 不同 LPIPS 设置，分开报告 |
| A15 | `docs/reproduction/*` 的 checkpoint_sha256 列是英文单词而非哈希 | **未解决**，无法校验 |

## B. 缺失的原始文件（paper-only）

WEAVE P2A/R5 行、Other5、SD1.5 插件、latent vs RGB、style-memory 适配、126 s 推理计时、训练日志 CSV、
大多数 baseline 的 P2A/R5 CLIP/LPIPS、Latent-WCT 全部、baseline 的远程运行脚本、DINO 聚合脚本、主 checkpoint 本身。
这些都在远程 Windows 机器（`G:\GitHub\Latent_Style`、`I:\...`）。**建议**：拷回 `WEAVE/docs/reproduction/`（只拷 JSON/CSV，不拷图像），并在 02 把状态改为 verified。

## C. 过期文档（保留原位，但不要引用其中数字）

| 文件 | 过期内容 |
|---|---|
| `WEAVE/aaai2027_v4/SUPPLEMENTARY_MATERIAL.md` | 主结果写的是旧 base 模型 0.4859 行 |
| `WEAVE/config.json::_main_table_metrics` | 指向旧 `brk_a` checkpoint；另 `t_sampling_mode=logit_normal` 字段实际未被 `flow.py` 使用（代码是 uniform） |
| `WEAVE/README.md` "Current Baseline" | 是 base 模型 15-epoch 复现（epoch 6），不是论文模型 |
| `WEAVE/docs/delivery/DELIVERY_SUMMARY.md` | 旧 base 行 |
| `WEAVE/docs/79/main_table_v4_remote.csv` | "903K / 3.08 min" 旧表 |
| `WEAVE/BASELINE_PROGRESS.md` | 2026-07-07 进度，全是 TODO |
| `results/README.md` | 指向 `SchrodingerBridge/aaai2027_v4/paper.tex` 与不存在的 `eval_protocol_750/` |
| `SchrodingerBridge/aaai2027_v4/` | AAAI 稿的旧副本（2026-07-15），与 `WEAVE/aaai2027_v4` 有约 625 行差异 |
| `WEAVE/aaai2027_v4/aaai_arch_diagram_v16_staggered_bundle.drawio.png` | 图中 "903K Params"、"Endpoint-only Injection"、"Diagonal WCT" 与方法不符；ICME 版改用 `WEAVE/icme2027/fig_arch.tex`（TikZ） |
| `Plan_Docs/`、`PAPER_REVISION_PLAN.md` | AAAI 2026 时期 |

## D. 建议的后续清理（本次**未**执行，避免破坏脚本中的硬编码路径）

1. 把 `SchrodingerBridge/experiments/rebuttal_20260716/` 与 `SchrodingerBridge/rebuttal_exps/{experiments,docs,configs}` 迁到 `WEAVE/docs/reproduction/rebuttal/`，同步修改 02/03 中的路径。
2. 删除空 submodule gitlink：`Related_Works/{CSGO,SCSA,SCSA_style_transfer,StyleGallery,StyleShot}`、根目录 `StyleShot/`。
3. 根目录杂物（`nomachine_*.deb`、`page*.png`、`__tmp_*.py`、`session-ses_11b4.md`、`1-21-r1.log`）移到 `archive/` 或删除。
4. 把 `Cycle-NCE/`、`latent_cyclegan/`、`final_works/`、`Plan_Docs/` 整体归入 `archive/history/`，只保留 05 中列出的汇总文档入口。
