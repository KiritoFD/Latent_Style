# 05 项目历史时间线

git 历史从 2026-05-16 开始（之前为导入/压缩），更早的日期取自文档头。

| 阶段 | 时间 | 核心思路 | 为什么被替代 | 最好结果 / 汇总文档 |
|---|---|---|---|---|
| SA-Flow（OT flow matching + Reflow） | 2026-01 前 | VAE 潜空间 flow matching、batch OT 匹配、Reflow 伪配对、AdaGN + 时间门控 | 只有 `readme.md` 描述，无数字 | 未找到 |
| DiT / Thermal / LGT | 2026-01-13 – 02-07 | DiT（控制弱）→ LGTUNet（AdaGN+ResBlock）；cross-attn 失败回滚；LGT 理论（Patch-SWD + Cosine-SSM） | 走向极简 U-Net，成为 Cycle-NCE | `Termal-dynamic.md`、`lgt.txt`、`verify_lgt.py` |
| Cycle-NCE / LatentAdaCUT | 2026-02-08 – 04-02 | Cycle/NCE/identity → Gram → SWD → Color loss、TextureDictAdaGN → cross-attn → 瘦身为 SWD+Color+Identity | 非分布级、注意力膨胀导致 identity shortcut、8 GB 显存 | DiT 5-style style 0.820；CNN 最好 0.716；**汇总**：`Cycle-NCE/ARCHAEOLOGY_FINAL_CN.md`、`History_Report.md` |
| Latent CycleGAN / final_works | ~2026-03 | 潜空间 CycleGAN；CUT/SaMST/StarGAN 评测 | GAN 不稳定，转为 baseline | `final_works/final_works_metrics.csv`（CUT sta_clip_style 0.754） |
| LANCET / SchrodingerBridge | ~2026-04 – 05 | backbone 作为速度场 v(z_t,t,style)，SWD 引导 OT 耦合 + 桥匹配 | K/C/W 消融显示动能项伤风格 | 7ep：CLIP-S 0.7161、LPIPS 0.4514、310 s（`Plan_Docs/RESULTS_SUMMARY.md`）；AAAI 2026 稿 `SchrodingerBridge/aaai_submission/` |
| Tokenizer 探针 / inmortal | 2026-05-27 – 06-15 | style tokenizer、split-axis geometry、xpred+kmanifold | "雾化/白化" | inmortal CLIP-S 0.7372（LPIPS 0.6069），`best.csv`、`experiment_analysis_summary.txt` |
| 620 SpatialBridge / 621 白化审计 / 630 | 2026-06-20 – 06-30 | transformer blocks、DINO cross-attn；诊断 endpoint 收缩 α≈0.16、style gate 0.05 | 催生 WEAVE 的小波/HF 路由 | `docs/621/README.md`（最佳 clip_style 0.7051） |
| **WEAVE** | 2026-07-10 起 | Haar 解耦、LL 源对齐、HF 子带目标码、逐步 AdaIN；DINO-S 为主指标 | 当前 | AAAI 2027 投稿 `WEAVE/aaai2027_v4/`；rebuttal 实验 2026-07-16~18；ICME 版 `WEAVE/icme2027/` |

目录改名：`SchrodingerBridge/` → `WEAVE/` 发生在 commit 68884b68（2026-07-15），旧目录仍被跟踪，包含 rebuttal 原始数据。

## 历史文档中哪些值得看

- Cycle-NCE：只看 `ARCHAEOLOGY_FINAL_CN.md` 与 `History_Report.md`；`ARCHAEOLOGY_CN_SUMMARY.md`、`ARCHAEOLOGY_COMPLETE_REPORT_CN(_FULL).md` 是重复稿，两个 `*PLAN.md` 是空壳。
- 6 月：`docs/621/README.md` 为索引（其中引用的 `ablation_results.md` 不存在）。
- 数据集：`style_data/` 就是 P2A-256 数据（train 10361：photo 6187、Hayao 1752、monet 972、cezanne 850、vangogh 600；test 200）。

## root 级旧数据库（冻结）

| 文件 | 行数 | 说明 |
|---|---:|---|
| `experiment_database_all.csv` | 2503 | 20 列旧 schema（clip_style/clip_content/lpips…），截止 2026-06-15，无 DINO；生成脚本疑为 `WEAVE/docs/archive/scan_and_dashboard.py`（未确认） |
| `experiment_database_best_per_config.csv` | 326 | 同上，每配置最佳 |
| `docs/results/*.csv` | 8 个板 × all/best | 同一 schema 按协议拆分（distinct5_512、strict_protocol_750、wikiart512、photo_monet_5x5、tokenizer_probes、legacy256_overfit50、early_experiments、aaai2027_inmortal） |
| `run_summary.csv/json`、`manifest.json` | — | 步数扫描 + lambda 网格日志，含占位计时 |
| `results/tables/main_table.csv` | — | 早于当前 DINO-S 列的主表，过期 |
