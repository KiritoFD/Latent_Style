# WEAVE 宣讲答辩 PPT

`WEAVE_defense.pptx`：36 页（正文 27 页 + 附录 9 页），16:9，沿用中期答辩模板（封面、章节页、校徽页眉），
每页都附讲稿备注（演讲者视图可见）。

| 部分 | 页码 | 内容 |
|---|---|---|
| 开场 | 1–4 | 封面、提纲、研究概览、对照中期计划的完成情况 |
| 01 问题 | 5–9 | 背景、恒等捷径、IDT–TGT 夹逼判据、11 个基线的审计 |
| 02 诊断 | 10–11 | Haar 频带的梯度能量与风格可分性 |
| 03 方法 | 12–16 | 总览、源锚定端点与命题 1、频带加权匹配与纹理码、逐步对齐与无指标早停 |
| 04 实验 | 17–26 | 设置、D5 主表、三基准、效率、定性、消融、稳健性、总结、下一步 |
| 结尾 | 27 | 致谢与提问 |
| 附录 | 28–36 | 研究历程，以及 P2A、提升幅度、参考图、公平性、基线核查、超参数、ArtFID 的问答页 |

## 重新生成

```bash
python make_slide_figures.py   # assets/*.png：中文图表（需 HarmonyOS Sans SC 字体与 matplotlib）
python make_equations.py       # assets/eq_*.png 与 architecture.png（需 pdflatex、pdftocairo）
python make_deck.py            # WEAVE_defense.pptx（需 python-pptx）
```

- `template.pptx`：中期答辩 PPT 中的三页（封面、章节页、内容页）及其内嵌的鸿蒙黑体字体；所有新页面都由这三页复制而来。
- 数字来源与论文一致：主表与图表直接读取 `../icme2027/data/*.csv`；消融、种子与早停读取
  `SchrodingerBridge/experiments/rebuttal_20260716/` 下的原始文件；参考池、未见风格族、SD1.5 插件三处数值按论文转录
  （出处见 `docs/experiments/02_results_of_record.md`）。
- SD-Turbo 因配置错误已从论文中移除，PPT 同样不含（`main_table.csv` 中 `in_paper=0`）；附录第 34 页说明了原因。
- 预览：`soffice --headless --convert-to pdf WEAVE_defense.pptx`。LibreOffice 与 PowerPoint 的自动换行可能略有差异，
  正式答辩前建议在 PowerPoint 中整体过一遍。
