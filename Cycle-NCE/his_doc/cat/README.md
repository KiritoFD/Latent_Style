# Cycle-NCE 文档总目录

更新时间：2026-04-09

这个目录是 `Cycle-NCE/his_doc` 的总入口。目标不是再写一份草稿，而是把已经散落的历史文档、实验记录、报告素材、辅助资产分层放好，后续无论是继续考古、写论文、做中期答辩，还是恢复某条实验线，都能先从这里定位材料。

## 1. 目录分层

### `cat`

作用：
- 放总索引、目录说明、使用建议
- 让整个资料库有统一入口

当前建议从这里开始读：
1. [README.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/README.md)
2. [MATERIAL_MAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MATERIAL_MAP.md)
3. [MIDTERM_DEFENSE_READING_ORDER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MIDTERM_DEFENSE_READING_ORDER.md)

### `raw`

作用：
- 存第一手原材料，不强行改写
- 作为后续正式报告的证据池

子目录说明：
- `legacy`
  - 早期历史考古草稿、旧总结、散装思路
- `gen49`
  - `docs_49` 系列模块化自动/半自动文档
- `assets`
  - 原先散落在项目根目录下的 PDF、PPT、HTML、DOCX 等资产
- `docs_old`
  - 原 `Cycle-NCE/docs` 整体归档

### `notes`

作用：
- 存工作过程中的中间整理稿
- 强调“结构化可追溯”，不追求最后语言最漂亮

当前核心目录：
- [2026-04-09_rebuilt_history](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history)
  - 这套是基于 git 历史和 `Y:\experiments` 重建出来的历史资料库

### `reports`

作用：
- 放适合直接给老师、组会、答辩使用的正式稿
- 允许高层叙事压缩，但必须能追溯回 `notes`

当前正式稿：
- [2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)

### `appx`

作用：
- 放附录、表格汇总、答辩备用材料
- 后续适合加入更多 commit 表、实验榜单、补充截图说明

## 2. 推荐使用方式

如果目的是继续补历史：
- 先看 [MATERIAL_MAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MATERIAL_MAP.md)
- 再进 [2026-04-09_rebuilt_history/README.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/notes/2026-04-09_rebuilt_history/README.md)

如果目的是准备中期答辩：
- 先看 [2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/reports/2026-04-09_MIDTERM_DEFENSE_EXPERIMENT_REPORT.md)
- 再按 [MIDTERM_DEFENSE_READING_ORDER.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MIDTERM_DEFENSE_READING_ORDER.md) 回到证据层

如果目的是找原始资产：
- 直接从 [MATERIAL_MAP.md](/g:/GitHub/Latent_Style/Cycle-NCE/his_doc/cat/MATERIAL_MAP.md) 对照进入 `raw`

## 3. 这次归档的原则

- 原始材料与正式结论分开
- 工作笔记与答辩文稿分开
- 历史证据尽量可追溯到 git / experiments / 旧文档
- 后续新增材料优先按“原材料 / 工作笔记 / 正式报告 / 附录”四层放置

## 4. 后续可继续扩写的位置

- `appx` 增补关键 commit 的源码差异表
- `reports` 增补论文式方法章节和实验章节
- `notes` 继续补实验曲线、epoch 甜点、失败模式样例
