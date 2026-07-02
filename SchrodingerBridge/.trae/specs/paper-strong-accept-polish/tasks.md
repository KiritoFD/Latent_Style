# Tasks

# Task 1: 正文删除所有"过程问题" + 叙事重构
- [x] SubTask 1.1: 读取当前 paper.tex，grep "645"、"22 modules"、"subtractive"、"ablation campaign"、"半年" 等过程问题表述
- [x] SubTask 1.2: 从 §1 Introduction 完全删除 645 次消融、22 模块、"减法消融"等任何提及
- [x] SubTask 1.3: 从 §4.4 Ablation 删除"减法消融"叙事，只保留方法组件消融（DWT 级数、base locking、EOTA、stochastic routing）
- [x] SubTask 1.4: 重写 §1 Introduction：从消费级硬件难处出发（diffusion 重 + CUT/SaMam 慢 + SaMam 不如 identity 0.5816<0.6933）→ 提出 SFM 方案 → 展示硬结果
- [x] SubTask 1.5: 重写 Abstract，第一句直接亮出最终结果（3 min、RTX 3060、CLIP-S 0.7213）
- [x] SubTask 1.6: 删除任何"邀功"语调

# Task 2: VAE latent 语义优势阐述
- [x] SubTask 2.1: 在 §3.1 Problem Setup 添加段落：VAE latent 不只降低开销，其压缩语义表示更有利于风格表征（vs 像素空间低层纹理）
- [x] SubTask 2.2: 引用相关文献支撑 VAE 语义优势论点（rombach2022highresolution 已加到 refs.bib）

# Task 3: 像素空间对比实验
- [x] SubTask 3.1: 检查现有配置是否支持像素空间模式（grep config 或代码）—— 不可行
- [~] SubTask 3.2: 若支持，跑像素空间 SFM —— 不可行（RTX 3060 12GB 无法处理 192× 数据量）
- [~] SubTask 3.3: 评估像素空间 SFM 的 CLIP-S/LPIPS —— 不可行
- [x] SubTask 3.4: 在 §4 添加像素 vs latent 对比（改用文献论证，§4.3 末尾段落）
- [x] SubTask 3.5: 若像素空间性能显著低于 latent，则实证 VAE 语义优势 —— 文献论证（Rombach 2022）

# Task 4: 语言全文审查
- [x] SubTask 4.1: 全文逐句检查，确保每句 ≤25 词，无嵌套从句
- [x] SubTask 4.2: 将被动语态堆叠改为主动语态
- [x] SubTask 4.3: 检查所有术语首次出现有清晰一句话定义
- [x] SubTask 4.4: grep 确认无内部代号（LBM, FC-SB, I7, U4, V6 等）—— 0 matches
- [x] SubTask 4.5: 删除复杂句式和"邀功"语调

# Task 5: 核实 baseline 数值口径 + Table 1 修正
- [x] SubTask 5.1: 读取 `docs/baseline/README.md` v5，提取 12 baseline 的 CLIP-S/LPIPS 真实值
- [x] SubTask 5.2: 读取 `distinct5_idt_bootstrap_extended.csv`，提取 CI95_low/CI95_high
- [x] SubTask 5.3: 逐行核对 paper.tex Table 1 数值与 README/CSV 是否一致 —— 一致
- [x] SubTask 5.4: 在 Table 1 caption 添加口径声明脚注（750-pair all-pairs protocol）
- [x] SubTask 5.5: 确认 SaMam 0.5816 < identity 0.6933 论点有数据支撑（已加 "on the 750-pair protocol" 限定）

# Task 6: Table 1 添加 bootstrap 95% CI + 3-seed std
- [x] SubTask 6.1: 从 CSV 提取 SFM/SaMam/Seedream 三行 CI95 —— CSV 无 SFM/SaMam 行，仅 SaMST/Seedream 有 transfer-pair CI
- [~] SubTask 6.2: 修改 Table 1，这三行改为 `mean [CI_low, CI_high]` —— 数据不可得，未添加（诚实声明）
- [~] SubTask 6.3: 检查 exp/FCSB/local_t/ 下是否有 T11 其他种子运行 —— 无
- [~] SubTask 6.4: 若无，补跑 2 次种子 —— 未执行（需后续补跑）
- [~] SubTask 6.5: 在 Table 1 SFM 行添加 ±std —— 数据不可得，supplement 有 TODO
- [x] SubTask 6.6: 更新 Checklist 第 8 项（诚实声明 single-seed + supplement 文档）

# Task 7: 重写 supplement_aaai2027.tex
- [x] SubTask 7.1: 读取当前 supplement，识别所有 LBM 旧术语位置
- [x] SubTask 7.2: 重写 §A 方法详述（与主文 §3 术语/符号一致）
- [x] SubTask 7.3: 重写 §B 减法消融审计（645 配置、22 模块清单）——客观补充材料
- [x] SubTask 7.4: 重写 §C 3-seed 稳定性数据（seed 1 填入，seeds 2-3 标 TODO）
- [x] SubTask 7.5: 重写 §D 失败案例与局限性
- [x] SubTask 7.6: supplement 独立编译 0 errors（7 页）

# Task 8: 编译验证 + git commit
- [x] SubTask 8.1: 编译主文，验证 7 正文 + 2 引文/checklist，0 errors（9 页，0 errors）
- [x] SubTask 8.2: 编译 supplement，0 errors（7 页）
- [x] SubTask 8.3: grep "645\|22 modules\|subtractive\|半年\|LBM\|lbm\|FC-SB" 主文返回 0（过程问题已清除）
- [x] SubTask 8.4: 验证 §1 叙事为"难处→方案→结果"，无过程问题
- [x] SubTask 8.5: git commit 分阶段

# Task Dependencies
- [Task 2] depends on [Task 1]（VAE 阐述在叙事重构后）✓
- [Task 3] 可与 [Task 1,2] 并行（像素实验独立）✓
- [Task 4] depends on [Task 1,2,3]（语言审查在内容定稿后）✓
- [Task 6] depends on [Task 5] ✓
- [Task 7] SubTask 7.3-7.4 依赖 [Task 5,6] ✓
- [Task 8] depends on [Task 1,2,3,4,5,6,7] ✓
- [Task 1] 和 [Task 5] 可并行 ✓
- [Task 7] SubTask 7.1-7.2 可与 [Task 1,2] 并行 ✓
