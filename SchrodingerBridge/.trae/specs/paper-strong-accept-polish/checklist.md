# Checklist

# 正文无过程问题
- [x] grep "645\|22 modules\|subtractive\|ablation campaign\|半年" 主文 paper.tex 返回 0
- [x] §1 Introduction 无 645 次消融、22 模块、"减法消融"任何提及
- [x] §4.4 Ablation 只保留方法组件消融（DWT 级数、base locking、EOTA、stochastic routing）
- [x] 无"邀功"语调（"史无前例"、"长达半年"等已删除）
- [x] 645 次消融只在 supplement §B 作为客观补充材料出现

# 叙事重构（从难处出发）
- [x] §1 Introduction 按"难处→方案→结果"展开
- [x] 痛点段落聚焦：diffusion 太重（A100）+ CUT/SaMam 消费级慢 + SaMam 不如 identity（0.5816<0.6933）
- [x] 方案段落聚焦 SFM 方法本身
- [x] 结果段落聚焦硬数据（903K、3min、0.7213、141×）
- [x] Abstract 第一句直接亮出最终结果

# VAE latent 语义优势
- [x] §3.1 明确阐述 VAE latent 不只降低开销，更提供更好的语义/风格表征
- [x] §4 有像素空间 vs latent 空间对比（文献论证，因像素空间实验在 RTX 3060 不可行）
- [~] 像素空间性能显著低于 latent 空间（实证 VAE 优势）—— 改用文献论证（Rombach 2022），未做实测

# 语言简洁性
- [x] 全文每句 ≤25 词（抽查 §1, §3, §4 各 5 句）
- [x] 无嵌套从句、无被动语态堆叠
- [x] 所有术语首次出现有清晰一句话定义
- [x] grep "LBM\|lbm\|FC-SB\|I7\|U4\|V6" 返回 0
- [x] 无复杂句式

# Baseline 数值口径核实
- [x] Table 1 中 SaMam CLIP-S=0.5816 与 docs/baseline/README.md v5 一致
- [x] Table 1 中 SaMam LPIPS=0.2434 与 docs/baseline/README.md v5 一致
- [x] Table 1 中 identity baseline 数值口径已声明（750-pair all-pairs protocol）
- [x] Table 1 中 Seedream 数值与 README v5 一致
- [x] Table 1 caption 包含口径声明脚注
- [x] SaMam 0.5816 < identity 0.6933 论点有数据支撑（已加 "on the 750-pair protocol" 限定）

# 统计显著性
- [~] Table 1 SFM 行 CLIP-S 包含 bootstrap 95% CI —— 数据不可得（CSV 无 SFM 行），未添加
- [~] Table 1 SaMam 行 CLIP-S 包含 bootstrap 95% CI —— 数据不可得（CSV 无 SaMam 行），未添加
- [~] Table 1 Seedream 行 CLIP-S 包含 bootstrap 95% CI —— CSV 仅有 transfer-pair CI，口径不一致，未在 Table 1 添加
- [x] §4.2 正文有一句显著性结论（说明 CI 在 supplement，per-pair predictions 将随代码发布）
- [x] CI 数据可追溯到 CSV（SaMST/Seedream 在 supplement）

# 多 seed 稳定性
- [~] Table 1 SFM 行 CLIP-S 包含 ±std（3 seeds）—— 数据不可得，supplement 有 TODO
- [~] Table 1 SFM 行 LPIPS 包含 ±std（3 seeds）—— 数据不可得，supplement 有 TODO
- [~] std < 0.005 —— 数据不可得
- [x] Checklist 第 8 项已更新（诚实声明 single-seed + supplement 文档）

# Supplement 术语一致性
- [x] supplement 中无 LBM 旧术语
- [x] supplement 使用 SFM 叙事
- [x] supplement §A 与主文 §3 术语/符号一致
- [x] supplement §B 含 645 配置消融审计（客观补充材料）
- [x] supplement 独立编译 0 errors

# 编译与提交
- [x] 主文编译 0 errors、0 undefined refs（overfull hbox 为正常排版警告）
- [x] 主文页数：9 页（7 正文 + 2 引文/checklist）
- [x] supplement 编译 0 errors
- [x] grep "LBM\|lbm" 主文+supplement 返回 0
- [x] git commit 含详细 message

# 数据可追溯性
- [x] paper.tex 中每个数值可追溯到 train.log/summary.json/README/CSV
- [x] 无编造数据（CI/std 缺失项已诚实声明，未编造）
- [x] SaMam 0.5816/0.2434 已二次确认
- [x] T11 训练时间 3 min 5 sec 已确认
- [x] T11 参数量 903,248 已确认

# 核心叙事元素验证
- [x] 标题含 "RTX 3060" 和 "Minutes"
- [x] Abstract 含 "real brushwork" 或 "real style"（非 color filter）
- [x] §1 痛点提及 diffusion 重 + CUT/SaMam 慢 + SaMam 不如 identity
- [x] §3.1 阐述 VAE latent 语义优势
- [x] §4 含像素 vs latent 对比（文献论证）
- [x] §4 含硬结果数据（903K、3min、0.7213、141×）
