# Overfit50 实验记录与结论

## 0. 研发约束（强制）
- 风格注入能力是首要目标与绝对优先级，任何“结构更稳但风格消失”的改动都按失败处理。
- 每次代码改动前先做 Git 备份提交（允许空提交），再实施改动。
- 每轮实验必须同时记录：定量指标、`collage.jpg` 定性结果、失败原因与下一步单变量改动。
- 禁止一次同时引入多个主改动，避免结论不可归因。


## 1. 目标与判定标准
- 目标：在 `overfit50` 上先验证“模型确实学到双向风格迁移”，再上大规模数据。
- 判定维度：
- 定量：`clip_style`、`content_lpips`、`classifier_acc`、`photo_to_art`。
- 定性：`collage.jpg`（是否只有色调变化、是否结构崩坏、是否双向都有效）。
- 工程：训练/评估过程稳定，无僵尸进程占显存。

## 2. 已完成实验（小规模、可复现实验）

### 2.1 overfit50_infra_v1
- 配置：`src/experiments/overfit50_infra_v1.json`
- 指标：
- `transfer clip_style = 0.4590`
- `transfer content_lpips = 0.3009`
- `transfer classifier_acc = 0.70`
- `photo_to_art clip_style = 0.5031`
- `photo_to_art classifier_acc = 0.54`
- 结论：作为 infra 修复后的新基线可用，但风格强度偏弱，方向不对称明显。

### 2.2 overfit50_model_d1_v1
- 配置：`src/experiments/overfit50_model_d1_v1.json`
- 核心改动：Decoder 端风格调制增强（`use_decoder_adagn=true`）。
- 指标：
- `transfer clip_style = 0.4671`
- `transfer content_lpips = 0.3056`
- `transfer classifier_acc = 0.84`
- `photo_to_art clip_style = 0.5122`
- `photo_to_art classifier_acc = 0.78`
- 结论：跨域可控性显著提升，是目前第一版有效模型。

### 2.3 overfit50_model_d2_v1
- 配置：`src/experiments/overfit50_model_d2_v1.json`
- 核心改动：Decoder 显式注入 32x 风格空间图。
- 指标：
- `transfer clip_style = 0.4671`
- `transfer content_lpips = 0.3064`
- `transfer classifier_acc = 0.85`
- `photo_to_art clip_style = 0.5113`
- `photo_to_art classifier_acc = 0.80`
- 结论：较 d1 小幅继续提升，当前稳定可用的候选基线之一。

### 2.4 overfit50_model_d3_v1
- 配置：`src/experiments/overfit50_model_d3_v1.json`
- 核心改动：增加 texture head 分支。
- 指标：
- `transfer clip_style = 0.4671`
- `transfer content_lpips = 0.3069`
- `transfer classifier_acc = 0.85`
- `photo_to_art clip_style = 0.5120`
- `photo_to_art classifier_acc = 0.80`
- 结论：与 d2 几乎持平，收益有限，证明“仅加分支”不足以解决风格强度天花板。

### 2.5 overfit50_model_d4_v1_quick
- 配置：`src/experiments/overfit50_model_d4_v1_quick.json`
- 指标：
- `transfer clip_style = 0.4413`
- `transfer content_lpips = 0.2438`
- `transfer classifier_acc = 0.08`
- `photo_to_art clip_style = 0.4734`
- `photo_to_art classifier_acc = 0.06`
- 结论：明显回退，模型趋向保守/近恒等映射，跨域几乎失效。

## 3. 关键观察（跨实验）
- 问题 1：风格强度上不去。即便跨域分类准确率提升，`clip_style` 仍停留在 `~0.51` 附近。
- 问题 2：模型容易走“低频/色调捷径”。视觉上会出现“有变化但偏颜色变换，不是纹理风格变化”。
- 问题 3：分类器引导存在单侧偏置风险。某些设置会把一个方向打穿，另一个方向仍弱。
- 问题 4：结构保持与风格注入存在拉扯。约束过强时变成恒等，过弱时结构漂移。

## 4. Infra 结论
- 之前的显存爆掉，主要是僵尸 `python3.12` 进程残留导致，不是模型本身显存常驻。
- 当前脚本链路（清理 + 再运行）后，`nvidia-smi` 可回到无进程占用，显存可正常释放。
- 训练中 GPU “锯齿”仍存在，但已经确认 DataLoader 等待时间低（毫秒级），主因是小批量短步长任务自身的 burst 形态，不是单点阻塞。

## 5. 最新模型级改进（已落地）
- 文件：`src/model.py`
- 改动：新增“残差高频偏置”机制，抑制亮度/低频捷径。
- 新增参数：
- `use_delta_highpass_bias`（默认 `true`）
- `style_delta_lowfreq_gain`（默认 `0.35`）
- 核心逻辑：将输出残差分解为高频+低频，保留高频细节并衰减低频分量；同时对 `style_texture_head` 输出做同样处理。
- 预期收益：
- 减少“只改亮度/色调”的投机路径。
- 在不明显增加结构破坏的前提下，提高纹理型风格注入占比。

## 6. 当前结论与建议
- 当前可用基线：`d2/d3`（综合最稳）。
- `d4_quick` 不可用于放大训练。
- 是否可直接上大规模：暂不建议。先在 overfit50 上验证“高频偏置改动”至少达到：
- `photo->Hayao classifier_acc >= 0.60`
- `style_transfer_ability classifier_acc >= 0.70`
- `photo_to_art clip_style >= 0.53`
- 拼图中能看到稳定纹理风格变化（不只是全局色调变化）。

## 7. 回归排查（d2/d3 有效，d4/d5 后风格失效）
- 结论：存在明确的配置生效回归，不是单纯超参数问题。
- 问题点 1：`trainer` 初始化模型时，未把 `use_delta_highpass_bias/style_delta_lowfreq_gain` 传入构造函数，导致配置文件里“关闭高频偏置”不生效，实际一直按默认值运行。
- 问题点 2：`inference` 初始化模型时同样遗漏这批参数，导致训练/评估模型结构与开关不一致。
- 问题点 3：`style_spatial_ref/id` 在模型内被强制高通化（归一化 + tanh），会压缩空间风格信号；现已改为可配置开关，默认关闭。

### 7.1 已修复代码位置
- `src/trainer.py`：补齐模型构造参数透传（`use_delta_highpass_bias`、`style_delta_lowfreq_gain`、`use_style_spatial_highpass`）。
- `src/utils/inference.py`：补齐与训练一致的模型构造参数透传，保证评估路径和训练路径一致。
- `src/model.py`：新增 `use_style_spatial_highpass` 开关；`encode_style_spatial_ref/id` 不再默认强制高通。

### 7.2 修复后快速回归（overfit50_e1_baseline_d2_ref_fixfull）
- 配置：`src/experiments/overfit50_e1_baseline_d2_ref.json`（8 epoch 回归）
- 训练现象：
- `style_pred_acc` 约 `0.36 -> 0.39`（未恢复到历史 d2 高值）
- `proto_cos_max` 约 `0.09 -> 0.56`（原型分离度恶化，存在风格原型塌缩风险）
- 评估结果：
- `style_transfer_ability.classifier_acc = 0.08`
- `photo_to_art.classifier_acc = 0.06`
- `conditional_sensitivity.delta_abs = 0.00264`（仍偏小）

### 7.3 解释
- 这次排查确认：d4/d5 之后确实存在“代码层回归（配置未生效 + 训练/评估开关不一致）”。
- 但修复回归后，当前模型仍未恢复历史最好 d2 性能，说明还叠加了另一个问题：风格原型/风格编码路径在当前损失组合下出现塌缩，导致跨域条件信号不足。

## overfit50_e1_baseline_d2_ref (2026-02-09 12:20:12)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e1_baseline_d2_ref.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.441247261762619`
- transfer content_lpips: `0.24389759046`
- transfer classifier_acc: `0.09`
- photo_to_art clip_style: `0.4734613049030304`
- photo_to_art classifier_acc: `0.06`

## overfit50_e2_highpass_default (2026-02-09 12:24:15)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e2_highpass_default.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e2_highpass_default/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e2_highpass_default/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.44101888746023177`
- transfer content_lpips: `0.2439115175`
- transfer classifier_acc: `0.09`
- photo_to_art clip_style: `0.4731108498573303`
- photo_to_art classifier_acc: `0.06`

## overfit50_e3_highpass_strong (2026-02-09 12:28:17)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e3_highpass_strong.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e3_highpass_strong/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e3_highpass_strong/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.4412660574913025`
- transfer content_lpips: `0.24394794434`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.4734386056661606`
- photo_to_art classifier_acc: `0.06`

## overfit50_e4_highpass_light (2026-02-09 12:32:17)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e4_highpass_light.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e4_highpass_light/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e4_highpass_light/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.44102913230657576`
- transfer content_lpips: `0.24390161465`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.47327480316162107`
- photo_to_art classifier_acc: `0.06`

## 7. E1-E4 收敛结论（按 Checklist）
- 结论：`E1~E4` 四组几乎同分，说明仅靠“delta 高频偏置系数”在当前损失形态下无法打破恒等解。
- `style_transfer_ability classifier_acc`：`0.08~0.09`，远低于门槛 `>=0.70`。
- `photo->Hayao classifier_acc`：稳定在 `0.06`，目标风格条件基本未生效。
- 新增条件敏感性指标：
- `delta_abs` 约 `0.00251~0.00254`，变化幅度极小。
- `delta_high_ratio` 约 `0.549~0.551`，频段占比变化不大，但在“总变化接近 0”时没有实质意义。
- 解释：模型当前主要问题不是“低频占比”，而是“跨域驱动力不足（几乎不改）”。
- 决策：
- 下一步回到 `d2/d3` 可用区间，优先恢复能跨域的驱动力（提高有效风格监督/跨域约束），再做频段精修。
- 暂不上大规模训练，继续在 overfit50 完成可控双向迁移后再放大。

## 8. Step B（风格监督上移）验证结果
- 实验：
- `overfit50_e5_stepb_feat_student`
- `overfit50_e6_stepb_feat_teacher`
- 目标：将风格监督主信号迁移到 `encode_style_feats` 多尺度高维特征，避免低维统计捷径。
- 结果：
- E5 `transfer classifier_acc = 0.08`，`photo_to_art classifier_acc = 0.06`。
- E6 `transfer classifier_acc = 0.09`，`photo_to_art classifier_acc = 0.06`。
- 条件敏感性：
- E5 `delta_abs = 0.0025338`，`delta_high_ratio = 0.5524`。
- E6 `delta_abs = 0.0026366`，`delta_high_ratio = 0.5486`。
- 结论：
- Step B 在当前损失组合下未打破近恒等解，跨域能力仍失效。
- 仅“上移风格监督”不足以恢复可用迁移，必须先恢复有效跨域驱动力（门槛型分类器约束或方向性约束），再叠加纹理监督。
## overfit50_e5_stepb_feat_student (2026-02-09 12:42:53)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e5_stepb_feat_student.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e5_stepb_feat_student/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e5_stepb_feat_student/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.4409090027213097`
- transfer content_lpips: `0.24388766420000002`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.473043612241745`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.002533837304217741`
- cond delta_high_ratio: `0.552445408677266`

## overfit50_e6_stepb_feat_teacher (2026-02-09 12:50:02)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e6_stepb_feat_teacher.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e6_stepb_feat_teacher/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e6_stepb_feat_teacher/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.44111956417560577`
- transfer content_lpips: `0.2439911685`
- transfer classifier_acc: `0.09`
- photo_to_art clip_style: `0.4734433740377426`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.002636616702657193`
- cond delta_high_ratio: `0.5486421574325153`
## overfit50_e1_baseline_d2_ref_fixcheck (2026-02-09 13:03:29)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e1_baseline_d2_ref_fixcheck.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref_fixcheck/full_eval/epoch_0004/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref_fixcheck/full_eval/epoch_0004/collage.jpg`
- transfer clip_style: `0.44092536717653275`
- transfer content_lpips: `0.24408357843999998`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.47288052797317504`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.0030763717915397136`
- cond delta_high_ratio: `0.5307256610506061`

## overfit50_e1_baseline_d2_ref_fixfull (2026-02-09 13:07:59)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e1_baseline_d2_ref_fixfull.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref_fixfull/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e1_baseline_d2_ref_fixfull/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.44142946928739546`
- transfer content_lpips: `0.24386850406`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.47367475509643553`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.0026404118712525814`
- cond delta_high_ratio: `0.5413608090681745`
## overfit50_e7_proto_sep_mid (2026-02-09 13:30:04)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e7_proto_sep_mid.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e7_proto_sep_mid/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e7_proto_sep_mid/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.4415360137820244`
- transfer content_lpips: `0.24378026265000002`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.47371913731098175`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.0036207712488248943`
- cond delta_high_ratio: `0.4947585201314335`

## overfit50_e8_proto_sep_strong (2026-02-09 13:33:53)
- config: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/src/experiments/overfit50_e8_proto_sep_strong.json`
- summary: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e8_proto_sep_strong/full_eval/epoch_0008/summary.json`
- collage: `/mnt/g/GitHub/Latent_Style/Cycle-NCE/small-exp-overfit50_e8_proto_sep_strong/full_eval/epoch_0008/collage.jpg`
- transfer clip_style: `0.4412860089540481`
- transfer content_lpips: `0.24370155675000005`
- transfer classifier_acc: `0.08`
- photo_to_art clip_style: `0.4736902046203613`
- photo_to_art classifier_acc: `0.06`
- cond pair_count: `100`
- cond delta_abs: `0.0028640731738414616`
- cond delta_high_ratio: `0.5344088328196556`

