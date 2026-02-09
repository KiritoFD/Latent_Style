# Overfit50 实验计划 V2（概率引导 + 风格注入优先）

## 1. 背景与目标
- 当前已确认：`cls CE` 容易导致模型学分类器捷径，不等于真实风格注入。
- 现阶段目标：在 `overfit50` 上优先验证“风格条件是否真实生效”，再考虑上大规模。
- 已执行约束：
- 训练默认禁用 `cls CE`（只保留概率引导）
- 训练/脚本均启用单实例锁，避免双进程爆显存

## 2. 硬性原则
- 风格注入优先于所有其它指标，出现“结构很稳但风格没变”即判失败。
- 每次只做单变量主改动，避免不可归因。
- 每轮实验必须输出三类证据：
- `summary.json` 定量指标
- `collage.jpg` 定性结果
- `conditional_sensitivity`（`delta_abs`, `delta_high_ratio`）

## 3. 本轮核心改动（已落地）
- `src/losses.py`
- 新增高维高频风格监督：`w_featmatch_hf`, `w_gram_hf`, `w_moment_hf`
- 新增特征层选择：`style_feat_min_level`
- 新增概率门控：`prob_gate_enabled`, `prob_gate_min`, `prob_gate_power`, `prob_gate_detach`
- 默认禁用 `cls CE`：`disable_cls_ce=true`
- `src/run.py`
- 新增单实例训练锁（`run_lock_path`）
- 脚本侧增加 `flock` 锁，避免并发实验

## 4. 实验矩阵（V2）
- `E1_ref_baseline`
- 配置：`src/experiments/overfit50_e1_baseline_d2_ref_fixfull.json`
- 目的：提供禁用 cls CE 后的基线参照
- `E3_highpass_strong`
- 配置：`src/experiments/overfit50_e3_highpass_strong.json`
- 目的：验证高频偏置本身是否带来可见风格提升
- `E9_hifeat_probgate_v1`
- 配置：`src/experiments/overfit50_e9_hifeat_probgate_v1.json`
- 目的：验证“高维高频监督 + 概率门控”联合作用

## 5. 判定标准（过线条件）
- `photo->Hayao classifier_acc >= 0.60`
- `style_transfer_ability classifier_acc >= 0.70`
- `photo_to_art clip_style >= 0.53`
- `conditional_sensitivity.delta_abs` 显著高于近恒等组
- `collage.jpg` 看到稳定纹理变化，非纯色调偏移

## 6. 失败判定
- 任一实验出现训练 `NaN`：该组直接标记为“无效结果”，不用于模型结论。
- `delta_abs` 接近 0 或 `pair_count` 异常：视为条件路径失效。
- 仅 `clip_style` 上升而 `photo_to_art classifier_acc` 接近 0：视为伪提升。

## 7. 执行顺序
1. 跑 `E1_ref_baseline`（确认当前基线）
2. 跑 `E3_highpass_strong`（单改高频偏置）
3. 跑 `E9_hifeat_probgate_v1`（完整 V2 路径）
4. 汇总 `EXPERIMENTS.md`，并按过线条件做 go/no-go

## 8. 注意事项
- full_eval 统一使用覆盖写 `metrics.csv`（默认行为），避免历史污染。
- 若手动重跑同一输出目录，务必传 `--force_regen`。
- 若脚本提示锁文件存在，先确认是否有在跑进程，不要强删锁文件后并发再起。
