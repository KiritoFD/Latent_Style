# Overfit50 实验计划（下一轮）

## 0. 执行纪律（强制）
- 风格注入是首要目标与绝对优先级：任何改动若削弱跨域风格注入，视为失败。
- 每次代码改动前必须做一次 Git 备份提交（允许空提交），再开始实现。
- 每轮实验必须同时给出：定量指标 + `collage.jpg` 定性结果 + 失败原因。
- 不在一次实验里混入多个主改动，保证可归因。


## A. 目标
- 在 overfit50 上先验证模型级改动是否有效，再上大规模。
- 优先解决：风格注入“颜色化”而非“纹理化”的问题。

## B. 当前基线
- 主基线：`overfit50_model_d2_v1` / `overfit50_model_d3_v1`
- 参考失败对照：`overfit50_model_d4_v1_quick`

## C. 新改动（模型级，非纯调参）
- 文件：`src/model.py`
- 新机制：残差高频偏置（`use_delta_highpass_bias`）
- 原理：
- 对 `delta` 与 `style_texture_head` 输出做 low/high 分离：`delta = high + lowfreq_gain * low`
- 限制低频分量主导，鼓励风格更多通过高频纹理进入输出

## D. 实验矩阵
- E1：基线复现（d2 参数不变）
- 目的：确认最新 infra 与旧结果一致，排除环境漂移。
- E2：开启高频偏置（默认 `style_delta_lowfreq_gain=0.35`）
- 目的：验证是否减少“只改色调”。
- E3：高频更强（`style_delta_lowfreq_gain=0.25`）
- 目的：进一步压低低频捷径，观察结构损失是否可控。
- E4：高频较弱（`style_delta_lowfreq_gain=0.45`）
- 目的：检查是否存在过抑制导致风格强度反降。

## E. 评估与门槛
- 每个实验都必须产出：
- `summary.json`
- `metrics.csv`
- `collage.jpg`
- overfit 通过门槛：
- `photo->Hayao classifier_acc >= 0.60`
- `Hayao->photo classifier_acc >= 0.80`
- `photo_to_art clip_style >= 0.53`
- `style_transfer_ability classifier_acc >= 0.70`
- 定性要求：拼图中风格变化可见且不以整图发灰/偏色为主。

## F. 决策规则
- 若 E2/E3/E4 任一达到门槛且视觉稳定：进入大规模训练准备。
- 若全部不达标：继续模型级改造（优先 decoder feature matching，不先盲目堆损失权重）。

## G. 统一实验 Checklist（每次 overfit50 必做）
- `1) 条件敏感性检查`
- 固定同一输入 `x`，生成 `G(x, photo)` 与 `G(x, Hayao)`。
- 记录 `delta_abs = mean(abs(y_photo - y_hayao))`。
- 对 `delta = y_hayao - y_photo` 做 low/high 分解，记录 `high_ratio = E_high / (E_low + E_high)`。
- 判定：`delta_abs` 不能接近 0，且 `high_ratio` 相对基线提升（避免纯低频/色调变化）。
- `2) 跨域有效性检查`
- 读取 `summary.json` 的 `photo->Hayao` 与 `Hayao->photo` 的 `classifier_acc`。
- 判定：两方向都必须有效，避免单边打穿（尤其防止只会转 photo）。
- `3) 风格强度检查`
- 关注 `photo_to_art clip_style` 与 `style_transfer_ability classifier_acc`。
- 判定：风格强度提升的同时，`content_lpips` 不出现异常飙升。
- `4) 定性检查`
- 必看 `collage.jpg`：要求出现稳定纹理变化，不接受仅亮度/色调变化。
- `5) 结论归档`
- 每个实验必须记录：通过项、失败项、怀疑原因、下一步只改一个关键因子。

## H. 三步收敛路线（执行顺序固定）
- `Step A`：先做条件敏感性与频段占比统计，确认“style_id 真的生效”。
- `Step B`：将风格监督主信号放在高维多尺度特征（不再依赖低维统计）。
- `Step C`：保持低频结构约束，同时把变化能量偏向中高频（delta 频段解耦）。

## I. 当前状态与下一步（2026-02-09）
- Step A：已完成，指标已接入 `summary.json`（`delta_abs` / `delta_high_ratio`）。
- Step B：已执行（E5/E6），结果未改善跨域迁移，仍接近恒等。
- 下一步执行策略：
- 先恢复“跨域驱动力”到可用区间（目标先回到 `d2/d3` 的有效水平）。
- 再叠加 Step C 的频段约束做精修，而不是单独依赖频段/特征监督。

