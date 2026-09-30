# 为什么我们的 Style 这么差 — 深层根因

> 实验数据: 全部7组 style 0.66-0.67, LPIPS 0.29-0.30. H1(线性FM)反而最好.
> 审计注记(2026-06-18): 当前实现里 `semantic_self_topology_blend=1.0` 并不等于 style value 完全被切断.
> `src/lancet_blocks.py` 的 `SemanticCrossAttn` 只把 attention logits 向 content topology 混合, 但 `v` 仍来自 style map.
> 另一个独立实现问题是: phase-616 旧版 `topogate_attention_gw` 实际只看了最后一个 semantic body block,
> 所以任何旧 h5/h6 结论都要先对照
> `docs/experiments/2026-06-18-topogate-multiblock-audit/README.md`.
> 另见:
> - `docs/experiments/2026-06-18-styleid-eval-probe/README.md`
> - `docs/experiments/2026-06-18-phase616-auto-family-mutation-audit/README.md`
> - `docs/experiments/2026-06-18-lowrank-code-map-order-audit/README.md`
> - `docs/experiments/2026-06-18-current-stage1-lowrank-recheck/README.md`

---

## 一层: TopoGate blend=1.0 强烈压制 Style routing, 但不是实现上的完全切断

**这是重要原因, 但不能再表述成“style 完全进不来”。**

`semantic_self_topology_blend=1.0` 意味着 UNet 的 self-attention 层:
$$A_{\text{final}} = 1.0 \times A_{\text{self-content}} + 0.0 \times A_{\text{cross}}$$

更准确地说:

- attention logits 的 routing 被 content topology 强烈主导
- 但 style values 仍然走 attention value path
- 所以它更像 **极强的 content prior**, 不是实现意义上的 total no-op

`docs/experiments/2026-06-18-bold-eval-graph-preflight/README.md` 已经证明:

- 在 repaired lowrank base 上, `blend=0.0/0.3` 会改 plain eval graph
- 只是 `plain_forward_delta` 只有大约 `7e-4 ~ 1e-3`
- 对 body-level style separation 的提升也很小

按 `docs/experiments/2026-06-18-family-validity-matrix/README.md` 里的 runtime bucket,
这已经属于 **weak_runtime_lever**, 不是 no-op:

- `exact_noop`: `plain_forward_delta = 0`
- `micro_runtime_lever`: `0 < delta <= 1e-4`
- `weak_runtime_lever`: `1e-4 < delta <= 2e-3`
- `moderate_runtime_lever`: `2e-3 < delta <= 2e-2`

所以 close results 现在不能被读成 “blend 把所有东西彻底锁死, 所有组都没变模型”, 而更该读成:

- blend 确实是 runtime lever
- 但它是 **弱 lever**

**证据**: H1(线性FM)反而最好(0.670) 仍然说明结构约束过强会伤 style, 但它不再足以支持 “vertical/TopoGate 让 style 完全不工作” 这个更强说法。

**这也修正了一个关键误读**: 垂直FM、结构OT、SDE、非平衡OT 的 close results, 现在更常见的解释不是 “attention 完全不看 style”, 而是:

1. 训练图真的变了
2. 但 plain no-reference eval path 没跟着变, 或者变得太弱

详见 `docs/618/close_result_triage.md`。

补一条现在已经落到真实远端产物上的证据:

- `docs/experiments/2026-06-18-remote-real-run-audit/README.md`
- `docs/experiments/2026-06-18-current-stage1-lowrank-recheck/README.md`

当前远端 repaired-base `ot_rerun_lowrank_auto` 在 `h0~h4` 上已经形成 `close_cluster`,
而自动诊断给的是 `train_eval_contract_gap`。
这让“训练图变了, 但 benchmarked plain no-reference path 没被拉起来”这条解释
不再只是本地 probe / family matrix 的推断, 而是已经在真实远端整组 run 上出现的现象。

补一条 2026-06-18 晚上的 corrected relaunch 证据:

- 旧的远端 phase618 pipeline 因为 family downgrade / multiblock stale / launcher 显存阈值不一致而整组不可信
- 清理后重新从头拉起的 repaired-base `h0_vertical_fm`:
  - `epoch_0001 -> 0.6709 / 0.3777`
  - `epoch_0002 -> 0.6620 / 0.3341`
  - `epoch_0004 -> 0.6606 / 0.3503`
  - `epoch_0006 -> 0.6611 / 0.3614`
  - validity audit: `artifact_status=valid`, `effect_contract=training_real_eval_inert`
  - convergence state at `epoch_0004`: `best_epoch=epoch_0001`, `since_best=3`, `converged=false`
- 同一 rerun 根上的 `h1_linear_fm` 目前已经明显更差:
  - `epoch_0001 -> 0.6526 / 0.3370`
  - `epoch_0002 -> 0.6493 / 0.4055`
  - `epoch_0003 -> 0.6471 / 0.4265`

这意味着 corrected rerun 的第一手 live run 也已经落在同一个解释框架里:

1. 这次不是“远端还在跑旧代码”
2. 也不是“repaired lowrank family 根本没接上”
3. 它先给出了一个 style 更高的真实点, 说明不是 exact no-op
4. `h0` 和 `h1` 在 corrected rerun 上也已经不是逐点重合, 进一步削弱了“组间没变模型”的解释
5. 但 `h0` 随后轨迹又往旧 close-result 带回落, 所以旧 OT 家族仍更像在训练目标层面发生变化, 而不是把 plain no-reference eval path 真正拉起来
6. 当前代码重跑的 `current-stage1-lowrank-recheck` 也再次证明:
   - `h0/h2/h3/h4/h5/h6` 对 shared-weight plain eval graph 仍是 `0.0` 级别同构
   - 但 training probe 依然把它们分成 `bridge_only_change` 和 `ot_or_target_change`
   - 所以“close 就是模型根本没变”这条解释在当前树上也站不住

同一轮 corrected rerun 里还暴露出一个**实验 infra 层**的实现裂缝, 但它现在也已经修掉:

- `src/run.py` 的 epoch-end early stop 只认 `round2_convergence.json["converged"]`
- `phase616_auto.py` 的外层 runner 却按 `objective_gap` 的 best + patience 在外部停 run
- 这会造成 “run 已经切到下一个变体, 但 convergence.json 还写着 false” 的双重现实

修复后 `round2_convergence.json` 会显式导出:

- `objective_best_epoch`
- `objective_epochs_since_best`
- `objective_patience_converged`
- `stop_ready`
- `stop_reason`

现在 trainer 内层和 auto runner 外层已经吃同一个 stop 信号。
这不是“模型没变”的根因, 但它确实是一个会污染 corrected rerun 阅读的实现问题。

补一条 2026-06-19 的 checkpoint-vs-init 证据:

- `docs/experiments/2026-06-19-checkpoint-style-response-audit/remote_h1_epoch18/README.md`

这里把拉回来的远端 `h1_linear_fm epoch_0018.pt` 和同 config 的随机初始化直接并排比较。
结果不是简单的 “trained suppression”, 而是:

- `overall_reading = matched_target_suppressed_styleid_amplified_body_dead`

关键数字:

- `matched_target_spatial_forward_delta`: `0.02726 -> 0.000678`
- `matched_target_both_forward_delta`: `0.02929 -> 0.000718`
- `topology_gate1_blend_effect_delta`: `0.02928 -> 0.000828`
- `styleid_max_forward_pair_delta`: `0.01079 -> 0.20629`
- `styleid_max_body_pair_delta`: `0.0 -> 0.0`

这说明:

1. 训练后的 checkpoint **绝不是“什么都没学到”**
2. 它把 matched-target / topology 这一侧的 style 响应明显压扁了
3. 但同时把 plain no-reference 的 `style_id -> decoder` 响应放大了十几倍
4. 可是 `h_body` 仍然完全不动

所以现在对 “style 为什么弱” 的更准确表述已经变成:

- 问题不只是 `blend=1.0` 太强
- 也不只是 tokenizer 只有 5 个向量
- 更是 **训练把 style 学到了错误的位置**

也就是说, 模型当前更像学会了:

- 晚期 decoder 的纹理/色彩调制

而没有学会:

- plain no-reference eval 真正需要的 body-level / spatial style transport

这会直接解释一种很容易误读的现象:

- 训练图里的 matched-target spatial path 明明是强的
- 训练后的 checkpoint 也明明对 `style_id` 有更强响应
- 但最终 benchmark 仍然 close

因为真正缺的不是 “任何 style 响应”, 而是 **早层、空间型、body-level 的 no-reference style actuation**。

## 二层: Legacy tokenizer 几乎没有风格表征能力

当前 tokenizer: `legacy_factorized` — style values 是 `Embedding(5, style_dim)` 的查表。5个向量代表5种风格。

**这意味着 model 看到的"风格"只是5个固定向量**, 没有任何内容适应性、没有任何实例级风格信息。这个 tokenizer 甚至比之前被我们废弃的 PureLatentSpatial 更弱——后者至少有 content query → routing 机制。

配合 blend=1.0: **形成双重压制**。Attention routing 被 content topology 强主导 + tokenizer输出的 style values 太弱。

## 三层: 和对比方法的本质差距

| 方法 | 风格来源 | 我们 |
|------|---------|------|
| StyleShot | 从参考图编码 (多尺度MoE, Transformer) | style_id查表 (5个向量) |
| CSGO | Perceiver Resampler从参考图压缩 | style_id查表 |
| StyleGallery | 扩散特征聚类 + 区域匹配 | 无参考图 |
| SCSA | 语义mask引导的attention约束 | TopoGate blend=1.0(过度) |

**所有人都有风格参考图, 我们没有。** 这是最根本的设定差异。

在无参考图设定下, 风格必须完全来自训练数据中学到的类别表征。但我们的:
1. 训练数据只有5类 (且 Impressionism 13030 vs Minimalism 1307, 严重不平衡)
2. Tokenizer 只有5个可学习的 style embedding
3. 没有任何从"风格实例"中提取信息的机制

## 四层: 整个架构偏向内容保持

| 机制 | 效果 | 对style的影响 |
|------|------|-------------|
| TopoGate blend=1.0 | 完美结构保持 | 强烈压制 style routing |
| Residual connections | 保留内容 | 稀释style变化 |
| Skip connections | 内容直通 | 绕过style注入 |
| Velocity prediction | 残差预测(Δx) | 默认锚定内容 |

**整个架构被设计为"在保证结构的前提下做最小的风格改变"**。

补一个新的实现侧证据:

- `docs/experiments/2026-06-18-style-injection-live-init-probe/README.md`

这份 probe 说明, 如果直接打开 `style_injection_mode` 这类 no-reference actuation 分支,
在默认 zero-init 下它们可以是 **exact no-op**.
只有给 style injection 一个小幅 live-init, plain no-reference eval graph 才会明显移动。

补一条 2026-06-18 的实现审计结果:

- 之前 `tools/probe_conditioning_sensitivity.py` 的 anatomy trace 漏掉了 runtime 里的
  `_apply_style_feature_injection(..., site="body"/"decoder")`
- 所以会出现 `forward()` 已经变了, 但 anatomy 行还像没变的假阴性
- 修完之后可以确认: 在 repaired lowrank base 上, baseline 自己已经有 active 的 lowrank code-map 路径
- 因此现在读 style-injection close result 时, 关键不再是“某层是不是绝对为 0”, 而是“variant 是否仍与 baseline 完全相同”

同一天又补了一处更隐蔽的 probe fidelity 问题:

- `code_only_no_reference` anatomy 分支之前没有复现 runtime 里的
  `model._compute_style_code(...)` +
  `model._structured_style_from_sidecar(...)`
- 所以它会把 repaired lowrank 的 code-only anatomy 读得偏大, 同时漏掉一部分 structured style-map 参与
- 修完后见:
  `docs/experiments/2026-06-18-current-state-conditioning-probe/README.md`
- 当前随机初始化 lowrank base 上:
  - `conditioning_code_forward_delta = 0.0022138`
  - `anatomy_code_only_delta = 0.0022138`
  - `code_only_no_reference.style_map_a_vs_b_mean_abs = 0.0025965`

这意味着:

1. repaired lowrank 的 matched-target code-only path 不是死的
2. 但它比旧 anatomy 读法暗示的要弱得多
3. “旧 OT 家族 close result” 继续更像 `train_eval_contract_gap`, 而不是“模型根本没改到”

也就是说:

1. `zero-init` style-injection close result 仍然不能当负证据
2. `live-init` style-injection 若与 baseline 仍完全重合, 才更接近实现无效
3. 若只出现 `1e-4 ~ 1e-3` 级增量, 那是弱杠杆, 不是 no-op

当前已有校准样本:

- `mixed + live_init` 在 plain path 上大约是 `7e-3`, 属于 **moderate_runtime_lever**
- `spatial_carrier_gate + live_init` 在 plain path 上只有 `6e-5`, 更接近 **micro_runtime_lever**

所以 “style injection 不行” 这种话, 现在至少要拆成:

1. exact no-op control
2. micro / weak lever
3. moderate lever 但指标仍然 close

所以现在要更谨慎地区分两种 close result:

1. 杠杆真的很弱
2. 新风格分支还没醒

至少在随机初始化阶段, 这两者不能混读。

---

## 从相关工作能学到什么

### 学到的1: 风格表征必须生动

**StyleShot 的核心洞察**: 风格表征的质量决定了风格迁移的上限。CLIP encoder 不够好 → 需要专门的 Style-Aware Encoder。

**我们的问题**: 5 个 style_id embedding → 风格表征极度贫乏。**这是最根本的瓶颈, 比 TopoGate blend 更深层。**

**可做的**: 从 OT 匹配后的 `matched_target` 中提取风格特征 → 注入 tokenizer。这样即使没有参考图, tokenizer 也能看到"具体的风格实例"。

### 学到的2: 结构约束必须有度

**SCSA 的启示**: 硬约束 (G1/G2 $-\infty$ mask) 和软约束 (blending) 应该在**不同层用不同强度**。所有层都用最强约束 = 过度压制。

**我们的问题**: blend=1.0 在所有层 → 全局过度约束。

**可做的**: 多尺度 blend — 粗尺度 (8×8) blend=1.0 (保大局), 细尺度 (64×64) blend=0.2 (放笔触)。

### 学到的3: 无参考图设定下, 数据量和分布是关键

**StyleShot 的 StyleGallery 数据集**: 风格均衡、多样。LAION 只有 7.7% stilized → 训练效果差。

**我们的问题**: 5 类, Impressionism 13030 vs Minimalism 1307 (10:1 不平衡), 且没有文本描述、没有风格标注。

**可做的**: 扩大风格类别, 平衡采样, 或使用 WikiArt 更多类 (如 27 类)。

---

## 突破路径优先级

1. **降 blend**: 0.4/0.5/0.6 sweep — 当前最快的验证
2. **多尺度 blend**: 不同层不同强度
3. **matched_target style encoding**: 让 tokenizer 从实际风格图像中学 (中等代码量)
4. **数据扩充**: 更多风格类别, 均衡分布 (需要数据处理)
