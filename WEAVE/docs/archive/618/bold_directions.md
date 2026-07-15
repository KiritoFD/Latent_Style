# 大胆的改进方向 — 不调参, 改范式

> 当前死局: blend=1.0 锁死 attention, tokenizer 5 向量查表, 所有人有参考图我们没有.
> 以下方向不追求"fine-tune 参数", 而是改变模型的底层设计逻辑.
> 审计注记(2026-06-18): 先不要把 `blend=1.0` 理解为"style 完全进不来".
> 现实现更接近"content topology 主导 routing, 但 style values 仍在走 attention value path".
> 此外, phase618 auto launcher 曾有 base family 被静默改回 `legacy_factorized` 的问题; 在读旧结果前先看:
> - `docs/experiments/2026-06-18-styleid-eval-probe/README.md`
> - `docs/experiments/2026-06-18-phase616-auto-family-mutation-audit/README.md`
> - `docs/experiments/2026-06-18-lowrank-code-map-order-audit/README.md`
> - `docs/experiments/2026-06-18-topogate-multiblock-audit/README.md`
> - `docs/experiments/2026-06-18-current-stage1-lowrank-recheck/README.md`
>
> 审计注记(2026-06-19):
> `tools/probe_checkpoint_style_response.py` 在拉回的
> `h1_linear_fm epoch_0018.pt` 上给出的总读法是
> `matched_target_suppressed_styleid_amplified_body_dead`。
> 这意味着训练后模型不是“没有学 style”, 而是把 style 更多学成了
> decoder-only late actuation, 同时 body-level no-reference path 仍然死着。
> 因而优先级最高的方向不再是继续加强晚期 decoder 标量,
> 而是:
> - body-level no-reference carrier
> - matched-target -> plain-path distillation
> - feature-level transfer

---

## 方向1: 反转 TopoGate — Style-Locked Attention

**现状**: TopoGate 锁 content self-attention → style 进不来.
**反转**: 锁 style self-attention, 把 content 作为调制信号.

> 审计注记(2026-06-18):
> 当前实现里 `semantic_self_topology_blend` 本质上是在
> `style-routing logits` 和 `content-topology logits` 之间做 lerp。
> 在 repaired lowrank base 上, `blend=0.0` / `0.3` 确实会改 plain eval graph,
> 但 `plain vs_base_forward_mean_abs` 只在约 `7e-4 ~ 1e-3` 量级,
> 对 `max_body_pair_delta` 的提升也很小。
> 按 family matrix 的 bucket, 这已经是 **weak_runtime_lever**,
> 不是 no-op, 但也远谈不上 paradigm shift。
> 见 `docs/experiments/2026-06-18-bold-eval-graph-preflight/README.md`。

$$A_{\text{final}} = 0.3 \times A_{\text{self-content}} + 0.7 \times A_{\text{style}}$$

先用 style features 生成一个"风格画布", 内容只作为微调信号.
Content 从"不可破坏的铁律"降级为"软约束".

**预期**: style 直接突破 0.72, LPIPS 可能升到 0.40-0.50. 需要配合更强的 content correction (PC solver).

**代码量**: 改一个参数 + 扫描.

---

## 方向2: 抛弃 Tokenizer — 直接匹配传输

**现状**: tokenizer 输出 spatial_map → UNet modulation. Tokenizer 是瓶颈.
**替代**: 不用 tokenizer, 直接用 OT-matched 目标图做 **多尺度 latent 特征匹配**.

训练时:
1. OT 匹配找到 matched_target $z_t$
2. 同时从 content $z_0$ 和 matched $z_t$ 提取 UNet encoder 多尺度特征
3. Content 的低频特征 + target 的高频特征 → 混合后解码
4. Loss = 混合特征的 SWD vs target 分布

**不需要 tokenizer, 不需要 style_id, 不需要 spatial_map**. 风格信息直接从 matched target 的特征中"借"过来.

**预期**: style 0.72+, LPIPS 取决于频率分离的质量.

**代码量**: 中等 — 需要改 model forward 的特征混合逻辑.

---

## 方向3: 对抗风格鉴别器 — Fool-the-Discriminator

**现状**: Terminal SWD 是弱风格信号. SWD 只能匹配分布矩, 对高频纹理不敏感.
**替代**: 加一个轻量风格鉴别器 $D_s$, 判别输出是否属于目标风格.

$$\mathcal{L}_{\text{style}} = -\mathbb{E}_{z \sim p_{\text{generated}}}[\log D_s(z, \text{style\_id})]$$

$D_s$ 是一个浅层 CNN (比 UNet 小 10 倍), 只判断"这张 latent 是否像目标风格".

**这提供了比 SWD 强得多的风格梯度**. 鉴别器可以捕捉 SWD 漏掉的高频笔触特征.

**预期**: style 突破 0.73+. 训练不稳定风险 → 加 gradient penalty.

**代码量**: 低 — 加一个鉴别器类 + 对抗 loss.

---

## 方向4: 内容约束后置 — Content as Corrector, Not Anchor

**现状**: 整个架构围绕内容保持设计 — residual + skip + TopoGate + velocity.
**反转**: 训练时放开风格 (降低 blend 到 0, 自由风格化), 推理时用 PC solver 修正.

**训练**: blend=0 (完全自由风格化), 不加 kinetic, 不加任何结构约束
**推理**: solver_pc + latent_lowpass corrector 把宏观结构拉回 content

> 审计注记(2026-06-18):
> `solver_pc` 不是 no-op, 但它主要改的是 `integrate()` 分支而不是 `forward()`
> / `predict_transport_base()`。在当前 repaired lowrank base 上,
> `plain vs_base_integrate_mean_abs` 可以到约 `0.013`, 但 style-id 分离并没有增强。
> 所以 "blend=0 + PC solver" 更像结构修正杠杆, 不是 style actuation 主杠杆。

**这将大幅释放 style 能力** — 训练时模型可以自由地"画"任何风格, 不受 content 约束.
推理时 PC solver 作为事后校正.

**预期**: style 0.73+, LPIPS 取决于 PC solver 参数.

**代码量**: 极低 — 改 config 即可验证.

---

## 方向5: 从 matched_target 学风格 — Instance-Level Style Encoding

**现状**: tokenizer 的 style values 是 Embedding(5, D) — 5 个固定向量.
**替代**: 每次训练迭代, OT 匹配后, 用一个 StyleBankEncoder 从 matched_target 中编码风格特征:

```python
# 训练循环中:
matched_target = ot_match(content, target_style_images)
style_code = StyleBankEncoder(matched_target)  # 从实际风格图中编码
# style_code 替代 style_id lookup
```

StyleBankEncoder 是一个轻量网络 (几层 Conv), 从风格实例中提取纹理/笔触/色彩信息.

**这个改变让 tokenizer 从"查表"升级为"编码"** — 风格表征不再是 5 个固定向量, 而是从实际风格图中动态提取.

> 审计注记(2026-06-18):
> 代码里已经存在训练时的 matched-target 编码路径:
> `src/lancet_runtime.py::encode_target_style_latent(...)`,
> 并且 repaired lowrank base 已启用
> `matched_target_conditioning_mode=both` +
> `matched_target_style_encoder_mode=residual`。
> 所以方向5真正缺的不是"再加一个 matched_target encoder",
> 而是如何把这条 instance-level 风格信息蒸馏回 plain no-reference eval path。

> 另外, 如果下一轮想把 `style_injection_mode` / `style_injection_form`
> 当作新的 no-reference style actuation 机制, 先看:
> - `docs/experiments/2026-06-18-style-injection-live-init-probe/README.md`
>
> 当前实现里这类分支默认可能是 exact zero-init path.
> 不开 `style_injection_live_init=true`, close result 不能直接当作负证据。
>
> 另外, anatomy probe 本身已经修过一次:
> `tools/probe_conditioning_sensitivity.py` 之前漏掉了 runtime 的
> `_apply_style_feature_injection(..., site="body"/"decoder")`.
> 修完之后的证据更明确:
> - `mixed + live_init` 会在 repaired lowrank base 上给 plain no-reference path 带来可测增量
> - `spatial_carrier_gate + live_init` 更偏向改 spatial/matched-target path, plain path 杠杆更弱
> - 如果按 runtime bucket 来读: `mixed + live_init` 更接近 **moderate_runtime_lever**,
>   `spatial_carrier_gate + live_init` 在 plain path 上更接近 **micro_runtime_lever**
> 所以下一轮如果要押注“新 no-reference actuation”, 优先级应继续高于 spatial_carrier_gate。
>
> 同一天又补了第二处 anatomy fidelity 问题:
> `code_only_no_reference` 旧 trace 没有经过
> `model._compute_style_code(...)` + `model._structured_style_from_sidecar(...)`,
> 所以把 repaired lowrank 的 code-only anatomy 读大了。
> 修正后见:
> - `docs/experiments/2026-06-18-current-state-conditioning-probe/README.md`
>
> 当前更稳的读法是:
> - code-only path 是 live 的, 但弱
> - spatial matched-target path 仍显著更强
> - 所以如果下一轮要押注“把 instance-level style 蒸馏回 plain path”, 不能再拿旧 anatomy 里的 `~1e-2` code-only 数值当乐观前提

**预期**: style 0.71-0.73, 配合降低 blend 效果更佳.

**代码量**: 中等 — 新增 StyleBankEncoder 类, 修改 losses.py 的风格注入逻辑.

---

## 方向6: 多模态风格表征 — 参考图+ID 混合

**现状**: 只有 style_id, 没有参考图 → 风格表征极弱.
**方案**: 在**推理时**允许可选的参考图输入. 训练时仍然用 style_id, 但设计一个 reference encoder 分支.

```
训练: style = Embedding(style_id)  # 现有
推理: style = ReferenceEncoder(ref_image)  # 新, 可选
```

ReferenceEncoder 只在推理时激活, 编码参考图的风格 → 注入 tokenizer.
训练时仍用 style_id — 保持无配对训练的优势.

**这是最小代价获得最大收益的方向**. 代码量小 (加一个可选 encoder), 但直接把"无参考图"的劣势变成了"可选参考图"的优势.

**预期**: 有参考图时 style 0.74+, 无参考图时保持现有 0.67.

**代码量**: 低 — 加一个 encoder + 推理时的分支选择.

---

## 方向优先级

| 优先级 | 方向 | 现在的原因 |
|:---:|------|------|
| 1 | 方向5: matched_target编码 | 直接对准最新证据里的核心错位: 训练看的是 instance-level spatial style, eval 用的是 body-dead no-reference path |
| 2 | 方向2: 抛弃tokenizer | 直接绕开 5 向量查表 + body-dead spatial carrier, 把风格搬到 feature/body 路径里 |
| 3 | 方向6: 多模态 | 最小代价补强风格表征, 也能作为“是不是表征太贫”的快速现实校验 |
| 4 | 方向3: 对抗鉴别器 | 可能强化 style, 但如果只压在 decoder 末端, 容易重复“style_id 放大但 body 仍死”的旧问题 |
| 5 | 方向1: 反转TopoGate | 现在更像控制实验; 最新 checkpoint 审计已经说明 topology 路径本身会被训练压扁 |
| 6 | 方向4: blend=0 + PC solver | 对结构修正仍有价值, 但已知不是主要的 style actuation 主杠杆 |

---

## 读近似结果时的筛选规则

在继续推进这些 bold directions 之前, 先用
`docs/618/close_result_triage.md` 的决策图过滤实验家族。

当前已经明确:

1. `stage3_style_r1_r10_old_base`
   - **整组作废**
   - 原因不是理论失败, 而是 base repair 和 bold direction 混在一起
2. `stage3_style_r1_r10_repaired_lowrank`
   - 只有真正还会改 plain eval graph 的 repaired-base levers 才值得看
   - `r7/r8/r10` 这类纯 carrier-repair 方向在 repaired base 上已经塌成 `no_effect`
3. `bold_r11_r16_repaired_lowrank`
   - runtime 和 training 都是真改
   - 但已经被证实是 **弱 runtime lever**
   - 更具体地说, 当前都落在 `weak_runtime_lever` bucket (`plain_forward_delta` 约 `7e-4 ~ 1e-3`)
   - 所以它们更适合作为“config-only rescue 不够”的负证据, 不适合作为主线救火方案
4. `style_injection_*_zero_init`
   - 不能再拿 close result 当负证据
   - 因为这类分支在默认 init 下可以与 baseline **逐点重合**
5. `style_injection_*_live_init`
   - 现在可以作为公平的实现检查样本
   - 但还要看 bucket: `micro`/`weak` 是小杠杆, `moderate` 才更值得进入完整训练
   - 但要区分 `mixed` 与 `spatial_carrier_gate` 的杠杆位置: 前者更像 plain-path actuation, 后者更像 spatial-path actuation
6. `plain_path_distill_lowrank`
   - 不能因为 init plain eval 不动就判死
   - 它本来就是 training-only-by-design, 必须看完整训练后的学到的结果

补一条现在已经落到真实远端 run 的约束:

- `docs/experiments/2026-06-18-remote-real-run-audit/README.md`
- `docs/experiments/2026-06-18-current-stage1-lowrank-recheck/README.md`
- 历史 repaired-base `ot_rerun_lowrank_auto` 整组结果已经出现过 `close_cluster`
- 清理 family / multiblock / launcher 之后重新从头拉起的 corrected `h0_vertical_fm`
  目前也已经给出:
  - `epoch_0001 -> 0.6709 / 0.3777`
  - `epoch_0002 -> 0.6620 / 0.3341`
  - `epoch_0004 -> 0.6606 / 0.3503`
  - `epoch_0006 -> 0.6611 / 0.3614`
- 同一 corrected rerun 根上的 `h1_linear_fm` 目前是:
  - `epoch_0001 -> 0.6526 / 0.3370`
  - `epoch_0002 -> 0.6493 / 0.4055`
  - `epoch_0003 -> 0.6471 / 0.4265`
  - audit: `artifact_status=valid`, `effect_contract=training_real_eval_inert`
- current-code recheck 还进一步钉死了:
  - `h0/h2/h3/h4/h5/h6` 对 shared-weight plain eval graph 仍是 `0.0` 级别同构
  - 但 training probe 依然显示 `bridge_only_change` 或 `ot_or_target_change`
- 所以自动读法仍然更接近 `train_eval_contract_gap`

更关键的是: corrected trajectory 不是逐点重合, 而是“先抬 style, 再回落 toward close band”。
这类形状更像旧 OT / bridge 假说没有把 plain no-reference eval path 维持住,
而不是实现上完全没改模型。

这意味着继续把 GPU 花在“旧 OT 家族是不是其实根本没改到模型”上,
证据价值已经越来越低。现在更像是在重复确认同一个 contract-gap 现象,
而不是在逼近新的 style actuation 机制。

这意味着:

- 继续花大量 GPU 在 blend/solver 小扫参上, 性价比已经很低
- 真正值得代码量的是:
  - 把 matched-target 的 instance-level style 信息蒸馏回 plain no-reference eval path
  - 建立更强的 no-reference style actuation 机制
  - 或者直接引入 reference-conditioned / feature-transfer / discriminator 式新范式
