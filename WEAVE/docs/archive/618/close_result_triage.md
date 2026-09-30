# 618 近似结果排查图

> 更新时间: 2026-06-18
> 目的: 当不同实验组的结果非常接近时, 先判断这是实现/证据问题, 还是已经可以当成理论负证据。

这份文档把当前 phase-618 的 probe、family matrix、以及 validity auditor 结论收成一个可执行的决策图。

核心问题不是:

> "这些点靠得很近, 所以是不是全都没用?"

而是:

> "这些点靠得很近, 到底是因为根本没改到我们实际评估的模型路径, 还是因为确实改到了, 但杠杆本身就弱?"

---

## 1. 先看哪一类近似

统一先跑:

```bash
py -3.12 tools/audit_phase618_run_validity.py --config <base_config> --variant-spec <variant_spec> --variant-name <name>
```

或者对本地已有 run:

```bash
py -3.12 tools/audit_phase618_run_validity.py --run-dir <run_dir>
```

然后按 `artifact_status + effect_contract` 解释:

| 组合 | 含义 | 该不该把“结果接近”当负证据 |
| --- | --- | --- |
| `confounded` | 多个因果改动混在一起 | 不该 |
| `stale` | 证据早于修复/探针合同 | 不该 |
| `suspect` | 配置和要验证的机制矛盾 | 不该 |
| `valid + training_real_eval_inert` | 训练图变了, plain no-reference eval 图没变 | 不该 |
| `valid + training_only_by_design` | 本来就只打算改训练, init eval 不该动 | 不该 |
| `valid + runtime_and_training_real` | 训练和运行图都真变了 | 可以 |

---

## 1.1 如果怀疑“训练后把响应学歪了”, 不要只看 random-init

当 random-init probe 明明显示某条杠杆是 live 的, 但训练后的结果仍然和别组非常接近时,
现在有一个更直接的复核工具:

```bash
py -3.12 tools/probe_checkpoint_style_response.py \
  --config <config.json> \
  --checkpoint <epoch_xxxx.pt> \
  --output-dir <audit_dir>
```

当前标准样本见:

- `docs/experiments/2026-06-19-checkpoint-style-response-audit/remote_h1_epoch18/README.md`

这一步回答的不是 “init 时有没有 live lever”, 而是:

> 训练后 checkpoint 到底把 style 响应压没了、放大了, 还是把它改道到了另一个位置?

现在重点看三类读法:

1. `trained_style_suppression`
   - 说明训练把原本 live 的 style / topology 杠杆压扁了
2. `trained_style_amplification`
   - 说明训练把该分支越学越强
3. `matched_target_suppressed_styleid_amplified_body_dead`
   - 这是目前最关键的新模式
   - 含义不是 “模型没变”
   - 而是:
     - matched-target / topology 路径被压扁
     - plain no-reference 的 `style_id -> decoder` 路径被放大
     - 但 `h_body` 仍然完全不动

第三种情况尤其重要, 因为它会让 close result 看起来像 “组间没变模型”,
但真相更接近:

- 模型确实变了
- 只是把 style 学成了 late decoder actuation
- 没有学成 benchmark 真正需要的 body-level no-reference style carrier

所以这类 close result 应该优先读成:

- train/eval contract 继续错位
- 或者 style actuation 学到了错误的位置

而不是 universal no-op。

---

## 2. 当前七大家族/校准族的最终读法

基于:

- `docs/experiments/2026-06-18-family-validity-matrix/README.md`
- `docs/experiments/2026-06-18-phase618-validity-auditor-smoke/README.md`
- `docs/experiments/2026-06-18-current-stage1-lowrank-recheck/README.md`

### A. `stage1_h0_h6_old_base`

结论:

- **不是实现全死**
- 但 **plain no-reference eval 路径没变**

证据:

- config probe: `plain_eval_change_count = 0`
- training probe: `training_bridge_only_count = 2`, `training_ot_change_count = 4`

读法:

- 这组结果接近, 主要说明旧 base 上存在明显 train/eval contract gap
- 不能拿它证明 “OT 没执行” 或 “blend 把所有东西全锁死”

### B. `stage1_h0_h6_repaired_lowrank`

结论:

- **在 repaired lowrank base 上, 旧 OT 家族仍然是 training-real**
- 但 **pairwise plain eval 依然 inert**

证据:

- `plain_eval_change_count = 0`
- `training_ot_change_count = 4`
- `h5/h6` 的 `ot_topogate_descriptor_blocks = 4`, 说明 multiblock TopoGate 修复已经生效
- current-code recheck 里 `h0/h2/h3/h4/h5/h6` 仍全部满足 `max_vs_base_forward_mean_abs = 0.0`,
  但 training probe 继续把它们分成 `bridge_only_change` 与 `ot_or_target_change`

读法:

- 现在已经不能再把 “style carrier 死了” 当作旧 OT 家族接近的主要解释
- 这是目前最强的证据, 表明旧 OT 假说即使是真的在训练里动了, 对 benchmark 的 plain no-reference path 仍然太弱

### C. `stage3_style_r1_r10_old_base`

结论:

- **整组作废**

证据:

- `artifact_status = confounded`
- old base + lowrank repair + style sweep 被混在一起

读法:

- 这组不能再拿来判断 bold directions
- 任何“r7/r8/r10 好像更强”的结论都要丢掉

### D. `stage3_style_r1_r10_repaired_lowrank`

结论:

- 这才是 **真正可读的 repaired-base style sweep**

当前可读分层:

- `r1-r6`: `plain_eval_change`
- `r7/r8/r10`: `no_effect`
- `r9`: `plain_eval_change`

读法:

- 纯 lowrank code-map repair 变体在 repaired base 上已经塌成 `no_effect`
- 说明它们先前的“收益”主要来自修 carrier, 不是来自理论本身
- 真正还在动 plain eval 的, 只剩 blend 系和 `blend + lowrank` 的混合项

### E. `bold_r11_r16_repaired_lowrank`

结论:

- **runtime-real**
- **training-real**
- **但很弱**

证据:

- 全部 `plain_eval_change`
- training side 分成 `conditioning_or_loss_change` 和 `bridge_only_change`
- `plain_forward_delta` 只有大约 `7e-4 ~ 1e-3`

读法:

- 这里如果结果接近, 就可以比较有把握地读成 “杠杆真实存在, 但力度太小”
- 这是负证据, 不是实现死掉
- 更细一点, 它们现在都落在 family matrix 的 `weak_runtime_lever` bucket

### F. `plain_path_distill_lowrank`

结论:

- **training-real and runtime-inert by design**

证据:

- config probe: 全部 `no_effect`
- training probe: loss / OT / bridge 变化都存在
- `plain_path_distill` 指标始终非零

读法:

- 这里如果结果接近, 暂时不能当负证据
- 因为它本来就不是要在 init 时改 plain eval graph
- 它必须靠完整训练后是否学到蒸馏效果来判

### G. `style_injection_live_init_probe`

结论:

- 这不是新的指标赢家家族, 而是一个 **实现/探针校准族**
- 它专门回答: “style injection close result 到底是 no-op, 还是弱但真实的 runtime lever?”

证据:

- `z1/z3` (`zero_init`) : `config_effect_classification = no_effect`
- `z2/z4` (`live_init`) : `config_effect_classification = plain_eval_change`
- `mixed + live_init` 的 `plain vs_base_forward_mean_abs` 约 `0.006 ~ 0.007`
- `spatial_carrier_gate + live_init` 也是真改, 但 plain path 更弱, 更偏 spatial/matched-target path

读法:

- 以后如果 style-injection 变体结果接近, 先看它是不是还停留在 default zero-init
- 若是 `zero-init + no_effect`, 这不是负证据, 只是 **exact no-op control**
- 若是 `live-init + plain_eval_change`, 才能开始把 close result 读成“杠杆真实存在但偏弱”

---

## 3. 现在到底哪些“接近”已经可以当成理论负证据

可以:

1. repaired lowrank 上的 `bold_r11_r16`
2. repaired lowrank 上的旧 OT 家族, 但要加限定:
   - 它们不是 no-op
   - 它们是 training-real 但 plain-eval inert
   - 所以负证据更准确地说是:
     - “这些 OT / bridge 改动没有把 no-reference benchmark path 变成更强的 style actuator”

不可以:

1. old-base style sweep
2. pre-multiblock 的 h5/h6
3. plain-path distill 在没完整训练前
4. `style_injection_mode` 仍是 default zero-init 的 close result
   - 这类结果先默认当“wake-up-limited control”, 不能直接当理论负证据

补充一个当前远端现实检查:

- 早期局面见 `docs/experiments/2026-06-18-remote-stage-summary-backfill/README.md`
  - 那时远端 `ot_rerun_lowrank_auto` 只完成了 `h0/h1`, 还 **不是** close cluster
- 当前实况见 `docs/experiments/2026-06-18-remote-real-run-audit/README.md`
  - 现在远端 repaired-base `ot_rerun_lowrank_auto` 已形成真正的 `close_cluster`
  - 自动诊断是 `train_eval_contract_gap`
  - 也就是: close 的不是“代码没变模型”, 而是“这族改动对 plain no-reference eval path 仍然 inert / 太弱”

所以现在对远端旧 OT 结果更准确的话已经变成:

- 早期 `h0/h1` 阶段: 还不能说远端已经 close
- 当前 `h0~h4` 阶段: 已经可以说它是一个 **repaired-base close cluster**, 但解释优先级仍是 `train_eval_contract_gap`, 不是 universal implementation no-op

---

## 4. 对 `why_style_weak` 和 `bold_directions` 的修正

### 已经不能再说的话

- “`blend=1.0` 把 style 完全堵死了”
- “结果接近, 所以这些组根本没改模型”
- “r7/r8/r10 强, 说明 lowrank code-map 就是正确方向”

### 现在更准确的话

- `blend=1.0` 更像 **强 content-topology prior**, 不是实现意义上的完全切断
- 旧 OT 家族很多改动 **确实发生在训练图里**, 只是没有把 plain no-reference eval path 拉起来
- repaired-base bold blend/solver 改动 **确实会改 runtime**, 只是幅度太小
- plain-path distill 是目前最干净的 **contract-gap-oriented** 杠杆
- style-injection close result 要先区分:
  - `zero-init exact no-op`
  - `live-init weak runtime lever`

补一个现在统一使用的 bucket 读法:

- `exact_noop`: `plain_forward_delta = 0`
- `micro_runtime_lever`: `0 < delta <= 1e-4`
- `weak_runtime_lever`: `1e-4 < delta <= 2e-3`
- `moderate_runtime_lever`: `2e-3 < delta <= 2e-2`

---

## 5. 现在最该做什么

优先级:

1. **继续完整训练 `plain_path_distill_lowrank`**
   - 这是当前最干净、最贴近真实问题的杠杆
2. **如果要把 style injection 当 no-reference 救火方向, 必须默认开 `style_injection_live_init=true`**
   - 不然很容易把 exact no-op control 误读成理论失败
3. **把旧 OT 家族只当作“对 plain eval path 不够强”的负证据**
   - 不要再继续围绕它做大扫参
4. **把 blend/solver bold sweep 只当作弱 runtime lever 的证据**
   - 不要再期待它们单独救 style
5. **真正值得投入代码量的, 是新范式**
   - stronger no-reference style actuation
   - matched-target instance style -> plain eval path distillation
   - feature-level transfer / reference-conditioned path / discriminator-style loss

一句话总结:

> 当前最需要避免的错误, 已经不是“把 no-op 当成有效实验”, 而是“把已经被审计证明确实很弱的杠杆, 继续误当成还没实现对”。 
