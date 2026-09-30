# 03 消融、稳健性与机制实验

全部在 D5-512、750 请求、8 步 Euler、AdaIN β=2.0 下评测，除非另注。
`RB/` = `SchrodingerBridge/experiments/rebuttal_20260716/`，`RX/` = `SchrodingerBridge/rebuttal_exps/`。

## 1. 组件消融（论文 Table II）

见 [02](02_results_of_record.md#消融表table-ii)。重训的三个变体（D3 λ_LL=1.0、D4 direct endpoint、D5 learned HH head）均为 seed 42、15 epoch、关闭早停但保留探针；
配置：`RX/configs/rebuttal_D3_wll_1p0.json`、`rebuttal_D4_direct_target.json`、`rebuttal_D5_hh_head.json`。
HH head 的 seed-7 配置存在（`rebuttal_D5_hh_head_seed7.json`），结果未入库。

## 2. 早停规则 vs oracle（supplement Table S3）

规则：每 epoch 后在固定 4 个 latent（t=0.5）上测共享 trunk 的 LL/HF 梯度比 ρ_e；首次满足 gate 增长且 ρ_e/ρ_{e-1} ≤ 0.65 时停止。
旧的"绝对值穿越 1"规则对尺度敏感（seed 7、probe batch 8 在 15 epoch 内不触发），已废弃。

| run | 训练 epoch | e_internal | e_oracle | DINO-S(oracle) | regret | 来源 |
|---|---:|---:|---:|---:|---:|---|
| seed 42 | 4 | 4 | 4 | 0.4917 | 0 | `RB/expA_seed42/` |
| seed 7 | 15 | 4（相对规则） | 4 | 0.4910 | 0 | `RB/expA_seed7/`（其 json 中 e_internal=null 为旧绝对规则）、`WEAVE/docs/reproduction/internal_dynamics_robustness.csv` |
| seed 123 | 3 | 3 | 3 | 0.4862 | 0 | `RB/expA_seed123/` |
| λ_LL=1.0 | 15 | 5 | 3 | 0.4910 | 6.8e-5 | `RB/expA_D3_seed42/` |
| direct endpoint | 15 | 5 | 13 | 0.4894 | 0.0039 | `RB/expA_D4_seed42/` |
| learned HH head | 15 | 4 | 4 | 0.4930 | 0 | `RB/expA_D5_seed42/` |

probe batch 2 / 8（相对规则）在 seed 42 上都选 epoch 4（ρ_step 0.2299 / 0.5329）。

## 3. 敏感性扫描（supplement Table S4）

来源：`RX/docs/stability_experiments_summary.md`（原始 `exp/rebuttal/*.json` 在远程，未入库）。

- **AdaIN scale**（同 checkpoint，只改推理，完全匹配）：1.0/1.25/1.5/2.0 → DINO-S 0.4831/0.4838/0.4844/0.4920。>2.0 未在当前 checkpoint 上测；base 模型 2.5 时内容崩溃（DINO-C 0.2586，`WEAVE/docs/reproduction/endpoint_adain_axis.csv`）。
- **λ_LL**（重训，固定 4–5 epoch 预算）：0.1–0.5 共 8 点 DINO-S 0.4847–0.4857；DINO-C 单调降、CLIP-S 单调升。0.0 与 2.0 两点来自旧 `WEAVE/exp/ablation_v2`（旧评测脚本），**不能**并入同一条曲线。
- **α**（LL blend，重训）：0.1/0.2/0.4/0.5 → DINO-S 0.4859/0.4864/0.4845/0.4843；DINO-C 与 LPIPS 同向下降（两个内容代理分歧）。
- 0.3/0.3 参照点来自内部早停主 run，不是两条扫描的匹配成员 → 不宣称 0.3 唯一最优。

## 4. 参考池稳定性

`RB/results/b1_reference_pool_corrected.json`：每目标风格从 30 张参考中无放回抽 m 张，所有 150 个请求共用，重复 1000 次。
WEAVE−IDT 的 board 均值 margin 100% 为正；逐请求正比例 68.9%（m=8）/ 69.2%（m=16）。
旧 `RX` Exp2（bootstrap，DINO-S 0.458、"IDT floor 0.8326"）实为参考池敏感性且 IDT 定义错误，**作废**。

## 5. ArtFID 审计

`RX/experiments/rebuttal_20260716/expC_canonical_artfid/canonical_artfid.json`（同一 750 源 manifest）：
IDT 216.5（FID 215.5）< WEAVE 295.3（FID 230.5，LPIPS 0.283）≈ SaMam 297.3 < Seedream 311.0（720 请求）< TGT 545.7±56.1。
Z-STAR、StyleAligned 输出来自另一份源列表（15/750 匹配），**排除**。结论：ArtFID 奖励 IDT，不能单独作为风格排名。
注意 ArtFID 里 WEAVE 的 LPIPS 0.283 与主表 0.2595 不同（不同 LPIPS 计算设置），不要混用。

## 6. 结构保持（MiDaS 深度 + Canny 边缘）

`RX/experiments/rebuttal_20260716/task3_topological/task3_summary.json`（脚本 `RX/scripts/task3_topological.py`）：
depth MSE WEAVE 0.0321 vs SaMam 0.0406；edge IoU WEAVE 0.140 vs SaMam 0.159；StyleAligned 仅 15 对可匹配，仅供参考。

## 7. HF 路由探针（supplement Table S7）

从 base checkpoint `brk_a_ll03_10ep` 微调、AdaIN 1.5；诊断用，不进主表。
来源：`WEAVE/docs/model_probe/target_hf_delta_eval_summary.json`、`WEAVE/docs/713/EXPERIMENT_SUMMARY_FOR_METHOD_AND_NEXT_PLAN.md`、`HF_ARCHITECTURE_PROBE_2026-07-13.md`。
结论：原始空间 target-HF 泄漏目标布局（DINO-C 0.404）；pooled subband codes 是可用路线，推理时置零则风格与内容同时下降（因果有效）；
标量放大、方向辅助损失、时间窗、低秩基、去掉 style memory 都不如简单 subband residual。

## 8. 其它

| 实验 | 结果 | 来源 | 状态 |
|---|---|---|---|
| SD1.5 img2img + Haar 对齐（LH/HL/HH 0.8） | 600 对 DINO-S 0.3974→0.4080，CLIP-S 0.7221→0.7366，p<1e-15；750 对 0.4525→0.4613；4.9 ms/张 | 仅论文 | paper-only |
| Other5 零样本 | 见 02 | 仅论文 | paper-only |
| VAE × Haar 层数泛化（Task1+） | "DINO-S" 0.912/0.898/0.814/0.596 | 仅 commit f25b280c 说明 | **不可引用**：脚本 `task1_generalization.py` 的 "DINO-S" 实为与**源图**的余弦（= DINO-C），且只有 5 对 |
| 小波基 / 层数（Exp4） | Haar l1 CLIP .7261/LPIPS .3288；l2 .7301/.3402；db2 l1 .7258/.3288；db2 l2 .7298/.3398 | `RX/docs/stability_experiments_summary.md`（复用 2026-07-01 Phase 4D/4E） | 旧架构旧协议，仅作历史诊断 |
| 旧 ablation_v2（21 个，base 架构） | 例：wo_endpoint_adain LPIPS 0.339 | `WEAVE/exp/ablation_v2/_results.json` | 旧评测，与 Table II 不可比 |
| 1-epoch capacity/loss/attention/stylefilm/endpoint/gate-init | clip_style≈0.702，差异 <0.002 | `WEAVE/results/*.csv` | 旧协议 smoke，废弃 |
| 推理加速（ONNX/TRT/compile） | base 模型，RTX 4070 Laptop，1 步 bf16 3.74 ms/张 | `WEAVE/experiments/inference_speed/` | 支线，非论文 |
| λ_kinetic × terminal_swd 网格、步数扫描、efficiency | SB 时代 review 补充实验 | `lambda_grid/`、`step_count_sweep/`、`efficiency/`、`review_additional_experiments_aggregates/` | 历史（SchrodingerBridge 方法） |
