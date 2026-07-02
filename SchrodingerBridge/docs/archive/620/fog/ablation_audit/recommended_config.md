# 620 消融审计：Phase 5 推荐配置说明

> 生成时间：2026-06-22  
> 配置文件：`configs/620_spatial_bridge_ablation_recommended.json`  
> 实验目录：`exp/620_spatial_bridge/620_ablation_recommended_smoke/`  
> 验收门：`wfi_score < 0.40`，`clip_style ≥ 0.695`，`content_lpips < 0.36`

---

## 1. 配置来源

本配置以 `exp/620_spatial_bridge/620_film_v5_endpoint_film_hd512_local_smoke/config.json` 为模板，严格按照 `docs/620/fog/ablation_audit/design_decisions.md` 中 Phase 4 的推荐取值生成，用于 Phase 5 的端到端训练与 WFI 验证。若该初始组合未通过验收门，将依据任务要求进行调整（如改回 `style_film_enabled=true` 或调整 `gate_init`），并在本文档中记录所有迭代。

---

## 2. 关键参数与选择依据

| 参数 | 推荐值 | 模板原值 | 选择依据 |
|---|---|---|---|
| `model.base_dim` | 64 | 64 | Phase 3.1 显示 `128×4/128×6` 对 CLIP-S 提升不足 0.001，且 WFI 未改善；64 为最小有效维度，显存与速度成本最低。 |
| `model.num_res_blocks` | 4 | 4 | `64×6` WFI 略优（0.3828 vs 0.3887），但训练时间增加约 14%，收益有限；默认 4 作为最小有效深度。 |
| `model.style_condition_source` | `latent` | `latent` | Phase 3.3 验证 `target_dino_patches` 导致 WFI 飙升至 0.61–0.64，严重白化；`latent` 是唯一通过 WFI 门的条件源。 |
| `model.style_dino_adapter_enabled` | false | false | Adapter 无法修复 DINO patches 的白化（WFI 仍 0.6076），成本收益不成正比。 |
| `model.style_moe_enabled` | false | false | Round 1 与当前消融均显示 MoE 无收益。 |
| `model.endpoint_head_mode` | `endpoint_lowhigh` | `endpoint_lowhigh` | `velocity` 在 smoke 1 epoch 更优，但历史验证较少、多 epoch 稳定性未知；保守保留 `endpoint_lowhigh`。 |
| `model.endpoint_film_enabled` | true | true | 关闭 FiLM 的 `endpoint_lowhigh_nofilm` WFI 接近门限（0.3957）；保留 FiLM 可维持风格表达上限。 |
| `model.endpoint_style_hidden_dim` | 128 | 512 | Phase 2.3 中 hd128 单因子 WFI（0.3801）优于 hd256（0.3990）和 hd512（0.3915），且参数量更小。 |
| `model.style_film_enabled` | false | true | Phase 2.2 显示 block 内 StyleFiLM 开/关差异极小（WFI 差 0.0003）；在已有 endpoint-FiLM 的情况下可关闭以简化模型。 |
| `model.style_attn_mode` | `gated` | `gated` | Phase 2.1 中 `softmax` 单因子 WFI 最低（0.3736），但 `gated` 更贴近历史默认且组合稳定性更好；本配置先以 `gated` 验证。 |
| `model.style_cross_attn_gate_init` | **0.3** | 0.3 | Phase 2.4 单因子显示 0.05 最优；但在 `hd128 + edge=0 + style_film=false` 组合下，0.05 导致 WFI=0.4062 未通过，调整为 **0.3** 后 WFI=0.3757 通过。 |
| `bridge.single_step_swd_weight` | 8.0 | 8.0 | SWD=8 在 WFI 门内取得较好平衡；SWD=16 单独使用会超门（0.4013），需配合 edge=0 才回落。 |
| `bridge.swd_noise_sigma` | 0.02 | 0.02 | 关闭 noise 会使 WFI 从 0.3959 升至 0.4105，是必备的白化抑制项。 |
| `bridge.single_step_edge_weight` | 0.0 | 0.1 | Phase 3.2 中 `edge=0` 是唯一的“三赢”开关：WFI 0.3786、CLIP-S 0.7020、LPIPS 0.3336 均优于 edge=0.1。 |
| `training.num_epochs` | 1 | 1 | 历史 E3 显示 3 epoch 在当前 lr 下会加剧白化；默认 1 epoch，多 epoch 需带 early stopping / 低 lr。 |
| `training.batch_size` | 4 | 4 | 与所有 smoke 实验保持一致。 |
| `training.accumulation_steps` | 16 | 16 | 与所有 smoke 实验保持一致，有效 batch=64。 |
| `model.velocity_hf_residual_enabled` | false | false | 历史 `hf_residual` WFI=0.4746，独立无效。 |

---

## 3. Phase 5 验证计划

1. 创建实验目录 `exp/620_spatial_bridge/620_ablation_recommended_smoke/`。
2. 将 `configs/620_spatial_bridge_ablation_recommended.json` 复制为该目录下的 `config.json`。
3. 运行训练：`python run.py --config exp/620_spatial_bridge/620_ablation_recommended_smoke/config.json`。
4. 训练完成后运行 WFI 评估：
   ```bash
   python tools/run_eval_with_wfi.py \
     --checkpoint exp/620_spatial_bridge/620_ablation_recommended_smoke/epoch_0001.pt \
     --output exp/620_spatial_bridge/620_ablation_recommended_smoke/full_eval_wfi/epoch_0001 \
     --test-dir f:/wikiart_distinct5_samam_512_classview_real/test \
     --cache-dir f:/eval_cache --clip-hf-cache-dir f:/eval_cache/hf \
     --source-dir f:/wikiart_distinct5_samam_512_classview_real/test \
     --batch-size 4 --target-chunk-size 2 --vae-decode-batch-size 4 \
     --eval-lpips-chunk-size 4 --clip-style-idt-baseline 0.639920825263 \
     --num-steps 8 --verbose
   ```
5. 验收标准：WFI < 0.40 且 CLIP-S ≥ 0.695。
6. 若未通过，分析原因并调整（如改回 `style_film_enabled=true` 或调整 `gate_init`），重新训练直到通过。

---

## 4. 验证结果

### 4.1 最终验证结果

| 指标 | 验收门 | 实测值 | 状态 |
|---|---|---:|---|
| WFI ↓ | `< 0.40` | **0.3757** | ✅ 通过 |
| CLIP-S ↑ | `≥ 0.695` | **0.6995** | ✅ 通过 |
| content LPIPS ↓ | `< 0.36` | **0.3422** | ✅ 通过 |
| ΔWFI (gen − source) | — | +0.0540 | 优于 hd512 基线的 +0.0689 |

验证命令：

```bash
python run.py --config exp/620_spatial_bridge/620_ablation_recommended_smoke/config.json
python tools/run_eval_with_wfi.py \
  --checkpoint exp/620_spatial_bridge/620_ablation_recommended_smoke/epoch_0001.pt \
  --output exp/620_spatial_bridge/620_ablation_recommended_smoke/full_eval_wfi/epoch_0001 \
  --test-dir f:/wikiart_distinct5_samam_512_classview_real/test \
  --cache-dir f:/eval_cache --clip-hf-cache-dir f:/eval_cache/hf \
  --source-dir f:/wikiart_distinct5_samam_512_classview_real/test \
  --batch-size 4 --target-chunk-size 2 --vae-decode-batch-size 4 \
  --eval-lpips-chunk-size 4 --clip-style-idt-baseline 0.639920825263 \
  --num-steps 8 --verbose --force-regen
```

### 4.2 调试迭代记录

| 迭代 | 关键参数 | WFI | CLIP-S | LPIPS | 说明 |
|---|---|---:|---:|---:|---|
| 初始 Phase 4 推荐 | hd128, gated, style_film=false, **gate_init=0.05**, edge=0 | 0.4062 | 0.6994 | 0.3186 | ❌ WFI 超门 0.40 |
| 调整后最终 | hd128, gated, style_film=false, **gate_init=0.3**, edge=0 | **0.3757** | 0.6995 | 0.3422 | ✅ 全部通过 |

**调整原因分析**：
- `gate_init=0.05` 在 Phase 2.4 单因子实验中最优，但那是基于 `endpoint_film_hd512 + style_film=true + edge=0.1` 的基线。
- 当组合改为 `hd128 + style_film=false + edge=0` 后，风格/端点信号的动态范围被压缩，需要更强的初始 cross-attention gate（0.3）来补偿，否则生成图出现白化反弹。
- 这一迭代说明 **单因子最优不等于组合最优**，也是 Phase 5 验证的核心价值。

---

## 5. 配置路径

- 配置文件：`configs/620_spatial_bridge_ablation_recommended.json`
- 训练产物：`exp/620_spatial_bridge/620_ablation_recommended_smoke/epoch_0001.pt`
- 评估报告：`exp/620_spatial_bridge/620_ablation_recommended_smoke/full_eval_wfi/epoch_0001/wfi_eval_report.json`
