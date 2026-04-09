# 配置命名与搜索空间备忘

## 1. 为什么要记配置命名

当前 `Cycle-NCE/src` 下已经有大量 `config_*.json`，它们本身就是一份“后期研究问题清单”。

## 2. 明显可见的命名簇

### 2.1 baseline / holy grail / sweetspot

代表：

- `config_01_baseline.json`
- `config_00_holy_grail.json`
- `config_09_real_holy_grail.json`
- `config_14_sweetspot_texture.json`
- `config_15_sweetspot_color.json`
- `config_16_sweetspot_balanced.json`

含义：

- 这些配置大概率对应主线候选解，不是单一 ablation。

### 2.2 macro / micro / patch

代表：

- `config_01_macro_base.json`
- `config_02_macro_idt_10.json`
- `config_06_patch_micro_only.json`
- `config_07_patch_macro_only.json`
- `config_10_patch_micro_only.json`
- `config_12_patch_bipolar_unleashed.json`

含义：

- patch 尺度与 macro/micro 频段分工是后期明确的搜索轴。

### 2.3 gate / attn / no_pos / high_temp

代表：

- `config_04_attn_gate_fixed.json`
- `config_05_attn_gate_learned.json`
- `config_07_attn_direct_qk.json`
- `config_08_attn_raw_v.json`
- `config_09_attn_no_smooth.json`
- `config_10_attn_fully_unleashed.json`
- `config_02_no_pos_emb.json`
- `config_09_no_pos_high_temp.json`

含义：

- 这是 Era D attention 系列搜索空间的直接证据。

### 2.4 color / swd / repulse / idt

代表：

- `config_04_color_down_20.json`
- `config_05_color_down_05.json`
- `config_06_swd_up_80.json`
- `config_07_repulse_off.json`
- `config_09_repulse_margin_2.json`
- `config_01_idt_08.json`
- `config_03_idt_16.json`

含义：

- 损失权重与 margin 级别的系统搜索已经配置化。

## 3. 这说明什么

从配置命名就能看出，后期研究不再只是“试一个新模块”，而是把搜索空间拆成了几条明确轴：

1. 风格频段：macro / micro / patch
2. 注意力形态：gate / qk / v / no_pos / high_temp
3. 损失权重：color / swd / idt / repulse
4. 最终候选：holy grail / sweetspot / balanced

## 4. 用途

后续如果继续补文档，可以直接按这些簇追加：

- “attention 配置族”
- “patch / macro / micro 配置族”
- “sweetspot 配置族”

