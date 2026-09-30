# WEAVE 代码精简记录

日期：2026-07-10

## 精简原则

- 保持 T11 默认执行路径不变。
- 不依据 MUSIQ 决定四指标模型结构。
- 删除无人引用的重复实现与只服务废弃实验的运行入口。
- 对仍可能影响 DINO-S/DINO-C 的 HH head、FiLM head、hard-region SWD 暂时保留，但要求通过 `README.md` 中的 matched A/B 决定去留。

## 本次删除

- 删除 `trainer_remote.py`：与 `trainer.py` 近乎完全重复，remote 差异应由启动参数处理。
- 删除 `blocks620_remote.py`：与 `blocks620.py` 仅有极少差异且无活跃引用。
- 删除 patch SWD、soft-mask SWD、Sinkhorn region SWD、hierarchical SWD、adaptive-K SWD、spectral-region SWD、attention-region SWD。
- semantic SWD 只保留 `off` 与 deterministic hard `region` 两种模式。
- 删除对应的废弃 schema 字段。
- 删除 `configs/semantic_swd_musiq/`、`musiq_s1` 至 `musiq_s8` 及旧 MUSIQ task runners。
- 删除指向上述配置的旧 remote ablation runners。

## 暂时保留

- Global SWD：T11 活跃训练目标的一部分。
- Hard-region SWD：等待 DINO-S/DINO-C matched ablation。
- HH velocity head：等待四指标 matched ablation。
- Style-FiLM heads：等待四指标 matched ablation。
- 通用评估后处理接口：为历史复现保留，但 710 主实验必须关闭。

## 后续代码目标

完成 710 消融后，将最终保留路径收敛到：

1. Haar DWT/IDWT；
2. stochastic HF routing；
3. 三个 velocity heads；
4. spectral FM + 单一 SWD 变体；
5. endpoint per-subband WCT；
6. 单一 trainer 与扁平 canonical config。

