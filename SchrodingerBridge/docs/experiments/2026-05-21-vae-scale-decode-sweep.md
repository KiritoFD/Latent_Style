# 2026-05-21 VAE Decode Scale Sweep Protocol

目的：把 VAE scaling factor 当成纯推理/解码超参验证，检查它是否能改善 `t00/t01` 的 750 full-eval。

## 关键经验

这件事不能通过“生成图片到磁盘，然后 `--reuse_generated` 重新读 JPEG”来做。

我们实际踩到的现象：

- 原始 `t00` full-eval: `CLIP-style = 0.72591`
- 直接用 `run_evaluation.py --checkpoint ... --force_regen` 重跑: `CLIP-style = 0.72579`
- 错误的 save/reload sweep: 默认 scale 下只有约 `0.705`

结论：

> VAE scale sweep 必须走原始 full-eval 的 in-memory tensor 路径。JPEG 保存/重读会显著改变 CLIP/LPIPS 结果，不能用于判断 decode scale。

## 已做修正

`decode_latent` 增加了可选 decode scale：

```python
decode_latent(vae, latent, device=device, scaling_factor=args.vae_decode_scale)
```

`run_evaluation.py` 增加两个参数：

```text
--vae_decode_scale 0.22319
--seed 42
```

其中：

- `--vae_decode_scale` 只影响 decode 时的除法 scale。
- encode 仍使用 VAE config scale。
- model latent scale 不变。
- `--seed` 用于固定 VAE latent sampling 和生成路径。

## 正确扫描方式

必须直接跑 checkpoint，不要 `--reuse_generated`：

```powershell
python src\utils\run_evaluation.py `
  --checkpoint exp\diffeomorphic_tangent_sweep\t00_ws0p03_g6_nl0\epoch_0008.pt `
  --output exp\vae_scale_decode_sweep\t00_s0p22319 `
  --force_regen `
  --seed 42 `
  --vae_decode_scale 0.22319 `
  --batch_size 16 `
  --eval_lpips_chunk_size 2
```

对 `t00/t01` 都要跑同一组候选值：

```text
0.18215  SD VAE default
0.20500  midpoint probe
0.22319  full train-set empirical calibration
0.23218  overfit50 empirical calibration
```

## Sanity Gate

在扫 scale 前，默认 `0.18215` 必须复现旧结果：

| checkpoint | expected CLIP-style |
|---|---:|
| `t00_ws0p03_g6_nl0` | about 0.7259 |
| `t01_ws0p03_g6_nl0p05` | about 0.7264 |

如果默认 scale 跑不到这个量级，说明扫描链路仍然不可信。

## Invalidated Results

之前 `exp/vae_scale_decode_sweep/t00_t01_scales_750` 这组结果无效，原因是它走了 JPEG save/reload 评估路径。不要用其中的 `0.22319` / `0.23218` 排名做结论。
