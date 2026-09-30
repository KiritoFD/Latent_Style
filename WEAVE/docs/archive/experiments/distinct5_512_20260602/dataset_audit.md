# Distinct5-512 Dataset Audit

更新时间：2026-06-02

## 核验结论

本地和远程用于新一轮 LANCET / SaMAM / SaMST 对比的是同一套 Distinct5-512 图像数据。正式评估口径固定为 all 5x5 / 750 images。

注意：`2026-06-02-distinct5-512-lancet-representation-speed.md` 早期段落里仍有 `/mnt/f/...` 路径，这是历史记录中的旧映射。远程当前有效数据路径是 `/mnt/i/...`。

## 类别与划分

| style | train | test |
|---|---:|---:|
| Early_Renaissance | 1000 | 30 |
| Impressionism | 1000 | 30 |
| Minimalism | 1000 | 30 |
| Rococo | 1000 | 30 |
| Ukiyo_e | 1000 | 30 |

## 本地路径

| 用途 | 路径 |
|---|---|
| 原始 class images train | `F:\wikiart_distinct5_512_images\train` |
| 原始 class images test | `F:\wikiart_distinct5_512_images\test` |
| SaMAM classview train | `F:\wikiart_distinct5_samam_512_classview\train` |
| SaMAM classview test | `F:\wikiart_distinct5_samam_512_classview\test` |
| EMA latent train | `F:\wikiart_distinct5_samam_512_latents_ema\train` |
| EMA latent test | `F:\wikiart_distinct5_samam_512_latents_ema\test` |

## 远程路径

| 用途 | 路径 |
|---|---|
| 原始 class images train | `/mnt/i/datasets/wikiart_distinct5_512_images/train` |
| 原始 class images test | `/mnt/i/datasets/wikiart_distinct5_512_images/test` |
| SaMAM flat train/test | `/mnt/i/datasets/wikiart_distinct5_samam_512` |
| EMA latent train class dirs | `/mnt/i/wikiart_distinct5_latents_512_ema` |
| EMA latent test class dirs | `/mnt/i/wikiart_distinct5_latents_512_ema_test` |
| LANCET packed latent train root | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train` |
| LANCET packed latent test root | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/test` |
| packed latent cache | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache` |
| prototype pairing cache | `/mnt/i/wikiart_distinct5_samam_512_latents_ema/train/.latent_cache/prototype_pairing_top8.pt` |

## 文件名集合 hash

统一使用 Python lexical sort，对每类文件名用 `\n` 拼接后做 SHA256。下面列的是原始 image class dirs；classview 文件名带 `Style__` 前缀，latent 文件名后缀为 `.pt`，因此 hash 不与原始 jpg 直接相等，但 basename 对齐。

### Train

| style | count | sha256_names |
|---|---:|---|
| Early_Renaissance | 1000 | `2adcce3c029d06fe7ccb2760ed0e6a8725db46babbbf82335f661a8ff086923f` |
| Impressionism | 1000 | `f244107768e9ad98f7d496438b1649b8a9f4477915b059969b0325e1e7981a71` |
| Minimalism | 1000 | `8529d684e516c09035b28f6e688bbb0c6dc88631aee829bdf45df7fa2e881160` |
| Rococo | 1000 | `0cd6a3906ae2b4dd7fbbb7867c983de0fb380d912a460ab22f79166a122834c1` |
| Ukiyo_e | 1000 | `43c2382d0208692d617524eea6dd169e383b68fef331020f9e61853aa3378fc7` |

### Test

| style | count | sha256_names |
|---|---:|---|
| Early_Renaissance | 30 | `8c61d9d9a45fcb4edc393c39b979c90116e7a57e228f13b8063b95a075ca6d17` |
| Impressionism | 30 | `9754cd610a5d90d3d51f15537b21a0fc5aa7aa89e1d934f001a101f752ef3464` |
| Minimalism | 30 | `0fead7e88af5379c0eac0bc378848696ce40fd6dd068a12d9f96784097f18c34` |
| Rococo | 30 | `1607695ca9ccff2adafc86b8023645133ee78b7005994a09eaad9331c312d6a8` |
| Ukiyo_e | 30 | `a18bd4b9dd055e7301f4f4e2c6f0cc0cf50a17f392536cd63ed51b09fcf3f2ae` |

## 操作备注

- Windows Python 直接遍历 `F:\wikiart_distinct5_samam_512_classview` 时可能遇到 reparse/symlink 访问错误；不要用它作为 dataset identity 的唯一核验方式。
- 原始 `F:\wikiart_distinct5_512_images` 和远程 `/mnt/i/datasets/wikiart_distinct5_512_images` 已经足够确认图像集合一致。
- LANCET 使用 EMA latent；SaMAM 使用图片 flat/classview；SaMST 计划使用图片训练。三者的 test split 均应保持同一 150 张。
