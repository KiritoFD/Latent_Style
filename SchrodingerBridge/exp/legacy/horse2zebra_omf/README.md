# Horse2Zebra OMF

独立实验目录，和当前艺术风格实验隔离。

目录约定：

- 原始数据与预处理输出：`../datasets/horse2zebra`
- 训练配置：`./config.json`
- checkpoint 与日志：`./artifacts`

使用顺序：

```bat
prepare_data.bat
train.bat
eval_checkpoints.bat
```

数据准备脚本会自动：

- 下载官方 `horse2zebra.zip`
- 解压到 `../datasets/horse2zebra/raw/horse2zebra`
- 编码训练 latent 到 `../datasets/horse2zebra/latents_train/{horse,zebra}`
- 编码测试 latent 到 `../datasets/horse2zebra/latents_test/{horse,zebra}`
- 复制测试图到 `../datasets/horse2zebra/test_images/{horse,zebra}`

当前配置默认：

- `num_styles = 2`
- `style_subdirs = ["horse", "zebra"]`
- `batch_size = 96`
- `num_epochs = 160`
- `save_interval = 20`
