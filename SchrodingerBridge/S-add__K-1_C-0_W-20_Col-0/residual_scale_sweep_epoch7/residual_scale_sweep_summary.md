# Residual Scale Sweep Epoch 7

| Run | all style | all content | all LPIPS | transfer style | transfer content | transfer LPIPS | photo->art style | photo->art content | photo->art LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_epoch7` | 0.7161 | 0.8086 | 0.4514 | 0.6911 | 0.8019 | 0.4609 | 0.6716 | 0.7643 | 0.4882 |
| `residual_1p25` | 0.7219 | 0.7635 | 0.5110 | 0.7016 | 0.7561 | 0.5223 | 0.6864 | 0.7247 | 0.5567 |
| `residual_1p5` | 0.7208 | 0.7212 | 0.5645 | 0.7047 | 0.7135 | 0.5767 | 0.6929 | 0.6951 | 0.6154 |
| `residual_2p0` | 0.7069 | 0.6558 | 0.6519 | 0.6957 | 0.6486 | 0.6639 | 0.6876 | 0.6471 | 0.7044 |

## Transfer CLIP-style by target

| Run | photo | Hayao | monet | vangogh | cezanne |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_epoch7` | 0.6855 | 0.6551 | 0.6868 | 0.7302 | 0.6979 |
| `residual_1p25` | 0.7058 | 0.6669 | 0.6961 | 0.7414 | 0.6977 |
| `residual_1p5` | 0.7166 | 0.6632 | 0.7021 | 0.7456 | 0.6957 |
| `residual_2p0` | 0.7140 | 0.6492 | 0.6998 | 0.7321 | 0.6836 |

## Transfer CLIP-style by source

| Run | photo | Hayao | monet | vangogh | cezanne |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base_epoch7` | 0.6716 | 0.5858 | 0.7470 | 0.7260 | 0.7251 |
| `residual_1p25` | 0.6864 | 0.6172 | 0.7461 | 0.7247 | 0.7335 |
| `residual_1p5` | 0.6929 | 0.6417 | 0.7347 | 0.7201 | 0.7340 |
| `residual_2p0` | 0.6876 | 0.6625 | 0.7088 | 0.7022 | 0.7175 |
