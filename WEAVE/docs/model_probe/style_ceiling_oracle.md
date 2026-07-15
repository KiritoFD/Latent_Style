# Style Ceiling Oracle Probe

Config: `configs\exp_brk_a_ll03_10ep.json`
Checkpoint: `exp\dino_s_break\brk_a_ll03_10ep\epoch_0010.pt`
Data: `G:\wikiart27_latents_compact\train`
Samples: 32
Load: `{'missing': 0, 'unexpected': 0}`

## Content-style delta energy

| band | share |
|---|---:|
| ll | 0.6832 |
| lh | 0.1055 |
| hl | 0.1134 |
| hh | 0.0979 |

## Remain after model@AdaIN=1.5

| band | energy share | stat L2 |
|---|---:|---:|
| ll | 0.6525 | 1.212505 |
| lh | 0.1132 | 0.078824 |
| hl | 0.1200 | 0.086874 |
| hh | 0.1142 | 0.107736 |

## Transfer ratio (1 - statL2(pred,style)/statL2(content,style))

| setting | full | ll | lh | hl | hh |
|---|---:|---:|---:|---:|---:|
| no_adain | 0.1257 | 0.1218 | 0.1366 | 0.1248 | 0.0000 |
| adain_1 | 0.1222 | 0.1131 | 0.2906 | 0.3033 | 0.0988 |
| adain_1.5 | 0.1219 | 0.1128 | 0.3344 | 0.3398 | 0.1851 |
| adain_2 | 0.1136 | 0.1121 | 0.0444 | 0.0617 | -0.1040 |

## Oracle SAT alpha sweep (LL blend + exact style HF)

| alpha | transfer | full rms->style | content rms | ll statL2 |
|---:|---:|---:|---:|---:|
| 0 | 0.0045 | 1.0405 | 0.6985 | 1.371805 |
| 0.3 | 0.3030 | 0.9240 | 0.7298 | 0.960264 |
| 0.5 | 0.5022 | 0.8647 | 0.7817 | 0.685903 |
| 1 | 1.0000 | 0.8055 | 0.9858 | 0.000000 |

## Oracle LL appearance + style HF

| setup | transfer | full rms | content rms | ll statL2 |
|---|---:|---:|---:|---:|
| LL-adain a=0.3+HF style | 0.3030 | 0.9240 | 0.7298 | 0.960264 |
| LL-adain a=0.5+HF style | 0.5022 | 0.8647 | 0.7817 | 0.685903 |
| LL-adain a=1+HF style | 1.0000 | 0.8055 | 0.9858 | 0.000000 |
| LL-wct a=0.3+HF style | 0.3029 | 0.9191 | 0.7322 | 0.960877 |
| LL-wct a=0.5+HF style | 0.5021 | 0.8580 | 0.7878 | 0.686266 |
| LL-wct a=1+HF style | 1.0000 | 0.8027 | 1.0046 | 0.000001 |

## Oracle single-band swap on model(no AdaIN) then AdaIN1.5

| swapped | transfer | full rms | full statL2 |
|---|---:|---:|---:|
| ll | 0.9107 | 0.6754 | 0.056660 |
| lh | 0.1283 | 1.1249 | 0.586182 |
| hl | 0.1281 | 1.1212 | 0.586350 |
| hh | 0.1278 | 1.1241 | 0.586593 |
