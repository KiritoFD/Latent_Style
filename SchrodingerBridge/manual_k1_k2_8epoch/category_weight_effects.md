# Manual Category Weight Effects

Weights: content `[photo 1.35, Hayao 1.25, monet/vangogh/cezanne 0.85]`; target `[photo 0.80, Hayao 1.35, monet/vangogh/cezanne 1.05]`. Positive deltas mean weighted run is higher; LPIPS lower is better.

## K1_weight_effect: all non-identity directions grouped by target

| target | ?style | ?content | ?LPIPS | ?EC | verdict |
|---|---:|---:|---:|---:|---|
| photo | -0.0134 | +0.0144 | -0.0057 | -0.0035 | style-, content+, lpips+, EC- |
| Hayao | -0.0043 | +0.0008 | -0.0086 | +0.0037 | style-, lpips+, EC+ |
| monet | -0.0093 | -0.0042 | +0.0120 | -0.0140 | style-, content-, lpips-, EC- |
| vangogh | -0.0094 | -0.0053 | -0.0013 | -0.0035 | style-, content-, EC- |
| cezanne | -0.0050 | +0.0086 | -0.0062 | +0.0009 | style-, content+, lpips+ |

## K1_weight_effect: photo-to-art directions

| target | base style | weighted style | ?style | ?content | ?LPIPS | ?EC |
|---|---:|---:|---:|---:|---:|---:|
| Hayao | 0.6665 | 0.6643 | -0.0023 | +0.0040 | -0.0179 | +0.0109 |
| monet | 0.6684 | 0.6645 | -0.0039 | +0.0045 | +0.0035 | -0.0045 |
| vangogh | 0.7025 | 0.6878 | -0.0147 | -0.0117 | -0.0027 | -0.0055 |
| cezanne | 0.6535 | 0.6518 | -0.0017 | +0.0179 | -0.0270 | +0.0168 |

## K2_weight_effect: all non-identity directions grouped by target

| target | ?style | ?content | ?LPIPS | ?EC | verdict |
|---|---:|---:|---:|---:|---|
| photo | +0.0226 | +0.0116 | -0.0093 | +0.0175 | style+, content+, lpips+, EC+ |
| Hayao | +0.0483 | -0.0199 | +0.0064 | +0.0212 | style+, content-, lpips-, EC+ |
| monet | -0.0028 | +0.0015 | -0.0011 | -0.0010 | style- |
| vangogh | -0.0024 | -0.0024 | -0.0038 | +0.0020 | style-, content-, lpips+ |
| cezanne | -0.0232 | +0.0116 | -0.0160 | -0.0033 | style-, content+, lpips+, EC- |

## K2_weight_effect: photo-to-art directions

| target | base style | weighted style | ?style | ?content | ?LPIPS | ?EC |
|---|---:|---:|---:|---:|---:|---:|
| Hayao | 0.6084 | 0.6542 | +0.0458 | +0.0159 | -0.0094 | +0.0290 |
| monet | 0.6436 | 0.6459 | +0.0024 | +0.0187 | -0.0256 | +0.0179 |
| vangogh | 0.6631 | 0.6755 | +0.0125 | +0.0044 | -0.0201 | +0.0202 |
| cezanne | 0.6560 | 0.6358 | -0.0202 | +0.0264 | -0.0447 | +0.0175 |
