# Exp Brightness/Color vs Train Report

- exp_root: `G:\GitHub\Latent_Style\Cycle-NCE\exp`
- stats_dir: `G:\GitHub\Latent_Style\Cycle-NCE\src\eval_cache\train_color_stats`
- runs analyzed: 90
- failures: 0

## Best Brightness Mean Alignment
| experiment | epoch | is_tokenized | image_count | style_brightness_mean_abs | style_brightness_std_abs | style_rgb_mean_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| abl_no_adagn | epoch_0080 | 0 | 750 | 4.145 | 16.288 | 9.797 |
| abl_rank64 | epoch_0040 | 0 | 750 | 4.336 | 14.957 | 9.715 |
| abl_rank1 | epoch_0080 | 0 | 750 | 4.424 | 10.647 | 10.128 |
| abl_rank64 | epoch_0080 | 0 | 750 | 4.454 | 11.415 | 9.859 |
| abl_rank8 | epoch_0080 | 0 | 750 | 4.638 | 9.823 | 9.781 |
| abl_naive_skip | epoch_0040 | 0 | 750 | 4.700 | 14.524 | 11.099 |
| abl_rank1 | epoch_0040 | 0 | 750 | 4.753 | 13.353 | 9.683 |
| abl_rank8 | epoch_0040 | 0 | 750 | 4.894 | 11.165 | 10.194 |
| abl_naive_skip | epoch_0080 | 0 | 750 | 4.996 | 13.455 | 12.302 |
| abl_no_adagn | epoch_0040 | 0 | 750 | 5.179 | 17.965 | 10.330 |

## Best Brightness Std Alignment
| experiment | epoch | is_tokenized | image_count | style_brightness_std_abs | style_brightness_mean_abs | style_rgb_std_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| style_oa | epoch_0120 | 0 | 750 | 9.780 | 7.438 | 21.773 |
| abl_rank8 | epoch_0080 | 0 | 750 | 9.823 | 4.638 | 20.579 |
| abl_rank1 | epoch_0080 | 0 | 750 | 10.647 | 4.424 | 22.021 |
| abl_rank8 | epoch_0040 | 0 | 750 | 11.165 | 4.894 | 23.055 |
| abl_rank64 | epoch_0080 | 0 | 750 | 11.415 | 4.454 | 23.426 |
| style_oa | epoch_0120 | 0 | 750 | 11.507 | 6.224 | 24.417 |
| style_oa | epoch_0120 | 0 | 750 | 11.780 | 9.688 | 25.187 |
| style_oa | epoch_0060 | 0 | 750 | 13.279 | 6.776 | 27.660 |
| weight_exp7_pseudo_hist_swd60_tv01_id20_r16_e60 | epoch_0060 | 0 | 750 | 13.349 | 10.031 | 27.410 |
| abl_rank1 | epoch_0040 | 0 | 750 | 13.353 | 4.753 | 27.143 |

## Best RGB Mean Alignment
| experiment | epoch | is_tokenized | image_count | style_rgb_mean_l2 | style_brightness_mean_abs | style_chroma_mean_l2 |
| --- | --- | --- | --- | --- | --- | --- |
| abl_rank1 | epoch_0040 | 0 | 750 | 9.683 | 4.753 | 2.772 |
| abl_rank64 | epoch_0040 | 0 | 750 | 9.715 | 4.336 | 3.079 |
| abl_rank8 | epoch_0080 | 0 | 750 | 9.781 | 4.638 | 3.249 |
| abl_no_adagn | epoch_0080 | 0 | 750 | 9.797 | 4.145 | 3.660 |
| abl_rank64 | epoch_0080 | 0 | 750 | 9.859 | 4.454 | 3.468 |
| abl_rank1 | epoch_0080 | 0 | 750 | 10.128 | 4.424 | 3.703 |
| abl_rank8 | epoch_0040 | 0 | 750 | 10.194 | 4.894 | 3.098 |
| abl_no_adagn | epoch_0040 | 0 | 750 | 10.330 | 5.179 | 3.253 |
| style_oa | epoch_0120 | 0 | 750 | 11.066 | 6.224 | 1.935 |
| abl_naive_skip | epoch_0040 | 0 | 750 | 11.099 | 4.700 | 4.380 |

