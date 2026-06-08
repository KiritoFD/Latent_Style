# Distinct5 Non-CLIP Style Probe

Date: 2026-06-08

Scope:

- dataset: `Distinct5-WikiArt`
- purpose: add a non-CLIP target-style verification path for the current paper-facing operating points
- classifier family: frozen/pretrained `ConvNeXt-Tiny` backbone with a trained image-level linear head

## Training packet

- script:
  - [classify.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/utils/classify.py)
- config anchor:
  - [distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/distinct5_512_ema_variant_k_content_adaptive_vq_queue_e3.json)
- train root:
  - `G:\GitHub\Latent_Style\Dataset\distinct5_512\train`
- val root:
  - `G:\GitHub\Latent_Style\Dataset\distinct5_512\test`
- output checkpoint:
  - [distinct5_convnext_style_classifier.pt](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_convnext_style_classifier.pt)
- output report:
  - [distinct5_convnext_style_classifier_report.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_convnext_style_classifier_report.json)

Held-out validation result:

- best val accuracy: `0.9600`
- best macro recall: `0.9600`
- best mean confidence: `0.9643`
- stop epoch: `4`

Interpretation:

- the five Distinct5 classes are cleanly separable by a non-CLIP image classifier on the held-out test split
- this is strong enough to use as a paper-facing auxiliary target-style check

## Evaluated operating points

Manifest:

- [operating_point_manifest.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/operating_point_manifest.csv)

Probe script:

- [eval_nonclip_style_probe.py](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/eval_nonclip_style_probe.py)

Outputs:

- [distinct5_nonclip_style_probe.csv](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.csv)
- [distinct5_nonclip_style_probe.json](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.json)
- [distinct5_nonclip_style_probe.md](/G:/GitHub/Latent_Style/SchrodingerBridge/aaai2027/distinct5_nonclip_style_probe.md)

## Key transfer-only results

| point | target acc | target prob | source prob | target-source margin | identity source acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| `IDT` | `0.0100` | `0.0168` | `0.9329` | `-0.9161` | `0.9600` |
| `LBM-K` | `0.1667` | `0.1524` | `0.7069` | `-0.5544` | `0.8733` |
| `LBM-Knee` | `0.2367` | `0.2123` | `0.5633` | `-0.3511` | `0.7667` |
| `SaMST e15` | `0.2483` | `0.2405` | `0.5953` | `-0.3548` | `0.9400` |
| `LBM-PS-v2` | `0.2717` | `0.2696` | `0.3064` | `-0.0368` | `0.3600` |
| `Seedream-4.5` | `0.3783` | `0.3758` | `0.4774` | `-0.1016` | `0.9467` |

## Paper-facing interpretation

- `IDT` remains a near-zero target-style mover under a non-CLIP classifier, consistent with the paper's diagnostic claim
- `LBM-K` clearly exceeds `IDT`, but is still conservative
- `LBM-Knee` and `SaMST e15` are close on non-CLIP target-style recognition
  - this is important because it shows Knee is not winning only through CLIP
- `LBM-Knee` remains easier to defend as the main closed point because it is much cleaner than `SaMST` on the artifact-sensitive diagnostics
- `LBM-PS-v2` is the strongest style-side LBM point under this non-CLIP probe, which supports keeping it as the explicit style ceiling row
- `Seedream-4.5` remains the strongest external large-prior reference on this non-CLIP target-style check

## Bottom line

The non-CLIP probe supports the current manuscript reading:

- `LBM-Knee` is the main closed Pareto point
- `LBM-PS-v2` is the stronger style ceiling
- `Seedream-4.5` is the stronger external style reference
- `IDT` remains a valid no-op failure control on Distinct5-WikiArt
