# Minimal WEAVE Ablations

All runs use five epochs, batch 24, and the same 750-image Distinct5 evaluation protocol on the remote 3060.

| Run | CLIP-S | LPIPS | DINO-S | DINO-C | Decision |
|---|---:|---:|---:|---:|---|
| Minimal baseline | 0.7266 | 0.3343 | 0.4813 | 0.7573 | Reference |
| Cross-attention off | 0.7141 | 0.3311 | 0.4808 | 0.7951 | Keep attention: style transfer drops materially |
| LL weight = 0 | 0.7127 | 0.3142 | 0.4776 | 0.8012 | Keep LL supervision: content rises at the expense of style |
| Bridge sigma = 0 | 0.7277 | 0.3471 | 0.4841 | 0.7523 | Keep noise default: style improves slightly, but content metrics worsen |

## Confirmed Removals

- Legacy spatial bridge and its objective.
- ASG, masking experiments, all SWD variants, edge loss, and low-pass/content auxiliary losses.
- Per-step versus last-step AdaIN distinction: no measurable difference; retain only final-step alignment in the next cleanup.

## Confirmed Retentions

- Rectified flow.
- Haar wavelet coordinates.
- LL flow supervision (`w_LL=0.3` current baseline).
- Cross-attention/style-token path. It improves style metrics even though it trades off DINO-C.
- High-pass AdaIN alignment.

## Pending

- `w_LL=1` checkpoint is trained; it is lower priority because both `w_LL=0` and the existing `w_LL=1` history are adverse to style metrics.
