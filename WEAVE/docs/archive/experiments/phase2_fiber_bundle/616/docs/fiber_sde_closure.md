# Fiber-SDE Closure

Date: 2026-06-14

## Scope

- Parent: `k070 epoch_0003`.
- Tested: matched isotropic vs fiber-aligned SDE solver noise at `sigma=0.01, 0.02, 0.03, 0.05`.
- Fixed variables: tokenizer, backbone, topogate, appearance head, loss, checkpoint, eval dataset, CLIP-S/LPIPS contract.

## Results

| sigma | mode | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | decision |
|---:|---|---:|---:|---:|---:|---|
| 0.01 | isotropic | 0.671501 | 0.313795 | 0.703024 | 0.311868 | control |
| 0.01 | fiber | 0.671581 | 0.313762 | 0.702954 | 0.311888 | tie |
| 0.02 | isotropic | 0.672031 | 0.314990 | 0.703432 | 0.313025 | control |
| 0.02 | fiber | 0.671818 | 0.314936 | 0.703320 | 0.313015 | not promoted |
| 0.03 | isotropic | 0.673391 | 0.316894 | 0.704514 | 0.314930 | control |
| 0.03 | fiber | 0.673405 | 0.316883 | 0.704633 | 0.314862 | marginal positive |
| 0.05 | isotropic | 0.675927 | 0.322953 | 0.706639 | 0.320868 | style upper |
| 0.05 | fiber | 0.675948 | 0.323189 | 0.706763 | 0.321093 | style upper, not promoted |

## Decision

- Fiber-SDE is not promoted as the core mechanism. It improves style relative to deterministic parent, but the best style point still falls far short of `0.74`, and higher sigma pays visible LPIPS cost.
- Fiber-aligned noise is not a strong positive result. It is marginally favorable at `sigma=0.03`, but the delta is too small; at `sigma=0.05` it improves style by only `+0.000021` transfer while worsening LPIPS by `+0.000237`.
- Keep `sigma=0.03 fiber` as the balanced eval-only option and `sigma=0.05 fiber` as the style-first diagnostic upper point.
- Next action: SMoE tokenizer training, changing only tokenizer mechanics.

## 2026-06-14 Fine Style-Ceiling Addendum

After SMoE and the kinetic short probe showed poor training cost/benefit, the SDE branch was reopened only as an eval-only style-ceiling scan. The parent, tokenizer, backbone, loss, topogate, appearance path, and checkpoint stayed fixed at `k070 epoch_0003`; only the solver noise scale changed. All runs used seed `42`.

| sigma | mode | transfer CLIP-S | transfer LPIPS | all-pairs CLIP-S | all-pairs LPIPS | decision |
|---:|---|---:|---:|---:|---:|---|
| 0.04 | isotropic | 0.674624 | 0.319512 | 0.705757 | 0.317374 | control |
| 0.04 | fiber | 0.674520 | 0.319587 | 0.705680 | 0.317446 | fiber worse |
| 0.06 | isotropic | 0.677688 | 0.327456 | 0.708146 | 0.325220 | control |
| 0.06 | fiber | 0.677541 | 0.327567 | 0.708073 | 0.325322 | fiber worse |
| 0.08 | isotropic | 0.681007 | 0.339036 | 0.710653 | 0.336767 | style ceiling |
| 0.08 | fiber | 0.681075 | 0.339063 | 0.710641 | 0.336797 | mixed tie |

Decision: the fine scan raises the reachable eval-only style ceiling, but the slope is not good enough. `sigma=0.08` is still far below the desired `0.74` style region and has already moved LPIPS from the parent `0.314618/0.312550` to about `0.339/0.337`. Fiber-aligned noise is not materially better than isotropic noise. Do not spend a long training lane on this direction without a stronger mechanism than noise amplitude.
