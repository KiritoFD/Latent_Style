# Timing Filled Report

This report supplements `timing_summary.md` with the currently missing values.

## Ours

- Train to `epoch_0007`: `309.902s` actual, from training log.
- Avg epoch time over epochs 1-7: `44.272s`.
- Inference (`epoch_0007`, generation-only, 750 images): `85.414s` actual.
- Inference sec/image: `0.113885`.

## SaMST

- Train probe: `1` epoch across `5` styles took `67.687s` actual.
- Extrapolated full train (`30` epochs, profile `4g`): `2030.610s`.
- Strict 750 inference actual: `39.826s`, or `0.053101s/image`.

## StyleID

- Training is not needed; method is training-free.
- Measured actual generation for `photo` target (`150` images): `603.267s`.
- Estimated fair full `750` inference: `3016.335s`.
- Estimated sec/image: `4.021780`.

## Notes

- `Ours` train time is taken from the existing training CSV log, not re-trained.
- `Ours` inference time was freshly measured in generation-only mode for `epoch_0007`.
- `SaMST` train time was measured for one epoch and extrapolated linearly, per your requested policy.
- `StyleID` full-750 time is still an estimate derived from the actually measured `photo` target runtime.
