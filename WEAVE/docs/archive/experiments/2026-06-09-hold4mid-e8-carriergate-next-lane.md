# `Hold4Mid e8 + Carrier-Gate Injection` Next-Lane Decision

Date: 2026-06-09

Why this line is promoted next:

- `Trust` is already a negative closure and the redundant rerun was stopped.
- `Hold4Mid + spatial_carrier_gate body+decoder` has now reproduced its weak plateau under a clean eval contract.
- the remaining coherent question is not another schedule-only tweak.
- the most defensible next family is:
  - keep the `Hold4Mid e8` geometry anchor
  - reopen style with the lighter `carrier_gate` branch only

Why this family is still worth GPU time:

- it is a real mechanism change, not a clamp-schedule micro-variation
- it is simpler than `spatial_carrier_gate body+decoder`
- it is more aligned with the visual diagnosis:
  - make style more explicit
  - but avoid generic fog and avoid global structure takeover

Current remote state:

- run root exists:
  - `/mnt/i/Github/Latent_Style/SchrodingerBridge/exp/inmortal-exp/aaai2027_inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2`
- checkpoints currently on disk:
  - `epoch_0001.pt`
  - `epoch_0002.pt`
  - `epoch_0003.pt`
  - `epoch_0004.pt`
  - `epoch_0005.pt`
- no formal `full_eval` closure is present yet
- remote training is not currently active for this lane

Updated active status:

- the lane has now been relaunched as the current formal remote training lane
- latest observed state:
  - remote process:
    - `src/run.py --config configs/aaai2027/inmortal_hold4mid_e8_carriergate_injection_seed42_b8a2.json`
  - latest visible epoch log:
    - training CSV currently contains completed rows through `epoch 3`
- latest retained training-side read:
  - `e3 loss = 8.7191`
  - `e3 terminal_swd = 5.1562`
  - `e3 samples_per_sec = 36.09`

Artifact caution:

- the relaunched run is currently writing into a save dir that already contains older checkpoints and older `full_eval` outputs
- current evidence suggests:
  - `epoch_0001` to `epoch_0006` are from the fresh rerun
  - `epoch_0007` to `epoch_0012` and earlier `full_eval` summaries are older artifacts
- therefore any mid-run read must use:
  - current training CSV rows
  - checkpoint modification times
  - and later overwritten summaries
  rather than assuming every file in the run root belongs to the fresh continuation

Immediate interpretation:

- the relaunched lane is healthy
- no new numerical instability or launch-path regression has appeared yet
- the next required step is a clean retained-point evaluation closure, not another launch fix

Observed partial training read:

- the packet reached at least mid-`epoch 6` in `remote_train.log`
- early logged epochs remained numerically stable:
  - `e3 loss = 8.7191`, `tswd = 5.1562`
  - `e4 loss = 8.6482`, `tswd = 4.9688`
  - `e5 loss = 8.8274`, `tswd = 5.0938`

Interpretation:

- the line is not obviously dead on first health
- but it is incomplete and unevaluated
- therefore it is the correct next lane after the current `bodydecoder IntroStyle v3` closure finishes

Immediate next action:

- relaunch or continue this packet under the formal remote single-lane contract
- then run the same `LPIPS + IntroStyle` closure used for the repaired bodydecoder audit
