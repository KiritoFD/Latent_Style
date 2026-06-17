# Clean OT Probe Round 4: Model Feature Maps

Date: 2026-06-17

## Purpose

The previous 616 OT rounds established one local winner:

- retain `self_affinity_gw`
- retire `lowedge_self_affinity_gw`
- retire `structure_only`

But there is still a valid implementation audit question:

- is the old `self_affinity_gw` actually testing the intended geometry
- or is it still too tied to a hand-built latent proxy

The current retained `self_affinity_gw` descriptor is computed from low/edge/high
statistics of the raw latent tensor, then converted into a self-affinity vector.
That is a useful proxy, but it is not the same thing as matching on the model's
own internal feature geometry.

## Implementation audit result

The earlier retained OT path was not a broken code path, but it did undershoot
the intended 616 design target.

- In [losses.py](/G:/GitHub/Latent_Style/SchrodingerBridge/src/losses.py:524),
  `self_affinity_gw` is built from raw latent lowpass, edge, and high-frequency
  magnitude statistics.
- That means the prior "GW-like" result was testing a hand-built latent proxy,
  not the encoder/down feature geometry and not the structured tokenizer routing
  geometry discussed in [design.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/616/design.md:222).
- The correct interpretation of earlier positive evidence is therefore:
  "structure-aware proxy OT helped relative to plain appearance OT", not
  "the model-feature OT hypothesis is already validated."

Round 4 is the first clean matched probe that answers the narrower audit
question: whether the useful structure signal should come from the retained
latent proxy, the model encoder, or the structured tokenizer sidecar.

Round 4 therefore keeps the OT contract fixed and changes only the structure
descriptor source.

## Matched contract

Fixed:

- `contract_family = phase616`
- `coupling_solver = sinkhorn_unbalanced`
- `coupling_cost_composition = appearance_plus_structure`
- `training_target_projection_mode = pure_vertical_flow`
- one epoch
- `stop_after_global_steps = 60`
- same transfer-only eval contract

## Candidates

- control: current retained latent-proxy `self_affinity_gw`
- candidate A: `encoder_self_affinity_gw`
- candidate B: `tokenizer_aux_self_affinity_gw`

Interpretation:

- `encoder_self_affinity_gw` asks whether OT should see the model encoder/down
  feature map rather than low/edge/high latent statistics
- `tokenizer_aux_self_affinity_gw` asks whether OT should see the structured
  tokenizer routing geometry directly

## Expected answer

If either feature-map candidate improves transfer and reduces hubness, that is
evidence that the previous OT implementation was directionally right but still
measuring the wrong surface.

If both feature-map candidates regress relative to the retained control, that is
evidence that the earlier implementation was not simply "wrong"; the raw latent
proxy may actually be the cleaner match surface under the current model.

## Matched configs

- control: [phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_selfaffgw_mix_faststep60_e1.json)
- candidate A: [phase616_clean_ot_probe_encoder_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_encoder_selfaffgw_mix_faststep60_e1.json)
- candidate B: [phase616_clean_ot_probe_tokenaux_selfaffgw_mix_faststep60_e1.json](/G:/GitHub/Latent_Style/SchrodingerBridge/configs/aaai2027/phase616_clean_ot_probe_tokenaux_selfaffgw_mix_faststep60_e1.json)
- launcher: [run_phase616_clean_ot_probe_round4_featuremaps.sh](/G:/GitHub/Latent_Style/SchrodingerBridge/tools/experiments/run_phase616_clean_ot_probe_round4_featuremaps.sh)

## Closure rule

- retain a feature-map candidate only if it matches or beats the current control
  on transfer while not worsening `ot_target_gini` / `ot_target_max_mass`
- if a feature-map candidate lowers train OT cost but regresses transfer, record
  it as diagnostic-only
- if both feature-map candidates regress on both transfer and white-box probes,
  keep the retained control and write the conclusion explicitly as an audit result

## Results

Round completed on 2026-06-17. All three matched probes closed successfully.

### Control: latent-proxy `self_affinity_gw`

Train closure:

- epoch wall: `87.14 s`
- `avg_optimizer_step_time_sec = 1.4769`
- `ot_cost = 2.7054`
- `ot_target_gini = 0.0594`
- `ot_target_max_mass = 0.3531`
- `ot_source_truncation = 0.6472`
- `ot_target_truncation = 0.6472`
- `fiber_energy_ratio = 0.4410`
- `low_freq_leak = 3.4510`

Transfer eval:

- `CLIP-S = 0.6682`
- `LPIPS = 0.7197`
- generated effective rank `= 1.0403`
- generated offdiag cosine `= 0.9740`
- eval wall `= 212.27 s`

### Candidate A: `encoder_self_affinity_gw`

Train closure:

- epoch wall: `96.64 s`
- `avg_optimizer_step_time_sec = 1.6379`
- `ot_cost = 2.7015`
- `ot_target_gini = 0.0582`
- `ot_target_max_mass = 0.3522`
- `ot_source_truncation = 0.6429`
- `ot_target_truncation = 0.6429`
- `fiber_energy_ratio = 0.4195`
- `low_freq_leak = 3.6646`

Transfer eval:

- `CLIP-S = 0.6826`
- `LPIPS = 0.8225`
- generated effective rank `= 1.0795`
- generated offdiag cosine `= 0.9511`
- eval wall `= 219.80 s`

Matched delta vs. control:

- `CLIP-S`: `+0.0144`
- `LPIPS`: `+0.1028`
- `ot_target_gini`: `-0.0012`
- `ot_target_max_mass`: `-0.0009`
- generated effective rank: `+0.0392`
- generated offdiag cosine: `-0.0229`
- epoch wall: `+9.50 s`

Interpretation:

- the encoder feature map does produce a "truer" internal-geometry OT signal
  than the latent proxy
- that signal increases style actuation and latent-delta diversity
- but it does so by paying a very large structure price, so it is not usable as
  the retained 616 OT repair

Decision: `negative_for_promotion`

### Candidate B: `tokenizer_aux_self_affinity_gw`

Train closure:

- epoch wall: `112.41 s`
- `avg_optimizer_step_time_sec = 1.9052`
- `ot_cost = 2.8034`
- `ot_target_gini = 0.0505`
- `ot_target_max_mass = 0.3477`
- `ot_source_truncation = 0.6718`
- `ot_target_truncation = 0.6718`
- `fiber_energy_ratio = 0.4166`
- `low_freq_leak = 3.5762`

Transfer eval:

- `CLIP-S = 0.6652`
- `LPIPS = 0.7195`
- generated effective rank `= 1.0657`
- generated offdiag cosine `= 0.9580`
- eval wall `= 214.19 s`

Matched delta vs. control:

- `CLIP-S`: `-0.0030`
- `LPIPS`: `-0.0002`
- `ot_target_gini`: `-0.0089`
- `ot_target_max_mass`: `-0.0054`
- generated effective rank: `+0.0254`
- generated offdiag cosine: `-0.0160`
- epoch wall: `+25.27 s`

Interpretation:

- the tokenizer-sidecar geometry is much safer than the encoder geometry
- it slightly improves hubness and keeps LPIPS effectively flat
- but it does not improve style, and it costs about `29%` more train time than
  the retained latent-proxy control

Decision: `diagnostic_only`

## Decision

Retain `self_affinity_gw` latent proxy as the active 616 OT control.

Round-4 audit conclusion:

- the previous implementation was not "wrong" in the sense of a broken code path
- it was narrower than the theory target because it matched on a hand-built
  latent proxy rather than true model feature maps
- after correcting that mismatch, neither internal feature-map candidate gave a
  better retained trade-off than the proxy control

What this means for the next 616 step:

- do not keep pushing encoder/tokenizer feature-map OT as the main lane
- keep the retained proxy OT repair as the current clean baseline
- if we revisit feature-map OT later, it should be as a constrained follow-up
  with stronger vertical/frequency protection, not as a direct replacement

## 2026-06-17 content-side cleanup rerun

After the implementation audit, the round-4 OT helper paths were cleaned up so
the content-side intent is explicit in code instead of implicit:

- `_ot_encoder_feature_map()` now uses a neutral OT style code together with
  `gate=0.0`
- `_ot_tokenizer_aux_feature_map()` now calls the latent-native tokenizer
  directly and prefers `aux_map` routing attention, instead of routing through
  the broader runtime sidecar path

This cleanup does **not** change the intended round-4 hypothesis. It removes a
source of audit ambiguity and makes the rerun easier to interpret.

Rerun launch record:

- launched on `2026-06-17 04:43:50 +08:00`
- task name:
  `phase616_clean_ot_probe_round4_featuremaps_rerun_contentside`
- remote launcher log:
  `/home/xy/Latent_Style/SchrodingerBridge_phase616/SchrodingerBridge/docs/experiments/phase2_fiber_bundle/616/logs/clean_ot_probe_round4_featuremaps/launcher_rerun_contentside.log`

Early health:

- remote WSL health check passed
- GPU at launch stayed well below the `11.2 GiB` guard band
- live training confirmed the control config started successfully on the cleaned
  code path

Decision pending:

- compare the rerun against the earlier round-4 closure
- if the deltas are materially unchanged, promote the cleaner interpretation and
  keep the earlier qualitative conclusion
- if the rerun shifts the ordering, replace the old round-4 closure with the
  cleaned result

### Interim control evidence

The rerun control `self_affinity_gw_mix` completed first on the cleaned
content-side helper path.

New control readout:

- epoch wall: `92.49 s`
- `avg_optimizer_step_time_sec = 1.5677`
- `ot_target_gini = 0.05935`
- `ot_target_max_mass = 0.35310`
- `ot_source_truncation = 0.64722`
- `ot_target_truncation = 0.64722`
- `fiber_energy_ratio = 0.44271`
- `low_freq_leak = 3.45661`
- `CLIP-S = 0.66843`
- `LPIPS = 0.75084`
- generated effective rank `= 1.01760`
- generated offdiag cosine `= 0.98843`
- eval wall `= 223.24 s`

Interim delta vs. the earlier round-4 control:

- white-box OT metrics are effectively unchanged
  - `ot_target_gini`: `-0.00005`
  - `ot_target_max_mass`: `-0.00000`
  - `ot_source_truncation`: `+0.00002`
  - `fiber_energy_ratio`: `+0.00171`
  - `low_freq_leak`: `+0.00561`
- style is effectively unchanged
  - `CLIP-S`: `+0.00023`
- the visible regression so far is on transfer LPIPS and eval/train wall
  - `LPIPS`: `+0.03114`
  - epoch wall: `+5.35 s`
  - eval wall: `+10.97 s`

Interim interpretation:

- the content-side cleanup did **not** invalidate the earlier structural OT
  reading for the control path
- the control rerun is close enough on white-box OT behavior that the old
  interpretation remains directionally stable
- however, because LPIPS moved materially while style stayed flat, the final
  rerun decision should still wait for candidate A/B before replacing the old
  round-4 closure

### Interim encoder evidence

The rerun candidate `encoder_self_affinity_gw_mix` also completed on the
cleaned content-side helper path.

New encoder readout:

- epoch wall: `98.58 s`
- `avg_optimizer_step_time_sec = 1.6709`
- `ot_target_gini = 0.05846`
- `ot_target_max_mass = 0.35241`
- `ot_source_truncation = 0.64256`
- `ot_target_truncation = 0.64256`
- `fiber_energy_ratio = 0.43417`
- `low_freq_leak = 3.59583`
- `CLIP-S = 0.67944`
- `LPIPS = 0.79631`
- generated effective rank `= 1.09521`
- generated offdiag cosine `= 0.93888`
- eval wall `= 217.11 s`

Interim delta vs. the earlier round-4 encoder candidate:

- `CLIP-S`: `-0.00316`
- `LPIPS`: `-0.02619`
- `ot_target_gini`: `+0.00026`
- `ot_target_max_mass`: `+0.00021`
- `ot_source_truncation`: `-0.00034`
- `fiber_energy_ratio`: `+0.01467`
- `low_freq_leak`: `-0.06877`
- epoch wall: `+1.94 s`
- eval wall: `-2.69 s`

Interim delta vs. the cleaned rerun control:

- `CLIP-S`: `+0.01101`
- `LPIPS`: `+0.04548`
- `ot_target_gini`: `-0.00089`
- `ot_target_max_mass`: `-0.00069`
- `ot_source_truncation`: `-0.00466`
- `fiber_energy_ratio`: `-0.00854`
- `low_freq_leak`: `+0.13923`

Interim interpretation:

- the encoder candidate still behaves like the same branch as before:
  slightly better style / hubness, but worse transfer LPIPS than the control
- the clean rerun did **not** reveal a hidden inversion where encoder OT
  suddenly becomes the retained winner
- compared with the earlier round-4 closure, the cleaned rerun actually narrows
  the LPIPS penalty, but the ordering remains directionally the same so far

### Final tokenaux evidence

The rerun candidate `tokenizer_aux_self_affinity_gw_mix` also completed on the
cleaned content-side helper path.

New tokenaux readout:

- epoch wall: `111.97 s`
- `avg_optimizer_step_time_sec = 1.8978`
- `ot_target_gini = 0.05033`
- `ot_target_max_mass = 0.34755`
- `ot_source_truncation = 0.67217`
- `ot_target_truncation = 0.67217`
- `fiber_energy_ratio = 0.41915`
- `low_freq_leak = 3.58141`
- `CLIP-S = 0.66521`
- `LPIPS = 0.73438`
- generated effective rank `= 1.03548`
- generated offdiag cosine `= 0.97682`
- eval wall `= 212.98 s`

Delta vs. the earlier round-4 tokenaux candidate:

- `CLIP-S`: `+0.00001`
- `LPIPS`: `+0.01488`
- `ot_target_gini`: `-0.00017`
- `ot_target_max_mass`: `-0.00015`
- `ot_source_truncation`: `+0.00037`
- `fiber_energy_ratio`: `+0.00255`
- `low_freq_leak`: `+0.00521`
- epoch wall: `-0.44 s`
- eval wall: `-1.21 s`

Delta vs. the cleaned rerun control:

- `CLIP-S`: `-0.00322`
- `LPIPS`: `-0.01645`
- `ot_target_gini`: `-0.00902`
- `ot_target_max_mass`: `-0.00555`
- `ot_source_truncation`: `+0.02495`
- `fiber_energy_ratio`: `-0.02356`
- `low_freq_leak`: `+0.12480`

## Clean rerun closure

The content-side cleanup rerun did **not** overturn the original round-4
ordering.

Final rerun ordering:

- control `self_affinity_gw_mix`: still the best retained OT baseline
- `encoder_self_affinity_gw_mix`: still style-positive but structure-negative
- `tokenizer_aux_self_affinity_gw_mix`: still the safer diagnostic branch, but
  not a promoted winner

What changed:

- the cleaned rerun makes the helper paths much easier to interpret
- the encoder candidate's LPIPS penalty is smaller than in the first round-4
  closure, but it still loses to the control on the retained trade-off
- the tokenaux candidate again improves hubness while failing to improve style,
  which keeps it in `diagnostic_only`

Final decision after clean rerun:

- retain `self_affinity_gw` latent proxy OT as the active 616 control
- keep the original round-4 qualitative conclusion
- promote the **cleaner interpretation**, not a different winner
