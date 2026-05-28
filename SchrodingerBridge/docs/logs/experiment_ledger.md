# Experiment Ledger

Format: Each experiment block logs hypothesis, config delta, results, and verdict.

---

## Experiment Note: 2026-05-27 Neutral Tokenizer Spiral

**Status**: Running on remote
**Remote task**: `LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL`
**Remote path**: `I:\Github\Latent_Style\SchrodingerBridge`
**Started**: 2026-05-27 18:05 local time

### Hypothesis

Style type should be represented as executable operator coordinates, not as a
single class embedding. The clean tokenizer queue should keep target exposure
balanced and test whether `identity / grammar / band / residual` fields can
separate target styles by themselves.

Hayao is a diagnostic slice only. It must not be manually oversampled or
upweighted. If Hayao remains poor, the diagnosis should identify whether the
vocabulary failed to separate Hayao or whether the backbone lacks the correct
flat-color / contour operator.

### Running Variants

```text
ema_style_vocab_neutral_w34
ema_style_vocab_neutral_w36_stylepush
```

Both variants use neutral tokenizer initialization and balanced style exposure.

### Current Remote Health Check

At 2026-05-27 18:09 local time:

```text
task: running
gpu: 8153 / 12288 MiB, 100% utilization
active variant: ema_style_vocab_neutral_w34
process: run_vae_backend_256_probe.py -> src/run.py
```

### Diagnostic Tool Update

`tools/experiments/summarize_style_tokenizer_debug.py` now writes:

```text
style_tokenizer_debug_readout.md
style_tokenizer_debug_by_style.csv
style_tokenizer_field_discrimination.csv
style_tokenizer_eval_overview.csv
style_tokenizer_checkpoint_vocab.csv
```

The tool reports:

- global and cross-target eval rows;
- per-target tokenizer/carrier responses;
- normalized field separability;
- Hayao delta versus the other styles;
- whether weak Hayao means tokenizer collapse or missing executable operator.

### Evidence From Earlier Tokenizer Probe

Running the updated diagnostic on the earlier `ema_style_vocab_texton_w34`
shows:

```text
global: clip_style=0.708368, content_lpips=0.514357
Hayao cross: clip_style=0.642931, content_lpips=0.565908
grammar normalized_range=3.442
Hayao flattening delta vs others=0.004876
```

Verdict: Hayao fields can separate, but the visible result remains weak. This
points to an insufficient or wrong executable Hayao operator rather than a
need for manual Hayao target weighting.

### First Clean Result

`ema_style_vocab_neutral_w34` finished on the remote task.

| epoch | global clip_style | content_lpips | EC | Hayao cross clip_style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|---:|
| 6 | 0.704802 | 0.514826 | 0.341952 | 0.647169 | 0.558829 |
| 7 | 0.707341 | 0.512390 | 0.344907 | 0.643466 | 0.566596 |
| 8 | 0.707817 | 0.514850 | 0.343397 | 0.643154 | 0.566445 |

Tokenizer readout at epoch 8:

```text
grammar normalized_range=3.443
Hayao flattening delta vs others=0.004895
Hayao band mean is below others
checkpoint grammar norm: Hayao=0.885814, Cezanne=0.567891, photo/Monet/VanGogh=0
checkpoint band norm: Hayao=0.355879, Cezanne=0.331349, others near zero
```

Interpretation: the neutral tokenizer can discover non-trivial Hayao and
Cezanne fields without manual style weighting. However, Hayao remains visually
and metrically weak, so the next architecture step should not be "more Hayao
weight". With a usable backbone already available, the immediate next step is
to treat tokenizer quality as its own component: measure vocabulary capacity,
coverage, sensitivity, and vocabulary-only refinement before changing the
backbone again.

### 2026-05-27 18:31 Follow-up Check

`ema_style_vocab_neutral_w36_stylepush` is still running. The remote GPU is
busy at roughly `4881 / 12288 MiB` and `98%` utilization, so the queue was left
untouched.

The completed `w34` diagnostic was regenerated on the remote:

```text
python tools\experiments\summarize_style_tokenizer_debug.py ^
  exp\vae_backend_256_probe\ema_style_vocab_neutral_w34 --limit-events 80
```

The readout confirms the main interpretation:

- style-token grammar is not collapsed;
- Hayao has the strongest grammar and flattening response;
- Hayao still has the weakest cross-target style score;
- therefore the current tokenizer has real but incomplete field separation.
  The next controlled change should focus on tokenizer/vocabulary quality
  rather than target-style reweighting or another backbone branch.

### Tokenizer Component Scorecard

The second clean tokenizer run, `ema_style_vocab_neutral_w36_stylepush`,
finished successfully.

| run | epoch | clip_style | LPIPS | EC | Hayao cross style | Hayao cross LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| `neutral_w34` | 8 | 0.707817 | 0.514850 | 0.343397 | 0.643154 | 0.566445 |
| `neutral_w36_stylepush` | 8 | 0.708146 | 0.519977 | 0.339926 | 0.645181 | 0.570138 |

Tokenizer/component diagnostics for `neutral_w36_stylepush`:

```text
grammar normalized_range: 3.443
band-gain normalized_range: 0.065
Hayao flatten delta vs others: +0.007022
Hayao low-delta vs others: +0.006179
Hayao high-delta vs others: -0.000042
grammar norm: Hayao=0.886319, Cezanne=0.567891, photo/Monet/VanGogh=0
band norm: Hayao=0.356546, Cezanne=0.330701, others near zero
```

Component verdict: the tokenizer is not collapsed globally, but it is not yet
a good style vocabulary. It mainly learns Hayao and Cezanne grammar/band
offsets, while Monet/VanGogh/photo remain nearly at neutral grammar. Increasing
style pressure gives only `+0.00033` global style and `+0.00203` Hayao style
while worsening LPIPS. The next tokenizer work should define and optimize
component metrics directly: vocabulary effective rank, per-style field
coverage, field-to-actuator sensitivity, and by-style downstream deltas.

### Tokenizer Component Scorecard Tool

Added:

```text
tools/experiments/evaluate_style_tokenizer_component.py
```

Remote output:

```text
exp\vae_backend_256_probe\tokenizer_component_scorecard\
```

The component scorecard treats tokenizer quality as separate from backbone
quality. It reports:

- vocabulary effective rank;
- active non-photo styles for grammar and band vocabularies;
- per-style vocabulary rows;
- field-to-actuator sensitivity;
- downstream style/LPIPS gates.

Current scorecard:

| run | style | LPIPS | Hayao style | grammar active | band active | erank g/b | coverage | sensitivity | component |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `neutral_w34` | 0.707817 | 0.514850 | 0.643154 | 2 | 2 | 0.370 / 0.632 | 0.500 | 1.000 | 0.676 |
| `neutral_w36_stylepush` | 0.708146 | 0.519977 | 0.645181 | 2 | 2 | 0.370 / 0.634 | 0.500 | 1.000 | 0.669 |

This makes the tokenizer bottleneck concrete: the carrier reads some fields,
but vocabulary coverage is poor. Only Hayao and Cezanne leave neutral
grammar/band rows. Monet and Van Gogh are not receiving explicit grammar
coordinates, so the current tokenizer is not a full style vocabulary yet.

### Tokenizer-Only Refit Route

Extended `tools/experiments/run_style_embedding_mainline_calibration.py` with
tokenizer-only recipes:

```text
m10_token_vocab_swd_anchor
m11_token_vocab_stylepush
```

These freeze the backbone, `style_emb`, and `style_spatial_id_16`, then optimize
only:

```text
style_tokenizer.grammar_vocab.weight
style_tokenizer.band_vocab.weight
```

A smoke run on `ema_style_vocab_neutral_w36_stylepush/epoch_0008.pt` completed
with `--max-iters-per-style 2 --skip-eval`. It confirmed nonzero tokenizer
gradient flow and saved `style_adapter.pt`.

The full remote queue was launched:

```text
task: LANCET_TOKENIZER_VOCAB_REFIT
checkpoint: exp\vae_backend_256_probe\ema_style_vocab_neutral_w36_stylepush\epoch_0008.pt
recipes: m10_token_vocab_swd_anchor,m11_token_vocab_stylepush
out: exp\style_tokenizer_vocab_refit\w36_epoch8
```

At 2026-05-27 18:58 local time it was running `m10`, with GPU around
`6959 / 12288 MiB` and `93%` utilization.

### Tokenizer-Only Refit Result

`LANCET_TOKENIZER_VOCAB_REFIT` finished successfully.

| recipe | clip_style | LPIPS | Hayao cross style | Hayao cross LPIPS | weakest cross target |
|---|---:|---:|---:|---:|---|
| `m10_token_vocab_swd_anchor` | 0.710066 | 0.466699 | 0.618145 | 0.517782 | Hayao |
| `m11_token_vocab_stylepush` | 0.710138 | 0.466697 | 0.618121 | 0.517815 | Hayao |

This is a useful negative result. The global LPIPS is excellent, but style is
still below the `0.72` target and Hayao is far below the other styles.
Increasing the tokenizer-only style pressure did not change the result.

Adapter scorecard:

| recipe | grammar active | band active | erank g/b | coverage | component |
|---|---:|---:|---:|---:|---:|
| `m10_token_vocab_swd_anchor` | 2 | 2 | 0.370 / 0.606 | 0.500 | 0.444 |
| `m11_token_vocab_stylepush` | 2 | 2 | 0.370 / 0.603 | 0.500 | 0.444 |

Direct checkpoint-vs-adapter diff shows why the score did not move:

```text
grammar delta: exactly 0 for all styles
band delta: mainly Cezanne, with only tiny movement elsewhere
```

Gradient audit under the `m10` objective:

| style | grammar grad norm | band grad norm | current grammar norm | current band norm |
|---|---:|---:|---:|---:|
| Hayao | 1.66e-05 | 3.85e-04 | 0.8863 | 0.3565 |
| Monet | 0.00e+00 | 3.38e-04 | 0.0000 | 0.0036 |
| Van Gogh | 0.00e+00 | 4.31e-04 | 0.0000 | 0.0014 |
| Cezanne | 0.00e+00 | 3.45e-04 | 0.5679 | 0.3307 |

Verdict: the current tokenizer vocabulary is under-executable. `grammar_vocab`
is mostly diagnostic state, not an effective training handle. In this backbone
configuration, grammar only reaches the flatten/high-frequency suppression
path, so vocabulary-only refit cannot create the missing style operators.

### Tokenizer Projector Route

The next tokenizer-only route keeps the backbone weights frozen but lets the
tokenizer fields produce an explicit style-code delta through
`style_tokenizer.code_projector`.

Code support added:

- `style_adapter.pt` now stores `style_tokenizer.project_code` and
  `style_tokenizer.code_projector.*`.
- inference and calibration loaders restore those tokenizer projector fields.
- `m12_token_projector_swd_anchor` and `m13_token_projector_stylepush` train
  `grammar_vocab`, `band_vocab`, and `code_projector` while freezing backbone,
  `style_emb`, and `style_spatial_id_16`.
- `tools/experiments/diagnose_style_tokenizer_gradients.py` audits per-style
  grammar/band gradient flow.

Smoke result: `m12` with two iterations per style completed and saved an
adapter. Anchor loss became nonzero after the first update, which confirms that
the tokenizer projector changes the endpoint instead of merely perturbing the
band gains.

Remote full run:

```text
process: hidden Start-Process batch
out: exp\style_tokenizer_projector_refit\w36_epoch8
recipes: m12_token_projector_swd_anchor,m13_token_projector_stylepush
started: 2026-05-27 19:37 local time
```

High-level tokenizer agenda:

```text
docs/maths/18_style_tokenizer_theory_agenda.md
```

This document records the current supervision signal, completed tokenizer
tests, negative evidence, and the theory problems that must be solved before
more backbone sweeps are justified.

Projector full run result:

| recipe | clip_style | LPIPS | Hayao style | Hayao LPIPS |
|---|---:|---:|---:|---:|
| `m12_token_projector_swd_anchor` | 0.709745 | 0.430403 | 0.614650 | 0.482358 |
| `m13_token_projector_stylepush` | 0.709595 | 0.434844 | 0.622817 | 0.488738 |

Scorecard:

| recipe | grammar active | band active | erank g/b | coverage | component |
|---|---:|---:|---:|---:|---:|
| `m12` | 2 | 2 | 0.384 / 0.599 | 0.500 | 0.445 |
| `m13` | 2 | 2 | 0.377 / 0.629 | 0.500 | 0.449 |

Metric-space diagnosis:

| recipe | identity-low rho | grammar-high rho | grammar-abs-high rho | band-energy rho | all-full rho |
|---|---:|---:|---:|---:|---:|
| `m12` | 0.000 | 0.139 | 0.321 | -0.103 | -0.055 |
| `m13` | 0.000 | 0.139 | 0.406 | -0.236 | -0.042 |

Verdict: `cat+project` is not the right tokenizer abstraction. It improves the
endpoint enough to lower LPIPS, but it does not create an isometric or
orthogonal style metric space. Move to the hard-bound operator route:

```text
identity -> pointwise color/channel operator
grammar  -> depthwise spatial operator
band     -> low/mid/high residual gains
```

Remote hard-binding run launched:

```text
task: LANCET_FACTORIZED_TOKENIZER
started: 2026-05-27 20:21 local time
variants: ema_style_vocab_factorized_w36, ema_style_vocab_factorized_w40_stylepush
output: exp\vae_backend_256_probe
status logs: exp\factorized_tokenizer_status\stdout.log / stderr.log
```

First health check after launch:

```text
variant: ema_style_vocab_factorized_w36
progress: epoch 3/8
GPU: ~8260 / 12288 MiB, 100% util
status: finite; no OOM/non-finite observed
```

---

## Experiment Note: 2026-05-27 Post-VAE EMA Verdict

**Status**: Evidence review
**Scope**: Remote post-VAE result CSVs under `exp/vae_backend*` and tokenizer
probe outputs.

### Conclusion

EMA is not fundamentally unusable. The current evidence shows two regimes:

- high-style EMA can reach the target band but sacrifices content:
  `ema_guard_w20_lowwarp e7 = clip_style 0.7245 / LPIPS 0.5526`;
- content-safe EMA is close but still short on style:
  `ema_transport_texton_w34_guard e6 = 0.7145 / 0.4826`,
  `ema_bodyblend_w28_guard e6 = 0.7158 / 0.4972`.

SDXL and KL-f4 are currently lower-style in the same post-VAE exploration:
SDXL is around `0.667` in the stable minimal line, and KL-f4 is around `0.654`.

### Important Boundary

Do not use the old pre-VAE-switch t00/t01 numbers as a clean EMA-vs-MSE A/B.
They are useful historical capacity evidence for the old system, but not proof
that the current EMA backend is worse or better than MSE.

### Per-Style Diagnosis

The current EMA problem is not global. Strong content-safe EMA rows already
reach high target-style means for Van Gogh, Cezanne, and Monet, but Hayao stays
near `0.66-0.67` and often has worse LPIPS. This points to a missing macro
flat-color / contour operator rather than a dead VAE.

### Tokenizer Probe

The tokenizer probe completed:

- `ema_style_vocab_texton_w34`: best `0.7084 / 0.5144`;
- `ema_style_vocab_hayao_w36`: best `0.7068 / 0.5445`.

Verdict: the first factorized-tokenizer readout is too weak and too generic.
The route remains conceptually valid, but it must be paired with a stronger
macro carrier rather than treated as a post-hoc style table.

### Next Hypothesis

Use EMA as the main post-VAE backend. Start from a content-safe carrier and add
explicit macro flat-color and edge-contour branches for Hayao. Continue to use
Seedream only as a diagnostic visual reference, not as a training objective.

---

## Experiment 000: armored_breakthrough_8ep_sinkhorn_baseline

**Status**: Planned
**Date**: 2026-05-19
**Config**: `exp/armored_breakthrough_proper/config_sinkhorn_baseline.json`

### Hypothesis
Sinkhorn alone (no SWD increase) should reproduce baseline Style (~0.703) with better LPIPS (~0.44-0.46). This is the control: measure the LPIPS improvement from Sinkhorn alone.

### Config Delta from Baseline (config.json)
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sinkhorn_baseline"
  }
}
```

### Predictions
- clip_style: 0.700-0.708 (slight regression from softmax)
- content_lpips: 0.44-0.46 (improvement from softmax)

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 2026-05-28: vae_backend_256_mse_controls

**Status**: Completed  
**Date**: 2026-05-28  
**Output**: `exp/vae_backend_256_mse_controls/vae_backend_256_results.csv`

### Hypothesis
If the current EMA carriers are merely mismatched to the EMA latent statistics,
then cloning the same architectures/losses/schedules onto the original MSE
backend should give a clean style lift without worsening LPIPS.

### Controlled Delta
Only the VAE backend and latent root changed:

```text
vae_model = mse
latent_root = latent-256
```

### Best Results

| variant | best epoch | clip_style | content_lpips | EC | readout |
|---|---:|---:|---:|---:|---|
| `mse_plain4_w20_anchor` | 6 | `0.703597` | `0.419917` | `0.408145` | small gain |
| `mse_dynamic_guard_w28` | 6 | `0.710273` | `0.446022` | `0.393476` | small gain |
| `mse_transport_texton_w34_guard` | 6 | `0.718588` | `0.483008` | `0.371505` | best content-safe MSE row |
| `mse_bodyblend_w28_guard` | 6 | `0.715295` | `0.485741` | `0.367846` | LPIPS improves, style does not |
| `mse_guard_w20_lowwarp` | 7 | `0.725233` | `0.553443` | `0.323858` | high style, over LPIPS budget |

### Verdict
MSE is a useful control and slightly helps the texton carrier, but it does not
solve the target. No MSE row satisfies `clip_style > 0.72` with
`content_lpips < 0.50`. The next mainline should remain architecture/tokenizer
work, using `mse_transport_texton_w34_guard` as a diagnostic near-miss rather
than promoting MSE as the default backend.

---

## Experiment 004: memory_residual_backbone_probe

**Status**: Completed diagnostic  
**Date**: 2026-05-28  
**Path**: `exp/memory_residual_backbone_probe`

### Hypothesis
If reference-memory works because it supplies a concrete style source, an
id-only routed prototype bank should help once it bypasses the frozen style-map
interface and enters the trainable actuator as a separate residual field.

### Config
```text
rs00_memory_residual_s22_e2:
  adapter = mr00_residual_hightex_k4_s22
  residual = routed_memory - base_map
  residual_strength = 0.22
  trainable = body_blocks, blender, skip_fusion, decoder_blocks, dec_post, dec_mod, output_head

rs01_memory_residual_hp_s32_e2:
  adapter = mr01_residual_hightex_k4_hp_s32
  residual = highpass(routed_memory - base_map), support gated
  residual_strength = 0.32
  trainable = same as rs00
```

### Results
```text
rs00: clip_style=0.707073, content_lpips=0.432237, Hayao=0.606759
rs01: clip_style=0.707358, content_lpips=0.429501, Hayao=0.606127
ag02: clip_style=0.710955, content_lpips=0.407269, Hayao=0.605668
```

### Verdict
Negative. The residual branch is active and finite, but it lowers global style
and worsens content relative to ag02. This rejects untyped prototype residual
injection and residual/highpass strength sweeps.

---

## Experiment 2026-05-28: reference_memory_generation_probe

**Status**: Completed diagnostic  
**Artifacts**: `exp/reference_memory_generation_probe_full`

### Hypothesis

If the m02/tokenizer plateau is caused by a centroid-like id-only style source,
then using an internal target-style latent as the runtime style source should
lift CLIP style before any loss or backbone change.

### Protocol

- checkpoint/style adapter frozen;
- no training, no loss modification;
- patched inference plumbing to pass `target_style_latent` into the existing
  runtime source path;
- 750 generated images per recipe, evaluated with reuse-generated full-eval.

### Results

| id | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `rm00_random_ref1` | 0.715127 | 0.477313 | 0.628046 | strong source-quality diagnostic |
| `rm01_lowfreq_match_k8` | 0.715447 | 0.477220 | 0.627415 | selector not decisive |

### Interpretation

The backbone can use real target-style source features. The missing piece is
not another scalar loss knob; it is a better tokenizer-controlled style source.
The next model change should internalize this as a local multi-prototype memory
bank rather than requiring an exemplar/reference image during final inference.

---

## Experiment 2026-05-28: tokenizer prototype carrier pc00/pc01

**Status**: Completed diagnostic  
**Config**: `exp/tokenizer_prototype_carrier_calibration`

### Hypothesis
If the tokenizer plateau is caused by residual source mismatch, a zero-start
carrier sourced from the style-routed `style_feat` should add visible textons
while preserving the m02 content anchor.

### Results
| run | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `pc00_m02_prototype_carrier_anchor` | 0.710468 | 0.406890 | 0.605174 | safe but below ag02 |
| `pc01_m02_prototype_carrier_push` | 0.710299 | 0.407741 | 0.605220 | stronger carrier worsens aggregate metrics |

### Verdict
Prototype-carrier energy is nonzero, but generic `style_feat` is not a
style-discriminative source. Do not add more amplitude to this branch.

---

## Experiment 2026-05-28: style memory bank mb00/mb02

**Status**: Completed diagnostic  
**Config**: `tools/experiments/run_style_memory_bank_probe.py`  
**Results**: `exp/style_memory_bank_probe/style_memory_bank_results.csv`

### Hypothesis
If m02 is style-limited by an averaged learned style source, blending
`style_spatial_id_16` with training-set body-feature prototypes should lift
CLIP style before any tokenizer/loss change.

### Results
| run | source | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---|---:|---:|---:|---|
| `mb00_body_mean_blend25` | 25% mean body prototype | 0.710612 | 0.407762 | 0.604948 | safe but below m02/ag02 |
| `mb02_body_exemplar_blend35` | 35% high-texture exemplar | 0.710516 | 0.409222 | 0.606350 | Hayao slightly up, LPIPS worse, global style flat |

### Verdict
Single-map memory replacement is not enough. Mean prototypes are too smooth;
single exemplars add texture but hurt content and do not lift global style.
If memory is pursued, it must be token-selected multi-prototype routing rather
than one blended `style_spatial_id_16` map.

---

## Experiment 2026-05-28: style_memory_bank_adapter_probe bm00/bm01/bm02

**Status**: Completed diagnostic / rejected as mainline  
**Config**: `tools/experiments/run_style_memory_bank_adapter_probe.py`  
**Results**: `exp/style_memory_bank_adapter_probe/style_memory_bank_adapter_results.csv`  
**Analysis**: `exp/analysis/style_memory_bank_adapter_probe_20260528/analysis.md`

### Hypothesis

If explicit reference latents work because they supply real target-style source
features, an id-only adapter-side multi-prototype bank should recover part of
that style lift without a test-time reference.

### Protocol

- checkpoint and `m02_embspatial_highpass_style` adapter frozen;
- no loss modification and no backbone training;
- build `style_memory_bank_16` from internal training-set body features;
- evaluate 750 generated images per recipe with reuse-generated full-eval.

### Results

| run | source | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---|---:|---:|---:|---|
| `bm00_hightex_k4_blend65` | 4 high-texture prototypes, blend 0.65 | 0.710854 | 0.407380 | 0.605454 | safe but below ag02 |
| `bm01_diverse_k4_blend65` | 4 low-frequency diverse prototypes, blend 0.65 | 0.710676 | 0.407408 | 0.605344 | diversity does not help |
| `bm02_hightex_k4_boost_blend75` | high-texture prototypes, boost 1.12, blend 0.75 | 0.710693 | 0.407397 | 0.605188 | stronger texture does not help |

### Verdict

The static id-only bank is a clean negative result. The prototypes have real
high-texture energy and the adapter path is active, but global style-level
mixtures still collapse into the m02 plateau. The reference-memory gain needs
local/content-conditioned routing, not a larger `style_spatial_id_16` centroid.
Do not push blend/boost/loss on this design.

---

## Experiment 2026-05-28: local route memory bank br00/br01

**Status**: Completed diagnostic / rejected as mainline  
**Config**: `tools/experiments/run_style_memory_bank_adapter_probe.py`  
**Results**: `exp/style_memory_bank_adapter_route_probe`  
**Analysis**: `exp/analysis/style_memory_bank_adapter_route_probe_20260528/analysis.md`

### Hypothesis

Static style-id mixtures failed because they collapse prototypes into one
centroid. Local content-conditioned dictionary attention over prototype tokens
could preserve the sample-local geometry that made reference-memory useful.

### Results

| run | route | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---|---:|---:|---:|---|
| `br00_route_hightex_k4_s45` | high-texture k4, route strength 0.45 | 0.710530 | 0.407402 | 0.604825 | below ag02 |
| `br01_route_hightex_k4_s65` | high-texture k4, route strength 0.65 | 0.710609 | 0.407408 | 0.604584 | below ag02; Hayao worse |

### Verdict

The route is active but too late/too weak: inserting a content-routed map into
the old frozen style-map interface does not recover the reference-memory lift.
The next valid route is not temperature/strength sweeping; it is a trainable
router-aware actuator or a backbone phase with the router present from the
start.

---

## Experiment 2026-05-28: tc00/tc01 tokenizer texton carrier

**Status**: Completed diagnostic / rejected as mainline  
**Date**: 2026-05-28  
**Launcher**: `tools/experiments/run_tokenizer_adain_gate_calibration.py`

### Hypothesis
The m02 tokenizer plateau is not caused by token values or scalar loss pressure.
It is caused by a missing high-frequency carrier: g56 made `grammar[5]`
executable, but the native high residual path stayed almost dead. Add a
zero-start token-selectable texton residual, train only tokenizer fields plus
that carrier, and keep m02 as the teacher anchor.

### Config Delta
```json
{
  "model": {
    "style_token_texton_carrier_enable": true,
    "style_token_texton_carrier_strength": "0.16 for tc00, 0.22 for tc01",
    "style_token_texton_carrier_hidden_mult": "0.75 for tc00, 0.85 for tc01",
    "style_token_texton_carrier_tanh_scale": 0.42,
    "style_token_grammar_texture_enable": true,
    "style_token_adain_gate_enable": true
  },
  "freeze_policy": "freeze m02 backbone/style_emb/style_spatial; train grammar_vocab, band_vocab, token_texton_carrier_mapper only"
}
```

### Decision Gate
- promote only if the first-class grid is not hazy and style rises above
  `ag02=0.710955`;
- strong candidate only if it approaches `0.72` while LPIPS stays below `0.50`;
- reject immediately if it reproduces the factorized-token haze failure.

### Results
```text
tc00_m02_texton_carrier_anchor: 0.7104315019 clip_style / 0.4073038044 LPIPS / Hayao 0.6047670975
tc01_m02_texton_carrier_push:   0.7106213341 clip_style / 0.4069454408 LPIPS / Hayao 0.6058331524
```

Both grids were non-hazy and structurally close to m02, but neither beat
`ag02_m02_g56_texture_anchor = 0.7109550617 / 0.4072693332`.

### Verdict
Simple zero-start residual carrier is not enough. The failure is not numerical
instability or haze; it is representational. The carrier is still derived from
content high bands and the same AdaIN residual, so it cannot inject a genuinely
new style texton source. Next carrier must be style-routed/prototype-based.

---

## 2026-05-27 Tokenizer No-Prior Correction

**Status**: Implemented code-level correction; no new GPU run yet.

### Trigger

The first style-tokenizer runs used structured fields, but the tokenizer
initialization also contained hand-coded Hayao / Van-Gogh style priors. That is
not the intended tokenizer problem. The intended problem is to give the model a
field schema and let the vocabulary values be learned.

### Evidence Before Correction

Remote tokenizer probes:

| run | best global clip_style | content_lpips | Hayao cross clip_style | verdict |
|---|---:|---:|---:|---|
| `ema_style_vocab_texton_w34` | 0.7084 | 0.5144 | 0.6429 | below anchor, weak Hayao |
| `ema_style_vocab_hayao_w36` | 0.7069 | 0.5445 | 0.6531 | mild Hayao lift but LPIPS worse |

The debug readout also showed that the field responses were small and generic.
This is not a clean test of tokenizer theory because the vocabulary started
with manual style priors.

### Code Change

- `src/style_tokenizer.py`: remove all manual per-style grammar/band priors.
- `src/lancet_blocks.py`: make the token flatten response signed and
  differentiable at zero, so neutral zero initialization is still trainable.
- `docs/maths/16_tokenizer_no_prior_spiral.md`: record the no-prior tokenizer
  spiral protocol.

### Current Hypothesis

Tokenizer progress should be spiral-shaped:

```text
neutral tokenizer backbone -> vocabulary-only refinement -> actuator diagnosis
-> revised backbone -> refined vocabulary
```

The next backbone run should not be judged only by one global average. It must
report per-target and cross-target metrics, especially Hayao. Success requires
Hayao to become a learned field pattern, not a manually initialized pattern.

### Follow-up Correction

The tokenizer mainline also avoids manual Hayao training pressure. The clean
queue is:

- `ema_style_vocab_neutral_w34`
- `ema_style_vocab_neutral_w36_stylepush`

Both use balanced style exposure. Hayao is evaluated separately because it is
the clearest failure slice, but it is not oversampled or upweighted in the
main tokenizer spiral.

### 2026-05-27 Tokenizer Metric-Space / Operator-Binding Update

**Status**: Local code support added; remote projector refit still running.

The tokenizer objective has been reframed as representation learning rather
than only stronger backbone control. A good tokenizer must be:

- algebraically orthogonal across `identity`, `grammar`, and `band`;
- bound to distinct operator families;
- diagnosed against frequency-separated training-data measures.

Code support added:

- `tools/experiments/diagnose_style_token_metric_space.py` now reports
  token/data distance correlations for full, low, high, abs-high, and
  low/mid/high energy-ratio distances.
- `src/lancet_backbone.py` now supports
  `dynamic_style_operator_mode="factorized_token"`, binding:
  - `identity` to pointwise `1x1` channel mixing and bias;
  - `grammar` to depthwise `3x3` spatial kernels;
  - `band_gains` to direct low/mid/high residual-band scaling.
- `run_vae_backend_256_probe.py` gained:
  - `ema_style_vocab_factorized_w36`;
  - `ema_style_vocab_factorized_w40_stylepush`.

This does not use Seedream or any external generated image as supervision.
Seedream remains diagnostic-only.

### 2026-05-27 Operator-Bound Tokenizer Cleanup

**Status**: Local implementation updated; remote factorized run that was already
active keeps running on the previously synced snapshot.

The old tokenizer projector path is now retired from the runnable tokenizer
implementation. `StyleTokenizer` no longer builds `code_projector`, no longer
stores `project_code`, and no longer concatenates `base_code`, `identity`,
`grammar`, and `band_logits` in `forward`. It returns `StyleTokenFields` plus
the deterministic residual base code only.

Reason: the projector experiment was a negative representation result. It can
lower LPIPS, but it re-mixes fields through an anonymous `style_code` and does
not produce a disentangled metric space. Future tokenizer runs should use
operator-bound fields or explicit field-statistic losses.

### 2026-05-27 Factorized w36 Result And Feature-Operator Follow-Up

**Status**: w36 and w40 completed on remote.

`ema_style_vocab_factorized_w36` completed epochs 6/7/8. Best result is not a
style solution:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.665652 | 0.323192 | 0.450519 |
| 7 | 0.666157 | 0.324339 | 0.450096 |
| 8 | 0.665982 | 0.324816 | 0.449660 |

Per-style epoch 8 confirms Hayao is still the weak slice:

| target | clip_style | LPIPS |
|---|---:|---:|
| Hayao | 0.586460 | 0.375123 |
| monet | 0.634582 | 0.296655 |
| vangogh | 0.662697 | 0.300282 |
| cezanne | 0.648391 | 0.312260 |

Diagnosis: final-head-only factorized binding strongly preserves content but
does not provide enough style actuator capacity. Component scorecard:
`coverage=0.500`, `component=0.598`, grammar active rows `4`, band active rows
`0`. Metric-space diagnosis shows useful grammar/abs-high alignment
(`spearman=0.6121`) but weak identity-low alignment (`0.0000`), inverted
band-energy alignment (`-0.4545`), and high cross-covariance across fields.

Code follow-up added locally: `dynamic_style_feature_operator`, a decoder
feature-level factorized operator using the same `StyleTokenFields`. The probe
variant is `ema_style_vocab_factorized_feature_w36`.

`ema_style_vocab_factorized_w40_stylepush` confirms the same failure mode:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.665316 | 0.321302 | 0.451549 |
| 7 | 0.665672 | 0.322517 | 0.450982 |
| 8 | 0.665615 | 0.323082 | 0.450567 |

W40 component scorecard is worse than W36 (`coverage=0.375`,
`component=0.500`, band active rows `0`). Metric-space diagnosis improves
grammar vs abs-high correlation (`spearman=0.7576`) but keeps identity-low at
`0.0000` and band-energy inverted (`spearman=-0.4545`). Raising terminal SWD
therefore does not solve the tokenizer; the actuator level and band supervision
are the blockers.

`ema_style_vocab_factorized_feature_w36` also completed and is a negative
result:

| epoch | clip_style | content_lpips | EC |
|---:|---:|---:|---:|
| 6 | 0.663724 | 0.325178 | 0.447895 |
| 7 | 0.664565 | 0.327200 | 0.447120 |
| 8 | 0.664501 | 0.327682 | 0.446756 |

The feature operator increased train-time velocity (`|v|` rose to about
`0.169`) but did not convert that motion into style. Component scorecard:
`coverage=0.500`, `component=0.544`, band active rows `0`. Metric-space
diagnosis degraded versus final-head-only: full-style isometry fell to
`spearman=0.0182`, grammar/high stayed weak (`0.2364`), and identity-low stayed
zero. Conclusion: actuator placement alone is insufficient. The next mainline
change should make the fields identifiable with data-derived field losses
before adding more operator capacity.

### 2026-05-27 Correction: Do Not Promote Hazy Factorized Tokenizer Runs

Visual review rejects the factorized output/feature runs as a mainline path.
They are content-safe but visibly hazy and de-stylized, with global style near
`0.665`, far below the style-normal adapter anchor.

Active anchor is restored to:

| anchor | clip_style | content_lpips | EC |
|---|---:|---:|---:|
| `m02_embspatial_highpass_style` | 0.71073 | 0.40735 | 0.84967 |

The local field-loss edit proposed after the factorized diagnosis was reverted
before launch and did not remain in `src/config_schema.py` or `src/losses.py`.
The remote scheduled tasks `LANCET_FACTORIZED_TOKENIZER` and
`LANCET_FACTORIZED_FEATURE` were disabled to prevent the hazy route from
restarting.

Updated rule: tokenizer research must be evaluated against the m02 visual
style gate first. Do not change the main OMF loss or promote a tokenizer route
unless it preserves visible style; low LPIPS alone is not a success.

### 2026-05-27 Tokenizer Band-Gate Calibration

Hypothesis:

```text
Freeze the texton backbone; train only tokenizer.band_vocab as the low/mid/high
texton carrier valve.
```

This was a tokenizer-only stage in the tokenizer/backbone spiral. It did not
modify the main OMF loss and did not bind tokenizer fields to the output head.
The execution surface is the existing
`StyleBlender._style_texton_band_allocation`, where `style_tokens.band_gains`
multiply low/mid/high texton deltas.

Script:

```text
tools/experiments/run_tokenizer_bandgate_calibration.py
```

Planned source checkpoint:

```text
exp\vae_backend\ema_transport_texton\ema_transport_texton_w34_guard\epoch_0006.pt
```

Reason: this texton checkpoint is the strongest style-normal backbone with an
explicit texton carrier (`0.71451 / 0.48261 / 0.36968` from prior eval). The
m02 adapter remains the LPIPS anchor, but it was produced on the AdaIN carrier
and does not expose the texton band valve.

Recipes:

| recipe | trainable parameters | purpose |
|---|---|---|
| `bg00_band_anchor` | `style_tokenizer.band_vocab.weight` only | conservative band-coordinate fit |
| `bg01_band_stylepush` | `style_tokenizer.band_vocab.weight` only | stronger style push with teacher anchor |

Results:

| recipe | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `bg00_band_anchor` | 0.71289 | 0.44403 | 0.60185 | safe but style-neutral |
| `bg01_band_stylepush` | 0.71264 | 0.44406 | 0.60096 | safe but style-neutral |

CSV copied locally:

```text
exp\tokenizer_bandgate_calibration\tokenizer_bandgate_results.csv
```

Decision gate: passed the metric style gate, but not enough to become a style
actuator. It is not the hazy `factorized_*` failure mode, but it also does not
increase style. Keep it as a diagnostic/safety coordinate only.

Rollback decision after visual/style review: the active style-normal anchor is
still `m02_embspatial_highpass_style` (`0.71073 / 0.40735 / 0.84967`). Do not
promote the factorized output/feature routes; they are hazy negative controls.
Do not change the main OMF loss to compensate for tokenizer weakness.

### 2026-05-27 Tokenizer-Gated Transport-AdaIN Plan

Hypothesis:

```text
Tokenizer should not replace the output head; it should act as a low-rank
valve over the proven m02 transport-AdaIN carrier.
```

Code changes:

- `model.style_token_adain_gate_enable` defaults to `false`;
- when enabled, `transport_adain` reads `style_tokens.band_gains` and multiplies
  low/mid/high AdaIN residual bands;
- when enabled, `style_tokens.grammar` also controls the existing flat-region
  high-pass suppression path;
- no main OMF loss changes.

Script:

```text
tools/experiments/run_tokenizer_adain_gate_calibration.py
```

Source:

```text
checkpoint: exp\vae_backend\ema_transport_moment\ema_transport_adain_w34_guard\epoch_0006.pt
init adapter: exp\style_embedding_mainline_calibration\ema_transport_adain_w34_e6_fulltrain\m02_embspatial_highpass_style\style_adapter.pt
```

Trainable parameters:

```text
style_tokenizer.grammar_vocab.weight
style_tokenizer.band_vocab.weight
```

Recipes:

| recipe | purpose |
|---|---|
| `ag00_m02_safe_gate` | conservative tokenizer gate over m02 |
| `ag01_m02_style_gate` | stronger tokenizer gate, acceptable only if style rises without fog |

Decision gate: reject immediately if the grid becomes hazy or global style
falls below the m02 style-normal level. Keep only if global style moves toward
`0.72+` or Hayao cross-style improves without LPIPS leaving the `0.47-0.50`
working band.

Result on 2026-05-27:

| recipe | clip_style | content_lpips | Hayao clip_style | verdict |
|---|---:|---:|---:|---|
| `ag00_m02_safe_gate` | 0.71076 | 0.40728 | 0.60489 | style-normal but neutral |
| `ag01_m02_style_gate` | 0.71061 | 0.40729 | 0.60514 | style-normal but neutral |

Interpretation: the tokenizer-gated AdaIN path is safe, but it does not lift
style. It should not be promoted as a breakthrough. The only active anchor is
still `m02_embspatial_highpass_style` (`0.71073 / 0.40735 / 0.84967`).

Correction after user review: do not pivot this into a scalar-loss edit. The
tokenizer question is whether fields are executable on top of the style-normal
carrier. Hazy/de-stylized factorized routes remain hard negative controls, not
candidate baselines. Remote scheduled tasks for the old tokenizer/factorized
queues were stopped and disabled:

```text
LANCET_FACTORIZED_FEATURE
LANCET_FACTORIZED_TOKENIZER
LANCET_STYLE_VOCAB_TOKENIZER
LANCET_STYLE_VOCAB_NEUTRAL_SPIRAL
LANCET_TOKENIZER_ADAIN_GATE
LANCET_TOKENIZER_BANDGATE
LANCET_TOKENIZER_PROJECTOR_REFIT
LANCET_TOKENIZER_VOCAB_REFIT
```

Next action is diagnostic only: compare tokenizer field movement and endpoint
sensitivity around the m02 anchor before changing backbone or loss.

### 2026-05-28 Tokenizer Spiral Registry

User requirement update: every tokenizer/backbone spiral experiment now needs a
structured document and CSV row stating configuration, purpose, result metrics,
what was verified, and what adjustment follows.

New registry files:

```text
docs/logs/tokenizer_spiral_experiment_registry.md
docs/logs/tokenizer_spiral_experiment_registry.csv
```

These files are now the compact decision ledger for the tokenizer spiral. The
long-form `experiment_ledger.md` keeps narrative context; the registry keeps
row-level experiment decisions.

### 2026-05-28 Stat-Initialized Tokenizer Probe

One-line hypothesis:

```text
If the tokenizer is a style coordinate system, measured training-pool frequency
coordinates should already move the m02 carrier without training the token values.
```

Script:

```text
tools/experiments/run_tokenizer_stat_vocab_probe.py
```

Configuration:

- checkpoint: `exp/vae_backend/ema_transport_moment/ema_transport_adain_w34_guard/epoch_0006.pt`
- init adapter: `exp/style_embedding_mainline_calibration/ema_transport_adain_w34_e6_fulltrain/m02_embspatial_highpass_style/style_adapter.pt`
- latent root: `I:\Github\Latent_Style\latent-256-sd15-ema`
- no optimization; map per-style latent statistics into tokenizer `grammar` and
  `band_vocab`
- two scales: `sv00_stat_m02_conservative`, `sv01_stat_m02_balanced`

Results:

| recipe | clip_style | content_lpips | Hayao cross style | verdict |
|---|---:|---:|---:|---|
| `sv00_stat_m02_conservative` | 0.710740 | 0.407393 | 0.605003 | safe but neutral |
| `sv01_stat_m02_balanced` | 0.710551 | 0.407434 | 0.604945 | safe but neutral |

Visual gate: first-grid review is not hazy and remains in the m02 visual
family.

Verified claim: the problem is not just bad token numeric values. The measured
token coordinates barely move the endpoint, so the m02 carrier is under-reading
the tokenizer.

Adjustment: freeze the stat vocabulary and m02 backbone, then train only a
small zero-initialized `token_reader` attached to the existing transport-AdaIN
band allocator.

### 2026-05-28 Stat Token Reader Probe

One-line hypothesis:

```text
Freeze the stat tokenizer and the m02 carrier; train only a zero-initialized
reader so the carrier learns how to interpret measured style coordinates.
```

Code changes:

- `model.style_token_reader_enable`, `style_token_reader_hidden`,
  `style_token_reader_scale`
- `StyleBlender.token_reader`, default off and zero-initialized
- `tools/experiments/run_tokenizer_stat_reader_probe.py`

The switch defaults to off. When on, the reader consumes
`identity + grammar + band_logits` and only modulates the existing low/mid/high
transport-AdaIN residual gains. It does not replace the output head and does
not change the main OMF loss.

Current result:

| recipe | clip_style | content_lpips | Hayao cross style | verdict |
|---|---:|---:|---:|---|
| `sr00_stat_reader_safe` | 0.710698 | 0.405284 | 0.604194 | safe but style-neutral |
| `sr01_stat_reader_style` | 0.710523 | 0.402585 | 0.604742 | safe but style-neutral |

`sr00` and `sr01` first-grid outputs are not hazy and remain in the m02 family,
but neither moves style. Treat the reader probe as a safety-control result, not
a breakthrough.

Correction after user review: do not turn this into a scalar-loss route. The
active rollback anchor is still `m02_embspatial_highpass_style`
(`0.71073 / 0.40735 / 0.84967`). The rejected factorized routes remain hard
negative controls because their low LPIPS came from haze/de-stylization. The
next tokenizer work must diagnose representation/readout around m02 before any
new training objective is considered.

### 2026-05-28 m02 Operator Binding g56 Diagnostic

One-line hypothesis:

```text
grammar[5]/grammar[6] should become executable style coordinates if they are
hard-bound to the existing m02 transport-AdaIN mid/high residual gains.
```

Implementation:

- added `model.style_token_grammar_texture_enable`, default `false`;
- added `model.style_token_grammar_texture_scale`, default `0.35`;
- `StyleBlender` now optionally maps `grammar[5]` to mid residual gain and
  `grammar[6]` to high residual gain inside `transport_adain`;
- main OMF loss is unchanged;
- diagnostic script:
  `tools/experiments/diagnose_m02_tokenizer_operator_binding.py`.

Planned diagnostic id: `m02_operator_binding_g56`.

Decision gate:

- promote only if grammar mid/high perturbations create measurable mid/high
  endpoint motion without a low-band fog response;
- reject if band-low remains the only strong route;
- next stage, if passed, is tokenizer-only training with m02 frozen.

Result:

| readout | endpoint response |
|---|---:|
| strongest `band_low` perturbation | `0.00730` RMS |
| strongest `grammar_mid_texton` perturbation | `0.00543` RMS |
| `stat_vocab_preview` range | `0.00493-0.00603` RMS |
| `grammar_high_texture` perturbation | about `0.0002` RMS |

Verdict: partial pass. `grammar[5]` now has a real mid-texton actuator, close
enough to `band_mid` to justify a tokenizer-only run. `grammar[6]` remains weak,
so the high-texture path is not solved. Next action is `ag02/ag03`: freeze m02
and train only grammar/band through the g56 binding. Reject immediately if the
first grid becomes foggy.

Tokenizer-only g56 results:

| recipe | clip_style | content_lpips | Hayao cross style | verdict |
|---|---:|---:|---:|---|
| `ag02_m02_g56_texture_anchor` | 0.710955 | 0.407269 | 0.605668 | safe but marginal |
| `ag03_m02_g56_texture_push` | 0.710725 | 0.407305 | 0.605254 | pressure does not help |

Conclusion: g56 is stable and not foggy, but the gain is too small. The active
bottleneck is not tokenizer field values or scalar loss pressure; it is that
the high-frequency carrier remains weak. A next attempt should add or expose a
real high-texture carrier, then train tokenizer fields against it with m02
frozen.

### 2026-05-28 Prototype-Carrier Follow-up

Texton carrier result:

| recipe | clip_style | content_lpips | Hayao cross style | verdict |
|---|---:|---:|---:|---|
| `tc00_m02_texton_carrier_anchor` | 0.710431 | 0.407304 | 0.604767 | safe but below m02/ag02 |
| `tc01_m02_texton_carrier_push` | 0.710621 | 0.406945 | 0.605833 | safe but no global style lift |

Verified claim:

```text
Adding a trainable carrier around content-high/AdaIN residuals is not enough;
the missing variable is likely carrier source/routing, not scalar amplitude.
```

Next experiment:

```text
pc00/pc01: freeze m02 backbone and style adapter, train grammar+band vocab plus
zero-start token_prototype_carrier_mapper sourced from style-routed style_feat.
```

This keeps Seedream out of the training path and does not alter the main loss.
Promotion gate is strict: beat `ag02_m02_g56_texture_anchor` visually and on
`clip_style`; reject immediately if grids are hazy/de-stylized.

---

## Experiment 001: armored_breakthrough_8ep_sinkhorn_sw60

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw60.json`

### Hypothesis (see docs/maths/03_predictive_models.md)
Sinkhorn routing protects LPIPS while terminal SWD weight at 0.60 pushes Style past 0.72.

### Config Delta from Baseline (config.json)
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.60,
    "w_kinetic": 1.0
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw60"
  }
}
```

### Predictions
- clip_style: 0.718-0.725
- content_lpips: 0.45-0.49

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 002: armored_breakthrough_8ep_sinkhorn_sw80

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw80.json`

### Hypothesis
Pushing SWD to 0.80 with kinetic at 0.5 should break the 0.72 barrier at the cost of slightly higher LPIPS.

### Config Delta from Baseline
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.80,
    "w_kinetic": 0.5
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw80"
  }
}
```

### Predictions
- clip_style: 0.725-0.735
- content_lpips: 0.48-0.54

### Results
(TBD)

### Verdict
(TBD)

---

## Experiment 003: armored_breakthrough_8ep_sinkhorn_sw30

**Status**: Planned  
**Date**: 2026-05-19  
**Config**: `exp/armored_breakthrough_proper/config_sw30.json`

### Hypothesis
Moderate SWD increase with Sinkhorn to find the "sweet spot" — the minimal SWD weight that pushes past 0.72.

### Config Delta
```json
{
  "model": {
    "semantic_attn_routing_mode": "sinkhorn"
  },
  "bridge": {
    "terminal_swd_weight": 0.30,
    "w_kinetic": 1.0
  },
  "training": {
    "num_epochs": 8,
    "save_interval": 8
  },
  "checkpoint": {
    "save_dir": "./exp/armored_breakthrough_proper/sw30"
  }
}
```

### Predictions
- clip_style: 0.710-0.718
- content_lpips: 0.44-0.47

### Results
(TBD)

### Verdict
(TBD)
