# Tokenizer Spiral Experiment Registry

Purpose: keep a durable, decision-oriented record for the tokenizer/backbone
spiral. Every row must state the configuration, purpose, result, what it
verified, and the next adjustment.

Target gate:

```text
clip_style > 0.73
content_lpips preferably < 0.50, ideally near 0.47
0.72 / 0.40 remains a strong Pareto point
```

Active visual anchor:

```text
m02_embspatial_highpass_style = 0.71073 clip_style / 0.40735 LPIPS / 0.84967 EC
```

Hard rejection rule: if a route becomes hazy, de-stylized, or collapses below
the m02 visible-style level, it is a negative control even when LPIPS improves.

## Current Read

The active rollback point is still `m02_embspatial_highpass_style`. Recent
tokenizer probes around that point were useful as diagnostics, but none should
replace the anchor:

- stat vocab values preserve the m02 visual family but barely move style;
- the tiny stat reader slightly improves LPIPS but does not raise style;
- factorized output/feature routes are rejected because the low LPIPS comes
  from haze and de-stylization.

Therefore the next mainline move is not a scalar loss edit. Tokenizer work
must first diagnose representation and reader executability around the m02
carrier, with a hard visual gate that rejects any foggy route before metric
chasing.

## Experiment Table

| id | status | config | purpose | result | verified | adjustment |
|---|---|---|---|---|---|---|
| `m02_embspatial_highpass_style` | active anchor | EMA transport-AdaIN W34 guard epoch 6 + adapter refit; backbone frozen during adapter calibration | restore style-normal visual quality with strong LPIPS | `clip_style=0.71073`, `LPIPS=0.40735`, `EC=0.84967` | safe content-preserving anchor; better than foggy tokenizer routes | keep as rollback anchor |
| `ema_style_vocab_factorized_w36` | rejected | hard-bound factorized token output head | test operator-orthogonal tokenizer binding | `0.66598 / 0.32482`, Hayao `0.58646` | LPIPS can improve by becoming hazy near-identity | do not promote; use as negative control |
| `ema_style_vocab_factorized_w40_stylepush` | rejected | factorized token output head with stronger style pressure | test whether more pressure fixes factorized route | `0.66562 / 0.32308`, Hayao `0.58484` | stronger pressure did not recover style | reject; no scalar-loss rescue |
| `ema_style_vocab_factorized_feature_w36` | rejected | feature-level factorized token operator | test whether placement, not head binding, caused failure | `0.66450 / 0.32768`, Hayao `0.58077` | feature injection still hazy and weak | reject; placement alone is not enough |
| `bg00_band_anchor` | diagnostic | texton backbone frozen; train only `band_vocab` | test low/mid/high texton valve | `0.71289 / 0.44403`, Hayao `0.60185` | band gate is safe but style-neutral | keep only as content-safety coordinate |
| `bg01_band_stylepush` | diagnostic | texton backbone frozen; stronger band stylepush | test whether band-only can lift style | `0.71264 / 0.44406`, Hayao `0.60096` | band-only is too low-rank for style | do not continue band-only sweeps |
| `ag00_m02_safe_gate` | diagnostic | m02 transport-AdaIN; train grammar+band tokenizer gate | test safe tokenizer valve over m02 | `0.71076 / 0.40728`, Hayao `0.60489` | safe but neutral | m02 carrier reads token weakly |
| `ag01_m02_style_gate` | diagnostic | m02 transport-AdaIN; stronger grammar+band gate | test stronger token gate over m02 | `0.71061 / 0.40729`, Hayao `0.60514` | stronger gate still neutral | diagnose field executability |
| `m02_tokenizer_sensitivity_full_light` | diagnostic | no training; perturb grammar/band around m02 | measure token Jacobian to endpoint | strongest response from `band_low`, `band_mid`; `band_high` near zero; grammar mostly dead | m02 reads low/mid band, not texture grammar | avoid token value tuning without reader |
| `sv00_stat_m02_conservative` | completed | no training; measured latent stats -> token fields; conservative scale | test if data-derived tokenizer coordinates are enough | `0.71074 / 0.40739`, Hayao `0.60500` | stat vocab is non-hazy but not executable enough | train a reader, not the token values |
| `sv01_stat_m02_balanced` | completed | no training; measured latent stats -> token fields; larger scale | test stronger stat coordinates | `0.71055 / 0.40743`, Hayao `0.60494` | larger stat coords still barely move output | reader bottleneck confirmed |
| `sr00_stat_reader_safe` | completed | stat vocab frozen; m02 frozen; train zero-init token_reader, safe weights | teach m02 to read measured coordinates without changing backbone/tokenizer | `0.71070 / 0.40528`, Hayao `0.60419` | safe grid and slightly better LPIPS, but style remains neutral | do not promote; reader-safe is useful only as a safety control |
| `sr01_stat_reader_style` | completed diagnostic | stat vocab frozen; m02 frozen; train token_reader with higher style pressure | test whether reader capacity can lift style | `0.71052 / 0.40258`, Hayao `0.60474` | LPIPS improves by staying close to m02, but style is still neutral | do not promote; stop reader/loss sweeps |
| `m02_operator_binding_g56` | completed diagnostic | no training; bind `grammar[5]`/`grammar[6]` to m02 transport-AdaIN mid/high residual gains | test whether tokenizer grammar has an executable texture operator | `grammar[5]` endpoint RMS up to `0.00543`; stat preview `0.00493-0.00603`; `grammar[6]` still near zero | g5 mid-texton is executable; high-texture path remains weak | train tokenizer fields through g56 while freezing m02 |
| `ag02_m02_g56_texture_anchor` | completed | m02 frozen; train grammar/band with g56 texture binding, conservative anchor | use executable g5 route to lift style without haze | `0.71096 / 0.40727`, Hayao `0.60567` | safe but only marginally above m02/ag01 | keep as evidence that g5 binding is stable, not sufficient |
| `ag03_m02_g56_texture_push` | completed | m02 frozen; train grammar/band with g56 texture binding, stronger HP-SWD/style pressure | test if stronger g5 texture pressure can approach `clip_style>0.72` | `0.71073 / 0.40730`, Hayao `0.60525` | extra pressure did not beat ag02 | stop scalar pressure here; need stronger high-frequency carrier |
| `tc00_m02_texton_carrier_anchor` | completed diagnostic | m02 frozen; train grammar/band plus zero-start token texton carrier, conservative carrier strength | test whether a real trainable mid/high texton carrier lifts style without fog | `0.71043 / 0.40730`, Hayao `0.60477` | non-hazy but below m02/ag02; simple residual carrier does not add useful style energy | reject as mainline; carrier source/routing, not amplitude, is the bottleneck |
| `tc01_m02_texton_carrier_push` | completed diagnostic | m02 frozen; same carrier with stronger HP-SWD and high carrier branch | test whether the new carrier can move toward `0.72+` while LPIPS remains below `0.50` | `0.71062 / 0.40695`, Hayao `0.60583` | stronger carrier recovers Hayao slightly but still below ag02 globally | reject simple texton-residual route; next carrier must be style-routed/prototype-based |
| `pc00_m02_prototype_carrier_anchor` | completed diagnostic | m02 frozen; train grammar/band plus zero-start prototype carrier sourced from style-routed `style_feat` | test whether missing style energy is a source/routing problem rather than residual amplitude | `0.71047 / 0.40689`, Hayao `0.60517` | carrier energy is nonzero and grids are safe, but it does not beat ag02 | reject style_feat-source carrier; next source must be explicit target-style memory/prototype bank |
| `pc01_m02_prototype_carrier_push` | completed diagnostic | same prototype carrier with stronger high-band allocation and slightly lower anchor | stress-test whether prototype source can move toward `clip_style>0.72` while LPIPS remains below `0.50` | `0.71030 / 0.40774`, Hayao `0.60522` | stronger prototype energy worsens aggregate style/LPIPS | stop strengthening this branch; build data-derived style memory source instead |
| `mb00_body_mean_blend25` | completed diagnostic | m02 adapter with `style_spatial_id_16` blended 25% toward training-set body-feature mean prototypes | test whether the learned m02 style source under-represents the real target style manifold | `0.71061 / 0.40776`, Hayao `0.60495` | safe but slightly below m02/ag02; averaged body source is too smooth | do not use single mean source |
| `mb02_body_exemplar_blend35` | completed diagnostic | m02 adapter with `style_spatial_id_16` blended 35% toward high-texture body-feature exemplar prototypes | test whether the mean prototype is too smooth and a texture atlas source is needed | `0.71052 / 0.40922`, Hayao `0.60635` | exemplar raises Hayao slightly but hurts LPIPS and not global style | single blended style map is insufficient; memory needs token-selected multi-prototype routing |
| `rm00_random_ref1` | completed diagnostic | m02 adapter frozen; generation uses one random internal target-style latent as `target_style_latent` reference source | test whether explicit internal style source breaks the id-only m02/tokenizer plateau | `0.71513 / 0.47731`, Hayao `0.62805` | reference source raises style and Hayao substantially, but changes protocol and risks reference imprinting | convert this into token-selected multi-prototype memory; do not promote exemplar-guided inference as final protocol |
| `rm01_lowfreq_match_k8` | completed diagnostic | m02 adapter frozen; target reference chosen from high-texture candidates by source low-frequency match | test whether content-compatible reference selection improves the reference-memory gain | `0.71545 / 0.47722`, Hayao `0.62742` | selector only adds `+0.00032` clip over random; source availability matters more than this global selector | learn local/token-selected routing instead of hand-picking one global reference |
| `bm00_hightex_k4_blend65` | completed diagnostic | m02 frozen; adapter-side id-only 4-prototype high-texture style bank, blend `0.65` | test whether internal prototypes recover the reference-memory lift without a test-time reference | `0.71085 / 0.40738`, Hayao `0.60545` | safe but below ag02; static style-id mixture collapses back to the m02 plateau | reject static bank; next source must be local/content-conditioned |
| `bm01_diverse_k4_blend65` | completed diagnostic | m02 frozen; id-only 4-prototype low-frequency diverse style bank, blend `0.65` | test whether prototype diversity fixes centroid smoothing | `0.71068 / 0.40741`, Hayao `0.60534` | diverse global mixture also fails to lift style | routing must be spatially adaptive, not one style-level mixture |
| `bm02_hightex_k4_boost_blend75` | completed diagnostic | m02 frozen; high-texture bank with highpass boost `1.12`, blend `0.75` | stress-test whether stronger prototype texture approaches rm01 | `0.71069 / 0.40740`, Hayao `0.60519` | stronger texture/blend still does not raise style | stop static-bank amplitude pushes; move prototype choice into actuator/routing |
| `br00_route_hightex_k4_s45` | completed diagnostic | m02 frozen; high-texture prototype bank selected by local content-token attention before the style-map actuator, route strength `0.45` | test whether local prototype routing recovers reference-memory lift without test-time reference | `0.71053 / 0.40740`, Hayao `0.60483` | route is active but below ag02; frozen style-map actuator absorbs it | reject route-only adapter; train router-aware actuator/backbone instead |
| `br01_route_hightex_k4_s65` | completed diagnostic | same local route with strength `0.65` | stress-test whether stronger local routing lifts style while LPIPS stays under `0.50` | `0.71061 / 0.40741`, Hayao `0.60458` | stronger route still below ag02 and weaker on Hayao | stop temperature/strength sweeps for this route-only design |
| `ra00_route_actuator_s45_e2` | completed diagnostic | fixed `br00` local router source; train body/blender/decoder/output head for 2 epochs | test whether the routed source only failed because the actuator was frozen | `0.71034 / 0.43584`, Hayao `0.61408` | Hayao rises but global style and LPIPS both worsen vs ag02 | old style-map interface is still too entangled; do not continue route-strength sweeps |
| `rs00_memory_residual_s22_e2` | completed diagnostic | high-texture K4 bank; `routed_memory - base_map` enters as body residual, strength `0.22`; train actuator 2 epochs | bypass the collapsed style-map interface with a separate source field | `0.70707 / 0.43224`, Hayao `0.60676` | residual is active and non-hazy but lowers global style hard | reject untyped memory residual; source must be style-field typed or OT-aligned |
| `rs01_memory_residual_hp_s32_e2` | completed diagnostic | same residual source with highpass kernel `5`, support gate gamma `4`, strength `0.32` | test if highpass-gated texton residual can add style without color fog | `0.70736 / 0.42950`, Hayao `0.60613` | cleaner than rs00 on LPIPS but still below ag02 by style | reject residual-strength/highpass sweeps; need semantic/contrastive prototype assignment |
| `rt00_typed_fet_s18_e2` | completed diagnostic | typed flat/edge/texton prototype bank; content-nearest routing; residual strength `0.18`; train actuator 2 epochs | test whether explicit flat/edge/texton roles fix untyped residual collapse | `0.70806 / 0.43225`, Hayao `0.60899` | typed gates are finite and active, but global style still drops below ag02 | reject typed/content-nearest residual; assignment must be style-measure aligned |
| `rt01_typed_fet_hp_s24_e2` | completed diagnostic | typed flat/edge/texton prototype bank with highpass support; residual strength `0.24`; train actuator 2 epochs | test whether highpass support adds texture without content drift | `0.70665 / 0.43126`, Hayao `0.60710` | highpass typed route loses more style while only slightly improving LPIPS | reject highpass/strength sweep for typed residual |
| `rt02_typed_uniform_s20_e2` | completed diagnostic | typed bank with uniform within-type routing; residual strength `0.20`; train actuator 2 epochs | test whether content-nearest selection inside each type caused style dilution | `0.70690 / 0.42789`, Hayao `0.60752` | uniform within type improves LPIPS vs rt00 but remains below ag02 on style | reject uniform typing; atoms need target-style OT/contrastive selection |
| `rt03_typed_uniform_hp_s24_e2` | completed diagnostic | typed uniform routing plus highpass support; residual strength `0.24`; train actuator 2 epochs | stress-test typed uniform highpass route | `0.70727 / 0.42896`, Hayao `0.60793` | best typed variant is still below ag02 and far below rm01 | stop typed residual variants; optimize prototype assignment against target style measure before injection |
| `mf00_fisher_k6_s20` | completed diagnostic | adapter-only Fisher style projection; 6 prototypes per style | test whether raw style descriptor inseparability caused style-average prototype selection | adapter only; prototype purity abs_mean `1.063` | Fisher projection makes atoms separable vs raw ma00 `0.0015` | promote only to consumer test |
| `mf01_typed_fisher_k9_s22` | completed diagnostic | adapter-only Fisher style projection plus flat/edge/texton roles | test whether typed atoms become style-discriminative after Fisher projection | adapter only; prototype purity abs_mean `0.913` | Fisher projection makes typed atoms separable vs raw ma01 `0.0043` | promote only to consumer test |
| `rf00_fisher_stylepure_s20_e2` | completed diagnostic | fixed mf00 source; train body/blender/skip_fusion/decoder/output head for 2 epochs | test whether separable style atoms recover reference-memory lift in the current residual actuator | `0.70785 / 0.43553`, Hayao `0.60823` | separable atoms still become non-style perturbation in residual dynamics | reject residual-bank consumption; use operator-bound style source next |
| `rf01_typed_fisher_s22_e2` | completed diagnostic | fixed mf01 typed Fisher source; train body/blender/skip_fusion/decoder/output head for 2 epochs | test whether typed roles plus Fisher projection fix typed residual collapse | `0.70663 / 0.43156`, Hayao `0.60671` | typed Fisher route is below ag02 and rt00 | stop residual-bank variants; bind style coordinates directly to executable operators |
| `fo00_fisher_operator_readout` | completed diagnostic | no training; Fisher axes aligned to measured mid/high energy and written into `band[0:3]`, `grammar[1,5,6,7]` | test whether discriminative Fisher coordinates are executable when bound directly to transport-AdaIN tokenizer fields | endpoint RMS `0.01036`, detail/low `1.160`, high fraction `0.430` | not pure low-frequency fog; `grammar[5]` perturbation is active | promote to tokenizer-only training; keep residual-bank branch closed |
| `fo01_fisher_operator_token_swd80` | completed diagnostic | Fisher-token init; train only tokenizer grammar/band for 80 iters per target style | test whether executable Fisher-bound scalar gates can lift style under SWD/HP-SWD with m02 anchor | `0.71030 / 0.40875`, Hayao `0.60461` | visually safe but below `ag02=0.71096 / 0.40727`; token values barely move | reject scalar-gate SWD sweeps; next tokenizer route needs a stronger operator than scalar band/grammar gates |
| `fo10_depthwise_operator_readout` | completed diagnostic | no training; `grammar[8:15]` controls a fixed 3x3 depthwise filter bank on transport-AdaIN detail residual | test whether grammar-bound spatial filters provide a higher-capacity executable operator | endpoint RMS `0.01087`, detail/low `1.261`, high fraction `0.432` | filter perturbations are detail-dominant but still small; preview is safe | promote to tokenizer-only training, then stop if metrics remain flat |
| `fo11_depthwise_filter_swd80` | completed diagnostic | Fisher-token init plus fixed depthwise filter bank; train only tokenizer grammar/band for 80 iters per target style | test whether spatial filter grammar breaks the m02/ag02 style plateau | `0.71030 / 0.40877`, Hayao `0.60472` | depthwise delta is active (`Hayao dw ~= 0.024`) but metrics are unchanged vs fo01 | stop tokenizer-only operator tuning; next spiral step is backbone/actuator consumption with tokenizer frozen |
| `fo12_depthwise_consumer_guard_e2` | pending | resume fo11 checkpoint; freeze Fisher/depthwise tokenizer and style identity; train body/blender/skip/decoder/output consumer for 2 epochs | test whether the depthwise operator needs a trained consumer rather than more tokenizer updates | pending | fo11 localized the bottleneck to frozen m02 consumption | if style rises without LPIPS/grid failure, continue consumer/tokenizer spiral; otherwise reject consumer-only route |

## 2026-05-28 Fisher Style-Measure Prototype Probe

Hypothesis:

```text
The raw body-feature style descriptor was too weakly separable, so prototype
assignment was effectively style-average. A Fisher discriminant projection
trained only from internal style labels should make style atoms separable.
```

Result:

```text
mf00_fisher_k6_s20 prototype purity abs_mean:       1.063
mf01_typed_fisher_k9_s22 prototype purity abs_mean: 0.913
raw ma00/ma01/ma02 purity abs_mean:                 0.0015-0.0044

rf00_fisher_stylepure_s20_e2: 0.707854 / 0.435530, Hayao 0.608229
rf01_typed_fisher_s22_e2:     0.706630 / 0.431558, Hayao 0.606712
ag02 tokenizer anchor:        0.710955 / 0.407269, Hayao 0.605668
```

Theory correction:

```text
Fisher projection fixes style atom separability, but separable atoms still fail
after being injected as a generic residual source. Therefore the bottleneck has
moved from atom selection to operator execution: selected style coordinates
must be bound to an executable flow/operator path, not consumed as an
unstructured memory residual.
```

Decision:

```text
Reject rf00/rf01 as mainline improvements. Do not continue memory residual
variants, even with better selection. The next valid spiral step is
operator-bound tokenizer consumption: use the discriminative style coordinate
to modulate an explicit low/mid/high or depthwise/pointwise operator, and run
a local actuator-readout diagnostic before full eval.
```

## 2026-05-28 Fisher Operator Tokenizer Probe

Hypothesis:

```text
Fisher style coordinates are only useful if they are directly bound to an
executable operator. Align Fisher axes to measured mid/high energy, write them
into transport-AdaIN tokenizer fields, and check local endpoint motion before
training.
```

No-training readout:

```text
fo00_fisher_operator_readout
endpoint_delta_rms: 0.01036
detail_over_low:    1.160
high_fraction:      0.430
grammar_mid perturb endpoint_rms: 0.00222
```

This passed the actuator gate: the movement was not just low/color drift, and
the `grammar[5]` mid-texton binding remained executable.

Tokenizer-only training:

```text
fo01_fisher_operator_token_swd80: 0.710301 / 0.408750, Hayao 0.604612
ag02_m02_g56_texture_anchor:      0.710955 / 0.407269, Hayao 0.605668
```

Theory correction:

```text
Direct operator binding is necessary but still not sufficient when the operator
is only a small set of scalar gates. The Fisher token initialization moves the
endpoint in a measurable mid/high direction, but SWD/HP-SWD training barely
updates those fields and does not raise aggregate style. The gradient path from
global style discrepancy to scalar band/grammar gates is too low-rank and too
indirect.
```

Decision:

```text
Reject scalar-gate Fisher tokenizer sweeps. Do not keep increasing SWD weight,
token scale, or iteration count on this branch. The next tokenizer route should
keep the successful rule--Fisher/physical coordinates must bind to an
executable operator--but replace scalar gates with a higher-capacity operator,
for example a depthwise spatial kernel for grammar or a direct style-measure
target for tokenizer fields.
```

## 2026-05-28 Depthwise Grammar Operator Probe

Hypothesis:

```text
If scalar grammar gates are too low-rank, bind grammar[8:15] to a fixed
depthwise 3x3 filter bank. This gives the tokenizer an explicit local spatial
operator without changing the main OMF loss or adding a trainable residual bank.
```

Readout:

```text
fo10_depthwise_operator_readout
endpoint_delta_rms: 0.01087
detail_over_low:    1.261
high_fraction:      0.432
```

Training result:

```text
fo11_depthwise_filter_swd80: 0.710301 / 0.408772, Hayao 0.604718
fo01_scalar_gate_swd80:      0.710301 / 0.408750, Hayao 0.604612
ag02 tokenizer anchor:       0.710955 / 0.407269, Hayao 0.605668
```

Theory correction:

```text
The new spatial operator is active but still does not improve style when only
the tokenizer is trained. This localizes the current bottleneck to the frozen
m02 backbone/actuator: it can tolerate token-controlled detail filters, but it
does not turn them into a stronger target-style measure. The next spiral step
should freeze the tokenizer and train the backbone/actuator consumer.
```

## 2026-05-28 Fisher Depthwise Consumer Probe

Hypothesis:

```text
fo11 failed because the frozen m02 actuator does not know how to consume the
new depthwise grammar operator. Keep the tokenizer fixed and train only the
consumer path; if this cannot move style, the issue is not tokenizer training
but the operator interface itself.
```

Launch row:

```text
fo12_depthwise_consumer_guard_e2
source checkpoint: exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/checkpoint_fisher_operator_tokenizer.pt
source adapter:    exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/style_adapter.pt
freeze:            style_emb, style_spatial_id_16, style_tokenizer
train:             body_blocks, blender, skip_fusion, decoder_blocks, dec_post, dec_mod, output_head
gate:              reject if style does not beat ag02 or LPIPS/grid worsens materially
```

Result:

```text
fo12_depthwise_consumer_guard_e2: 0.707308 / 0.433830, Hayao 0.609139
ag02 tokenizer anchor:           0.710955 / 0.407269, Hayao 0.605668
fo11 frozen-tokenizer source:     0.710301 / 0.408772, Hayao 0.604718
```

Theory correction:

```text
The depthwise grammar operator can be consumed if the backbone/actuator is
unfrozen, because Hayao rises from 0.6047 to 0.6091. However, training the full
consumer path is too broad: global clip_style falls and LPIPS worsens sharply.
The operator is being absorbed as repaint/content drift instead of style
semantics. Reject broad body/decoder consumer training; the next valid consumer
test is a narrow operator interface such as blender-only or a tiny zero-init
operator gate.
```

## 2026-05-28 Narrow Depthwise Consumer Interface Probe

Hypothesis:

```text
fo12 failed because the trainable consumer surface was too wide. If the
depthwise grammar operator is real, the smallest valid actuator is the
StyleBlender interface itself. Train only `blender` while freezing tokenizer,
body, skip, decoder, and output_head.
```

Launch row:

```text
fo14_depthwise_blender_only_e2
source checkpoint: exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/checkpoint_fisher_operator_tokenizer.pt
source adapter:    exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/style_adapter.pt
freeze:            style_emb, style_spatial_id_16, style_tokenizer, body, skip, decoder, output_head
train:             blender
gate:              promote only if global style rises or Hayao rises without LPIPS/grid damage
```

Result:

```text
fo14_depthwise_blender_only_e2 failed before optimization:
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

Correction:

```text
This is not an OOM or optimizer problem. In the active `transport_adain` mode,
`StyleBlender`'s named trainable parameters (`alpha`, `conv`, `mod_mapper`) are
not part of the actual frequency transport branch. The branch is currently
function-style: it applies low/mid/high gates, token allocations, flattening,
and depthwise-filter deltas without using those module weights. Therefore
`blender-only` is a dead trainable surface.
```

Next launch row:

```text
fo16_depthwise_gate_only_e2
source checkpoint: exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/checkpoint_fisher_operator_tokenizer.pt
source adapter:    exp/fisher_operator_tokenizer_probe/fo11_depthwise_filter_swd80/style_adapter.pt
model change:      enable two-scalar learnable mid/high gate on depthwise grammar operator
freeze:            fo11 tokenizer, style identity, body, skip, decoder, output_head
train:             token_depthwise_filter_gate_logits
gate:              promote only if style/Hayao improves without LPIPS/grid damage
```

## Logging Rule

Any new experiment must add one CSV row and one table row with:

```text
id, status, config, purpose, freeze policy, metrics, visual gate, verified claim, adjustment
```

Do not launch a follow-up if it cannot be explained by one sentence tied to a
previous verified claim.

Additional safety rule: tokenizer experiments must not modify the main OMF
loss to rescue a weak reader. If a route needs new loss pressure to look good,
it is a diagnostic, not a mainline improvement.

## 2026-05-28 fo16 Depthwise Gate Result

| id | status | clip_style | LPIPS | Hayao style | claim | adjustment |
|---|---|---:|---:|---:|---|---|
| `fo16_depthwise_gate_only_e2` | completed_diagnostic | 0.710135 | 0.408745 | 0.604389 | The two-scalar depthwise gate is connected and trainable, but too low-rank to move style. It is slightly below `fo11` and `ag02`. | Stop scalar gate-only. Next minimal interface should be gate + output_head or a per-style/per-basis depthwise gate. |

The learned gate moved to approximately mid gain `0.812` and high gain `1.049`
under scale `0.75`. This is useful diagnostics: the optimizer can reach the
operator, but a global two-number gain cannot express style-specific texton
geometry.

## 2026-05-28 fo17 Launch

| id | status | config | purpose | decision gate |
|---|---|---|---|---|
| `fo17_depthwise_gate_head_e2` | pending | resume `fo11`; enable depthwise mid/high gate; train `token_depthwise_filter_gate_logits + output_head`; gate LR multiplier `50x`, output head `1x` | Test the smallest connected consumer surface where depthwise grammar transport can become latent delta. | Promote only if global style beats `ag02` without LPIPS/grid damage. Reject if it stays flat like `fo16` or drifts like `fo12`. |

One-sentence hypothesis: `fo16` reached the operator but had only two scalar
degrees of freedom; adding only `output_head` gives that operator a narrow
image-space decoder without reopening the broad body/decoder repaint path.

Result:

| id | status | clip_style | LPIPS | Hayao style | claim | adjustment |
|---|---|---:|---:|---:|---|---|
| `fo17_depthwise_gate_head_e2` | completed_diagnostic | 0.710782 | 0.438083 | 0.614275 | The output head can consume the depthwise operator enough to lift Hayao, but global style remains below `ag02` and LPIPS worsens by `+0.0308`. The learned gate stays near identity, so the change is mostly output-head repaint. | Reject as mainline. Next step should be per-style/per-basis depthwise gating or style-discriminative assignment, not broader output-head/body training. |

Learned gate after training:

```text
token_depthwise_filter_gate_logits = [0.0162, 0.0123]
with scale 0.75, mid gain ~= 1.012 and high gain ~= 1.009
```

This validates that `output_head` is a connected consumer, but it does not solve
the style representation problem. It selectively improves Hayao/Van Gogh while
damaging content and leaving the averaged global style below the safe anchor.

## 2026-05-28 fo18 Launch

| id | status | config | purpose | decision gate |
|---|---|---|---|---|
| `fo18_depthwise_style_basis_gate_e2` | pending | resume `fo11`; add zero-init `style_id x depthwise_basis` gate inside the grammar operator; train only `token_depthwise_filter_style_basis_gate_logits` | Move expressivity from the shared decoder back into an operator-bound style-local gate. | Promote only if global style beats `ag02` or Hayao rises without LPIPS drift. Reject if it stays flat or needs output-head/body reopening. |

One-sentence hypothesis: `fo17` proved the downstream path can move, but it
moved by repaint; a per-style depthwise-basis gate should change the operator's
style geometry directly while preserving the frozen endpoint decoder.

Result:

| id | status | clip_style | LPIPS | Hayao style | claim | adjustment |
|---|---|---:|---:|---:|---|---|
| `fo18_depthwise_style_basis_gate_e2` | completed_diagnostic | 0.710152 | 0.408805 | 0.604769 | The style-local basis gate is connected and learns nonzero per-style allocations without opening the output-head repaint route, but it remains metric-flat and below `ag02`. | Stop fixed-basis gate-only. The next operator route needs tokenizer grammar/band joint training or a learned style-discriminative operator alphabet. |

Learned gate:

```text
shape = [5, 8]
abs_mean = 0.1611
abs_max  = 0.4795
photo row stays zero
Hayao row = [-0.2890, -0.4140, 0.1270, 0.2541, 0.2792, -0.1566, 0.1301, 0.2890]
```

This corrects the hypothesis: the style-local gate is not silent. It moves in a
style-specific pattern while preserving the frozen endpoint decoder. The
failure is that the current fixed depthwise basis is not sufficiently
style-discriminative to raise the global style measure.

## 2026-05-28 fo19 Superseded Before Launch

| id | status | config | purpose | decision gate |
|---|---|---|---|---|
| `fo19_depthwise_style_basis_token_joint_e2` | superseded_before_launch | resume `fo18`; train `token_depthwise_filter_style_basis_gate_logits + style_tokenizer.grammar_vocab + style_tokenizer.band_vocab`; freeze style identity source, backbone, decoder and output head | Test whether `fo18` is flat because the frozen `fo11` tokenizer coordinates point into the wrong depthwise-basis directions. | Superseded because it still assumes the fixed Sobel/Laplace alphabet is adequate. |

Reason: this is too incremental after `fo18`. The stronger theory read is that
the alphabet itself is wrong or too small, not merely that the tokenizer points
into it poorly.

## 2026-05-28 fo20 Launch

| id | status | config | purpose | decision gate |
|---|---|---|---|---|
| `fo20_learned_style_operator_alphabet_e2` | running full remote run, PID `26476` | resume `fo18`; train style-local zero-mean high-pass depthwise basis deltas plus tokenizer grammar/band assignment; freeze style identity source, backbone, decoder and output head | Test whether the bottleneck is the fixed depthwise operator alphabet rather than tokenizer assignment or output-head consumption. | Promote if global style beats `ag02` or Hayao rises without LPIPS drift. Reject if learned alphabet remains flat; then the bottleneck is upstream style source/OT assignment. |

One-sentence hypothesis: `fo18` proved the gate is alive but the fixed
Sobel/Laplace basis is not style-discriminative enough; learn a constrained
high-pass operator alphabet while keeping the endpoint decoder frozen.

Training constraints:

```text
loss: unchanged main OMF/SWD
train: token_depthwise_filter_style_basis_delta, token_depthwise_filter_style_basis_gate_logits,
       style_tokenizer.grammar_vocab, style_tokenizer.band_vocab
freeze: style_emb, style_spatial_id_16, body, skip path, decoder, output_head
operator constraint: style-local 3x3 kernel deltas are zero-mean high-pass residuals
LR: grammar 4.5e-4, band 1.575e-4, style-basis gate 1.35e-3, basis delta 6.75e-4
source: fo18 epoch_0002 checkpoint
launch: 2026-05-28 14:53 on remote 3060 after deleting two-batch smoke output
```
