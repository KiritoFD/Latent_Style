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
weight"; it should expose an executable macro flat-plane / clean-contour
operator that these fields can control.

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
- therefore the current texton/flatten carrier exposes too weak a Hayao
  operator. The next controlled change should be a tokenizer-driven
  flat-plane / contour branch rather than target-style reweighting.

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
