# AAAI 2027 Theory Owner Audit - 2026-06-03

Source: continuous theory-owner lane (`Darwin`)

This memo separates object-level truth from paper-safe wording. Its purpose is
to stop the manuscript from over-claiming what current experiments actually
prove.

## Global verdict

The safest and most accurate current statement is not:

- `latent-space MSE is broadly wrong`

but rather:

- pointwise Euclidean reconstruction is high-risk when it is used as the
  **style-endpoint alignment objective**
- L2 / MSE remains natural for **energy**, **velocity**, and some
  **teacher-student** or **local chart** objectives

The completed `mse / huber / l1` Distinct5 trio does not currently arbitrate
that thesis because the resolved configs keep:

- `objective_mode = omf`
- `w_flow = 0.0`

Under the active implementation, changing `loss_type` therefore does not engage
the intended branch. The trio should be treated as a near-null operational
control, not as metric-theory evidence.

## Whole-model mathematical map

1. `style tokenizer`
   - style id to control code
   - decides what style is requested, not how much motion is safe

2. `LANCET backbone`
   - content-conditioned latent vector-field executor
   - turns style control into latent motion or endpoint delta

3. `OT endpoint construction`
   - training-side unpaired endpoint constructor
   - not a full global solver claim

4. `active Distinct5 objective family`
   - object-level reading is closer to OMF-style endpoint-delta training than a
     clean textbook random-time flow-matching family

5. `kinetic regularization`
   - L2 acts as energy or action regularization
   - this is a natural use of L2, not the problematic endpoint-style use

6. `SA-SWD terminal matching`
   - distributional endpoint-style pressure
   - distinct from pointwise target-latent reconstruction

7. `evaluation`
   - `CLIP-S`: target-style affinity
   - `LPIPS`: content displacement proxy
   - `ArtFID`: broad artifact or realism diagnostic
   - `idt`: unchanged-image control
   - `delta_idt`: no-op-adjusted style gain, especially important on Distinct5

## Well-supported theory

- endpoint style pressure should come from OT plus terminal distribution
  matching rather than paired latent endpoint reconstruction
- kinetic regularization shapes the style-content tradeoff
- raw `CLIP-S` on Distinct5 must be interpreted together with `idt` and
  `delta_idt`
- tokenizer weakness currently looks more like an execution bottleneck than a
  raw code-capacity bottleneck

## Under-supported or wrong theory

Under-supported:

- `SA-SWD semantic axes are necessary` without a semantic-vs-random matched
  ablation
- `endpoint kinetic approximates full path energy` beyond a weak local reading
- `next tokenizer must be carrier + risk gate` as anything stronger than a
  hypothesis

Wrong or over-claimed:

- `all latent-space MSE terms are bad`
- `current mse/huber/l1 Distinct5 runs prove the broader latent metric thesis`
- `current Distinct5 result family is already a clean empirical proof of the
  random-time flow-matching story`
- `ArtFID` as a direct target-style gain metric
- strong universal speedup rhetoric from operating-point wall-clock observations

## Paper-safe claim boundary

Safe:

- not all latent-space L2 terms are problematic
- the main risk is using pointwise Euclidean endpoint reconstruction as the
  style-alignment objective
- measured gains are consistent with separating endpoint style supervision from
  path regularity
- Distinct5 requires unchanged-image controls for safe CLIP-style reading

Unsafe:

- `Huber/L1 beat MSE in latent space`
- `the flow-loss ablation proves Euclidean latent metrics are wrong`
- `current active Distinct5 family empirically validates the broad latent
  metric thesis`
- `22x speedup`-style rhetoric without normalized time-to-parity

## Minimum experiment set implied by the theory audit

Remote 3060 required:

1. endpoint-objective metric ablation on the actual endpoint-style object
2. semantic-vs-random SA-SWD axis ablation
3. path-statistics probe for kinetic or trajectory claims

Local analysis or rewriting only:

1. split OMF and FM wording in the method section
2. harden metric semantics in the evaluation section
3. rewrite efficiency claims into operating-point wall-clock language

Do not spend more remote budget on:

- additional seeds for the current `mse / huber / l1` trio in its invalidated
  form
