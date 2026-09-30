# Phase-616 Only Execution Plan

Date: 2026-06-16

## 1. Goal

Build the new 616 line directly from the three new docs:

- `docs/616/design.md`
- `docs/616/debug.md`
- `docs/616/infra.md`

This plan does not inherit any phase-2 narrative, promotion rule, or architectural compromise except where code compatibility requires a default-off switch.

Primary target:

- push transfer style toward `0.74`
- keep transfer LPIPS around `0.30-0.35`
- prioritize style over LPIPS once structure stays inside the acceptable band

Non-negotiable constraints:

- train from scratch, no promoted resume parent for the new main line
- run from remote WSL on EXT4, not from `/mnt/i/`
- keep all new mechanisms behind switches and preserve legacy checkpoint compatibility
- use training-time `CLIP-S + LPIPS` eval as the convergence authority
- use white-box probes as first-class convergence evidence, not just post-hoc plots
- do not dilute the 616 design into an OT-only or solver-only partial rewrite

## 2. What The Three 616 Docs Require

After digesting `design.md`, `debug.md`, and `infra.md`, the practical reading is:

1. The current bottleneck is not only solver stochasticity.
   The training target itself is structurally contaminated because minibatch OT produces targets with horizontal drift.

2. The new line must be full-stack.
   It must cover:
   - OT matching quality
   - vertical target geometry
   - stochastic fiber solver behavior
   - tokenizer expressivity
   - global statistics alignment
   - white-box diagnostics
   - EXT4/remote infra

3. The execution order must still be staged.
   Full-stack does not mean uncontrolled. It means all 616 mechanisms are in scope, but introduced in dependency order.

4. The debug doc is not optional.
   White-box probes are part of the method, not a side utility.

5. The infra doc changes the operating contract.
   Remote training must live on WSL EXT4, with evaluation optimized enough to support checkpoint-by-checkpoint convergence reads.

## 3. Theory To Preserve In Code

The theory docs contain some maximal forms. For execution, we keep the core ideas but implement them in stability-first order.

### 3.1 Fiber Geometry

Retained:

- content structure is the base manifold
- style lives in fiber directions
- TopoGate is a useful practical content-preserving gate, even if it is not a strict mathematical Ehresmann connection

Engineering interpretation:

- "vertical" means high-pass or structure-preserving target construction in latent space
- "horizontal contamination" means OT-matched targets inject structure displacement into the supervised target

### 3.2 OT Diagnosis

Retained:

- current OT can collapse toward mediocre or hub targets
- latent/L2-only similarity is too weak for unpaired style transfer
- balanced Sinkhorn can force bad matches in small minibatches

Engineering interpretation:

- implement a practical structure-aware cost first
- expose the full path toward routing-affinity/GW-like matching as a switch family
- replace balanced-only GPU Sinkhorn with a relaxed-marginal variant

### 3.3 Solver Diagnosis

Retained:

- deterministic ODE style tends toward conditional averages
- stochasticity is likely needed for stronger brushstroke emergence

Engineering interpretation:

- stochastic solver scans remain important, but they should be tested only after OT and target geometry are cleaned

### 3.4 Tokenizer Diagnosis

Retained:

- lookup-style spatial maps are likely actuation-limited
- content-conditioned linear transforms are more faithful than fixed style-value lookup

Engineering interpretation:

- `AffineConnectionTokenizer` is part of the 616 target design, not an optional appendix
- however, it should land after the OT and vertical-target scaffolding it depends on

### 3.5 Photometry Diagnosis

Retained:

- brightness and contrast mismatch are not accidental decoder noise
- normalization layers and low-pass anchoring can suppress style photometry transfer
- global statistics and local texture should be modeled separately

Engineering interpretation:

- add an explicit global-stats alignment path
- do not rely on late-stage heuristic affine hacks as the main answer

## 4. Unified 616 Plan

## 4.1 Phase 0: Infra And Observability First

Purpose:

- make the later experiment conclusions trustworthy

Required changes:

- remote codebase runs from WSL EXT4 clone
- checkpoint and logs stay on EXT4 during training
- eval remains in-training, not post-hoc only
- all retained checkpoints get fast transfer eval

Instrumentation to add or standardize:

- OT probes
  - `ot_plan_entropy`
  - `ot_barycentric_entropy`
  - `ot_target_gini`
  - `ot_target_max_mass`
  - `ot_cost_mean`
  - `ot_cost_var`
  - `ot_structure_cost_mean`
  - `ot_appearance_cost_mean`
- tokenizer probes
  - `spatial_svd_entropy`
  - `effective_experts`
  - `translation_delta_from_identity`
  - `offdiag_cosine` where applicable
- solver probes
  - `trajectory_curvature`
  - `velocity_norm_t`
  - `drift_norm_t`
  - `noise_norm_t`
- fiber leakage probes
  - `base_structural_drift`
  - `fiber_energy_ratio`
  - lightweight `low_freq_leak`

Artifacts:

- per-run `summary.json`
- machine-readable `numeric_debug.jsonl`
- checkpoint curve csv/json

Do not proceed past Phase 0 until:

- training and eval are running from EXT4
- probes are being recorded
- fast eval completes without blocking the lane excessively

## 4.2 Phase 1: OT Repair

Purpose:

- reduce many-to-one collapse and make matched targets structurally more meaningful before touching the objective

Mechanism A: structure-aware OT cost

- keep current appearance transport cost as one branch
- add a structure branch built from latent low-pass, edge energy, and content complexity
- optional later extension:
  - tokenizer-routing-derived structure descriptors

Implementation contract:

- new switch family, default off
- first form can be additive blended cost
- the interface must leave room for richer routing-affinity or GW-style costs

Proposed config fields:

- `bridge.ot_structure_cost_mode = none | lowedge | routing_affinity`
- `bridge.ot_structure_cost_weight`
- `bridge.ot_structure_lowpass_kernel`
- `bridge.ot_structure_edge_weight`

Mechanism B: unbalanced Sinkhorn

- add GPU-side relaxed-marginal coupling
- no CPU Hungarian in the active line

Proposed config fields:

- `bridge.coupling_solver = sinkhorn | sinkhorn_unbalanced | hungarian`
- `bridge.sinkhorn_unbalanced_tau_src`
- `bridge.sinkhorn_unbalanced_tau_tgt`

Success criteria:

- lower `ot_target_gini`
- lower hubness without exploding OT variance
- same-band or better `transfer CLIP-S / LPIPS`

## 4.3 Phase 2: Vertical Target Geometry

Purpose:

- stop supervising the network with structurally displaced endpoints

Mechanism:

- project matched targets into a source-anchored geometry before bridge construction

First implementation:

- explicit vertical target construction from content base + target fiber
- support low-pass and wavelet-like base/fiber splits
- keep legacy objective path only as a compatibility control

Target contract:

- low-frequency structure remains content-anchored
- high-frequency style remains target-driven

Proposed config family:

- `bridge.training_target_projection_mode = legacy | source_low_target_high | wavelet_source_low_target_high | pure_vertical_flow`
- `bridge.training_target_projection_kernel`
- `bridge.training_target_projection_low_anchor`
- `bridge.training_target_projection_low_mode`

Success criteria:

- lower `base_structural_drift`
- lower `low_freq_leak`
- style does not collapse relative to Phase-1 OT repair

## 4.4 Phase 3: Solver Validation

Purpose:

- test whether stochasticity becomes useful once OT and target geometry are no longer dirty

Order:

1. eval-only stochastic scans to validate the noise directionality contract
2. scratch training with the best solver on top of the vertical-target line

Families:

- `solver_unsb_cycle` with fiber-aligned noise
- `solver_pc` as structure-preserving stochastic corrector baseline
- `solver_i2sb` once the phase-2 geometry line is stable

Rule:

- no solver conclusion is valid if tested on a geometrically dirty target line

Success criteria:

- higher style at acceptable LPIPS cost
- improved `fiber_energy_ratio` without runaway `base_structural_drift`

## 4.5 Phase 4: Tokenizer Upgrade

Purpose:

- increase style actuation capacity only after the transport path is trustworthy

Tokenizer order:

1. control tokenizer for compatibility
2. `smoe_translator` as the translation-style intermediate
3. `affine_connection_tokenizer` as the full 616 tokenizer target

Rationale:

- `smoe_translator` preserves continuity better than lookup values and is already fiber-compatible
- `affine_connection_tokenizer` is the destination architecture because it acts directly on fiber features rather than replacing them with static style vectors

Observability requirements:

- `translation_delta_from_identity`
- `effective_experts`
- `spatial_svd_entropy`
- `offdiag_cosine`

Success criteria:

- style lift that persists after full-curve convergence
- no severe LPIPS band escape

## 4.6 Phase 5: Global Statistics Alignment

Purpose:

- address brightness and contrast mismatch only after the transport and tokenizer lines are understood

Interpretation of the design doc:

- do not start with more post-hoc hacks
- first use minimal latent statistic alignment that is explicitly modeled and measurable

Execution choice:

- implement an explicit stats-alignment path that can operate in normalized space with style-conditioned mean/std restoration
- keep legacy appearance alignment only as compatibility fallback, not as the main 616 answer

## 5. Strict Execution Order

The execution queue is:

1. Phase 0 infra and probes
2. Phase 1 OT repair
3. Phase 2 vertical target geometry
4. Phase 3 solver scans and solver training
5. Phase 4 tokenizer upgrade
6. Phase 5 stats-alignment refinement
7. integration run from scratch

What we will not do:

- no resumed parent for the new 616 mainline
- no hiding behind phase-2 retained parents as the reference architecture
- no skipping debug probes because scalar metrics look acceptable
- no post-hoc-only evaluation culture

## 6. First Concrete Implementation Bundle

The first code bundle to ship under this plan is:

1. EXT4 remote run contract
2. white-box probe plumbing
3. structure-aware OT cost
4. unbalanced Sinkhorn
5. vertical target projection family

This is the minimum bundle that makes later solver and tokenizer results interpretable.

## 7. Convergence And Decision Rules

For each family:

- save each retained checkpoint
- run fast transfer `CLIP-S + LPIPS`
- do not close if the current best checkpoint is within the newest 2 retained checkpoints
- close only after patience is exhausted and probes agree with the curve direction

Promotion rule:

- a mechanism is promoted only if the metric curve and white-box probes both improve

Retirement rule:

- a mechanism is retired if it fails matched-control comparison after full curve closure, even if individual cherry-picked points look attractive

## 8. Immediate Next Actions

1. Implement probe logging in the active training path.
2. Add `structure-aware OT cost` as a default-off branch.
3. Add `sinkhorn_unbalanced` as a default-off branch.
4. Add `pure_vertical_flow` target projection mode.
5. Write the first 616 scratch config on the EXT4 remote lane.
6. Launch the first strict-by-plan remote run.

That is the first implementation wave. The next wave is solver and tokenizer, but only after this bundle is stable.
