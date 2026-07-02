# `inmortal.md` Theory Reconciliation for `LANCET`

Date: 2026-06-07

Purpose:

- preserve the mathematical spirit of [inmortal.md](/G:/GitHub/Latent_Style/SchrodingerBridge/docs/inmortal.md)
- translate it into code-safe mechanism families for the current `LANCET/LBM` codebase
- correct the places where the prose is looser than the current implementation reality

## Preserved spirit

The following ideas are kept as the governing theory for the next mechanism round:

1. A single low-energy deterministic field is a bad place to ask for both:
   - semantic structure preservation
   - high-frequency stylization
2. A uniform global `L2` kinetic penalty suppresses the wrong thing:
   - it penalizes structure drift and texture injection in the same norm
3. OT target jitter can raise variance and encourage conservative low-energy solutions.
4. The right ceiling-push direction is:
   - low-energy transport for structure and coarse color
   - high-frequency refinement for texture and brushstroke detail

## Corrected readings

### 1. Current mainline is not pure flow matching

The current paper-facing mainline is effectively an endpoint-transport regime with:

- `w_flow = 0.0`
- terminal SA-SWD pressure
- kinetic regularization
- few-step integration pressure

So the `conditional expectation collapse` story from raw pointwise flow matching is only a partial explanation here. The more accurate diagnosis is:

- endpoint transport is kept low-energy,
- isotropic kinetic suppresses useful high-frequency motion,
- terminal style pressure alone is not enough to force a rich field,
- and few-step integration further rewards smooth, low-risk motion.

### 2. Direct EMA target replacement is too collapse-prone

The line

`v_target = EMA(z_matched) - z_c`

should not be used as the sole supervised target in the current system.

Reason:

- style targets are multi-modal
- direct EMA across matched targets risks averaging away valid style modes
- that would reinforce the same mean-collapse pathology we are trying to break

Corrected implementation target:

- keep structure-aware OT
- add barycentric target smoothing
- add EMA target teacher only as an auxiliary stabilizer

### 3. “Do not remove kinetic” is correct, but kinetic must be redefined

The code should not interpret `release high frequency` as `drop kinetic`.

Correct reading:

- remove global uniform dominance of `||v||^2`
- replace it with:
  - low-frequency dominant kinetic
  - optional high-frequency weak kinetic
  - optional edge-aware anisotropic kinetic
  - optional Jacobian / Stokes smoothing

### 4. Proximal refinement must be high-pass constrained

The refinement branch must not be allowed to absorb the whole task.

If it does, the transport field collapses and the refinement branch becomes a bypass.

Correct implementation:

- transport produces `z_base`
- proximal branch produces residual `delta`
- refined output is:
  - `z_final = z_base + highpass(delta)`

## Resulting implementation families

This reconciliation defines the formal mechanism families for the next round:

1. kinetic rewrite
   - spatial split
   - spectral split
   - manifold-adaptive split
2. structure-aware penalties
   - anisotropic
   - stokes / jacobian
3. proximal refinement
   - high-pass residual
   - norm-free modulation residual
   - cross-attention texture residual
4. target variance reduction
   - structure-aware OT cost
   - barycentric target smoothing
   - EMA teacher auxiliary
   - queue-side smoothing bundle

## Claim discipline

Future experiment notes should describe the mechanism round as:

- a ceiling-push mechanism program
- grounded by `inmortal.md`
- but implemented under the corrected theory contract above

They should not claim that the raw `EMA(z_matched)` target replacement itself is theoretically validated for the current codebase.
