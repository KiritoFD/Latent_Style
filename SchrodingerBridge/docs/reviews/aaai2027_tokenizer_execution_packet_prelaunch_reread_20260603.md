# AAAI 2027 Tokenizer Execution Packet Prelaunch Reread

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: prelaunch reread of the tokenizer code-to-execution alignment packet only

## Verdict

This packet is **directionally aligned with the intended claim boundary**, but in its current form it does **not automatically close that boundary**. It is a plausible narrowing probe, not yet a self-sealing paper packet.

The packet can support a narrow post-run claim only if the landed evidence shows that code-space separation and executed separation were measured on a genuinely matched source set, with source-content dominance either controlled or shown not to dominate the result.

## 1) Does the packet actually close the intended claim boundary?

**Not by design alone.** At present it can at most test the boundary; it does not guarantee closure.

Why:

1. The intended claim is about whether tokenizer distinctions **survive execution strongly enough to predict real no-op-adjusted style gain**.
2. The script measures:
   - tokenizer code geometry,
   - generated delta geometry,
   - style-wise `delta_idt`,
   - and their correlations.
3. That is the right family of readouts, but the current implementation still leaves open a major alternative explanation:
   - executed geometry may be dominated by mixed source-content effects rather than style survival itself.

So the packet is capable of producing useful evidence, but it is **not yet a guaranteed closure packet** for the paper-side renderer-vs-tokenizer diagnosis.

## 2) Hidden path / logging / measurement holes

### A. Source-set matching is underspecified

The protocol says the probe should use `the same source set used by current Distinct5 paper-facing runs`, but the script currently draws content latents from `--latent-root` and then takes the first `N` files per style.

Reviewer risk:

- unless the exact content file list is logged and matched to the paper-facing evaluation scope, this can be attacked as a different source pool rather than a true paper-side probe.

### B. Source-content dominance is not explicitly controlled

The protocol itself says the README packet must state whether source-content dominance was removed or controlled.  
The current script averages deltas over a mixed content pool, but it does **not**:

- remove per-source means,
- report same-source cross-style separation directly,
- or emit a dedicated content-dominance diagnostic.

Reviewer risk:

- a weak executed-separation result could still be attributed to content mixing rather than genuine execution attenuation.

### C. Launch provenance is incomplete

The manifest says the exact `.pt` checkpoint path must be resolved at launch, because the local repository does not retain the payload.  
That is a real prelaunch hole until it is concretely logged.

Also, the script summary currently records:

- checkpoint path,
- latent root,
- eval/idt CSV paths,
- classes,
- outputs and correlations,

but it does **not** record several critical run-contract details explicitly enough for paper use, including:

- exact selected content paths,
- `delta_probe_max_content_per_style`,
- `delta_probe_batch_size`,
- `delta_probe_num_steps`,
- `delta_probe_step_size`,
- `delta_probe_style_strength`,
- execution device/runtime metadata,
- commit or resolved-config provenance.

Reviewer risk:

- rerunability and exact probe identity are weaker than they should be for a paper-facing mechanism packet.

### D. Manifest / protocol / script contract is slightly loose

The manifest lists required switches, but it omits the required `--output-dir` switch even though the script requires it.

Also, the protocol asks for a short landed README packet with a paper-safe conclusion boundary; the script does not generate that artifact by itself.

Reviewer risk:

- the packet can land technically while still remaining manuscript-unsafe because the narrative boundary artifact is missing.

### E. Correlation evidence will be low-sample and fragile

On Distinct5 this packet ultimately operates over only:

- 5 stylewise points, and
- 10 pairwise style pairs.

Reviewer risk:

- any manuscript use based on one attractive correlation alone will be easy to attack as underpowered unless multiple readouts agree.

## 3) What must be true after the run before the paper can cite it

Before any manuscript use, **all** of the following should be true:

1. **Exact provenance is logged**
   - exact remote checkpoint path,
   - exact latent root,
   - exact content-file list used in the delta probe,
   - all delta-probe arguments,
   - and the output directory contents.

2. **Source-set equivalence is explicit**
   - the run must use the same paper-facing Distinct5 source set, or the landed README must explicitly justify any deviation.

3. **Source-content dominance is addressed**
   - either the packet includes an explicit control/removal analysis,
   - or the landed conclusion must remain narrow enough to say only that the probe is suggestive rather than closing execution attenuation.

4. **Multiple readouts tell the same story**
   - code geometry,
   - executed delta geometry,
   - pairwise code-to-delta alignment,
   - and relation to `delta_idt`
   must converge on the same interpretation.

5. **The paper uses only the supported boundary**
   - positive case:
     - `code-space separation exists, but executed separation weakens materially and tracks style gain weakly`
   - or negative case:
     - `this packet does not isolate execution attenuation cleanly enough, so tokenizer weakness remains plausible`

## Narrow paper-safe bottom line

Prelaunch, this is a **good next mechanism packet**, but not yet a manuscript-ready closure.  
If it lands without stronger source-set/provenance/control logging, the safest paper use will still be limited to:

- `the packet probes executed style survival directly`,
- not
- `the packet proves renderer-side attenuation is the next bottleneck`.
