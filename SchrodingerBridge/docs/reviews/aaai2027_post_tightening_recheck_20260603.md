# AAAI 2027 Post-Tightening Recheck

Date: 2026-06-03  
Lane: `adversarial_review`  
Scope: manuscript-only recheck after the latest tightening pass; no new experiment evidence added

## 1. Remaining overclaims

There are no longer major broad overclaims of the earlier kind. The remaining risk is narrow and concentrated in two places:

### A. SA-SWD still carries mild contribution-positioning risk

Location:

- `Our contributions` bullet for `A semantic-aligned terminal-matching design used in the current mainline (SA-SWD)`

Why it still matters:

- the wording is much safer than before;
- however, Gate B is still open, so placing SA-SWD in contribution space can still be read as slightly ahead of fully isolated evidence.

Reviewer-safe reading:

- acceptable only as `current mainline design`;
- not yet safe as a closed semantic-axis necessity claim.

### B. Tokenizer bottleneck wording is still a little broader than the tested family

Locations:

- `Abstract`: `the current measured bottleneck is less about raw tokenizer size than about faithful style execution through the latent renderer`
- `Related Work` / tokenizer discussion: `our current probes point to renderer-side style execution, rather than raw tokenizer size alone, as the immediate bottleneck`
- `Conclusion`: `Current probes suggest that the bottleneck is less about raw token capacity ...`

Why it still matters:

- this is now mostly safe, but the evidence is still concentrated in the current Distinct5 tokenizer family and current renderer setting;
- a skeptical reviewer could still ask for stronger scope pinning such as `within the tested Distinct5 tokenizer variants`.

Bottom line:

- residual overclaim risk is now **mild**, not structural;
- the manuscript is materially safer than it was before this tightening pass.

## 2. Should a new formal review-cycle row be logged now?

Recommendation: **no new formal review-cycle row yet; log only after Gate B closes or if the manuscript materially re-expands its claims again.**

Reason:

- this pass is primarily wording-tightening, not evidence movement;
- it does not appear to change the current formal state already captured by `R20260603C`:
  - overall stance remains `weak_reject`
  - claim safety remains `narrow_only`
  - evidence closure remains `partial`
- adding a new formal row now would mostly create bookkeeping noise rather than a new decision boundary.

This memo is sufficient as the manuscript-only checkpoint between `R20260603C` and the next evidence-changing event.

## 3. Next single most important blocker

The next single most important blocker is still:

- **Gate B: matched semantic-vs-random SA-SWD isolation**

Why this is the blocker:

- Gate A is already negatively closed;
- Gate C is still open, but Gate B is the remaining mechanism-side blocker that most directly controls whether SA-SWD can remain a differentiated contribution or must be downgraded to a current-mainline design choice;
- once Gate B closes, the manuscript can either keep the narrowed SA-SWD contribution line with cleaner support or further demote it if the matched control is weak/null.

Reviewer-safe action boundary:

- if the abnormal random-axis run finishes with clean matched summaries, Gate B may close on **quality-only** evidence;
- if not, a rerun remains necessary before elevating the SA-SWD mechanism claim.

## Addendum after final wording-tightening pass

After the subsequent manuscript-only edit that:

- recast SA-SWD explicitly as a `documented current-mainline design choice`, and
- pinned tokenizer bottleneck wording to the `tested Distinct5 tokenizer variants`
  `in this setting`,

the adversarial spot-check no longer found any structural manuscript-only
overclaim in these lines.

Updated manuscript-only reading:

- residual risk is now `very mild`, not material;
- no further paper-only wording edit is required before the next evidence move;
- the next meaningful blocker remains Gate B rather than another local prose
  tweak.
